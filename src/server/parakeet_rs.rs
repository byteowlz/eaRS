//! parakeet-rs (Nemotron) streaming ASR engine.
//!
//! Uses the stateful cache-aware Nemotron streaming model from `parakeet-rs`.
//! Each session owns an independent `Nemotron` decoder state while the ONNX
//! session is shared through a single `NemotronHandle`. Mic audio arrives from
//! the server at 24 kHz and is resampled to 16 kHz; 560 ms chunks are fed to
//! `Nemotron::transcribe_chunk`, which returns finalized text deltas that we
//! emit as eaRS `Word` events.
//!
//! See reports/parakeet-rs-spike/streaming_models_report.md for the model
//! benchmarks that selected Nemotron over the 120M EOU model.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, select, unbounded};
use parakeet_rs::{Nemotron, NemotronHandle, NemotronMode};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::server::SessionSink;
use crate::server::engine::{Engine, EngineKind, EngineSession};
use crate::{TranscriptionOptions, TranscriptionSink, WebSocketMessage, WordTimestamp};

const SERVER_SAMPLE_RATE: usize = 24_000;
const NEMOTRON_SAMPLE_RATE: usize = 16_000;
/// Nemotron streaming chunk: 560 ms at 16 kHz (matches the upstream example
/// and parakeet-rs's CHUNK_SIZE mel-frame schedule).
const CHUNK_SAMPLES_16K: usize = 8_960;
/// kaudio resampler needs a minimum input window; mirror sherpa's 1.6k floor.
const RESAMPLE_MIN_SAMPLES: usize = 1_600;

#[derive(Debug, Clone)]
pub struct ParakeetRsEngineConfig {
    pub model_dir: PathBuf,
    /// Target language for the multilingual variant (e.g. "de", "auto").
    /// Ignored by the English-only variant.
    pub lang: Option<String>,
}

pub struct ParakeetRsEngine {
    handle: Arc<NemotronHandle>,
    lang: Option<String>,
    #[allow(dead_code)]
    options: TranscriptionOptions,
    semaphore: Arc<Semaphore>,
}

impl ParakeetRsEngine {
    pub fn load(
        cfg: ParakeetRsEngineConfig,
        options: TranscriptionOptions,
        max_sessions: usize,
    ) -> Result<Self> {
        let handle = NemotronHandle::from_pretrained(&cfg.model_dir, None)
            .map_err(|e| anyhow::anyhow!("parakeet-rs model load failed for {}: {e}", cfg.model_dir.display()))?;
        let variant = match handle.mode() {
            NemotronMode::Multilingual => "multilingual",
            NemotronMode::EnglishOnly => "english-only",
        };
        eprintln!(
            "[parakeet-rs] loaded {variant} model from {} ({})",
            cfg.model_dir.display(),
            match cfg.lang.as_deref() {
                Some(lang) => format!("lang={lang}"),
                None => "lang=auto/default".to_string(),
            }
        );
        Ok(Self {
            handle: Arc::new(handle),
            lang: cfg.lang,
            options,
            semaphore: Arc::new(Semaphore::new(max_sessions.max(1))),
        })
    }
}

impl Engine for ParakeetRsEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::ParakeetRs
    }

    fn allocate(&self, sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
        let permit = match self.semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };

        // Each session gets an independent decoder state over the shared model.
        let mut model = Nemotron::from_shared(&self.handle);
        if let Some(lang) = self.lang.as_deref() {
            if matches!(model.mode(), NemotronMode::Multilingual) {
                if let Err(err) = model.set_target_lang(lang) {
                    eprintln!("[parakeet-rs] set_target_lang({lang}) failed: {err}");
                }
            }
        }

        let (audio_tx, audio_rx) = unbounded::<Vec<f32>>();
        let (lang_tx, lang_rx) = unbounded::<String>();
        let (control_tx, control_rx) = unbounded::<ParakeetRsControl>();
        let handle = ParakeetRsSessionHandle {
            audio_tx,
            lang_tx,
            control_tx,
        };

        std::thread::spawn(move || {
            run_parakeet_rs_session(model, audio_rx, lang_rx, control_rx, sink, permit);
        });

        Ok(Some(Box::new(handle) as Box<dyn EngineSession>))
    }
}

#[derive(Debug)]
enum ParakeetRsControl {
    Stop,
}

struct ParakeetRsSessionHandle {
    audio_tx: Sender<Vec<f32>>,
    lang_tx: Sender<String>,
    control_tx: Sender<ParakeetRsControl>,
}

impl ParakeetRsSessionHandle {
    pub fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.audio_tx
            .send(pcm)
            .context("failed to send audio chunk to parakeet-rs engine")
            .map(|_| ())
    }

    pub fn set_language(&self, lang: String) -> Result<()> {
        self.lang_tx
            .send(lang)
            .context("failed to send language command to parakeet-rs engine")
    }

    pub fn request_stop(&self) {
        let _ = self.control_tx.send(ParakeetRsControl::Stop);
    }
}

impl Drop for ParakeetRsSessionHandle {
    fn drop(&mut self) {
        // Drop closes the senders; the session loop observes the disconnect.
    }
}

impl EngineSession for ParakeetRsSessionHandle {
    fn engine(&self) -> EngineKind {
        EngineKind::ParakeetRs
    }

    fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        ParakeetRsSessionHandle::send_audio(self, pcm)
    }

    fn set_language(&self, lang: String) -> Result<()> {
        ParakeetRsSessionHandle::set_language(self, lang)
    }

    fn request_stop(&self) {
        ParakeetRsSessionHandle::request_stop(self);
    }

    fn supports_language(&self) -> bool {
        true
    }
}

#[allow(clippy::too_many_arguments)]
fn run_parakeet_rs_session(
    mut model: Nemotron,
    audio_rx: Receiver<Vec<f32>>,
    lang_rx: Receiver<String>,
    control_rx: Receiver<ParakeetRsControl>,
    mut sink: SessionSink,
    _permit: OwnedSemaphorePermit,
) {
    let mut buffer_24k: Vec<f32> = Vec::new();
    let mut buffer_16k: Vec<f32> = Vec::new();
    let mut stop_requested = false;
    let mut total_input_secs: f64 = 0.0;
    let mut committed_words: Vec<WordTimestamp> = Vec::new();
    let _ = lang_rx; // language changes applied at session start; mid-stream ignored for now.

    while !stop_requested {
        select! {
            recv(audio_rx) -> msg => match msg {
                Ok(chunk) => buffer_24k.extend_from_slice(&chunk),
                Err(_) => stop_requested = true,
            },
            recv(control_rx) -> msg => match msg {
                Ok(ParakeetRsControl::Stop) | Err(_) => stop_requested = true,
            },
            default(Duration::from_millis(10)) => {}
        }

        // Resample 24k -> 16k in windows the resampler is happy with.
        while buffer_24k.len() >= RESAMPLE_MIN_SAMPLES {
            let drained: Vec<f32> = buffer_24k.drain(..buffer_24k.len()).collect();
            match kaudio::resample(&drained, SERVER_SAMPLE_RATE, NEMOTRON_SAMPLE_RATE) {
                Ok(samples_16k) => {
                    if !samples_16k.is_empty() {
                        total_input_secs += samples_16k.len() as f64 / NEMOTRON_SAMPLE_RATE as f64;
                        buffer_16k.extend_from_slice(&samples_16k);
                    }
                }
                Err(err) => eprintln!("[parakeet-rs] resample failed: {err}"),
            }
        }

        // Feed the model fixed 560 ms chunks.
        while buffer_16k.len() >= CHUNK_SAMPLES_16K {
            let chunk: Vec<f32> = buffer_16k.drain(..CHUNK_SAMPLES_16K).collect();
            match model.transcribe_chunk(&chunk) {
                Ok(text) => emit_delta(&text, total_input_secs, &mut committed_words, &mut sink),
                Err(err) => eprintln!("[parakeet-rs] transcribe_chunk failed: {err}"),
            }
        }

        if stop_requested {
            // Flush the tail: pad the remainder to a full chunk, then feed a few
            // zero chunks like the upstream streaming example.
            if !buffer_16k.is_empty() || true {
                let zeros = vec![0.0f32; CHUNK_SAMPLES_16K];
                if !buffer_16k.is_empty() {
                    let mut tail = std::mem::take(&mut buffer_16k);
                    tail.resize(CHUNK_SAMPLES_16K, 0.0);
                    if let Ok(text) = model.transcribe_chunk(&tail) {
                        emit_delta(&text, total_input_secs, &mut committed_words, &mut sink);
                    }
                }
                for _ in 0..3 {
                    match model.transcribe_chunk(&zeros) {
                        Ok(text) => emit_delta(&text, total_input_secs, &mut committed_words, &mut sink),
                        Err(err) => eprintln!("[parakeet-rs] flush transcribe_chunk failed: {err}"),
                    }
                }
            }

            let final_text = committed_words
                .iter()
                .map(|w| w.word.clone())
                .collect::<Vec<_>>()
                .join(" ");
            sink.handle_message(WebSocketMessage::Final {
                text: final_text,
                words: committed_words.clone(),
            });
            sink.close();
        }
    }
}

/// Split a transcription delta on whitespace and emit each token as a `Word`.
fn emit_delta(
    text: &str,
    start_time: f64,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    for token in trimmed.split_whitespace() {
        if token.is_empty() {
            continue;
        }
        sink.handle_message(WebSocketMessage::Word {
            word: token.to_string(),
            start_time,
            end_time: None,
        });
        committed_words.push(WordTimestamp {
            word: token.to_string(),
            start_time,
            end_time: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_size_is_560ms_at_16k() {
        let ms = CHUNK_SAMPLES_16K as f64 / NEMOTRON_SAMPLE_RATE as f64 * 1000.0;
        assert!((ms - 560.0).abs() < 1e-6, "chunk must be 560ms, got {ms}ms");
    }

    fn make_sink() -> SessionSink {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        SessionSink::new(tx)
    }

    #[test]
    fn emit_delta_splits_words_and_accumulates() {
        let mut words = Vec::new();
        let mut sink = make_sink();
        emit_delta("  hello   world  ", 1.0, &mut words, &mut sink);
        emit_delta("", 1.1, &mut words, &mut sink);
        emit_delta("again.", 1.2, &mut words, &mut sink);

        let ws: Vec<String> = words.into_iter().map(|w| w.word).collect();
        assert_eq!(ws, vec!["hello", "world", "again."]);
    }
}
