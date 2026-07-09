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
use std::time::{Duration, Instant};

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
/// Wait longer than one 560 ms Nemotron chunk before treating an unbounded
/// trailing fragment as a completed word during a genuine speaking pause.
const PARTIAL_WORD_FLUSH_TIMEOUT: Duration = Duration::from_millis(800);

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
        let handle = NemotronHandle::from_pretrained(&cfg.model_dir, None).map_err(|e| {
            anyhow::anyhow!(
                "parakeet-rs model load failed for {}: {e}",
                cfg.model_dir.display()
            )
        })?;
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
    // Nemotron returns incremental SentencePiece fragments. Hold the trailing
    // unbounded fragment so it is not typed as a separate word (e.g. `spli` +
    // `t` becoming `spli t`).
    let mut pending_text = String::new();
    let mut last_nonempty_delta_at = Instant::now();
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
                Ok(text) => {
                    if !text.is_empty() {
                        last_nonempty_delta_at = Instant::now();
                    }
                    emit_delta(
                        &text,
                        false,
                        total_input_secs,
                        &mut pending_text,
                        &mut committed_words,
                        &mut sink,
                    );
                }
                Err(err) => eprintln!("[parakeet-rs] transcribe_chunk failed: {err}"),
            }
        }

        if should_flush_pending(&pending_text, last_nonempty_delta_at, Instant::now()) {
            eprintln!(
                "[parakeet-rs] flushing trailing fragment after {}ms without new text",
                PARTIAL_WORD_FLUSH_TIMEOUT.as_millis()
            );
            emit_delta(
                "",
                true,
                total_input_secs,
                &mut pending_text,
                &mut committed_words,
                &mut sink,
            );
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
                        emit_delta(
                            &text,
                            false,
                            total_input_secs,
                            &mut pending_text,
                            &mut committed_words,
                            &mut sink,
                        );
                    }
                }
                for _ in 0..3 {
                    match model.transcribe_chunk(&zeros) {
                        Ok(text) => emit_delta(
                            &text,
                            false,
                            total_input_secs,
                            &mut pending_text,
                            &mut committed_words,
                            &mut sink,
                        ),
                        Err(err) => eprintln!("[parakeet-rs] flush transcribe_chunk failed: {err}"),
                    }
                }
            }
            emit_delta(
                "",
                true,
                total_input_secs,
                &mut pending_text,
                &mut committed_words,
                &mut sink,
            );

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

fn should_flush_pending(pending_text: &str, last_delta_at: Instant, now: Instant) -> bool {
    !pending_text.is_empty() && now.duration_since(last_delta_at) >= PARTIAL_WORD_FLUSH_TIMEOUT
}

/// Buffer a transcription delta and emit only text ending at a word boundary.
///
/// SentencePiece deltas are not guaranteed to end on a word boundary. For
/// example, consecutive calls can return `" spli"` and `"t weird"`. Holding
/// the final fragment until the next whitespace avoids the dictation client
/// typing `spli t` with an irreversible space between the fragments.
fn emit_delta(
    text: &str,
    flush_pending: bool,
    start_time: f64,
    pending_text: &mut String,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
) {
    pending_text.push_str(text);
    let boundary = if flush_pending {
        Some(pending_text.len())
    } else {
        pending_text
            .char_indices()
            .rev()
            .find(|(_, ch)| ch.is_whitespace())
            .map(|(idx, ch)| idx + ch.len_utf8())
    };
    let Some(boundary) = boundary else {
        return;
    };

    let ready = pending_text[..boundary].to_string();
    pending_text.replace_range(..boundary, "");
    for token in ready.split_whitespace() {
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
    fn partial_word_timeout_waits_for_one_model_chunk_then_flushes() {
        let now = Instant::now();
        let chunk_duration =
            Duration::from_secs_f64(CHUNK_SAMPLES_16K as f64 / NEMOTRON_SAMPLE_RATE as f64);
        assert!(PARTIAL_WORD_FLUSH_TIMEOUT > chunk_duration);
        assert!(!should_flush_pending("split", now, now + chunk_duration));
        assert!(should_flush_pending(
            "split",
            now,
            now + PARTIAL_WORD_FLUSH_TIMEOUT,
        ));
        assert!(!should_flush_pending(
            "",
            now,
            now + PARTIAL_WORD_FLUSH_TIMEOUT,
        ));
    }

    #[test]
    fn emit_delta_holds_and_rejoins_split_word_fragments() {
        let mut words = Vec::new();
        let mut sink = make_sink();
        let mut pending = String::new();

        emit_delta(" spli", false, 1.0, &mut pending, &mut words, &mut sink);
        assert!(words.is_empty());
        assert_eq!(pending, "spli");

        emit_delta("t weird", false, 1.1, &mut pending, &mut words, &mut sink);
        assert_eq!(words[0].word, "split");
        assert_eq!(pending, "weird");

        emit_delta("ly fast ", false, 1.2, &mut pending, &mut words, &mut sink);
        emit_delta("again.", false, 1.3, &mut pending, &mut words, &mut sink);
        emit_delta("", true, 1.4, &mut pending, &mut words, &mut sink);

        let ws: Vec<String> = words.into_iter().map(|w| w.word).collect();
        assert_eq!(ws, vec!["split", "weirdly", "fast", "again."]);
        assert!(pending.is_empty());
    }
}
