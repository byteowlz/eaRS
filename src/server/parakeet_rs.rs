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

use std::path::{Path, PathBuf};
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
/// Energy gate: a 560 ms chunk counts as speech if any 50 ms window exceeds
/// this RMS. Windowed so a quiet utterance onset in the chunk tail still
/// registers.
const SPEECH_RMS_THRESHOLD: f32 = 0.0025;
const SPEECH_RMS_WINDOW: usize = 800; // 50 ms @ 16 kHz
/// Gate the model after this many consecutive silent chunks (~2.2 s). Long
/// silence saturates Nemotron's rolling encoder cache with silence statistics,
/// which mangles the first words after the pause; instead we stop feeding,
/// reset the model, and replay one chunk of pre-roll when speech resumes.
const SILENCE_GATE_AFTER_CHUNKS: usize = 4;

const HF_REPO: &str = "altunenes/parakeet-rs";
const HF_MULTILINGUAL_DIR: &str = "nemotron-3.5-asr-streaming-0.6b-onnx";
const HF_ENGLISH_DIR: &str = "nemotron-speech-streaming-en-0.6b";
const MODEL_FILES: [&str; 4] = [
    "encoder.onnx",
    "encoder.onnx.data",
    "decoder_joint.onnx",
    "tokenizer.model",
];

/// Download any missing Nemotron model files from Hugging Face into
/// `model_dir`. The variant is picked from the directory name: names
/// containing "speech-streaming-en" map to the English-only export, anything
/// else to the multilingual 3.5 export.
fn ensure_model_files(model_dir: &Path) -> Result<()> {
    let missing: Vec<&str> = MODEL_FILES
        .iter()
        .copied()
        .filter(|f| !model_dir.join(f).is_file())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let subfolder = match model_dir.file_name().and_then(|n| n.to_str()) {
        Some(name) if name.contains("speech-streaming-en") => HF_ENGLISH_DIR,
        _ => HF_MULTILINGUAL_DIR,
    };
    eprintln!(
        "[parakeet-rs] {} model file(s) missing in {}; downloading from {HF_REPO}/{subfolder}",
        missing.len(),
        model_dir.display()
    );
    std::fs::create_dir_all(model_dir)
        .with_context(|| format!("failed to create model dir {}", model_dir.display()))?;
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("failed to build Hugging Face API client")?;
    let repo = api.model(HF_REPO.to_string());
    for file in missing {
        let cached = repo
            .get(&format!("{subfolder}/{file}"))
            .with_context(|| format!("failed to download {subfolder}/{file} from {HF_REPO}"))?;
        // hf_hub snapshots are relative symlinks into its blobs dir; resolve
        // them so we link/copy the actual file instead of a dangling symlink.
        let cached = std::fs::canonicalize(&cached)
            .with_context(|| format!("failed to resolve {}", cached.display()))?;
        let target = model_dir.join(file);
        if target.is_symlink() {
            std::fs::remove_file(&target)?;
        }
        if std::fs::hard_link(&cached, &target).is_err() {
            std::fs::copy(&cached, &target).with_context(|| {
                format!("failed to copy {} to {}", cached.display(), target.display())
            })?;
        }
    }
    Ok(())
}

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
        if let Err(err) = ensure_model_files(&cfg.model_dir) {
            eprintln!("[parakeet-rs] auto-download failed: {err:#}");
        }
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

        let multilingual = matches!(model.mode(), NemotronMode::Multilingual);
        let (audio_tx, audio_rx) = unbounded::<Vec<f32>>();
        let (lang_tx, lang_rx) = unbounded::<String>();
        let (control_tx, control_rx) = unbounded::<ParakeetRsControl>();
        let handle = ParakeetRsSessionHandle {
            audio_tx,
            lang_tx,
            control_tx,
            multilingual,
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
    multilingual: bool,
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
        self.multilingual
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
    let mut silent_run: usize = 0;
    let mut gated = false;
    // Last silent chunk seen while gated; replayed as pre-roll on speech resume.
    let mut preroll: Option<Vec<f32>> = None;

    while !stop_requested {
        select! {
            recv(audio_rx) -> msg => match msg {
                Ok(chunk) => buffer_24k.extend_from_slice(&chunk),
                Err(_) => stop_requested = true,
            },
            recv(lang_rx) -> msg => match msg {
                Ok(lang) => {
                    // A language prompt is session-scoped, like Kyutai's
                    // set_language. Treat a live change as an utterance boundary:
                    // commit pending text, discard at most one unprocessed chunk,
                    // and reset cache/decoder state before applying the prompt.
                    emit_delta(
                        "",
                        true,
                        total_input_secs,
                        &mut pending_text,
                        &mut committed_words,
                        &mut sink,
                    );
                    buffer_24k.clear();
                    buffer_16k.clear();
                    match model.set_target_lang(&lang) {
                        Ok(()) => {
                            model.reset();
                            sink.handle_message(WebSocketMessage::LanguageChanged {
                                lang: lang.clone(),
                            });
                            eprintln!("[parakeet-rs] switched session language to {lang}");
                        }
                        Err(err) => {
                            eprintln!("[parakeet-rs] set_language({lang}) failed: {err}");
                        }
                    }
                }
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

        // Feed the model fixed 560 ms chunks, gating out long silence.
        while buffer_16k.len() >= CHUNK_SAMPLES_16K {
            let chunk: Vec<f32> = buffer_16k.drain(..CHUNK_SAMPLES_16K).collect();
            let speech = chunk_has_speech(&chunk);
            if gated {
                if !speech {
                    preroll = Some(chunk);
                    continue;
                }
                gated = false;
                silent_run = 0;
                eprintln!("[parakeet-rs] speech resumed, replaying pre-roll");
                if let Some(pre) = preroll.take() {
                    feed_chunk(
                        &mut model,
                        &pre,
                        total_input_secs,
                        &mut pending_text,
                        &mut committed_words,
                        &mut sink,
                        &mut last_nonempty_delta_at,
                    );
                }
            } else if speech {
                silent_run = 0;
            } else {
                silent_run += 1;
                if silent_run >= SILENCE_GATE_AFTER_CHUNKS {
                    eprintln!(
                        "[parakeet-rs] {}+ chunks of silence, gating model input",
                        SILENCE_GATE_AFTER_CHUNKS
                    );
                    emit_delta(
                        "",
                        true,
                        total_input_secs,
                        &mut pending_text,
                        &mut committed_words,
                        &mut sink,
                    );
                    model.reset();
                    gated = true;
                    preroll = Some(chunk);
                    continue;
                }
            }
            feed_chunk(
                &mut model,
                &chunk,
                total_input_secs,
                &mut pending_text,
                &mut committed_words,
                &mut sink,
                &mut last_nonempty_delta_at,
            );
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

fn chunk_has_speech(chunk: &[f32]) -> bool {
    chunk.chunks(SPEECH_RMS_WINDOW).any(|w| {
        let mean_sq = w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32;
        mean_sq.sqrt() > SPEECH_RMS_THRESHOLD
    })
}

fn feed_chunk(
    model: &mut Nemotron,
    chunk: &[f32],
    total_input_secs: f64,
    pending_text: &mut String,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
    last_nonempty_delta_at: &mut Instant,
) {
    match model.transcribe_chunk(chunk) {
        Ok(text) => {
            if !text.is_empty() {
                *last_nonempty_delta_at = Instant::now();
            }
            emit_delta(
                &text,
                false,
                total_input_secs,
                pending_text,
                committed_words,
                sink,
            );
        }
        Err(err) => eprintln!("[parakeet-rs] transcribe_chunk failed: {err}"),
    }
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
