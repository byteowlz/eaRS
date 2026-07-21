//! transcribe.cpp streaming ASR engine.
//!
//! Wraps the `transcribe-cpp` bindings (ggml-based GGUF models such as
//! multitalker-parakeet or nemotron-3.5). Audio arrives from the server at
//! 24 kHz and is resampled to the model's native rate; the stream API returns
//! a committed/tentative text split. Append-only clients consume committed
//! eaRS `Word` events; revisable-preview clients consume authoritative
//! `Interim { text }` snapshots (`committed + tentative`).
//!
//! The library allows one active stream per loaded model (the stream holds
//! the model's compute lease from begin to finalize), so this engine serves
//! one session at a time regardless of `--max-sessions`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, select, unbounded};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use transcribe_cpp::{Model as TcModel, RunOptions, StreamOptions};

use crate::server::engine::{Engine, EngineKind, EngineSession};
use crate::server::{SessionSink, current_timestamp};
use crate::{TranscriptionSink, WebSocketMessage, WordTimestamp};

const SERVER_SAMPLE_RATE: usize = 24_000;
/// kaudio resampler needs a minimum input window; 1.6k floor.
const RESAMPLE_MIN_SAMPLES: usize = 1_600;
/// Feed the stream in 560 ms slices (matches the benchmark configuration).
const FEED_CHUNK_MS: usize = 560;

/// A known streaming GGUF model, downloadable by short slug from the
/// `handy-computer` Hugging Face org. This is data, not judgment: each entry is
/// just a repo + default quant, mirroring handy's bundled catalog. An explicit
/// file path always bypasses this table, and any `handy-computer/*-gguf` repo
/// works by passing its slug even if it is not listed here.
use crate::server::catalog;

const HF_ORG: &str = "handy-computer";

/// Download a catalog model (or `slug@QUANT`) into the shared HF cache and
/// return its on-disk path. Reuses the same resolution as the engine loader.
///
/// # Errors
///
/// Returns an error if the download fails.
pub fn pull(spec: &str) -> Result<PathBuf> {
    resolve_model(Path::new(spec))
}

/// Resolve a `--transcribe-cpp-model` spec to an on-disk GGUF path.
///
/// - An existing file path is used verbatim.
/// - Otherwise the spec is treated as a catalog slug (optionally `slug@QUANT`)
///   and the matching GGUF is downloaded from `handy-computer/{slug}-gguf`
///   into the shared Hugging Face cache on first use, like handy does.
fn resolve_model(spec: &Path) -> Result<PathBuf> {
    if spec.is_file() {
        return Ok(spec.to_path_buf());
    }
    let raw = spec.to_string_lossy();
    // A path-looking spec that does not exist is a mistake, not a slug.
    if raw.contains('/') || raw.ends_with(".gguf") {
        bail!(
            "transcribe.cpp model file not found: {}\n\
             Pass an existing .gguf path, or a known model slug ({})",
            spec.display(),
            catalog_slugs()
        );
    }
    let (slug, quant) = match raw.split_once('@') {
        Some((s, q)) => (s.to_string(), Some(q.to_string())),
        None => (raw.to_string(), None),
    };
    let quant = quant
        .or_else(|| {
            catalog::load()
                .into_iter()
                .find(|m| m.slug == slug)
                .map(|m| m.default_quant)
        })
        .unwrap_or_else(|| "Q8_0".to_string());
    let repo_id = format!("{HF_ORG}/{slug}-gguf");
    let filename = format!("{slug}-{quant}.gguf");
    eprintln!("[transcribe-cpp] resolving model {slug} ({quant}) from {repo_id}");
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("failed to build Hugging Face API client")?;
    let path = api.model(repo_id.clone()).get(&filename).with_context(|| {
        format!(
            "failed to download {filename} from {repo_id}\n\
             (known slugs: {}; or pass an explicit .gguf path)",
            catalog_slugs()
        )
    })?;
    Ok(path)
}

fn catalog_slugs() -> String {
    catalog::load()
        .into_iter()
        .map(|m| m.slug)
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone)]
pub struct TranscribeCppEngineConfig {
    /// Either an existing `.gguf` path or a catalog slug (auto-downloaded).
    pub model_path: PathBuf,
    /// Language hint (e.g. "de"). "auto"/None lets the model autodetect.
    pub lang: Option<String>,
}

pub struct TranscribeCppEngine {
    model: TcModel,
    lang: Option<String>,
    native_sample_rate: usize,
    supported_languages: Vec<String>,
    semaphore: Arc<Semaphore>,
}

fn normalize_lang(lang: Option<&str>) -> Option<String> {
    match lang {
        None | Some("auto") | Some("") => None,
        Some(l) => Some(l.to_string()),
    }
}

impl TranscribeCppEngine {
    pub fn load(cfg: TranscribeCppEngineConfig) -> Result<Self> {
        let model_path = resolve_model(&cfg.model_path)?;
        let model = TcModel::load(&model_path).map_err(|e| {
            anyhow::anyhow!(
                "transcribe.cpp model load failed for {}: {e}",
                model_path.display()
            )
        })?;
        let caps = model.capabilities();
        if !caps.supports_streaming {
            bail!(
                "model {} ({}) does not support streaming; pick a streaming-capable GGUF \
                 (e.g. multitalker-parakeet-streaming or nemotron-3.5-asr-streaming)",
                model_path.display(),
                model.arch()
            );
        }
        let lang = normalize_lang(cfg.lang.as_deref());
        eprintln!(
            "[transcribe-cpp] loaded {} ({}, backend {}, {} Hz, lang={}, languages={:?})",
            model_path.display(),
            model.arch(),
            model.backend(),
            caps.native_sample_rate,
            lang.as_deref().unwrap_or("auto"),
            caps.languages,
        );
        Ok(Self {
            native_sample_rate: caps.native_sample_rate as usize,
            supported_languages: caps.languages,
            model,
            lang,
            // One active stream per model: the stream holds the compute lease.
            semaphore: Arc::new(Semaphore::new(1)),
        })
    }
}

impl Engine for TranscribeCppEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::TranscribeCpp
    }

    fn allocate(&self, sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
        let permit = match self.semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };

        let model = self.model.clone();
        let lang = self.lang.clone();
        let native_rate = self.native_sample_rate;
        let (audio_tx, audio_rx) = unbounded::<Vec<f32>>();
        let (lang_tx, lang_rx) = unbounded::<String>();
        let (control_tx, control_rx) = unbounded::<TranscribeCppControl>();
        let handle = TranscribeCppSessionHandle {
            audio_tx,
            lang_tx,
            control_tx,
            supported_languages: self.supported_languages.clone(),
        };

        std::thread::spawn(move || {
            if let Err(err) = run_transcribe_cpp_session(
                model,
                lang,
                native_rate,
                audio_rx,
                lang_rx,
                control_rx,
                sink,
                permit,
            ) {
                eprintln!("[transcribe-cpp] session failed: {err:#}");
            }
        });

        Ok(Some(Box::new(handle) as Box<dyn EngineSession>))
    }
}

#[derive(Debug)]
enum TranscribeCppControl {
    Stop,
    /// Acoustic end-of-utterance from the server ingress VAD. Finalize the
    /// current native stream, emit its complete snapshot, then begin a fresh
    /// stream while keeping the client session alive.
    UtteranceEnd,
}

struct TranscribeCppSessionHandle {
    audio_tx: Sender<Vec<f32>>,
    lang_tx: Sender<String>,
    control_tx: Sender<TranscribeCppControl>,
    supported_languages: Vec<String>,
}

impl EngineSession for TranscribeCppSessionHandle {
    fn engine(&self) -> EngineKind {
        EngineKind::TranscribeCpp
    }

    fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.audio_tx
            .send(pcm)
            .context("failed to send audio chunk to transcribe-cpp engine")
    }

    fn set_language(&self, lang: String) -> Result<()> {
        let normalized = lang.to_ascii_lowercase();
        let resolved = if normalized == "auto" {
            Some("auto".to_string())
        } else {
            self.supported_languages
                .iter()
                .find(|candidate| candidate.eq_ignore_ascii_case(&normalized))
                .or_else(|| {
                    let prefix = format!("{normalized}-");
                    self.supported_languages
                        .iter()
                        .find(|candidate| candidate.to_ascii_lowercase().starts_with(&prefix))
                })
                .cloned()
        };
        let resolved = resolved.with_context(|| {
            format!(
                "transcribe-cpp model does not support language {lang:?}; supported: {:?}",
                self.supported_languages
            )
        })?;
        self.lang_tx
            .send(resolved)
            .context("failed to send language command to transcribe-cpp engine")
    }

    fn send_speech_boundary(&self, active: bool) -> Result<bool> {
        if active {
            return Ok(false);
        }
        self.control_tx
            .send(TranscribeCppControl::UtteranceEnd)
            .context("failed to send utterance boundary to transcribe-cpp engine")?;
        // The engine owns Speech(false): it emits it only after finalization so
        // trailing words cannot arrive after the gateway commits the turn.
        Ok(true)
    }

    fn request_stop(&self) {
        let _ = self.control_tx.send(TranscribeCppControl::Stop);
    }

    fn supports_language(&self) -> bool {
        !self.supported_languages.is_empty()
    }
}

/// Tracks how much of the stream's committed text has been emitted as words,
/// holding back a trailing fragment that does not end on whitespace.
struct CommittedEmitter {
    emitted_bytes: usize,
    pending: String,
}

impl CommittedEmitter {
    fn new() -> Self {
        Self {
            emitted_bytes: 0,
            pending: String::new(),
        }
    }

    /// Absorb newly committed text and return words ready to emit.
    fn absorb(&mut self, committed: &str, flush: bool) -> Vec<String> {
        if committed.len() > self.emitted_bytes {
            self.pending.push_str(&committed[self.emitted_bytes..]);
            self.emitted_bytes = committed.len();
        }
        let boundary = if flush {
            Some(self.pending.len())
        } else {
            self.pending
                .char_indices()
                .rev()
                .find(|(_, ch)| ch.is_whitespace())
                .map(|(idx, ch)| idx + ch.len_utf8())
        };
        let Some(boundary) = boundary else {
            return Vec::new();
        };
        let ready = self.pending[..boundary].to_string();
        self.pending.replace_range(..boundary, "");
        ready.split_whitespace().map(str::to_string).collect()
    }
}

fn emit_interim(text: String, sink: &mut SessionSink) {
    if !text.trim().is_empty() {
        sink.handle_message(WebSocketMessage::Interim { text });
    }
}

fn emit_words(
    words: Vec<String>,
    start_time: f64,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
) {
    for word in words {
        sink.handle_message(WebSocketMessage::Word {
            word: word.clone(),
            start_time,
            end_time: None,
        });
        committed_words.push(WordTimestamp {
            word,
            start_time,
            end_time: None,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn run_transcribe_cpp_session(
    model: TcModel,
    mut lang: Option<String>,
    native_rate: usize,
    audio_rx: Receiver<Vec<f32>>,
    lang_rx: Receiver<String>,
    control_rx: Receiver<TranscribeCppControl>,
    mut sink: SessionSink,
    _permit: OwnedSemaphorePermit,
) -> Result<()> {
    let feed_samples = native_rate * FEED_CHUNK_MS / 1000;
    let mut session = model.session().context("failed to create session")?;
    let mut buffer_24k: Vec<f32> = Vec::new();
    let mut buffer_native: Vec<f32> = Vec::new();
    let mut committed_words: Vec<WordTimestamp> = Vec::new();
    let mut total_input_secs: f64 = 0.0;
    let mut stop_requested = false;

    'stream: while !stop_requested {
        let run = RunOptions {
            language: lang.clone(),
            ..Default::default()
        };
        let mut stream = session
            .stream(&run, &StreamOptions::default())
            .context("failed to begin stream")?;
        let mut emitter = CommittedEmitter::new();
        let mut pending_lang: Option<String> = None;
        let mut utterance_end_requested = false;

        loop {
            select! {
                recv(audio_rx) -> msg => match msg {
                    Ok(chunk) => buffer_24k.extend_from_slice(&chunk),
                    Err(_) => stop_requested = true,
                },
                recv(lang_rx) -> msg => match msg {
                    Ok(new_lang) => {
                        // Session-scoped language change: finalize the current
                        // stream at an utterance boundary and restart with the
                        // new language hint.
                        pending_lang = Some(new_lang);
                    }
                    Err(_) => stop_requested = true,
                },
                recv(control_rx) -> msg => match msg {
                    Ok(TranscribeCppControl::UtteranceEnd) => utterance_end_requested = true,
                    Ok(TranscribeCppControl::Stop) | Err(_) => stop_requested = true,
                },
                default(Duration::from_millis(10)) => {}
            }

            // A file client can outpace the feed loop and its Stop can win the
            // select race against queued audio; always drain audio first so
            // stop/finalize never discards buffered speech.
            for chunk in audio_rx.try_iter() {
                buffer_24k.extend_from_slice(&chunk);
            }

            while buffer_24k.len() >= RESAMPLE_MIN_SAMPLES {
                let drained: Vec<f32> = buffer_24k.drain(..).collect();
                match kaudio::resample(&drained, SERVER_SAMPLE_RATE, native_rate) {
                    Ok(samples) => {
                        total_input_secs += samples.len() as f64 / native_rate as f64;
                        buffer_native.extend_from_slice(&samples);
                    }
                    Err(err) => eprintln!("[transcribe-cpp] resample failed: {err}"),
                }
            }

            while buffer_native.len() >= feed_samples {
                let chunk: Vec<f32> = buffer_native.drain(..feed_samples).collect();
                match stream.feed(&chunk) {
                    Ok(update) => {
                        if std::env::var("EARS_DEBUG_ENGINE").is_ok() {
                            let t = stream.text();
                            eprintln!(
                                "[transcribe-cpp] feed: committed={:?} tentative={:?} changed={}",
                                t.committed.len(),
                                t.tentative,
                                update.committed_changed
                            );
                        }
                        if update.committed_changed {
                            let words = emitter.absorb(&stream.text().committed, false);
                            emit_words(words, total_input_secs, &mut committed_words, &mut sink);
                        }
                        if update.committed_changed || update.tentative_changed {
                            emit_interim(stream.text().display(), &mut sink);
                        }
                    }
                    Err(err) => eprintln!("[transcribe-cpp] stream feed failed: {err}"),
                }
            }

            if stop_requested || pending_lang.is_some() || utterance_end_requested {
                // Feed the resampled tail, then finalize to promote all
                // remaining tentative text.
                if !buffer_native.is_empty() {
                    let tail = std::mem::take(&mut buffer_native);
                    if let Err(err) = stream.feed(&tail) {
                        eprintln!("[transcribe-cpp] tail feed failed: {err}");
                    }
                }
                match stream.finalize() {
                    Ok(_) => {
                        let final_text = stream.text().full;
                        let words = emitter.absorb(&final_text, true);
                        emit_words(words, total_input_secs, &mut committed_words, &mut sink);
                        emit_interim(final_text, &mut sink);
                    }
                    Err(err) => eprintln!("[transcribe-cpp] finalize failed: {err}"),
                }
                drop(stream);

                if utterance_end_requested {
                    sink.handle_message(WebSocketMessage::Speech {
                        active: false,
                        timestamp: current_timestamp(),
                    });
                    eprintln!("[transcribe-cpp] utterance finalized; restarting stream");
                    continue 'stream;
                }

                if let Some(new_lang) = pending_lang.take() {
                    lang = normalize_lang(Some(&new_lang));
                    sink.handle_message(WebSocketMessage::LanguageChanged {
                        lang: new_lang.clone(),
                    });
                    eprintln!("[transcribe-cpp] switched session language to {new_lang}");
                    continue 'stream;
                }
                break 'stream;
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
        words: committed_words,
    });
    sink.close();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_emitter_holds_trailing_fragment() {
        let mut e = CommittedEmitter::new();
        assert_eq!(e.absorb("hello wor", false), vec!["hello".to_string()]);
        assert_eq!(
            e.absorb("hello world again", false),
            vec!["world".to_string()]
        );
        assert_eq!(
            e.absorb("hello world again", true),
            vec!["again".to_string()]
        );
        assert!(e.absorb("hello world again", true).is_empty());
    }

    #[test]
    fn committed_emitter_never_splits_a_word_before_a_real_boundary() {
        let mut e = CommittedEmitter::new();
        assert!(e.absorb("Irwi", false).is_empty());
        assert_eq!(e.absorb("Irwin ", false), vec!["Irwin".to_string()]);
    }

    #[test]
    fn session_rejects_unsupported_language_without_sending_command() {
        let (audio_tx, _audio_rx) = unbounded();
        let (lang_tx, lang_rx) = unbounded();
        let (control_tx, _control_rx) = unbounded();
        let handle = TranscribeCppSessionHandle {
            audio_tx,
            lang_tx,
            control_tx,
            supported_languages: vec!["en-US".to_string(), "de-DE".to_string()],
        };
        assert!(handle.set_language("xx".to_string()).is_err());
        assert!(lang_rx.try_recv().is_err());
        assert!(handle.set_language("en".to_string()).is_ok());
        assert_eq!(lang_rx.try_recv().unwrap(), "en-US");
        assert!(handle.set_language("de-DE".to_string()).is_ok());
        assert_eq!(lang_rx.try_recv().unwrap(), "de-DE");
        assert!(handle.set_language("auto".to_string()).is_ok());
        assert_eq!(lang_rx.try_recv().unwrap(), "auto");
    }

    #[test]
    fn normalize_lang_auto_is_none() {
        assert_eq!(normalize_lang(None), None);
        assert_eq!(normalize_lang(Some("auto")), None);
        assert_eq!(normalize_lang(Some("")), None);
        assert_eq!(normalize_lang(Some("de")), Some("de".to_string()));
    }
}
