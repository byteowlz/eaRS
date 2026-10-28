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

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, bounded};
use parakeet_rs::{Nemotron, NemotronHandle, NemotronMode};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::server::engine::{Engine, EngineKind, EngineSession, queue_command};
use crate::server::{SessionSink, current_timestamp};
use crate::{TranscriptionOptions, TranscriptionSink, WebSocketMessage, WordTimestamp};

const SERVER_SAMPLE_RATE: usize = 24_000;
const NEMOTRON_SAMPLE_RATE: usize = 16_000;
/// Nemotron streaming chunk: 560 ms at 16 kHz (matches the upstream example
/// and parakeet-rs's CHUNK_SIZE mel-frame schedule).
const CHUNK_SAMPLES_16K: usize = 8_960;
/// kaudio resampler needs a minimum input window; 1.6k floor.
const RESAMPLE_MIN_SAMPLES: usize = 1_600;

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
                format!(
                    "failed to copy {} to {}",
                    cached.display(),
                    target.display()
                )
            })?;
        }
    }
    Ok(())
}

/// ONNX Runtime minor version the pinned `ort` (2.0.0-rc.12 with parakeet-rs's
/// `api-24`) requires; older runtimes are rejected.
const ORT_MIN_MINOR: u32 = 24;

#[cfg(target_os = "macos")]
const ORT_LIB_NAMES: &[&str] = &["libonnxruntime.dylib", "libonnxruntime.1.dylib"];
#[cfg(target_os = "windows")]
const ORT_LIB_NAMES: &[&str] = &["onnxruntime.dll"];
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const ORT_LIB_NAMES: &[&str] = &["libonnxruntime.so", "libonnxruntime.so.1"];

/// Directories searched for ONNX Runtime when `ORT_DYLIB_PATH` is unset:
/// next to the executable (where `just install-all` copies dylibs), its
/// sibling `lib/`, then the usual package-manager prefixes.
fn ort_search_dirs(exe_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = exe_dir {
        dirs.push(dir.to_path_buf());
        dirs.push(dir.join("..").join("lib"));
    }
    for dir in [
        "/opt/homebrew/lib",
        "/usr/local/lib",
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ] {
        dirs.push(PathBuf::from(dir));
    }
    dirs
}

/// First candidate library that exists and passes `probe`, or the reasons
/// every existing candidate was rejected.
fn find_onnx_runtime(
    dirs: &[PathBuf],
    probe: impl Fn(&Path) -> Result<String>,
) -> std::result::Result<(PathBuf, String), Vec<String>> {
    let mut rejected = Vec::new();
    for path in dirs
        .iter()
        .flat_map(|dir| ORT_LIB_NAMES.iter().map(move |name| dir.join(name)))
        .filter(|path| path.is_file())
    {
        match probe(&path) {
            Ok(version) => return Ok((path, version)),
            Err(err) => rejected.push(format!("{err:#}")),
        }
    }
    Err(rejected)
}

/// Load `path` and check its ONNX Runtime version the way `ort` does, but
/// without `ort`: given an unloadable or too-old library, `ort` 2.0.0-rc.12
/// deadlocks in its own API initialisation instead of returning an error.
#[cfg(unix)]
fn probe_onnx_runtime(path: &Path) -> Result<String> {
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::os::unix::ffi::OsStrExt;

    #[repr(C)]
    struct OrtApiBase {
        _get_api: unsafe extern "C" fn(u32) -> *const c_void,
        get_version_string: unsafe extern "C" fn() -> *const c_char,
    }

    let c_path = CString::new(path.as_os_str().as_bytes())?;
    // SAFETY: plain dlopen/dlsym on a NUL-terminated path. The handle stays
    // open on purpose: `ort` loads the same library right afterwards.
    unsafe {
        let handle = libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            let err = libc::dlerror();
            let reason = if err.is_null() {
                "unknown dlopen error".to_string()
            } else {
                CStr::from_ptr(err).to_string_lossy().into_owned()
            };
            anyhow::bail!("{} cannot be loaded: {reason}", path.display());
        }
        let symbol = libc::dlsym(handle, c"OrtGetApiBase".as_ptr());
        if symbol.is_null() {
            anyhow::bail!("{} is not ONNX Runtime (no OrtGetApiBase)", path.display());
        }
        let get_base: unsafe extern "C" fn() -> *const OrtApiBase = std::mem::transmute(symbol);
        let base = get_base();
        if base.is_null() {
            anyhow::bail!("{} returned no ONNX Runtime API", path.display());
        }
        let version = CStr::from_ptr(((*base).get_version_string)())
            .to_string_lossy()
            .into_owned();
        let minor = version
            .split('.')
            .nth(1)
            .and_then(|m| m.parse::<u32>().ok())
            .unwrap_or(0);
        if minor < ORT_MIN_MINOR {
            anyhow::bail!(
                "{} is ONNX Runtime {version}, but >= 1.{ORT_MIN_MINOR} is required",
                path.display()
            );
        }
        Ok(version)
    }
}

#[cfg(not(unix))]
fn probe_onnx_runtime(_path: &Path) -> Result<String> {
    Ok("unchecked".to_string())
}

fn ort_install_hint() -> String {
    format!(
        "parakeet-rs loads ONNX Runtime >= 1.{ORT_MIN_MINOR} at runtime. Install it \
         (macOS: `brew install onnxruntime`; Arch: `pacman -S onnxruntime`) or set \
         ORT_DYLIB_PATH to the library, e.g. ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib"
    )
}

/// Make sure `ort`'s dynamic loader gets a usable ONNX Runtime before the
/// model loads. Left alone, `ort` panics when the bare library name is not on
/// the loader path, which killed the server (trx-zef0).
fn ensure_onnx_runtime() -> Result<(PathBuf, String)> {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            anyhow::bail!(
                "ORT_DYLIB_PATH points to {}, which does not exist. {}",
                path.display(),
                ort_install_hint()
            );
        }
        let version = probe_onnx_runtime(&path)
            .map_err(|err| anyhow::anyhow!("ORT_DYLIB_PATH: {err:#}. {}", ort_install_hint()))?;
        return Ok((path, version));
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let dirs = ort_search_dirs(exe_dir.as_deref());
    let (path, version) = find_onnx_runtime(&dirs, probe_onnx_runtime).map_err(|rejected| {
        let searched = dirs
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if rejected.is_empty() {
            anyhow::anyhow!(
                "ONNX Runtime not found (searched {searched}). {}",
                ort_install_hint()
            )
        } else {
            anyhow::anyhow!(
                "no usable ONNX Runtime found ({}). {}",
                rejected.join("; "),
                ort_install_hint()
            )
        }
    })?;
    // SAFETY: called once during server startup before any session exists;
    // nothing else reads or writes the environment concurrently here.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &path) };
    Ok((path, version))
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
        let (ort_path, ort_version) = ensure_onnx_runtime()?;
        eprintln!(
            "[parakeet-rs] using ONNX Runtime {ort_version} from {}",
            ort_path.display()
        );
        if let Err(err) = ensure_model_files(&cfg.model_dir) {
            eprintln!("[parakeet-rs] auto-download failed: {err:#}");
        }
        // Backstop: any remaining `ort` load panic becomes a startup error
        // instead of a crashed server.
        let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            NemotronHandle::from_pretrained(&cfg.model_dir, None)
        }))
        .map_err(|panic| {
            let reason = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            anyhow::anyhow!(
                "failed to load ONNX Runtime from {}: {reason}. {}",
                ort_path.display(),
                ort_install_hint()
            )
        })?;
        let handle = loaded.map_err(|e| {
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
        // Audio and controls share one FIFO: separate select! channels cannot
        // establish an accepted-audio drain barrier.
        let (tx, rx) = bounded(64);
        let handle = ParakeetRsSessionHandle { tx, multilingual };
        std::thread::spawn(move || {
            let mut sink = sink;
            if let Err(err) = run_parakeet_rs_session(model, rx, &mut sink, permit) {
                eprintln!("[parakeet-rs] session failed: {err:#}");
            }
            sink.close();
        });
        Ok(Some(Box::new(handle)))
    }
}

#[derive(Debug)]
enum ParakeetRsCommand {
    Audio(Vec<f32>),
    Language(String),
    Stop,
    UtteranceEnd,
    Drain(u64),
}

struct ParakeetRsSessionHandle {
    tx: Sender<ParakeetRsCommand>,
    multilingual: bool,
}

impl EngineSession for ParakeetRsSessionHandle {
    fn engine(&self) -> EngineKind {
        EngineKind::ParakeetRs
    }

    fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        queue_command(&self.tx, ParakeetRsCommand::Audio(pcm))
    }

    fn set_language(&self, lang: String) -> Result<()> {
        queue_command(&self.tx, ParakeetRsCommand::Language(lang))
    }

    fn send_speech_boundary(&self, active: bool) -> Result<bool> {
        if active {
            return Ok(false);
        }
        queue_command(&self.tx, ParakeetRsCommand::UtteranceEnd)?;
        Ok(true)
    }

    fn supports_ordered_drain(&self) -> bool {
        true
    }

    fn request_drain(&self, request_id: u64) -> Result<()> {
        if request_id == 0 {
            anyhow::bail!("drain request_id must be nonzero");
        }
        queue_command(&self.tx, ParakeetRsCommand::Drain(request_id))
    }

    fn request_stop(&self) {
        // Bound admission; subsequent handle drop disconnects even a full FIFO.
        let _ = queue_command(&self.tx, ParakeetRsCommand::Stop);
    }

    fn supports_language(&self) -> bool {
        self.multilingual
    }
}

/// Tests use the production FIFO, resampling, tail flush and message emission
/// with a deterministic decoder, without loading ONNX or any audio files.
trait StreamingDecoder {
    fn transcribe_chunk(&mut self, chunk: &[f32]) -> Result<String>;
    fn set_target_lang(&mut self, lang: &str) -> Result<()>;
    fn reset(&mut self);
}

impl StreamingDecoder for Nemotron {
    fn transcribe_chunk(&mut self, chunk: &[f32]) -> Result<String> {
        Ok(Nemotron::transcribe_chunk(self, chunk)?)
    }
    fn set_target_lang(&mut self, lang: &str) -> Result<()> {
        Ok(Nemotron::set_target_lang(self, lang)?)
    }
    fn reset(&mut self) {
        Nemotron::reset(self);
    }
}

fn run_parakeet_rs_session(
    mut model: impl StreamingDecoder,
    rx: Receiver<ParakeetRsCommand>,
    sink: &mut SessionSink,
    _permit: OwnedSemaphorePermit,
) -> Result<()> {
    let mut buffer_24k = Vec::new();
    let mut buffer_16k = Vec::new();
    let mut total_input_secs = 0.0;
    let mut committed_words = Vec::new();
    let mut pending_text = String::new();

    loop {
        // Disconnection is legacy shutdown, never an acknowledged drain.
        let command = rx.recv().unwrap_or(ParakeetRsCommand::Stop);
        if let ParakeetRsCommand::Audio(chunk) = &command {
            buffer_24k.extend_from_slice(chunk);
        }
        // Also resample a short (<1600 sample) tail at every boundary/terminal
        // command. Previously that accepted tail was abandoned on shutdown.
        let flush_input = !matches!(command, ParakeetRsCommand::Audio(_));
        if !buffer_24k.is_empty() && (flush_input || buffer_24k.len() >= RESAMPLE_MIN_SAMPLES) {
            let input = std::mem::take(&mut buffer_24k);
            let samples = kaudio::resample(&input, SERVER_SAMPLE_RATE, NEMOTRON_SAMPLE_RATE)
                .context("parakeet-rs resampling accepted audio")?;
            total_input_secs += input.len() as f64 / SERVER_SAMPLE_RATE as f64;
            buffer_16k.extend(samples);
        }
        while buffer_16k.len() >= CHUNK_SAMPLES_16K {
            let chunk: Vec<f32> = buffer_16k.drain(..CHUNK_SAMPLES_16K).collect();
            feed_chunk(
                &mut model,
                &chunk,
                total_input_secs,
                &mut pending_text,
                &mut committed_words,
                sink,
            )?;
        }
        match command {
            ParakeetRsCommand::Audio(_) => {}
            ParakeetRsCommand::Language(lang) => {
                flush_utterance(
                    &mut model,
                    &mut buffer_16k,
                    total_input_secs,
                    &mut pending_text,
                    &mut committed_words,
                    sink,
                    1,
                )?;
                match model.set_target_lang(&lang) {
                    Ok(()) => {
                        model.reset();
                        sink.handle_message(WebSocketMessage::LanguageChanged { lang });
                    }
                    Err(err) => eprintln!("[parakeet-rs] set_language failed: {err}"),
                }
            }
            ParakeetRsCommand::UtteranceEnd => {
                flush_utterance(
                    &mut model,
                    &mut buffer_16k,
                    total_input_secs,
                    &mut pending_text,
                    &mut committed_words,
                    sink,
                    1,
                )?;
                sink.handle_message(WebSocketMessage::Speech {
                    active: false,
                    timestamp: current_timestamp(),
                });
            }
            terminal @ (ParakeetRsCommand::Stop | ParakeetRsCommand::Drain(_)) => {
                flush_utterance(
                    &mut model,
                    &mut buffer_16k,
                    total_input_secs,
                    &mut pending_text,
                    &mut committed_words,
                    sink,
                    3,
                )?;
                sink.handle_message(WebSocketMessage::Final {
                    text: committed_words
                        .iter()
                        .map(|w| w.word.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    words: committed_words,
                });
                if let ParakeetRsCommand::Drain(request_id) = terminal {
                    sink.handle_message(WebSocketMessage::Drained { request_id });
                }
                return Ok(());
            }
        }
    }
}

fn flush_utterance(
    model: &mut impl StreamingDecoder,
    buffer_16k: &mut Vec<f32>,
    total_input_secs: f64,
    pending_text: &mut String,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
    zero_chunks: usize,
) -> Result<()> {
    // Pad the remaining speech to one model chunk, then feed enough zero chunks
    // to release delayed tokens. Boundaries use one to avoid polluting the
    // retained cache; final shutdown uses upstream's full three-chunk drain.
    if !buffer_16k.is_empty() {
        let mut tail = std::mem::take(buffer_16k);
        tail.resize(CHUNK_SAMPLES_16K, 0.0);
        feed_chunk(
            model,
            &tail,
            total_input_secs,
            pending_text,
            committed_words,
            sink,
        )?;
    }
    let zeros = vec![0.0f32; CHUNK_SAMPLES_16K];
    for _ in 0..zero_chunks {
        feed_chunk(
            model,
            &zeros,
            total_input_secs,
            pending_text,
            committed_words,
            sink,
        )?;
    }
    emit_delta(
        "",
        true,
        total_input_secs,
        pending_text,
        committed_words,
        sink,
    );
    Ok(())
}

fn feed_chunk(
    model: &mut impl StreamingDecoder,
    chunk: &[f32],
    total_input_secs: f64,
    pending_text: &mut String,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
) -> Result<()> {
    // A swallowed decoder error would make a subsequent drain ACK dishonest.
    let text = model
        .transcribe_chunk(chunk)
        .context("parakeet-rs transcribe_chunk")?;
    emit_delta(
        &text,
        false,
        total_input_secs,
        pending_text,
        committed_words,
        sink,
    );
    Ok(())
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

    struct FakeDecoder {
        calls: Arc<std::sync::Mutex<Vec<bool>>>,
        fail_on: Option<usize>,
    }

    impl StreamingDecoder for FakeDecoder {
        fn transcribe_chunk(&mut self, chunk: &[f32]) -> Result<String> {
            assert_eq!(chunk.len(), CHUNK_SAMPLES_16K);
            let mut calls = self.calls.lock().unwrap();
            let speech = chunk.iter().any(|value| value.abs() > 1e-8);
            calls.push(speech);
            if self.fail_on == Some(calls.len()) {
                anyhow::bail!("injected decode failure");
            }
            Ok(if speech { " first" } else { " tail" }.into())
        }
        fn set_target_lang(&mut self, _: &str) -> Result<()> {
            Ok(())
        }
        fn reset(&mut self) {}
    }

    fn fake_worker(
        commands: Vec<ParakeetRsCommand>,
        fail_on: Option<usize>,
    ) -> (Result<()>, Vec<serde_json::Value>, Vec<bool>, usize) {
        let (tx, rx) = bounded(16);
        for command in commands {
            tx.send(command).unwrap();
        }
        drop(tx);
        let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut sink = SessionSink::new(msg_tx);
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let decoder = FakeDecoder {
            calls: calls.clone(),
            fail_on,
        };
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.clone().try_acquire_owned().unwrap();
        let result = run_parakeet_rs_session(decoder, rx, &mut sink, permit);
        // Same unconditional close used by the native worker wrapper.
        sink.close();
        let mut messages = Vec::new();
        let mut closes = 0;
        while let Ok(message) = msg_rx.try_recv() {
            match message {
                tokio_tungstenite::tungstenite::Message::Text(text) => {
                    messages.push(serde_json::from_str(&text).unwrap())
                }
                tokio_tungstenite::tungstenite::Message::Close(_) => closes += 1,
                _ => panic!("unexpected worker output"),
            }
        }
        assert_eq!(closes, 1);
        assert_eq!(semaphore.available_permits(), 1);
        let calls = calls.lock().unwrap().clone();
        (result, messages, calls, closes)
    }

    #[tokio::test]
    async fn websocket_uses_real_parakeet_fifo_and_worker_with_fake_decoder() {
        use futures::{SinkExt, StreamExt};
        use std::time::Duration;
        use tokio_tungstenite::{connect_async, tungstenite::Message};

        struct FakeNativeEngine {
            calls: Arc<std::sync::Mutex<Vec<bool>>>,
            semaphore: Arc<Semaphore>,
        }
        impl Engine for FakeNativeEngine {
            fn kind(&self) -> EngineKind {
                EngineKind::ParakeetRs
            }
            fn allocate(&self, mut sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
                let permit = self.semaphore.clone().try_acquire_owned()?;
                let (tx, rx) = bounded(64);
                let decoder = FakeDecoder {
                    calls: self.calls.clone(),
                    fail_on: None,
                };
                std::thread::spawn(move || {
                    let result = run_parakeet_rs_session(decoder, rx, &mut sink, permit);
                    sink.close();
                    result.unwrap();
                });
                Ok(Some(Box::new(ParakeetRsSessionHandle {
                    tx,
                    multilingual: true,
                })))
            }
        }
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let semaphore = Arc::new(Semaphore::new(1));
        let mut manager = super::super::engine::EngineManager::new();
        manager.register(Arc::new(FakeNativeEngine {
            calls: calls.clone(),
            semaphore: semaphore.clone(),
        }));
        let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = tcp.local_addr().unwrap();
        let registry = super::super::listener::StreamRegistry::new();
        let reg = registry.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = tcp.accept().await.unwrap();
            super::super::run_connection(
                stream,
                manager,
                Some(reg),
                None,
                EngineKind::ParakeetRs,
                TranscriptionOptions {
                    boundary_vad: false,
                    ..Default::default()
                },
                Duration::from_secs(2),
            )
            .await
        });
        let (mut client, _) = connect_async(format!("ws://{addr}")).await.unwrap();
        let pcm: Vec<u8> = vec![0.1f32; 800]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        client.send(Message::binary(pcm)).await.unwrap();
        client
            .send(Message::text(r#"{"type":"drain","request_id":42}"#))
            .await
            .unwrap();
        let mut messages: Vec<serde_json::Value> = Vec::new();
        loop {
            match tokio::time::timeout(Duration::from_secs(3), client.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
            {
                Message::Text(text) => messages.push(serde_json::from_str(&text).unwrap()),
                Message::Close(_) => break,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(messages[0]["engine"], "parakeet-rs");
        assert_eq!(
            messages[1]["features"],
            serde_json::json!(["dictation-drain/1"])
        );
        let final_idx = messages.iter().position(|m| m["type"] == "final").unwrap();
        assert_eq!(messages[final_idx]["text"], "first tail tail tail");
        assert_eq!(
            messages[final_idx + 1],
            serde_json::json!({"type":"drained","request_id":42})
        );
        assert_eq!(messages.len(), final_idx + 2);
        tokio::time::timeout(Duration::from_secs(3), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(registry.list_active_streams().is_empty());
        assert_eq!(semaphore.available_permits(), 1);
        assert_eq!(*calls.lock().unwrap(), [true, false, false, false]);
    }

    #[test]
    fn drain_flushes_short_accepted_tail_before_final_and_ack() {
        let (result, messages, calls, _) = fake_worker(
            vec![
                ParakeetRsCommand::Audio(vec![0.1; 800]),
                ParakeetRsCommand::Drain(42),
                // Must never be decoded after the terminal FIFO command.
                ParakeetRsCommand::Audio(vec![0.7; 24_000]),
                ParakeetRsCommand::Drain(99),
            ],
            None,
        );
        result.unwrap();
        assert_eq!(calls, [true, false, false, false]);
        let final_idx = messages.iter().position(|m| m["type"] == "final").unwrap();
        assert_eq!(messages[final_idx]["text"], "first tail tail tail");
        assert_eq!(
            messages[final_idx + 1],
            serde_json::json!({"type":"drained","request_id":42})
        );
        assert_eq!(final_idx + 2, messages.len());
    }

    #[test]
    fn fifo_boundaries_and_language_commands_cannot_overtake_audio_or_drain() {
        let (result, messages, calls, _) = fake_worker(
            vec![
                ParakeetRsCommand::Audio(vec![0.1; 800]),
                ParakeetRsCommand::UtteranceEnd,
                ParakeetRsCommand::Language("de".into()),
                ParakeetRsCommand::Audio(vec![0.2; 800]),
                ParakeetRsCommand::Drain(7),
            ],
            None,
        );
        result.unwrap();
        assert_eq!(calls, [true, false, false, true, false, false, false]);
        let speech = messages.iter().position(|m| m["type"] == "speech").unwrap();
        let language = messages
            .iter()
            .position(|m| m["type"] == "languagechanged")
            .unwrap();
        let final_idx = messages.iter().position(|m| m["type"] == "final").unwrap();
        assert!(speech < language && language < final_idx);
        assert_eq!(messages.last().unwrap()["request_id"], 7);
    }

    #[test]
    fn decoder_failure_during_audio_or_tail_never_acks() {
        for fail_on in [1, 4] {
            let (result, messages, _, _) = fake_worker(
                vec![
                    ParakeetRsCommand::Audio(vec![0.1; 24_000]),
                    ParakeetRsCommand::Drain(42),
                ],
                Some(fail_on),
            );
            assert!(result.is_err());
            assert!(!messages.iter().any(|m| m["type"] == "drained"));
            assert!(!messages.iter().any(|m| m["type"] == "final"));
        }
    }

    #[test]
    fn legacy_stop_and_disconnection_never_ack() {
        for commands in [
            vec![
                ParakeetRsCommand::Audio(vec![0.1; 800]),
                ParakeetRsCommand::Stop,
            ],
            vec![ParakeetRsCommand::Audio(vec![0.1; 800])],
        ] {
            let (result, messages, calls, _) = fake_worker(commands, None);
            result.unwrap();
            assert_eq!(calls, [true, false, false, false]);
            assert_eq!(messages.last().unwrap()["type"], "final");
            assert!(!messages.iter().any(|m| m["type"] == "drained"));
        }
    }

    #[test]
    fn session_handle_queues_audio_boundary_language_and_drain_in_order() {
        let (tx, rx) = bounded(4);
        let handle = ParakeetRsSessionHandle {
            tx,
            multilingual: true,
        };
        assert!(handle.supports_ordered_drain());
        assert!(handle.request_drain(0).is_err());
        assert!(rx.is_empty());
        handle.send_audio(vec![0.1]).unwrap();
        handle.send_speech_boundary(false).unwrap();
        handle.set_language("de".into()).unwrap();
        handle.request_drain(42).unwrap();
        drop(handle);
        assert!(matches!(rx.recv().unwrap(), ParakeetRsCommand::Audio(_)));
        assert!(matches!(
            rx.recv().unwrap(),
            ParakeetRsCommand::UtteranceEnd
        ));
        assert!(matches!(rx.recv().unwrap(), ParakeetRsCommand::Language(_)));
        assert!(matches!(rx.recv().unwrap(), ParakeetRsCommand::Drain(42)));
        assert!(rx.recv().is_err());
    }

    #[test]
    fn drain_admission_is_bounded_and_cleanup_does_not_block() {
        let (tx, rx) = bounded(1);
        let handle = ParakeetRsSessionHandle {
            tx,
            multilingual: true,
        };
        handle.send_audio(vec![0.1]).unwrap();
        assert!(handle.request_drain(42).is_err());
        handle.request_stop();
        drop(handle);
        assert!(matches!(rx.recv().unwrap(), ParakeetRsCommand::Audio(_)));
        assert!(rx.recv().is_err());
    }

    #[test]
    fn onnx_runtime_search_skips_rejected_candidates() {
        let tmp = std::env::temp_dir().join(format!("ears-ort-probe-{}", std::process::id()));
        let bad = tmp.join("bad");
        let good = tmp.join("good");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::create_dir_all(&good).unwrap();
        std::fs::write(bad.join(ORT_LIB_NAMES[0]), b"").unwrap();
        std::fs::write(good.join(ORT_LIB_NAMES[0]), b"").unwrap();

        let dirs = vec![tmp.join("missing"), bad.clone(), good.clone()];
        let probe = |path: &Path| {
            if path.starts_with(&bad) {
                anyhow::bail!("{} is ONNX Runtime 1.17.0", path.display())
            }
            Ok("1.31.0".to_string())
        };
        let (path, version) = find_onnx_runtime(&dirs, probe).unwrap();
        assert_eq!(path, good.join(ORT_LIB_NAMES[0]));
        assert_eq!(version, "1.31.0");

        let rejected = find_onnx_runtime(&dirs[..2], probe).unwrap_err();
        assert_eq!(rejected.len(), 1);
        assert!(rejected[0].contains("1.17.0"));
        assert!(find_onnx_runtime(&dirs[..1], probe).unwrap_err().is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn onnx_runtime_search_starts_next_to_executable() {
        let exe_dir = PathBuf::from("/opt/ears/bin");
        let dirs = ort_search_dirs(Some(&exe_dir));
        assert_eq!(dirs[0], exe_dir);
        assert_eq!(dirs[1], exe_dir.join("..").join("lib"));
    }

    #[cfg(unix)]
    #[test]
    fn probe_rejects_a_file_that_is_not_onnx_runtime() {
        let fake = std::env::temp_dir().join(format!("ears-fake-ort-{}.so", std::process::id()));
        std::fs::write(&fake, b"not a library").unwrap();
        let err = probe_onnx_runtime(&fake).unwrap_err();
        assert!(format!("{err:#}").contains("cannot be loaded"), "{err:#}");
        std::fs::remove_file(&fake).unwrap();
    }

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
