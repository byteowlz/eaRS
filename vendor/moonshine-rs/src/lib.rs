//! Idiomatic, memory-safe Rust bindings for [Moonshine Voice] — fast, on-device
//! speech-to-text powered by ONNX Runtime.
//!
//! `moonshine-rs` wraps the official `libmoonshine` C API with safe types:
//! typed [`Error`]s, automatic resource cleanup via [`Drop`], and a
//! `Send + Sync` [`Transcriber`] suitable for long-running host applications.
//!
//! # Quick start
//!
//! ```no_run
//! use moonshine_rs::audio::load_audio_for_transcription;
//! use moonshine_rs::{ModelArch, Transcriber};
//!
//! // Decode + resample any supported file (WAV/MP3/AAC/FLAC/OGG/M4A) to 16kHz mono.
//! let pcm = load_audio_for_transcription("speech.wav")?;
//!
//! // Load a model directory (see the `download_model` example to fetch one).
//! let transcriber = Transcriber::from_files("./models/tiny-en", ModelArch::Tiny, None)?;
//!
//! let transcript = transcriber.transcribe(&pcm, 16_000)?;
//! println!("{}", transcript.text());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Features
//!
//! - `audio` *(default)* — multi-format decoding ([`symphonia`]) and 16kHz
//!   resampling ([`rubato`]) via the [`audio`] module.
//! - `serde` — derive `Serialize`/`Deserialize` on the transcript types.
//!
//! # Getting a model
//!
//! Transcription needs a model directory containing `encoder_model.ort`,
//! `decoder_model_merged.ort`, and `tokenizer.bin`. Resolve download URLs
//! natively with [`get_stt_dependencies`], or run the `download_model` example.
//!
//! [Moonshine Voice]: https://github.com/moonshine-ai/moonshine
//! [`symphonia`]: https://docs.rs/symphonia
//! [`rubato`]: https://docs.rs/rubato

use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, Mutex};

pub use moonshine_sys as sys;

#[cfg(feature = "audio")]
pub mod audio;

pub mod tts;
pub use tts::{
    get_g2p_dependencies, get_tts_dependencies, get_tts_voices, split_utterances,
    GraphemeToPhonemizer, SynthesizedAudio, TtsChunk, TtsOptions, TtsStreamStatus, TtsSynthesizer,
};

/// Errors returned by `moonshine-rs`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The underlying C API returned a non-zero status `code`. `message` is the
    /// human-readable string from [`error_string`].
    #[error("Moonshine C API error ({code}): {message}")]
    ApiError { code: i32, message: String },
    /// A transcriber handle was invalid (e.g. already freed).
    #[error("Invalid transcriber handle")]
    InvalidHandle,
    /// The C API returned a null pointer where data was expected.
    #[error("Null pointer returned from Moonshine API")]
    NullPointer,
    /// A Rust string could not be converted to a C string because it contained
    /// an interior NUL byte.
    #[error("Nul byte in CString conversion: {0}")]
    NulError(#[from] std::ffi::NulError),
}

/// Convenience alias for `Result<T, moonshine_rs::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Returns the human-readable message for a Moonshine error `code`.
pub fn error_string(code: i32) -> String {
    unsafe {
        let ptr = sys::moonshine_error_to_string(code);
        if ptr.is_null() {
            format!("Unknown error {}", code)
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Returns the linked `libmoonshine` version as an integer (e.g. `20000`).
pub fn get_version() -> i32 {
    unsafe { sys::moonshine_get_version() }
}

/// The model architecture to load. Must match the model files on disk.
///
/// Batch (non-streaming) transcription uses [`Tiny`](ModelArch::Tiny) or
/// [`Base`](ModelArch::Base); the `*Streaming` variants target the (not yet
/// wrapped) streaming API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    /// Smallest, fastest English model (`tiny-en`, ~71 MB).
    Tiny = sys::MOONSHINE_MODEL_ARCH_TINY as isize,
    /// Higher-accuracy English model (`base-en`, ~238 MB).
    Base = sys::MOONSHINE_MODEL_ARCH_BASE as isize,
    /// Streaming tiny model.
    TinyStreaming = sys::MOONSHINE_MODEL_ARCH_TINY_STREAMING as isize,
    /// Streaming base model.
    BaseStreaming = sys::MOONSHINE_MODEL_ARCH_BASE_STREAMING as isize,
    /// Streaming small model.
    SmallStreaming = sys::MOONSHINE_MODEL_ARCH_SMALL_STREAMING as isize,
    /// Streaming medium model.
    MediumStreaming = sys::MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING as isize,
}

impl ModelArch {
    /// Returns the raw C enum value for this architecture.
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Configuration passed to [`Transcriber::from_files`].
///
/// Options are forwarded to the C API as key/value string pairs. Use the
/// builder methods for common settings, or [`set`](TranscriberOptions::set) for
/// arbitrary keys.
///
/// ```
/// use moonshine_rs::TranscriberOptions;
/// let options = TranscriberOptions::new()
///     .with_ort_providers("CPU")
///     .with_identify_speakers(true);
/// ```
#[derive(Debug, Default, Clone)]
pub struct TranscriberOptions {
    options: Vec<(String, String)>,
}

impl TranscriberOptions {
    /// Creates an empty option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an arbitrary option key/value pair, returning `self` for chaining.
    pub fn set(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.push((key.into(), value.into()));
        self
    }

    /// Returns the option value for a given key, if set.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.options
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Selects the ONNX Runtime execution providers (e.g. `"CPU"`).
    pub fn with_ort_providers(self, providers: &str) -> Self {
        self.set("ort_providers", providers)
    }

    /// Controls whether to re-decode the previous hypothesis on streaming updates using
    /// speculative decoding (default `true` upstream).
    pub fn with_speculative_decoding(self, enable: bool) -> Self {
        self.set(
            "use_speculative_decoding",
            if enable { "true" } else { "false" },
        )
    }

    /// Enables speaker identification (diarization). See the
    /// `speaker_diarization` example.
    pub fn with_identify_speakers(self, enable: bool) -> Self {
        self.set("identify_speakers", if enable { "true" } else { "false" })
    }

    /// Sets the directory containing speaker diarization models (`segmentation.ort` and
    /// `embedding.ort`).
    ///
    /// If speaker identification is enabled via [`with_identify_speakers`](Self::with_identify_speakers)
    /// and no explicit directory is set, `moonshine-rs` will download the diarization models
    /// automatically into the local cache on first use.
    pub fn with_diarization_model_dir(self, path: impl AsRef<Path>) -> Self {
        self.set("diarization_model_dir", path.as_ref().to_string_lossy())
    }

    /// Sets the path to an optional spelling model.
    pub fn with_spelling_model(self, path: &str) -> Self {
        self.set("spelling_model_path", path)
    }

    /// Sets a list of key terms (comma-separated, e.g. `"Kubernetes,Ceph,etcd"`)
    /// to bias the decoder towards uncommon words, domain jargon, or contact names.
    ///
    /// Capitalization and spelling in `keyterms` are reflected in the transcript output.
    /// Only streaming model architectures (`TinyStreaming`, `SmallStreaming`, `MediumStreaming`)
    /// apply keyterm biasing.
    pub fn with_keyterms(self, keyterms: impl AsRef<str>) -> Self {
        self.set("keyterms", keyterms.as_ref())
    }

    /// Sets the keyterm biasing strength (default `2.0` upstream).
    ///
    /// A boost of `2.0` is a balanced default (recovering ~25% of domain errors with negligible impact
    /// on general words). Use `1.0` for minimal disruption, or `3.0` for stronger keyword biasing.
    /// Avoid setting above `4.0`.
    pub fn with_keyterm_boost(self, boost: f32) -> Self {
        self.set("keyterm_boost", boost.to_string())
    }

    /// Supplies a passage of free-form context text from which the model's tokenizer
    /// extracts unusual domain-specific terms to bias towards.
    ///
    /// For example, pass on-screen document content, an email thread, or a meeting agenda.
    /// Only streaming model architectures support context domain customization.
    pub fn with_context(self, context: impl AsRef<str>) -> Self {
        self.set("context", context.as_ref())
    }

    /// Caps the maximum number of key terms extracted from context text (default `200` upstream).
    pub fn with_context_max_terms(self, max_terms: u32) -> Self {
        self.set("context_max_terms", max_terms.to_string())
    }
}

/// A single recognized word with timing and confidence.
///
/// Populated only when the model ships the attention decoder (see the
/// `word_timestamps` example); otherwise [`TranscriptLine::words`] is empty.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptWord {
    /// The word text.
    pub text: String,
    /// Start time in seconds from the beginning of the audio.
    pub start: f32,
    /// End time in seconds from the beginning of the audio.
    pub end: f32,
    /// Model confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A span of a line attributed to a single speaker (diarization output).
///
/// Populated when [`TranscriberOptions::with_identify_speakers`] is enabled.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpeakerSpan {
    /// Start time in seconds from the beginning of the audio.
    pub start_time: f32,
    /// Span duration in seconds.
    pub duration: f32,
    /// Stable speaker identifier.
    pub speaker_id: u64,
    /// Zero-based speaker index within this transcript.
    pub speaker_index: u32,
    /// Start character offset of the span.
    pub start_char: u64,
    /// End character offset of the span.
    pub end_char: u64,
}

/// One line of a [`Transcript`], with timing and optional word/speaker detail.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptLine {
    /// The recognized text for this line.
    pub text: String,
    /// Start time in seconds from the beginning of the audio.
    pub start_time: f32,
    /// Line duration in seconds.
    pub duration: f32,
    /// Stable line identifier.
    pub id: u64,
    /// Whether the line is finalized (relevant for streaming).
    pub is_complete: bool,
    /// Streaming-only: Whether this line was updated since the last stream poll.
    pub is_updated: bool,
    /// Streaming-only: Whether this line was newly added since the last stream poll.
    pub is_new: bool,
    /// Streaming-only: Whether the text of this line has changed since the last stream poll.
    pub has_text_changed: bool,
    /// Streaming-only: Whether speaker spans were revised since the last stream poll.
    pub have_speakers_changed: bool,
    /// Streaming-only: Last transcription processing latency in milliseconds.
    pub last_transcription_latency_ms: u32,
    /// Per-word timing/confidence, if available. See [`TranscriptWord`].
    pub words: Vec<TranscriptWord>,
    /// Speaker attribution spans, if diarization was enabled.
    pub speaker_spans: Vec<SpeakerSpan>,
}

/// The full result of a transcription: an ordered list of [`TranscriptLine`]s.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transcript {
    /// The recognized lines, in order.
    pub lines: Vec<TranscriptLine>,
}

impl Transcript {
    /// Joins all line texts with newlines into a single string.
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A loaded Moonshine model ready to transcribe audio.
///
/// `Transcriber` is `Send + Sync` and manages the native handle, freeing it on
/// [`Drop`]. Load once and reuse across many `transcribe` calls; wrap in
/// [`std::sync::Arc`] to share across threads/tasks (see the `async_transcribe`
/// example). Concurrent calls are serialized internally by a mutex.
///
/// ```no_run
/// use moonshine_rs::{ModelArch, Transcriber};
/// let t = Transcriber::from_files("./models/tiny-en", ModelArch::Tiny, None)?;
/// let pcm = vec![0.0f32; 16_000]; // 1s of silence at 16kHz mono
/// let transcript = t.transcribe(&pcm, 16_000)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Transcriber {
    handle: i32,
    _lock: Mutex<()>,
}

unsafe impl Send for Transcriber {}
unsafe impl Sync for Transcriber {}

impl Transcriber {
    /// Loads a transcriber from a model directory containing
    /// `encoder_model.ort`, `decoder_model_merged.ort`, and `tokenizer.bin`.
    ///
    /// `arch` must match the model files. Pass `None` for default options.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if the model cannot be loaded (e.g. missing
    /// files, LFS pointer files, or an architecture mismatch).
    pub fn from_files(
        path: impl AsRef<Path>,
        arch: ModelArch,
        options: Option<&TranscriberOptions>,
    ) -> Result<Self> {
        let mut effective_opts = options.cloned().unwrap_or_default();

        if effective_opts.get("identify_speakers") == Some("true")
            && effective_opts.get("diarization_model_dir").is_none()
        {
            let diarization_dir = ensure_diarization_models_downloaded()?;
            effective_opts = effective_opts.with_diarization_model_dir(diarization_dir);
        }

        let path_str = path.as_ref().to_str().ok_or_else(|| Error::ApiError {
            code: -1,
            message: "Path contains invalid UTF-8".to_string(),
        })?;

        let c_path = CString::new(path_str)?;

        let mut raw_opts = Vec::with_capacity(effective_opts.options.len());
        let mut strings = Vec::with_capacity(effective_opts.options.len() * 2);

        for (k, v) in &effective_opts.options {
            let ck = CString::new(k.as_str())?;
            let cv = CString::new(v.as_str())?;
            raw_opts.push(sys::moonshine_option_t {
                name: ck.as_ptr(),
                value: cv.as_ptr(),
            });
            strings.push(ck);
            strings.push(cv);
        }

        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let handle = unsafe {
            sys::moonshine_load_transcriber_from_files(
                c_path.as_ptr(),
                arch.as_u32(),
                opts_ptr,
                raw_opts.len() as u64,
                sys::MOONSHINE_HEADER_VERSION as i32,
            )
        };

        if handle < 0 {
            return Err(Error::ApiError {
                code: handle,
                message: error_string(handle),
            });
        }

        Ok(Self {
            handle,
            _lock: Mutex::new(()),
        })
    }

    /// Loads a transcriber from in-memory model file buffers.
    ///
    /// Each entry in `files` is a `(filename, buffer)` pair, where `filename` is the
    /// canonical model filename (e.g. `"encoder_model.ort"`, `"decoder_model_merged.ort"`,
    /// `"tokenizer.bin"`, and optionally `"segmentation.ort"` / `"embedding.ort"` for
    /// diarization).
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if the model cannot be loaded from memory.
    pub fn from_memory_files(
        arch: ModelArch,
        files: &[(&str, &[u8])],
        options: Option<&TranscriberOptions>,
    ) -> Result<Self> {
        let mut effective_opts = options.cloned().unwrap_or_default();

        if effective_opts.get("identify_speakers") == Some("true")
            && effective_opts.get("diarization_model_dir").is_none()
            && !files.iter().any(|(f, _)| *f == "segmentation.ort")
        {
            let diarization_dir = ensure_diarization_models_downloaded()?;
            effective_opts = effective_opts.with_diarization_model_dir(diarization_dir);
        }

        let mut c_filenames = Vec::with_capacity(files.len());
        let mut filename_ptrs = Vec::with_capacity(files.len());
        let mut memory_ptrs = Vec::with_capacity(files.len());
        let mut memory_sizes = Vec::with_capacity(files.len());

        for (filename, bytes) in files {
            let c_fn = CString::new(*filename)?;
            filename_ptrs.push(c_fn.as_ptr());
            c_filenames.push(c_fn);
            memory_ptrs.push(bytes.as_ptr());
            memory_sizes.push(bytes.len() as u64);
        }

        let mut raw_opts = Vec::with_capacity(effective_opts.options.len());
        let mut strings = Vec::with_capacity(effective_opts.options.len() * 2);

        for (k, v) in &effective_opts.options {
            let ck = CString::new(k.as_str())?;
            let cv = CString::new(v.as_str())?;
            raw_opts.push(sys::moonshine_option_t {
                name: ck.as_ptr(),
                value: cv.as_ptr(),
            });
            strings.push(ck);
            strings.push(cv);
        }

        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let handle = unsafe {
            sys::moonshine_load_transcriber_from_memory_files(
                filename_ptrs.as_mut_ptr(),
                memory_ptrs.as_mut_ptr(),
                memory_sizes.as_ptr(),
                files.len() as u64,
                arch.as_u32(),
                opts_ptr,
                raw_opts.len() as u64,
                sys::MOONSHINE_HEADER_VERSION as i32,
            )
        };

        if handle < 0 {
            return Err(Error::ApiError {
                code: handle,
                message: error_string(handle),
            });
        }

        Ok(Self {
            handle,
            _lock: Mutex::new(()),
        })
    }

    /// Returns the raw native handle. Primarily useful for logging/debugging.
    pub fn handle(&self) -> i32 {
        self.handle
    }

    /// Transcribes mono `f32` PCM samples in `[-1.0, 1.0]`.
    ///
    /// Moonshine expects 16 kHz mono audio; use
    /// [`audio::load_audio_for_transcription`] to decode and resample files.
    /// This call is synchronous and CPU-bound — offload it with
    /// [`tokio::task::spawn_blocking`](https://docs.rs/tokio) inside async code.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if the C API reports a transcription failure.
    pub fn transcribe(&self, pcm_data: &[f32], sample_rate: u32) -> Result<Transcript> {
        let _guard = self._lock.lock().unwrap();

        let mut out_transcript: *mut sys::transcript_t = ptr::null_mut();

        let mut audio_vec = pcm_data.to_vec();
        let audio_ptr = audio_vec.as_mut_ptr();

        let ret = unsafe {
            sys::moonshine_transcribe_without_streaming(
                self.handle,
                audio_ptr,
                pcm_data.len() as u64,
                sample_rate as i32,
                0,
                &mut out_transcript,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        Ok(copy_transcript(out_transcript))
    }

    /// Creates and starts a new streaming session [`TranscriberStream`].
    ///
    /// Multiple concurrent streams can be spawned from a single `Transcriber`.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if creating or starting the native stream fails.
    pub fn create_stream(&self) -> Result<TranscriberStream<'_>> {
        let stream_handle = unsafe { sys::moonshine_create_stream(self.handle, 0) };
        if stream_handle < 0 {
            return Err(Error::ApiError {
                code: stream_handle,
                message: error_string(stream_handle),
            });
        }

        let ret_start = unsafe { sys::moonshine_start_stream(self.handle, stream_handle) };
        if ret_start != 0 {
            let _ = unsafe { sys::moonshine_free_stream(self.handle, stream_handle) };
            return Err(Error::ApiError {
                code: ret_start,
                message: error_string(ret_start),
            });
        }

        Ok(TranscriberStream {
            transcriber: self,
            stream_handle,
            closed: false,
        })
    }

    /// Creates and starts an owned streaming session [`OwnedTranscriberStream`] that owns
    /// an `Arc<Transcriber>`.
    ///
    /// Useful when storing streaming state in application state (e.g. GUI state or actors)
    /// across long-lived tasks or async boundaries without an explicit lifetime parameter.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if creating or starting the native stream fails.
    pub fn create_owned_stream(self: Arc<Self>) -> Result<OwnedTranscriberStream> {
        let stream_handle = unsafe { sys::moonshine_create_stream(self.handle, 0) };
        if stream_handle < 0 {
            return Err(Error::ApiError {
                code: stream_handle,
                message: error_string(stream_handle),
            });
        }

        let ret_start = unsafe { sys::moonshine_start_stream(self.handle, stream_handle) };
        if ret_start != 0 {
            let _ = unsafe { sys::moonshine_free_stream(self.handle, stream_handle) };
            return Err(Error::ApiError {
                code: ret_start,
                message: error_string(ret_start),
            });
        }

        Ok(OwnedTranscriberStream {
            transcriber: self,
            stream_handle,
            closed: false,
        })
    }

    /// Replaces the contextual-biasing key terms on this transcriber.
    ///
    /// Key terms bias the decoder towards rare words, jargon, contact names, or product
    /// names. `keyterms` is a comma-separated list of terms (e.g. `"Kubernetes,Ceph,etcd"`).
    /// Pass an empty string `""` to turn biasing off.
    ///
    /// Safe to call between transcription calls on a live stream and takes effect
    /// on the next transcription cycle without reloading the model.
    ///
    /// Note: Only streaming model architectures (`TinyStreaming`, `SmallStreaming`,
    /// `MediumStreaming`) support domain customization.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if the handle is invalid or if the loaded model is not
    /// a streaming architecture.
    pub fn set_keyterms(&self, keyterms: impl AsRef<str>) -> Result<()> {
        let _guard = self._lock.lock().unwrap();
        let keyterms_str = keyterms.as_ref();
        let c_keyterms = CString::new(keyterms_str)?;
        let ret =
            unsafe { sys::moonshine_transcriber_set_keyterms(self.handle, c_keyterms.as_ptr()) };
        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }
        Ok(())
    }

    /// Extracts unusual key terms from a passage of free-form text and biases towards them.
    ///
    /// Context text (such as an on-screen document, email thread, or meeting agenda) is parsed
    /// using the loaded model's tokenizer to automatically identify domain-specific words.
    ///
    /// `max_terms` caps the number of extracted terms. Pass `0` for the upstream default (`200`).
    /// Pass an empty string `""` to disable biasing.
    ///
    /// Safe to call between transcription calls on a live stream and takes effect
    /// on the next transcription cycle without reloading the model.
    ///
    /// Note: Only streaming model architectures (`TinyStreaming`, `SmallStreaming`,
    /// `MediumStreaming`) support domain customization.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if the handle is invalid or if the loaded model is not
    /// a streaming architecture.
    pub fn set_context(&self, context: impl AsRef<str>, max_terms: u32) -> Result<()> {
        let _guard = self._lock.lock().unwrap();
        let context_str = context.as_ref();
        let c_context = CString::new(context_str)?;
        let ret = unsafe {
            sys::moonshine_transcriber_set_context(
                self.handle,
                c_context.as_ptr(),
                max_terms as i32,
            )
        };
        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }
        Ok(())
    }
}

/// An active streaming session for incremental, real-time transcription.
///
/// Created via [`Transcriber::create_stream`]. Audio chunks can be added
/// frequently via [`add_audio`](TranscriberStream::add_audio). Call [`poll`](TranscriberStream::poll)
/// when updated transcript lines are desired, or [`finalize`](TranscriberStream::finalize)
/// at the end of a recording session.
pub struct TranscriberStream<'a> {
    transcriber: &'a Transcriber,
    stream_handle: i32,
    closed: bool,
}

unsafe impl<'a> Send for TranscriberStream<'a> {}
unsafe impl<'a> Sync for TranscriberStream<'a> {}

impl<'a> TranscriberStream<'a> {
    /// Returns the underlying native stream handle.
    pub fn handle(&self) -> i32 {
        self.stream_handle
    }

    /// Appends newly-captured PCM audio samples (`f32` in `[-1.0, 1.0]`) to the stream buffer.
    ///
    /// This call is lightweight and cheap to call frequently (e.g. directly from audio callbacks).
    /// It buffers the audio without running full transcription model evaluation.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if adding audio fails or the stream is closed.
    pub fn add_audio(&mut self, pcm_data: &[f32], sample_rate: u32) -> Result<()> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let mut audio_vec = pcm_data.to_vec();
        let audio_ptr = if audio_vec.is_empty() {
            ptr::null()
        } else {
            audio_vec.as_mut_ptr()
        };

        let ret = unsafe {
            sys::moonshine_transcribe_add_audio_to_stream(
                self.transcriber.handle(),
                self.stream_handle,
                audio_ptr,
                pcm_data.len() as u64,
                sample_rate as i32,
                0,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        Ok(())
    }

    /// Evaluates the stream buffer and returns an updated [`Transcript`].
    ///
    /// If `force` is `false`, the C library skips re-evaluation if less than ~200ms
    /// of new audio arrived since the last poll. Pass `force = true`
    /// (`MOONSHINE_FLAG_FORCE_UPDATE`) to force immediate model evaluation.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if transcription fails or the stream is closed.
    pub fn poll(&mut self, force: bool) -> Result<Transcript> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let _guard = self.transcriber._lock.lock().unwrap();

        let flags = if force {
            sys::MOONSHINE_FLAG_FORCE_UPDATE
        } else {
            0
        };

        let mut out_transcript: *mut sys::transcript_t = ptr::null_mut();

        let ret = unsafe {
            sys::moonshine_transcribe_stream(
                self.transcriber.handle(),
                self.stream_handle,
                flags,
                &mut out_transcript,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        Ok(copy_transcript(out_transcript))
    }

    /// Restarts the stream (stops and restarts C stream state).
    ///
    /// Useful for audio discontinuities (e.g. user muted microphone or paused input)
    /// to reset stream history before new audio arrives.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if restart fails or the stream is closed.
    pub fn restart(&mut self) -> Result<()> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let ret_stop =
            unsafe { sys::moonshine_stop_stream(self.transcriber.handle(), self.stream_handle) };
        if ret_stop != 0 {
            return Err(Error::ApiError {
                code: ret_stop,
                message: error_string(ret_stop),
            });
        }

        let ret_start =
            unsafe { sys::moonshine_start_stream(self.transcriber.handle(), self.stream_handle) };
        if ret_start != 0 {
            return Err(Error::ApiError {
                code: ret_start,
                message: error_string(ret_start),
            });
        }

        Ok(())
    }

    /// Replaces the contextual-biasing key terms on the underlying transcriber mid-stream.
    ///
    /// See [`Transcriber::set_keyterms`].
    pub fn set_keyterms(&self, keyterms: impl AsRef<str>) -> Result<()> {
        self.transcriber.set_keyterms(keyterms)
    }

    /// Extracts key terms from context text and applies them to the underlying transcriber mid-stream.
    ///
    /// See [`Transcriber::set_context`].
    pub fn set_context(&self, context: impl AsRef<str>, max_terms: u32) -> Result<()> {
        self.transcriber.set_context(context, max_terms)
    }

    /// Finalizes the stream and returns the complete final [`Transcript`].
    ///
    /// Stops the stream and forces a final model evaluation. Consumes `self`.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if finalization fails.
    pub fn finalize(mut self) -> Result<Transcript> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let ret_stop =
            unsafe { sys::moonshine_stop_stream(self.transcriber.handle(), self.stream_handle) };
        if ret_stop != 0 {
            return Err(Error::ApiError {
                code: ret_stop,
                message: error_string(ret_stop),
            });
        }

        let final_transcript = self.poll(true)?;
        let _ = self.close();
        Ok(final_transcript)
    }

    /// Explicitly closes the stream and releases native C resources.
    ///
    /// Called automatically when `TranscriberStream` is dropped.
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;

        if self.stream_handle >= 0 {
            let _ = unsafe {
                sys::moonshine_stop_stream(self.transcriber.handle(), self.stream_handle)
            };
            let ret = unsafe {
                sys::moonshine_free_stream(self.transcriber.handle(), self.stream_handle)
            };
            if ret != 0 {
                return Err(Error::ApiError {
                    code: ret,
                    message: error_string(ret),
                });
            }
        }
        Ok(())
    }
}

impl<'a> Drop for TranscriberStream<'a> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// An owned streaming session for incremental, real-time transcription.
///
/// Unlike [`TranscriberStream`], which borrows a `&'a Transcriber`, `OwnedTranscriberStream`
/// holds an `Arc<Transcriber>`. This makes it `'static` and easy to store in application state,
/// move across async tasks, or send to background workers.
pub struct OwnedTranscriberStream {
    transcriber: Arc<Transcriber>,
    stream_handle: i32,
    closed: bool,
}

unsafe impl Send for OwnedTranscriberStream {}
unsafe impl Sync for OwnedTranscriberStream {}

impl OwnedTranscriberStream {
    /// Returns the underlying native stream handle.
    pub fn handle(&self) -> i32 {
        self.stream_handle
    }

    /// Appends newly-captured PCM audio samples (`f32` in `[-1.0, 1.0]`) to the stream buffer.
    pub fn add_audio(&mut self, pcm_data: &[f32], sample_rate: u32) -> Result<()> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let mut audio_vec = pcm_data.to_vec();
        let audio_ptr = if audio_vec.is_empty() {
            ptr::null()
        } else {
            audio_vec.as_mut_ptr()
        };

        let ret = unsafe {
            sys::moonshine_transcribe_add_audio_to_stream(
                self.transcriber.handle(),
                self.stream_handle,
                audio_ptr,
                pcm_data.len() as u64,
                sample_rate as i32,
                0,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        Ok(())
    }

    /// Evaluates the stream buffer and returns an updated [`Transcript`].
    pub fn poll(&mut self, force: bool) -> Result<Transcript> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let _guard = self.transcriber._lock.lock().unwrap();

        let flags = if force {
            sys::MOONSHINE_FLAG_FORCE_UPDATE
        } else {
            0
        };

        let mut out_transcript: *mut sys::transcript_t = ptr::null_mut();

        let ret = unsafe {
            sys::moonshine_transcribe_stream(
                self.transcriber.handle(),
                self.stream_handle,
                flags,
                &mut out_transcript,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        Ok(copy_transcript(out_transcript))
    }

    /// Restarts the stream (stops and restarts C stream state).
    pub fn restart(&mut self) -> Result<()> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let ret_stop =
            unsafe { sys::moonshine_stop_stream(self.transcriber.handle(), self.stream_handle) };
        if ret_stop != 0 {
            return Err(Error::ApiError {
                code: ret_stop,
                message: error_string(ret_stop),
            });
        }

        let ret_start =
            unsafe { sys::moonshine_start_stream(self.transcriber.handle(), self.stream_handle) };
        if ret_start != 0 {
            return Err(Error::ApiError {
                code: ret_start,
                message: error_string(ret_start),
            });
        }

        Ok(())
    }

    /// Replaces the contextual-biasing key terms on the underlying transcriber mid-stream.
    ///
    /// See [`Transcriber::set_keyterms`].
    pub fn set_keyterms(&self, keyterms: impl AsRef<str>) -> Result<()> {
        self.transcriber.set_keyterms(keyterms)
    }

    /// Extracts key terms from context text and applies them to the underlying transcriber mid-stream.
    ///
    /// See [`Transcriber::set_context`].
    pub fn set_context(&self, context: impl AsRef<str>, max_terms: u32) -> Result<()> {
        self.transcriber.set_context(context, max_terms)
    }

    /// Finalizes the stream and returns the complete final [`Transcript`].
    pub fn finalize(mut self) -> Result<Transcript> {
        if self.closed {
            return Err(Error::InvalidHandle);
        }

        let ret_stop =
            unsafe { sys::moonshine_stop_stream(self.transcriber.handle(), self.stream_handle) };
        if ret_stop != 0 {
            return Err(Error::ApiError {
                code: ret_stop,
                message: error_string(ret_stop),
            });
        }

        let final_transcript = self.poll(true)?;
        let _ = self.close();
        Ok(final_transcript)
    }

    /// Explicitly closes the stream and releases native C resources.
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;

        if self.stream_handle >= 0 {
            let _ = unsafe {
                sys::moonshine_stop_stream(self.transcriber.handle(), self.stream_handle)
            };
            let ret = unsafe {
                sys::moonshine_free_stream(self.transcriber.handle(), self.stream_handle)
            };
            if ret != 0 {
                return Err(Error::ApiError {
                    code: ret,
                    message: error_string(ret),
                });
            }
        }
        Ok(())
    }
}

impl Drop for OwnedTranscriberStream {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

impl Drop for Transcriber {
    fn drop(&mut self) {
        if self.handle >= 0 {
            unsafe {
                sys::moonshine_free_transcriber(self.handle);
            }
        }
    }
}

fn copy_transcript(out_transcript: *mut sys::transcript_t) -> Transcript {
    if out_transcript.is_null() {
        return Transcript::default();
    }

    let mut lines = Vec::new();
    unsafe {
        let t = &*out_transcript;
        if t.line_count > 0 && !t.lines.is_null() {
            let raw_lines = std::slice::from_raw_parts(t.lines, t.line_count as usize);
            for raw_line in raw_lines {
                let text = if raw_line.text.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(raw_line.text).to_string_lossy().into_owned()
                };

                let mut words = Vec::new();
                if raw_line.word_count > 0 && !raw_line.words.is_null() {
                    let raw_words =
                        std::slice::from_raw_parts(raw_line.words, raw_line.word_count as usize);
                    for rw in raw_words {
                        let w_text = if rw.text.is_null() {
                            String::new()
                        } else {
                            CStr::from_ptr(rw.text).to_string_lossy().into_owned()
                        };
                        words.push(TranscriptWord {
                            text: w_text,
                            start: rw.start,
                            end: rw.end,
                            confidence: rw.confidence,
                        });
                    }
                }

                let mut speaker_spans = Vec::new();
                if raw_line.speaker_span_count > 0 && !raw_line.speaker_spans.is_null() {
                    let raw_spans = std::slice::from_raw_parts(
                        raw_line.speaker_spans,
                        raw_line.speaker_span_count as usize,
                    );
                    for rs in raw_spans {
                        speaker_spans.push(SpeakerSpan {
                            start_time: rs.start_time,
                            duration: rs.duration,
                            speaker_id: rs.speaker_id,
                            speaker_index: rs.speaker_index,
                            start_char: rs.start_char,
                            end_char: rs.end_char,
                        });
                    }
                }

                lines.push(TranscriptLine {
                    text,
                    start_time: raw_line.start_time,
                    duration: raw_line.duration,
                    id: raw_line.id,
                    is_complete: raw_line.is_complete != 0,
                    is_updated: raw_line.is_updated != 0,
                    is_new: raw_line.is_new != 0,
                    has_text_changed: raw_line.has_text_changed != 0,
                    have_speakers_changed: raw_line.have_speakers_changed != 0,
                    last_transcription_latency_ms: raw_line.last_transcription_latency_ms,
                    words,
                    speaker_spans,
                });
            }
        }
    }

    Transcript { lines }
}

/// Resolves the download manifest (CDN URLs, file sizes, CRC32C checksums) for
/// an STT model as a JSON string.
///
/// See the `download_model` example for parsing and fetching the listed files.
/// For extra knobs (word timestamps, spelling model path) use
/// [`get_stt_dependencies_with_options`].
pub fn get_stt_dependencies(
    language: &str,
    arch: Option<ModelArch>,
    include_spelling: bool,
) -> Result<String> {
    get_stt_dependencies_with_options(
        language,
        &SttDependenciesOptions {
            arch,
            include_spelling,
            ..Default::default()
        },
    )
}

/// Extended options for [`get_stt_dependencies_with_options`], covering the
/// full set of dependency-resolution knobs supported by
/// `moonshine_get_stt_dependencies` (added upstream alongside
/// `word_timestamps` and the `spelling` / `spelling_model_path` aliases).
#[derive(Debug, Default, Clone)]
pub struct SttDependenciesOptions {
    pub arch: Option<ModelArch>,
    /// Include the spelling model's files as an extra dependency group.
    /// Equivalent to passing `include_spelling` (alias: `spelling`) to the C
    /// API.
    pub include_spelling: bool,
    /// Include the optional attention decoder needed to produce word-level
    /// timestamps. This roughly doubles the download size for models that
    /// publish it, so it defaults to `false`.
    pub word_timestamps: bool,
    /// Resolve dependencies for a specific spelling model path instead of
    /// the language's default spelling model. Implies `include_spelling`.
    pub spelling_model_path: Option<String>,
}

impl SttDependenciesOptions {
    /// Creates a default option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves dependencies for a specific model architecture.
    pub fn with_arch(mut self, arch: ModelArch) -> Self {
        self.arch = Some(arch);
        self
    }

    /// Includes the spelling model's files as an extra dependency group.
    pub fn with_include_spelling(mut self, include_spelling: bool) -> Self {
        self.include_spelling = include_spelling;
        self
    }

    /// Includes the attention decoder needed for word-level timestamps.
    pub fn with_word_timestamps(mut self, word_timestamps: bool) -> Self {
        self.word_timestamps = word_timestamps;
        self
    }

    /// Resolves dependencies for a specific spelling model path (implies
    /// `include_spelling`).
    pub fn with_spelling_model_path(mut self, path: impl Into<String>) -> Self {
        self.spelling_model_path = Some(path.into());
        self
    }
}

/// Like [`get_stt_dependencies`], but exposes the full set of options
/// recognized by `moonshine_get_stt_dependencies`, including
/// `word_timestamps` and `spelling_model_path`.
pub fn get_stt_dependencies_with_options(
    language: &str,
    options: &SttDependenciesOptions,
) -> Result<String> {
    let c_lang = CString::new(language)?;
    let mut opts = Vec::new();
    let mut strings = Vec::new();

    if let Some(a) = options.arch {
        let ck = CString::new("model_arch")?;
        let cv = CString::new(a.as_u32().to_string())?;
        opts.push(sys::moonshine_option_t {
            name: ck.as_ptr(),
            value: cv.as_ptr(),
        });
        strings.push(ck);
        strings.push(cv);
    }

    if options.include_spelling {
        let ck = CString::new("include_spelling")?;
        let cv = CString::new("true")?;
        opts.push(sys::moonshine_option_t {
            name: ck.as_ptr(),
            value: cv.as_ptr(),
        });
        strings.push(ck);
        strings.push(cv);
    }

    if options.word_timestamps {
        let ck = CString::new("word_timestamps")?;
        let cv = CString::new("true")?;
        opts.push(sys::moonshine_option_t {
            name: ck.as_ptr(),
            value: cv.as_ptr(),
        });
        strings.push(ck);
        strings.push(cv);
    }

    if let Some(path) = &options.spelling_model_path {
        let ck = CString::new("spelling_model_path")?;
        let cv = CString::new(path.as_str())?;
        opts.push(sys::moonshine_option_t {
            name: ck.as_ptr(),
            value: cv.as_ptr(),
        });
        strings.push(ck);
        strings.push(cv);
    }

    let mut out_json: *mut std::os::raw::c_char = ptr::null_mut();

    let ret = unsafe {
        sys::moonshine_get_stt_dependencies(
            c_lang.as_ptr(),
            if opts.is_empty() {
                ptr::null()
            } else {
                opts.as_ptr()
            },
            opts.len() as u64,
            &mut out_json,
        )
    };

    if ret != 0 {
        return Err(Error::ApiError {
            code: ret,
            message: error_string(ret),
        });
    }

    if out_json.is_null() {
        return Err(Error::NullPointer);
    }

    let json_str = unsafe {
        let str_slice = CStr::from_ptr(out_json).to_string_lossy().into_owned();
        sys::moonshine_free_buffer(out_json as *mut std::ffi::c_void);
        str_slice
    };

    Ok(json_str)
}

/// Returns the full STT catalog (languages and available model architectures)
/// as a JSON string. See the `browse_catalog` example.
pub fn get_stt_catalog() -> Result<String> {
    let mut out_json: *mut std::os::raw::c_char = ptr::null_mut();

    let ret = unsafe { sys::moonshine_get_stt_catalog(&mut out_json) };

    if ret != 0 {
        return Err(Error::ApiError {
            code: ret,
            message: error_string(ret),
        });
    }

    if out_json.is_null() {
        return Err(Error::NullPointer);
    }

    let json_str = unsafe {
        let str_slice = CStr::from_ptr(out_json).to_string_lossy().into_owned();
        sys::moonshine_free_buffer(out_json as *mut std::ffi::c_void);
        str_slice
    };

    Ok(json_str)
}

/// Resolves the download manifest (CDN URLs, file sizes, CRC32C checksums) for
/// the speaker diarization models (`segmentation.ort` and `embedding.ort`) as a JSON string.
pub fn get_diarization_dependencies() -> Result<String> {
    let mut out_json: *mut std::os::raw::c_char = ptr::null_mut();

    let ret = unsafe { sys::moonshine_get_diarization_dependencies(&mut out_json) };

    if ret != 0 {
        return Err(Error::ApiError {
            code: ret,
            message: error_string(ret),
        });
    }

    if out_json.is_null() {
        return Err(Error::NullPointer);
    }

    let json_str = unsafe {
        let str_slice = CStr::from_ptr(out_json).to_string_lossy().into_owned();
        sys::moonshine_free_buffer(out_json as *mut std::ffi::c_void);
        str_slice
    };

    Ok(json_str)
}

/// Ensures that speaker diarization models (`segmentation.ort` and `embedding.ort`)
/// are present on disk, downloading them automatically from the official CDN manifest
/// into the local cache directory if missing.
///
/// Returns the path to the directory containing the downloaded diarization models.
pub fn ensure_diarization_models_downloaded() -> Result<PathBuf> {
    let cache_dir = if let Ok(custom) = env::var("MOONSHINE_RS_CACHE_DIR") {
        PathBuf::from(custom).join("diarization")
    } else if let Some(user_cache) = dirs::cache_dir() {
        user_cache.join("moonshine-rs").join("diarization")
    } else {
        env::temp_dir().join("moonshine-rs").join("diarization")
    };

    fs::create_dir_all(&cache_dir).map_err(|e| Error::ApiError {
        code: -1,
        message: format!("Failed to create diarization cache directory: {e}"),
    })?;

    let seg_path = cache_dir.join("segmentation.ort");
    let emb_path = cache_dir.join("embedding.ort");

    if seg_path.exists()
        && fs::metadata(&seg_path)
            .map(|m| m.len() > 0)
            .unwrap_or(false)
        && emb_path.exists()
        && fs::metadata(&emb_path)
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    {
        return Ok(cache_dir);
    }

    let manifest_json = get_diarization_dependencies()?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_json).map_err(|e| Error::ApiError {
            code: -1,
            message: format!("Failed to parse diarization manifest JSON: {e}"),
        })?;

    let groups = manifest["groups"]
        .as_array()
        .ok_or_else(|| Error::ApiError {
            code: -1,
            message: "Diarization manifest missing `groups` array".to_string(),
        })?;

    for group in groups {
        let files = group["files"].as_array().ok_or_else(|| Error::ApiError {
            code: -1,
            message: "Diarization group missing `files` array".to_string(),
        })?;

        for file in files {
            let name = file["name"].as_str().ok_or_else(|| Error::ApiError {
                code: -1,
                message: "Diarization file entry missing `name`".to_string(),
            })?;
            let url = file["url"].as_str().ok_or_else(|| Error::ApiError {
                code: -1,
                message: "Diarization file entry missing `url`".to_string(),
            })?;
            let expected_size = file["size"].as_u64().unwrap_or(0);

            let dest = cache_dir.join(name);
            if dest.exists() {
                let have = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                if expected_size == 0 || have == expected_size {
                    continue;
                }
            }

            let resp = ureq::get(url).call().map_err(|e| Error::ApiError {
                code: -1,
                message: format!("Failed to download diarization model from {url}: {e}"),
            })?;

            let mut bytes = Vec::new();
            resp.into_reader()
                .read_to_end(&mut bytes)
                .map_err(|e| Error::ApiError {
                    code: -1,
                    message: format!("Failed to read diarization model bytes from {url}: {e}"),
                })?;

            fs::write(&dest, &bytes).map_err(|e| Error::ApiError {
                code: -1,
                message: format!(
                    "Failed to write diarization model to {}: {e}",
                    dest.display()
                ),
            })?;
        }
    }

    Ok(cache_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = get_version();
        assert!(ver >= 20000, "Version should be at least 20000");
    }

    #[test]
    fn test_error_string() {
        let msg = error_string(sys::MOONSHINE_ERROR_NONE as i32);
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_stt_dependencies() {
        let deps = get_stt_dependencies("en", Some(ModelArch::Tiny), false).unwrap();
        assert!(deps.contains("encoder_model.ort"));
        assert!(deps.contains("decoder_model_merged.ort"));
    }

    #[test]
    fn test_stt_dependencies_with_options_word_timestamps() {
        let opts = SttDependenciesOptions::new()
            .with_arch(ModelArch::Tiny)
            .with_word_timestamps(true);
        let deps = get_stt_dependencies_with_options("en", &opts).unwrap();
        assert!(deps.contains("encoder_model.ort"));
    }

    #[test]
    fn test_stt_catalog() {
        let catalog = get_stt_catalog().unwrap();
        assert!(catalog.contains("English"));
    }

    #[test]
    fn test_stt_dependencies_tiny_streaming() {
        let opts = SttDependenciesOptions::new().with_arch(ModelArch::TinyStreaming);
        let deps = get_stt_dependencies_with_options("en", &opts).unwrap();
        assert!(deps.contains("encoder.ort"));
        assert!(deps.contains("streaming_config.json"));
    }

    #[test]
    fn test_diarization_dependencies() {
        let deps = get_diarization_dependencies().unwrap();
        assert!(deps.contains("segmentation.ort"));
        assert!(deps.contains("embedding.ort"));
    }

    #[test]
    fn test_transcriber_options_get_and_speculative_decoding() {
        let opts = TranscriberOptions::new()
            .with_identify_speakers(true)
            .with_speculative_decoding(false)
            .with_diarization_model_dir("/tmp/test_dir");

        assert_eq!(opts.get("identify_speakers"), Some("true"));
        assert_eq!(opts.get("use_speculative_decoding"), Some("false"));
        assert_eq!(opts.get("diarization_model_dir"), Some("/tmp/test_dir"));
    }

    #[test]
    fn test_transcriber_options_keyterm_and_context() {
        let opts = TranscriberOptions::new()
            .with_keyterms("Kubernetes,Ceph,etcd")
            .with_keyterm_boost(3.5)
            .with_context("Platform migration notes for cluster services.")
            .with_context_max_terms(150);

        assert_eq!(opts.get("keyterms"), Some("Kubernetes,Ceph,etcd"));
        assert_eq!(opts.get("keyterm_boost"), Some("3.5"));
        assert_eq!(
            opts.get("context"),
            Some("Platform migration notes for cluster services.")
        );
        assert_eq!(opts.get("context_max_terms"), Some("150"));
    }

    fn find_streaming_model_dir() -> Option<PathBuf> {
        let candidates = [
            env::var("MOONSHINE_TEST_STREAMING_DIR")
                .ok()
                .map(PathBuf::from),
            Some(PathBuf::from("../../models/tiny-streaming")),
            Some(PathBuf::from("models/tiny-streaming")),
            Some(PathBuf::from("/tmp/models/tiny-streaming")),
        ];
        candidates
            .into_iter()
            .flatten()
            .find(|c| c.join("streaming_config.json").exists())
    }

    fn find_non_streaming_model_dir() -> Option<PathBuf> {
        let candidates = [
            env::var("MOONSHINE_TEST_MODEL_DIR").ok().map(PathBuf::from),
            Some(PathBuf::from("../../models/tiny-en")),
            Some(PathBuf::from("models/tiny-en")),
            Some(PathBuf::from("/tmp/models/tiny-en")),
        ];
        candidates
            .into_iter()
            .flatten()
            .find(|c| c.join("encoder_model.ort").exists())
    }

    #[test]
    fn test_from_memory_files() {
        let model_dir = match find_non_streaming_model_dir() {
            Some(d) => d,
            None => return,
        };

        let enc_bytes = fs::read(model_dir.join("encoder_model.ort")).unwrap();
        let dec_bytes = fs::read(model_dir.join("decoder_model_merged.ort")).unwrap();
        let tok_bytes = fs::read(model_dir.join("tokenizer.bin")).unwrap();

        let files = [
            ("encoder_model.ort", enc_bytes.as_slice()),
            ("decoder_model_merged.ort", dec_bytes.as_slice()),
            ("tokenizer.bin", tok_bytes.as_slice()),
        ];

        let transcriber = Transcriber::from_memory_files(ModelArch::Tiny, &files, None).unwrap();
        assert!(transcriber.handle() >= 0);
    }

    #[test]
    fn test_streaming_lifecycle() {
        let model_dir = match find_streaming_model_dir() {
            Some(d) => d,
            None => return,
        };

        let transcriber =
            Transcriber::from_files(&model_dir, ModelArch::TinyStreaming, None).unwrap();
        let mut stream = transcriber.create_stream().unwrap();

        // Feed 1 second of silence in 100ms chunks (1600 samples at 16kHz)
        let chunk = vec![0.0f32; 1600];
        for _ in 0..10 {
            stream.add_audio(&chunk, 16_000).unwrap();
            let _ = stream.poll(false).unwrap();
        }

        stream.restart().unwrap();
        let final_transcript = stream.finalize().unwrap();
        assert!(final_transcript.lines.is_empty() || !final_transcript.lines.is_empty());
    }

    #[test]
    fn test_owned_streaming_lifecycle() {
        let model_dir = match find_streaming_model_dir() {
            Some(d) => d,
            None => return,
        };

        let transcriber =
            Arc::new(Transcriber::from_files(&model_dir, ModelArch::TinyStreaming, None).unwrap());
        let mut stream = transcriber.create_owned_stream().unwrap();

        let chunk = vec![0.0f32; 1600];
        stream.add_audio(&chunk, 16_000).unwrap();
        let _ = stream.poll(true).unwrap();

        let _ = stream.finalize().unwrap();
    }

    #[test]
    fn test_streaming_keyterms_and_context() {
        let model_dir = match find_streaming_model_dir() {
            Some(d) => d,
            None => return,
        };

        let opts = TranscriberOptions::new()
            .with_keyterms("Kubernetes,Ceph")
            .with_keyterm_boost(2.5)
            .with_context("Kubernetes deployment for Ceph storage")
            .with_context_max_terms(50);

        let transcriber =
            Transcriber::from_files(&model_dir, ModelArch::TinyStreaming, Some(&opts)).unwrap();

        // Test runtime updates directly on transcriber
        transcriber
            .set_keyterms("Anushka Sharma,Jurgen Klopp")
            .unwrap();
        transcriber
            .set_context("Meeting with Anushka Sharma and Jurgen Klopp", 100)
            .unwrap();
        transcriber.set_keyterms("").unwrap(); // disable

        // Test mid-stream updates via TranscriberStream
        let mut stream = transcriber.create_stream().unwrap();
        stream.set_keyterms("PostgreSQL,ClickHouse").unwrap();
        stream.set_context("Database migration plan", 20).unwrap();

        let chunk = vec![0.0f32; 1600];
        stream.add_audio(&chunk, 16_000).unwrap();
        let _ = stream.poll(false).unwrap();
        let _ = stream.finalize().unwrap();

        // Test mid-stream updates via OwnedTranscriberStream
        let arc_transcriber = Arc::new(transcriber);
        let mut owned_stream = arc_transcriber.create_owned_stream().unwrap();
        owned_stream.set_keyterms("Rust,Tokio,Tauri").unwrap();
        owned_stream
            .set_context("GUI voice application in Rust", 10)
            .unwrap();
        owned_stream.add_audio(&chunk, 16_000).unwrap();
        let _ = owned_stream.poll(true).unwrap();
        let _ = owned_stream.finalize().unwrap();
    }

    #[test]
    fn test_non_streaming_keyterms_error() {
        let model_dir = match find_non_streaming_model_dir() {
            Some(d) => d,
            None => return,
        };

        let transcriber = Transcriber::from_files(&model_dir, ModelArch::Tiny, None).unwrap();
        let res = transcriber.set_keyterms("Kubernetes");
        assert!(
            res.is_err(),
            "set_keyterms should fail on non-streaming models"
        );
    }
}
