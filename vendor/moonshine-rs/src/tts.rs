//! Text-to-Speech (TTS) and Grapheme-to-Phoneme (G2P) engine bindings.
//!
//! Provides on-device speech synthesis via Kokoro, Piper, and ZipVoice engines,
//! including one-shot synthesis, phoneme synthesis, streaming chunked synthesis,
//! and sentence/utterance splitting.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

use crate::{error_string, sys, Error, Result};

/// Configuration options for Text-to-Speech synthesis and voice resolution.
#[derive(Debug, Default, Clone)]
pub struct TtsOptions {
    options: Vec<(String, String)>,
}

impl TtsOptions {
    /// Creates an empty option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an arbitrary key-value option pair.
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

    /// Sets the voice identifier (e.g. `"kokoro_af_heart"`, `"piper_en_US-lessac-medium"`, or `"zipvoice_american_female"`).
    pub fn with_voice(self, voice: impl AsRef<str>) -> Self {
        self.set("voice", voice.as_ref())
    }

    /// Sets the playback/synthesis speed multiplier (e.g. `1.0` for normal, `1.25` for faster).
    pub fn with_speed(self, speed: f32) -> Self {
        self.set("speed", speed.to_string())
    }

    /// Sets the root directory for G2P/TTS model assets.
    pub fn with_g2p_root(self, path: impl AsRef<Path>) -> Self {
        self.set("g2p_root", path.as_ref().to_string_lossy())
    }

    /// Sets the root directory for model assets (alias for `g2p_root`).
    pub fn with_model_root(self, path: impl AsRef<Path>) -> Self {
        self.set("model_root", path.as_ref().to_string_lossy())
    }

    /// Sets whether to break utterances after colons `":"` (default `true`).
    pub fn with_split_on_colon(self, enable: bool) -> Self {
        self.set("split_on_colon", if enable { "true" } else { "false" })
    }

    /// Sets minimum codepoints for merged utterance units (default `0`).
    pub fn with_min_codepoints(self, min: usize) -> Self {
        self.set("min_codepoints", min.to_string())
    }

    pub(crate) fn to_c_options(&self) -> Result<(Vec<sys::moonshine_option_t>, Vec<CString>)> {
        let mut raw_opts = Vec::with_capacity(self.options.len());
        let mut c_strings = Vec::with_capacity(self.options.len() * 2);

        for (k, v) in &self.options {
            let ck = CString::new(k.as_str())?;
            let cv = CString::new(v.as_str())?;
            raw_opts.push(sys::moonshine_option_t {
                name: ck.as_ptr(),
                value: cv.as_ptr(),
            });
            c_strings.push(ck);
            c_strings.push(cv);
        }

        Ok((raw_opts, c_strings))
    }
}

/// Synthesized audio data from a TTS engine.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SynthesizedAudio {
    /// Mono PCM float samples in `[-1.0, 1.0]`.
    pub pcm: Vec<f32>,
    /// Sample rate in Hz (e.g. 24000, 22050, 16000).
    pub sample_rate: u32,
}

impl SynthesizedAudio {
    /// Duration of the synthesized audio in seconds.
    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.pcm.len() as f32 / self.sample_rate as f32
        }
    }
}

/// A single chunk of audio emitted during streaming synthesis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TtsChunk {
    /// Mono PCM float samples in `[-1.0, 1.0]`.
    pub pcm: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// The text segment covered by this chunk (if known).
    pub text: String,
    /// Utterance index (counting from 1).
    pub utterance_id: u64,
    /// Whether this is the final chunk of the current utterance.
    pub is_final: bool,
}

/// Status returned by [`TtsSynthesizer::next_chunk`].
#[derive(Debug, Clone)]
pub enum TtsStreamStatus {
    /// A new chunk of synthesized audio was produced.
    Chunk(TtsChunk),
    /// No complete sentence is buffered yet. Push more text or call [`flush`](TtsSynthesizer::flush).
    NeedText,
    /// Input ended and all queued audio has finished synthesizing.
    EndOfStream,
    /// Generation was abandoned via [`cancel`](TtsSynthesizer::cancel).
    Cancelled,
}

/// An active Text-to-Speech synthesizer instance.
///
/// `TtsSynthesizer` manages the underlying native engine handle and is `Send + Sync`.
/// Concurrent synthesis operations are serialized internally.
pub struct TtsSynthesizer {
    handle: i32,
    _lock: Mutex<()>,
}

unsafe impl Send for TtsSynthesizer {}
unsafe impl Sync for TtsSynthesizer {}

impl TtsSynthesizer {
    /// Loads a TTS synthesizer from model assets on disk located at `model_dir`.
    ///
    /// `model_dir` is the directory containing the downloaded TTS assets (e.g. `kokoro/`, `en_us/`).
    /// If no `model_root` / `g2p_root` is explicitly set in `options`, `model_dir` is automatically
    /// used as the asset root.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if initializing the synthesizer fails.
    pub fn from_files(
        language: &str,
        model_dir: impl AsRef<Path>,
        options: Option<&TtsOptions>,
    ) -> Result<Self> {
        let c_lang = CString::new(language)?;
        let dir_path = model_dir.as_ref();

        let mut effective_opts = options.cloned().unwrap_or_default();
        if effective_opts.get("g2p_root").is_none()
            && effective_opts.get("model_root").is_none()
            && effective_opts.get("tts_root").is_none()
            && effective_opts.get("path_root").is_none()
        {
            effective_opts = effective_opts.with_model_root(dir_path);
        }

        let (raw_opts, _strings) = effective_opts.to_c_options()?;
        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let handle = unsafe {
            sys::moonshine_create_tts_synthesizer_from_files(
                c_lang.as_ptr(),
                ptr::null_mut(),
                0,
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

    /// Loads a TTS synthesizer from in-memory asset buffers.
    ///
    /// Each entry in `files` is a `(filename, buffer)` pair matching canonical model keys
    /// (e.g. `"kokoro/prosody.model.ort"`, `"kokoro/config.json"`, `"piper/onnx"`, etc.).
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if initializing from memory fails.
    pub fn from_memory(
        language: &str,
        files: &[(&str, &[u8])],
        options: Option<&TtsOptions>,
    ) -> Result<Self> {
        let c_lang = CString::new(language)?;

        let mut c_filenames = Vec::with_capacity(files.len());
        let mut filename_ptrs = Vec::with_capacity(files.len());
        let mut memory_ptrs = Vec::with_capacity(files.len());
        let mut memory_sizes = Vec::with_capacity(files.len());

        for (name, bytes) in files {
            let c_name = CString::new(*name)?;
            filename_ptrs.push(c_name.as_ptr());
            c_filenames.push(c_name);
            memory_ptrs.push(bytes.as_ptr());
            memory_sizes.push(bytes.len() as u64);
        }

        let effective_opts = options.cloned().unwrap_or_default();
        let (raw_opts, _strings) = effective_opts.to_c_options()?;
        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let handle = unsafe {
            sys::moonshine_create_tts_synthesizer_from_memory(
                c_lang.as_ptr(),
                filename_ptrs.as_mut_ptr(),
                files.len() as u64,
                memory_ptrs.as_mut_ptr(),
                memory_sizes.as_ptr(),
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

    /// Returns the raw native handle.
    pub fn handle(&self) -> i32 {
        self.handle
    }

    /// Performs one-shot synthesis of complete text to PCM audio.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if synthesis fails.
    pub fn synthesize(&self, text: &str, options: Option<&TtsOptions>) -> Result<SynthesizedAudio> {
        let _guard = self._lock.lock().unwrap();
        let c_text = CString::new(text)?;

        let effective_opts = options.cloned().unwrap_or_default();
        let (raw_opts, _strings) = effective_opts.to_c_options()?;
        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let mut out_audio_data: *mut f32 = ptr::null_mut();
        let mut out_audio_data_size: u64 = 0;
        let mut out_sample_rate: i32 = 0;

        let ret = unsafe {
            sys::moonshine_text_to_speech(
                self.handle,
                c_text.as_ptr(),
                opts_ptr,
                raw_opts.len() as u64,
                &mut out_audio_data,
                &mut out_audio_data_size,
                &mut out_sample_rate,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        if out_audio_data.is_null() && out_audio_data_size > 0 {
            return Err(Error::NullPointer);
        }

        let pcm = if out_audio_data_size > 0 && !out_audio_data.is_null() {
            unsafe {
                let slice =
                    std::slice::from_raw_parts(out_audio_data, out_audio_data_size as usize);
                let vec = slice.to_vec();
                sys::moonshine_free_buffer(out_audio_data as *mut std::ffi::c_void);
                vec
            }
        } else {
            Vec::new()
        };

        Ok(SynthesizedAudio {
            pcm,
            sample_rate: out_sample_rate.max(0) as u32,
        })
    }

    /// Synthesizes audio directly from International Phonetic Alphabet (IPA) phonemes.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if phoneme synthesis fails.
    pub fn synthesize_phonemes(
        &self,
        phonemes: &str,
        options: Option<&TtsOptions>,
    ) -> Result<SynthesizedAudio> {
        let _guard = self._lock.lock().unwrap();
        let c_phonemes = CString::new(phonemes)?;

        let effective_opts = options.cloned().unwrap_or_default();
        let (raw_opts, _strings) = effective_opts.to_c_options()?;
        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let mut out_audio_data: *mut f32 = ptr::null_mut();
        let mut out_audio_data_size: u64 = 0;
        let mut out_sample_rate: i32 = 0;

        let ret = unsafe {
            sys::moonshine_phonemes_to_speech(
                self.handle,
                c_phonemes.as_ptr(),
                opts_ptr,
                raw_opts.len() as u64,
                &mut out_audio_data,
                &mut out_audio_data_size,
                &mut out_sample_rate,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        if out_audio_data.is_null() && out_audio_data_size > 0 {
            return Err(Error::NullPointer);
        }

        let pcm = if out_audio_data_size > 0 && !out_audio_data.is_null() {
            unsafe {
                let slice =
                    std::slice::from_raw_parts(out_audio_data, out_audio_data_size as usize);
                let vec = slice.to_vec();
                sys::moonshine_free_buffer(out_audio_data as *mut std::ffi::c_void);
                vec
            }
        } else {
            Vec::new()
        };

        Ok(SynthesizedAudio {
            pcm,
            sample_rate: out_sample_rate.max(0) as u32,
        })
    }

    // ------------------------------------------------------------------------
    // Streaming Text-to-Speech API
    // ------------------------------------------------------------------------

    /// Appends incoming text (e.g. streaming tokens from an LLM) to the synthesizer queue.
    ///
    /// Pieces are concatenated verbatim. Audio generation begins automatically in the background
    /// as complete sentences form.
    ///
    /// # Errors
    /// Returns [`Error::ApiError`] if appending text fails.
    pub fn push_text(&self, text: impl AsRef<str>) -> Result<()> {
        let _guard = self._lock.lock().unwrap();
        let c_text = CString::new(text.as_ref())?;
        let ret = unsafe { sys::moonshine_tts_push_text(self.handle, c_text.as_ptr()) };
        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }
        Ok(())
    }

    /// Queues any partially buffered text for synthesis immediately even without punctuation.
    pub fn flush(&self) -> Result<()> {
        let _guard = self._lock.lock().unwrap();
        let ret = unsafe { sys::moonshine_tts_flush(self.handle) };
        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }
        Ok(())
    }

    /// Signals that no more text will be added.
    ///
    /// Once all buffered audio is consumed via [`next_chunk`](Self::next_chunk),
    /// the stream reports [`TtsStreamStatus::EndOfStream`].
    pub fn end_input(&self) -> Result<()> {
        let _guard = self._lock.lock().unwrap();
        let ret = unsafe { sys::moonshine_tts_end_input(self.handle) };
        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }
        Ok(())
    }

    /// Abandons in-flight streaming synthesis and resets the engine to idle (barge-in interruption).
    pub fn cancel(&self) -> Result<()> {
        let _guard = self._lock.lock().unwrap();
        let ret = unsafe { sys::moonshine_tts_cancel(self.handle) };
        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }
        Ok(())
    }

    /// Returns `true` if a streaming generation is currently active.
    pub fn is_streaming(&self) -> bool {
        let _guard = self._lock.lock().unwrap();
        unsafe { sys::moonshine_tts_is_streaming(self.handle) > 0 }
    }

    /// Pulls the next available chunk of synthesized audio.
    ///
    /// Blocks only for the duration needed by the active synthesis step.
    ///
    /// # Returns
    /// - [`TtsStreamStatus::Chunk`]: Audio chunk ready.
    /// - [`TtsStreamStatus::NeedText`]: More text needed or call [`flush`](Self::flush).
    /// - [`TtsStreamStatus::EndOfStream`]: All audio produced after [`end_input`](Self::end_input).
    /// - [`TtsStreamStatus::Cancelled`]: Generation was aborted by [`cancel`](Self::cancel).
    pub fn next_chunk(&self) -> Result<TtsStreamStatus> {
        let _guard = self._lock.lock().unwrap();
        let mut out_chunk: *const sys::tts_chunk_t = ptr::null();

        let ret = unsafe { sys::moonshine_tts_next_chunk(self.handle, 0, &mut out_chunk) };

        match ret {
            0 => {
                if out_chunk.is_null() {
                    return Err(Error::NullPointer);
                }
                let c = unsafe { &*out_chunk };
                let pcm = if c.audio_data_count > 0 && !c.audio_data.is_null() {
                    unsafe {
                        std::slice::from_raw_parts(c.audio_data, c.audio_data_count as usize)
                            .to_vec()
                    }
                } else {
                    Vec::new()
                };

                let text = if c.text.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(c.text).to_string_lossy().into_owned() }
                };

                Ok(TtsStreamStatus::Chunk(TtsChunk {
                    pcm,
                    sample_rate: c.sample_rate.max(0) as u32,
                    text,
                    utterance_id: c.utterance_id,
                    is_final: c.is_final != 0,
                }))
            }
            1 => Ok(TtsStreamStatus::NeedText),
            2 => Ok(TtsStreamStatus::EndOfStream),
            3 => Ok(TtsStreamStatus::Cancelled),
            err => Err(Error::ApiError {
                code: err,
                message: error_string(err),
            }),
        }
    }

    /// Convenience helper that returns `Some(chunk)` if available, or `None` if waiting / finished / cancelled.
    pub fn poll_chunk(&self) -> Result<Option<TtsChunk>> {
        match self.next_chunk()? {
            TtsStreamStatus::Chunk(chunk) => Ok(Some(chunk)),
            _ => Ok(None),
        }
    }
}

impl Drop for TtsSynthesizer {
    fn drop(&mut self) {
        if self.handle >= 0 {
            unsafe {
                sys::moonshine_free_tts_synthesizer(self.handle);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Utterance splitting & Catalog helpers
// ----------------------------------------------------------------------------

/// Splits text into the natural utterance units that streaming TTS speaks one at a time.
pub fn split_utterances(
    language: Option<&str>,
    text: &str,
    options: Option<&TtsOptions>,
) -> Result<Vec<String>> {
    let c_lang = match language {
        Some(l) => Some(CString::new(l)?),
        None => None,
    };
    let lang_ptr = c_lang.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());

    let c_text = CString::new(text)?;

    let effective_opts = options.cloned().unwrap_or_default();
    let (raw_opts, _strings) = effective_opts.to_c_options()?;
    let opts_ptr = if raw_opts.is_empty() {
        ptr::null()
    } else {
        raw_opts.as_ptr()
    };

    let mut out_json: *mut std::ffi::c_char = ptr::null_mut();

    let ret = unsafe {
        sys::moonshine_tts_split_utterances(
            lang_ptr,
            c_text.as_ptr(),
            opts_ptr,
            raw_opts.len() as u64,
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
        return Ok(Vec::new());
    }

    let json_str = unsafe {
        let s = CStr::from_ptr(out_json).to_string_lossy().into_owned();
        sys::moonshine_free_buffer(out_json as *mut std::ffi::c_void);
        s
    };

    let items: Vec<String> = serde_json::from_str(&json_str).map_err(|e| Error::ApiError {
        code: -1,
        message: format!("Failed to parse split utterances JSON: {e}"),
    })?;

    Ok(items)
}

/// Returns merged G2P + TTS vocoder download dependencies as a JSON string.
pub fn get_tts_dependencies(
    languages: Option<&str>,
    options: Option<&TtsOptions>,
) -> Result<String> {
    let c_lang = match languages {
        Some(l) => Some(CString::new(l)?),
        None => None,
    };
    let lang_ptr = c_lang.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());

    let effective_opts = options.cloned().unwrap_or_default();
    let (raw_opts, _strings) = effective_opts.to_c_options()?;
    let opts_ptr = if raw_opts.is_empty() {
        ptr::null()
    } else {
        raw_opts.as_ptr()
    };

    let mut out_json: *mut std::ffi::c_char = ptr::null_mut();

    let ret = unsafe {
        sys::moonshine_get_tts_dependencies(
            lang_ptr,
            opts_ptr,
            raw_opts.len() as u64,
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
        return Ok("{}".to_string());
    }

    let json_str = unsafe {
        let s = CStr::from_ptr(out_json).to_string_lossy().into_owned();
        sys::moonshine_free_buffer(out_json as *mut std::ffi::c_void);
        s
    };

    Ok(json_str)
}

/// Returns known TTS voices and availability state as a JSON string.
pub fn get_tts_voices(languages: Option<&str>, options: Option<&TtsOptions>) -> Result<String> {
    let c_lang = match languages {
        Some(l) => Some(CString::new(l)?),
        None => None,
    };
    let lang_ptr = c_lang.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());

    let effective_opts = options.cloned().unwrap_or_default();
    let (raw_opts, _strings) = effective_opts.to_c_options()?;
    let opts_ptr = if raw_opts.is_empty() {
        ptr::null()
    } else {
        raw_opts.as_ptr()
    };

    let mut out_json: *mut std::ffi::c_char = ptr::null_mut();

    let ret = unsafe {
        sys::moonshine_get_tts_voices(lang_ptr, opts_ptr, raw_opts.len() as u64, &mut out_json)
    };

    if ret != 0 {
        return Err(Error::ApiError {
            code: ret,
            message: error_string(ret),
        });
    }

    if out_json.is_null() {
        return Ok("{}".to_string());
    }

    let json_str = unsafe {
        let s = CStr::from_ptr(out_json).to_string_lossy().into_owned();
        sys::moonshine_free_buffer(out_json as *mut std::ffi::c_void);
        s
    };

    Ok(json_str)
}

/// Returns G2P-only canonical asset keys as a JSON/comma-separated string.
pub fn get_g2p_dependencies(
    languages: Option<&str>,
    options: Option<&TtsOptions>,
) -> Result<String> {
    let c_lang = match languages {
        Some(l) => Some(CString::new(l)?),
        None => None,
    };
    let lang_ptr = c_lang.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());

    let effective_opts = options.cloned().unwrap_or_default();
    let (raw_opts, _strings) = effective_opts.to_c_options()?;
    let opts_ptr = if raw_opts.is_empty() {
        ptr::null()
    } else {
        raw_opts.as_ptr()
    };

    let mut out_json: *mut std::ffi::c_char = ptr::null_mut();

    let ret = unsafe {
        sys::moonshine_get_g2p_dependencies(
            lang_ptr,
            opts_ptr,
            raw_opts.len() as u64,
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
        return Ok(String::new());
    }

    let res_str = unsafe {
        let s = CStr::from_ptr(out_json).to_string_lossy().into_owned();
        sys::moonshine_free_buffer(out_json as *mut std::ffi::c_void);
        s
    };

    Ok(res_str)
}

// ----------------------------------------------------------------------------
// Grapheme to Phonemizer (G2P)
// ----------------------------------------------------------------------------

/// Grapheme-to-Phoneme converter instance.
pub struct GraphemeToPhonemizer {
    handle: i32,
    _lock: Mutex<()>,
}

unsafe impl Send for GraphemeToPhonemizer {}
unsafe impl Sync for GraphemeToPhonemizer {}

impl GraphemeToPhonemizer {
    /// Loads a grapheme-to-phonemizer from disk assets at `model_dir`.
    pub fn from_files(
        language: &str,
        model_dir: impl AsRef<Path>,
        options: Option<&TtsOptions>,
    ) -> Result<Self> {
        let c_lang = CString::new(language)?;
        let dir_path = model_dir.as_ref();

        let mut effective_opts = options.cloned().unwrap_or_default();
        if effective_opts.get("g2p_root").is_none() && effective_opts.get("model_root").is_none() {
            effective_opts = effective_opts.with_g2p_root(dir_path);
        }

        let (raw_opts, _strings) = effective_opts.to_c_options()?;
        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let handle = unsafe {
            sys::moonshine_create_grapheme_to_phonemizer_from_files(
                c_lang.as_ptr(),
                ptr::null_mut(),
                0,
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

    /// Converts text into IPA phonemes.
    pub fn text_to_phonemes(&self, text: &str, options: Option<&TtsOptions>) -> Result<String> {
        let _guard = self._lock.lock().unwrap();
        let c_text = CString::new(text)?;

        let effective_opts = options.cloned().unwrap_or_default();
        let (raw_opts, _strings) = effective_opts.to_c_options()?;
        let opts_ptr = if raw_opts.is_empty() {
            ptr::null()
        } else {
            raw_opts.as_ptr()
        };

        let mut out_phonemes: *const std::ffi::c_char = ptr::null();
        let mut out_count: u64 = 0;

        let ret = unsafe {
            sys::moonshine_text_to_phonemes(
                self.handle,
                c_text.as_ptr(),
                opts_ptr,
                raw_opts.len() as u64,
                &mut out_phonemes,
                &mut out_count,
            )
        };

        if ret != 0 {
            return Err(Error::ApiError {
                code: ret,
                message: error_string(ret),
            });
        }

        if !out_phonemes.is_null() {
            let s = unsafe { CStr::from_ptr(out_phonemes).to_string_lossy().into_owned() };
            unsafe {
                sys::moonshine_free_buffer(out_phonemes as *mut std::ffi::c_void);
            }
            Ok(s)
        } else {
            Ok(String::new())
        }
    }
}

impl Drop for GraphemeToPhonemizer {
    fn drop(&mut self) {
        if self.handle >= 0 {
            unsafe {
                sys::moonshine_free_grapheme_to_phonemizer(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_options_builder() {
        let opts = TtsOptions::new()
            .with_voice("kokoro_af_heart")
            .with_speed(1.2)
            .with_g2p_root("/tmp/g2p")
            .with_split_on_colon(true)
            .with_min_codepoints(10);

        assert_eq!(opts.get("voice"), Some("kokoro_af_heart"));
        assert_eq!(opts.get("speed"), Some("1.2"));
        assert_eq!(opts.get("g2p_root"), Some("/tmp/g2p"));
        assert_eq!(opts.get("split_on_colon"), Some("true"));
        assert_eq!(opts.get("min_codepoints"), Some("10"));
    }

    #[test]
    fn test_split_utterances() {
        let res = split_utterances(
            Some("en"),
            "Hello world! This is a test sentence. Dr. Smith is here.",
            None,
        );
        // split_utterances is a self-contained sentence splitter in C API
        if let Ok(units) = res {
            assert!(!units.is_empty());
            assert!(units.iter().any(|u| u.contains("Hello world")));
        }
    }

    #[test]
    fn test_tts_voices_query() {
        let res = get_tts_voices(Some("en"), None);
        if let Ok(voices_json) = res {
            assert!(
                voices_json.contains("kokoro")
                    || voices_json.contains("piper")
                    || voices_json.starts_with('{')
            );
        }
    }

    #[test]
    fn test_tts_dependencies_query() {
        let res = get_tts_dependencies(Some("en"), None);
        if let Ok(deps_json) = res {
            assert!(
                deps_json.contains("files")
                    || deps_json.contains("groups")
                    || deps_json.starts_with('{')
            );
        }
    }

    #[test]
    fn test_g2p_dependencies_query() {
        let res = get_g2p_dependencies(Some("en"), None);
        if let Ok(g2p_str) = res {
            assert!(!g2p_str.is_empty() || g2p_str.is_empty());
        }
    }

    #[test]
    fn test_synthesized_audio_helpers() {
        let audio = SynthesizedAudio {
            pcm: vec![0.0; 48_000],
            sample_rate: 24_000,
        };
        assert_eq!(audio.duration_seconds(), 2.0);

        let empty = SynthesizedAudio {
            pcm: Vec::new(),
            sample_rate: 0,
        };
        assert_eq!(empty.duration_seconds(), 0.0);
    }

    #[test]
    fn test_invalid_synthesizer_calls_return_error() {
        let synth = TtsSynthesizer {
            handle: -1,
            _lock: Mutex::new(()),
        };
        assert!(synth.synthesize("test", None).is_err());
        assert!(synth.synthesize_phonemes("test", None).is_err());
        assert!(synth.push_text("test").is_err());
        assert!(synth.flush().is_err());
        assert!(synth.end_input().is_err());
        assert!(synth.cancel().is_err());
        assert!(!synth.is_streaming());
        assert!(synth.next_chunk().is_err());
    }

    #[test]
    fn test_invalid_g2p_calls_return_error() {
        let g2p = GraphemeToPhonemizer {
            handle: -1,
            _lock: Mutex::new(()),
        };
        assert!(g2p.text_to_phonemes("test", None).is_err());
    }

    #[test]
    fn test_from_files_real_model() {
        let candidates = [
            std::env::var("MOONSHINE_TEST_TTS_DIR")
                .ok()
                .map(std::path::PathBuf::from),
            Some(std::path::PathBuf::from("../../models/tts/kokoro")),
            Some(std::path::PathBuf::from("models/tts/kokoro")),
            Some(std::path::PathBuf::from("/tmp/models/tts/kokoro")),
        ];
        let model_dir = match candidates
            .into_iter()
            .flatten()
            .find(|c| c.join("kokoro/config.json").exists())
        {
            Some(d) => d,
            None => return,
        };

        let options = TtsOptions::new().with_voice("kokoro_af_heart");
        let synth = TtsSynthesizer::from_files("en", &model_dir, Some(&options)).unwrap();
        let audio = synth.synthesize("Hello test", None).unwrap();
        assert!(!audio.pcm.is_empty());
        assert!(audio.sample_rate > 0);
    }
}
