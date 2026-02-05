use anyhow::Result;
use candle::Device;
use hf_hub::Cache;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;

use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use uuid::Uuid;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
// unix-only stderr redirection helpers

use crate::WordTimestamp;
use crate::config::{SentenceDetectionConfig, WhisperConfig};

#[derive(Clone)]
pub struct WhisperModel {
    context: Arc<Mutex<WhisperContext>>,
    config: WhisperConfig,
    model_path: PathBuf,
    model_name: String,
    forced_lang: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceBuffer {
    pub id: String,
    pub audio_samples: Vec<f32>,
    pub start_time: f64,
    pub end_time: f64,
    pub kyutai_text: String,
    pub words: Vec<WordTimestamp>,
}

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    samples: VecDeque<f32>,
    timestamps: VecDeque<f64>,
    max_duration: f64,
    sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct SentenceDetector {
    current_sentence: Option<SentenceBuffer>,
    silence_start: Option<f64>,
    last_word_time: f64,
    config: SentenceDetectionConfig,
    accumulated_text: String,
    accumulated_words: Vec<WordTimestamp>,
}

#[derive(Debug, Clone, Serialize)]
pub enum WhisperStatus {
    Pending,
    Processing,
    Corrected(String),
    Confirmed,
}

#[derive(Debug, Clone, Serialize)]
pub enum WhisperMessage {
    Processing {
        sentence_id: String,
        original_text: String,
        start_time: f64,
        end_time: f64,
    },
    Complete {
        sentence_id: String,
        original_text: String,
        corrected_text: String,
        confidence: f32,
        changed: bool,
    },
}

impl WhisperModel {
    pub async fn load(
        config: &WhisperConfig,
        model_override: Option<&str>,
        quantization_override: Option<&str>,
        _device: Device, // Not used with whisper-rs, but kept for compatibility
        forced_lang: Option<&str>,
    ) -> Result<Self> {
        let model_name = model_override.unwrap_or(&config.default_model);
        let quantization = quantization_override.unwrap_or(&config.quantization);

        let model_path = Self::download_model(model_name, quantization, config).await?;

        // Load the model using whisper-rs
        let ctx_params = WhisperContextParameters::default();
        let context = with_stderr_silenced(|| {
            WhisperContext::new_with_params(model_path.to_str().unwrap(), ctx_params)
        })
        .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {}", e))?;

        Ok(Self {
            context: Arc::new(Mutex::new(context)),
            config: config.clone(),
            model_path,
            model_name: model_name.to_string(),
            forced_lang: forced_lang.map(|s| s.to_string()),
        })
    }

    async fn download_model(
        model: &str,
        quantization: &str,
        config: &WhisperConfig,
    ) -> Result<PathBuf> {
        let storage_path = if config.storage_dir == "default" {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("~/.cache"))
                .join("huggingface")
                .join("whisper-models")
        } else {
            PathBuf::from(&config.storage_dir)
        };

        std::fs::create_dir_all(&storage_path)?;

        // For GGUF models, use ggerganov's whisper.cpp repo
        let repo_name = "ggerganov/whisper.cpp".to_string();

        let filename = if config.model_format == "gguf" {
            // Correct GGML filename format based on model and quantization (using .bin extension)
            let name = match model {
                "large-v3-turbo" => match quantization {
                    "Q5_0" | "Q5" => "ggml-large-v3-turbo-q5_0.bin",
                    "Q8_0" | "Q8" => "ggml-large-v3-turbo-q8_0.bin",
                    "f32" | "F32" => "ggml-large-v3-turbo.bin",
                    _ => "ggml-large-v3-turbo-q5_0.bin", // Default to Q5_0
                },
                "large-v3" => match quantization {
                    "Q5_0" | "Q5" => "ggml-large-v3-q5_0.bin",
                    "Q8_0" | "Q8" => "ggml-large-v3-q8_0.bin",
                    _ => "ggml-large-v3.bin",
                },
                "medium" => match quantization {
                    "Q5_0" | "Q5" => "ggml-medium-q5_0.bin",
                    _ => "ggml-medium.bin",
                },
                _ => return Err(anyhow::anyhow!("Unsupported model: {}", model)),
            };
            name.to_string()
        } else {
            "model.safetensors".to_string()
        };

        let model_file_path = storage_path.join(&filename);

        // Check if model already exists
        if model_file_path.exists() {
            return Ok(model_file_path);
        }

        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()?;
        let repo = api.model(repo_name.clone());
        let cache = Cache::from_env();
        let cache_repo = cache.model(repo_name.clone());

        let downloaded_path = if cache_repo.get(&filename).is_some() {
            repo.get(&filename)?
        } else {
            let pb = ProgressBar::new(0);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "[{elapsed_precise}] {bar:40.cyan/blue} {bytes:>7}/{total_bytes:7} {msg}",
                    )?
                    .progress_chars("##-"),
            );
            pb.set_message(format!("Downloading {}", filename));
            repo.download_with_progress(&filename, pb)?
        };
        std::fs::copy(&downloaded_path, &model_file_path)?;

        Ok(model_file_path)
    }

    pub fn transcribe_audio(&self, audio_samples: &[f32]) -> Result<String> {
        // Resample from 24kHz to 16kHz (Whisper's expected rate)
        let resampled = if audio_samples.len() > 0 {
            let ratio = 16000.0 / 24000.0;
            let new_len = (audio_samples.len() as f32 * ratio) as usize;
            let mut resampled = Vec::with_capacity(new_len);

            for i in 0..new_len {
                let pos = i as f32 / ratio;
                let idx = pos as usize;
                let frac = pos - idx as f32;

                if idx + 1 < audio_samples.len() {
                    let sample = audio_samples[idx] * (1.0 - frac) + audio_samples[idx + 1] * frac;
                    resampled.push(sample);
                } else if idx < audio_samples.len() {
                    resampled.push(audio_samples[idx]);
                }
            }
            resampled
        } else {
            audio_samples.to_vec()
        };

        // Lock the context and create a new state for this transcription
        let context = self.context.lock().unwrap();
        let mut state = context
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {}", e))?;

        // Set up parameters with all output suppressed
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Configure parameters
        if let Some(lang) = &self.forced_lang {
            params.set_language(Some(lang));
        } else {
            params.set_language(Some("auto"));
        }
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_single_segment(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_suppress_blank(true);
        params.set_suppress_non_speech_tokens(true);

        // Process the audio (whisper-rs expects f32 samples)
        with_stderr_silenced(|| state.full(params, &resampled))
            .map_err(|e| anyhow::anyhow!("Whisper processing failed: {}", e))?;

        // Get the transcribed text
        let num_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed to get segments: {}", e))?;

        let mut text = String::new();
        for i in 0..num_segments {
            let segment_text = state
                .full_get_segment_text(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment text: {}", e))?;
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&segment_text);
        }

        Ok(text.trim().to_string())
    }
}

// Suppress noisy stderr logs from whisper.cpp / ggml during critical calls
#[inline]
fn with_stderr_silenced<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    #[cfg(unix)]
    {
        use libc::{O_WRONLY, STDERR_FILENO, close, dup, dup2, fflush, open};
        unsafe {
            // Flush stderr and redirect to /dev/null
            fflush(std::ptr::null_mut());
            let old_stderr_fd = dup(STDERR_FILENO);
            let devnull = open(b"/dev/null\0".as_ptr() as *const i8, O_WRONLY);
            if devnull >= 0 {
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
            let result = f();
            // Restore stderr
            if old_stderr_fd >= 0 {
                dup2(old_stderr_fd, STDERR_FILENO);
                close(old_stderr_fd);
            }
            result
        }
    }
    #[cfg(not(unix))]
    {
        f()
    }
}

impl AudioBuffer {
    pub fn new(max_duration: f64, sample_rate: u32) -> Self {
        Self {
            samples: VecDeque::new(),
            timestamps: VecDeque::new(),
            max_duration,
            sample_rate,
        }
    }

    pub fn push_samples(&mut self, samples: &[f32], timestamp: f64) {
        // Ensure we have proper timestamping
        let sample_duration = 1.0 / self.sample_rate as f64;
        for (i, &sample) in samples.iter().enumerate() {
            self.samples.push_back(sample);
            self.timestamps
                .push_back(timestamp + (i as f64 * sample_duration));
        }

        // Remove old samples that exceed max_duration
        let cutoff_time = timestamp - self.max_duration;
        while let Some(&front_time) = self.timestamps.front() {
            if front_time < cutoff_time {
                self.timestamps.pop_front();
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn extract_segment(&self, start_time: f64, end_time: f64) -> Vec<f32> {
        let mut segment = Vec::new();

        // Add a small tolerance for timestamp matching
        let tolerance = 0.01; // 10ms tolerance

        for (i, &timestamp) in self.timestamps.iter().enumerate() {
            if timestamp >= (start_time - tolerance) && timestamp <= (end_time + tolerance) {
                if let Some(&sample) = self.samples.get(i) {
                    segment.push(sample);
                }
            }
        }

        // Ensure we have at least some samples
        if segment.is_empty() && !self.samples.is_empty() {
            // Fallback: try to get samples around the expected time
            let start_idx = ((start_time * 24000.0) as usize).min(self.samples.len());
            let end_idx = ((end_time * 24000.0) as usize).min(self.samples.len());

            if start_idx < end_idx {
                for i in start_idx..end_idx {
                    if let Some(&sample) = self.samples.get(i) {
                        segment.push(sample);
                    }
                }
            }
        }

        segment
    }
}

impl SentenceDetector {
    pub fn new(config: SentenceDetectionConfig) -> Self {
        Self {
            current_sentence: None,
            silence_start: None,
            last_word_time: 0.0,
            config,
            accumulated_text: String::new(),
            accumulated_words: Vec::new(),
        }
    }

    pub fn process_word(
        &mut self,
        word: &WordTimestamp,
        vad_confidence: Option<f32>,
    ) -> Option<SentenceBuffer> {
        // Add word to current accumulation
        if !self.accumulated_text.is_empty() {
            self.accumulated_text.push(' ');
        }
        self.accumulated_text.push_str(&word.word);
        self.accumulated_words.push(word.clone());

        // Check for sentence boundary conditions
        let has_punctuation = self
            .config
            .punctuation_markers
            .iter()
            .any(|marker| word.word.ends_with(marker));

        let silence_duration = if let Some(end_time) = word.end_time {
            end_time - self.last_word_time
        } else {
            word.start_time - self.last_word_time
        };

        let has_long_pause = silence_duration > self.config.silence_duration;
        let has_vad_pause =
            vad_confidence.map_or(false, |conf| conf > self.config.vad_pause_threshold);

        let sentence_duration = if let Some(first_word) = self.accumulated_words.first() {
            word.start_time - first_word.start_time
        } else {
            0.0
        };

        let is_min_duration =
            sentence_duration >= self.config.min_duration || self.accumulated_words.len() >= 3;
        let is_max_duration = sentence_duration >= self.config.max_duration;

        // Determine if we should complete the sentence
        let should_complete = is_min_duration
            && (has_punctuation || has_long_pause || has_vad_pause)
            || is_max_duration;

        if should_complete && !self.accumulated_words.is_empty() {
            let sentence = SentenceBuffer {
                id: Uuid::new_v4().to_string(),
                audio_samples: vec![], // Will be filled by audio buffer
                start_time: self.accumulated_words.first().unwrap().start_time,
                end_time: word.end_time.unwrap_or(word.start_time),
                kyutai_text: self.accumulated_text.trim().to_string(),
                words: self.accumulated_words.clone(),
            };

            // Reset for next sentence
            self.accumulated_text.clear();
            self.accumulated_words.clear();
            self.last_word_time = word.end_time.unwrap_or(word.start_time);

            Some(sentence)
        } else {
            self.last_word_time = word.end_time.unwrap_or(word.start_time);
            None
        }
    }
}

pub struct WhisperProcessor {
    model: Arc<WhisperModel>,
    sentence_tx: mpsc::UnboundedSender<SentenceBuffer>,
    result_rx: mpsc::UnboundedReceiver<WhisperMessage>,
}

impl WhisperProcessor {
    pub fn new(
        model: WhisperModel,
    ) -> (
        Self,
        mpsc::UnboundedReceiver<SentenceBuffer>,
        mpsc::UnboundedSender<WhisperMessage>,
    ) {
        let (sentence_tx, sentence_rx) = mpsc::unbounded_channel();
        let (result_tx, result_rx) = mpsc::unbounded_channel();

        let processor = Self {
            model: Arc::new(model),
            sentence_tx,
            result_rx,
        };

        (processor, sentence_rx, result_tx)
    }

    pub fn process_sentence(&self, sentence: SentenceBuffer) -> Result<()> {
        self.sentence_tx.send(sentence)?;
        Ok(())
    }

    pub async fn start_processing_loop(
        model: Arc<WhisperModel>,
        mut sentence_rx: mpsc::UnboundedReceiver<SentenceBuffer>,
        result_tx: mpsc::UnboundedSender<WhisperMessage>,
    ) {
        while let Some(sentence) = sentence_rx.recv().await {
            let model_clone = model.clone();
            let result_tx_clone = result_tx.clone();
            let sentence_clone = sentence.clone();

            tokio::spawn(async move {
                // Send processing message
                let processing_msg = WhisperMessage::Processing {
                    sentence_id: sentence_clone.id.clone(),
                    original_text: sentence_clone.kyutai_text.clone(),
                    start_time: sentence_clone.start_time,
                    end_time: sentence_clone.end_time,
                };
                let _ = result_tx_clone.send(processing_msg);

                // Process with Whisper
                match model_clone.transcribe_audio(&sentence_clone.audio_samples) {
                    Ok(result_text) => {
                        let changed = result_text != sentence_clone.kyutai_text;
                        let complete_msg = WhisperMessage::Complete {
                            sentence_id: sentence_clone.id.clone(),
                            original_text: sentence_clone.kyutai_text.clone(),
                            corrected_text: result_text,
                            confidence: 0.95, // whisper-rs doesn't provide confidence scores
                            changed,
                        };
                        let _ = result_tx_clone.send(complete_msg);
                    }
                    Err(_e) => {
                        // Send unchanged result on error
                        let complete_msg = WhisperMessage::Complete {
                            sentence_id: sentence_clone.id.clone(),
                            original_text: sentence_clone.kyutai_text.clone(),
                            corrected_text: sentence_clone.kyutai_text.clone(),
                            confidence: 0.0,
                            changed: false,
                        };
                        let _ = result_tx_clone.send(complete_msg);
                    }
                }
            });
        }
    }
}
