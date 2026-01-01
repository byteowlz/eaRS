use anyhow::Result;
use candle::{Device, Tensor};
use crossbeam_channel::Receiver;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub mod config;
pub mod virtual_keyboard;
pub use kaudio;
#[cfg(feature = "whisper")]
pub mod whisper;
#[cfg(not(feature = "whisper"))]
pub mod whisper {
    use anyhow::Result;
    use candle::Device;
    use tokio::sync::mpsc;

    #[derive(Clone)]
    pub struct WhisperModel;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct SentenceBuffer {
        pub id: String,
        pub audio_samples: Vec<f32>,
        pub start_time: f64,
        pub end_time: f64,
        pub kyutai_text: String,
        pub words: Vec<crate::WordTimestamp>,
    }

    #[derive(Debug, Clone, serde::Serialize)]
    pub enum WhisperStatus {
        Pending,
        Processing,
        Corrected(String),
        Confirmed,
    }

    #[derive(Debug, Clone, serde::Serialize)]
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
            _config: &crate::config::WhisperConfig,
            _model_override: Option<&str>,
            _quantization_override: Option<&str>,
            _device: Device,
            _forced_lang: Option<&str>,
        ) -> Result<Self> {
            Err(anyhow::anyhow!("whisper feature not enabled"))
        }

        pub fn transcribe_audio(&self, _audio_samples: &[f32]) -> Result<String> {
            Err(anyhow::anyhow!("whisper feature not enabled"))
        }
    }

    #[derive(Debug, Clone)]
    pub struct AudioBuffer;
    impl AudioBuffer {
        pub fn new(_max_duration: f64, _sample_rate: u32) -> Self {
            Self
        }
        pub fn push_samples(&mut self, _samples: &[f32], _timestamp: f64) {}
        pub fn extract_segment(&self, _start_time: f64, _end_time: f64) -> Vec<f32> {
            Vec::new()
        }
    }

    #[derive(Debug, Clone)]
    pub struct SentenceDetector;
    impl SentenceDetector {
        pub fn new(_config: crate::config::SentenceDetectionConfig) -> Self {
            Self
        }
        pub fn process_word(
            &mut self,
            _word: &crate::WordTimestamp,
            _vad_confidence: Option<f32>,
        ) -> Option<SentenceBuffer> {
            None
        }
    }

    pub struct WhisperProcessor;
    impl WhisperProcessor {
        pub fn new(
            _model: WhisperModel,
        ) -> (
            Self,
            mpsc::UnboundedReceiver<SentenceBuffer>,
            mpsc::UnboundedSender<WhisperMessage>,
        ) {
            let (_sent_tx, sent_rx) = mpsc::unbounded_channel();
            let (res_tx, _res_rx) = mpsc::unbounded_channel();
            (Self, sent_rx, res_tx)
        }

        pub fn process_sentence(&self, _sentence: SentenceBuffer) -> Result<()> {
            Ok(())
        }

        pub async fn start_processing_loop(
            _model: std::sync::Arc<WhisperModel>,
            _sentence_rx: mpsc::UnboundedReceiver<SentenceBuffer>,
            _result_tx: mpsc::UnboundedSender<WhisperMessage>,
        ) {
        }
    }
}
pub mod display;
pub mod server;

#[derive(Debug, serde::Deserialize)]
pub struct SttConfig {
    pub audio_silence_prefix_seconds: f64,
    pub audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    pub mimi_name: String,
    pub tokenizer_name: String,
    pub card: usize,
    pub text_card: usize,
    pub dim: usize,
    pub n_q: usize,
    pub context: usize,
    pub max_period: f64,
    pub num_heads: usize,
    pub num_layers: usize,
    pub causal: bool,
    pub stt_config: SttConfig,
}

impl Config {
    pub fn model_config(&self, vad: bool) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        let extra_heads = if vad {
            Some(moshi::lm::ExtraHeadsConfig {
                num_heads: 4,
                dim: 6,
            })
        } else {
            None
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

pub struct Model {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    timestamps: bool,
    vad: bool,
    config: Config,
    dev: Device,
    vad_timeout: Option<f64>,
    whisper_model: Option<std::sync::Arc<whisper::WhisperModel>>,
    whisper_enabled: bool,
    injection_end_time: Option<std::time::Instant>,
    verbose_injection: bool,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_time: f64,
    pub end_time: Option<f64>,
}

/// WebSocket interface for real-time transcription streaming.
///
/// ## Connection
/// Connect to `ws://localhost:<port>/` where `<port>` is specified via the `--ws` option.
///
/// ## Message Format
/// Messages are sent as JSON objects using the enum variant name as the key:
///
/// ### Word Message
/// Sent for each transcribed word as it's recognized:
/// ```json
/// {
///   "Word": {
///     "word": "hello",
///     "start_time": 1.23,
///     "end_time": 1.45  // null for real-time words without end time yet
///   }
/// }
/// ```
///
/// ### Pause Message
/// Sent when voice activity detection detects a pause (requires --vad flag):
/// ```json
/// {
///   "Pause": {
///     "timestamp": 1234567890.123
///   }
/// }
/// ```
///
/// ### Final Message
/// Sent at the end of transcription with complete results:
/// ```json
/// {
///   "Final": {
///     "text": "complete transcribed text",
///     "words": [
///       {"word": "hello", "start_time": 1.23, "end_time": 1.45},
///       {"word": "world", "start_time": 1.46, "end_time": null}
///     ]
///   }
/// }
/// ```
///
/// ## Commands
/// Send commands to the server as simple JSON strings:
///
/// ### Restart Command
/// Restart transcription session:
/// ```json
/// "Restart"
/// ```
///
/// ### Pause Command
/// Pause transcription:
/// ```json
/// "Pause"
/// ```
///
/// ### Resume Command
/// Resume transcription:
/// ```json
/// "Resume"
/// ```
///
/// ## Usage Example
/// ```bash
/// # Start transcription with WebSocket on port 8080
/// ears --live --ws 8080
///
/// # With timestamps and VAD
/// ears --live --ws 8080 --timestamps --vad
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WebSocketMessage {
    Word {
        word: String,
        start_time: f64,
        end_time: Option<f64>,
    },
    Pause {
        timestamp: f64,
    },
    Final {
        text: String,
        words: Vec<WordTimestamp>,
    },
    LanguageChanged {
        lang: String,
    },
    EngineChanged {
        engine: String,
    },
    Status {
        paused: bool,
        vad: bool,
        timestamps: bool,
        vad_timeout: Option<f64>,
        lang: Option<String>,
        engine: Option<String>,
    },
    WhisperProcessing {
        sentence_id: String,
        original_text: String,
        start_time: f64,
        end_time: f64,
    },
    WhisperComplete {
        sentence_id: String,
        original_text: String,
        corrected_text: String,
        confidence: f32,
        changed: bool,
    },
}

pub trait TranscriptionSink: Send {
    fn handle_message(&mut self, message: WebSocketMessage);
}

pub struct NullSink;

impl TranscriptionSink for NullSink {
    fn handle_message(&mut self, _message: WebSocketMessage) {}
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WebSocketCommand {
    Restart,
    Pause,
    Resume,
    SetLanguage { lang: String },
    GetStatus,
    SetVadTimeout { seconds: f64 },
    SetEngine { engine: String },
}

#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    pub timestamps: bool,
    pub vad: bool,
    pub save_audio: Option<String>,
    pub vad_timeout: Option<f64>,
    pub whisper_enabled: bool,
    pub whisper_model: Option<String>,
    pub whisper_quantization: Option<String>,
    pub whisper_languages: Option<Vec<String>>,
    pub whisper_force_lang: Option<String>,
    pub verbose_injection: bool,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            timestamps: false,
            vad: false,
            save_audio: None,
            vad_timeout: None,
            whisper_enabled: false,
            whisper_model: None,
            whisper_quantization: None,
            whisper_languages: None,
            whisper_force_lang: None,
            verbose_injection: false,
        }
    }
}

impl Model {
    pub(crate) fn batch_size(&self) -> usize {
        self.state.batch_size()
    }

    pub(crate) fn device(&self) -> &Device {
        &self.dev
    }

    pub(crate) fn timestamps_enabled(&self) -> bool {
        self.timestamps
    }

    pub(crate) fn vad_enabled(&self) -> bool {
        self.vad
    }

    pub(crate) fn vad_timeout_seconds(&self) -> Option<f64> {
        self.vad_timeout
    }

    pub(crate) fn audio_delay_samples(&self) -> usize {
        (self.config.stt_config.audio_delay_seconds * 24_000.0) as usize
    }

    pub(crate) fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        Ok(self
            .text_tokenizer
            .decode_piece_ids(tokens)
            .unwrap_or_else(|_| String::new()))
    }

    pub(crate) fn step_pcm_with_mask(
        &mut self,
        pcm: Tensor,
        mask: &moshi::StreamMask,
    ) -> Result<Vec<moshi::asr::AsrMsg>> {
        Ok(self.state.step_pcm(pcm, None, mask, |_, _, _| ())?)
    }

    pub(crate) fn reset_batch_slot(&mut self, batch_idx: usize) -> Result<()> {
        Ok(self.state.reset_batch_idx(batch_idx)?)
    }

    pub async fn load_from_hf(
        hf_repo: &str,
        cpu: bool,
        options: TranscriptionOptions,
        model_dir: Option<&std::path::Path>,
    ) -> Result<Self> {
        Self::load_from_hf_with_batch(hf_repo, cpu, options, model_dir, 1).await
    }

    pub async fn load_from_hf_with_batch(
        hf_repo: &str,
        cpu: bool,
        options: TranscriptionOptions,
        model_dir: Option<&std::path::Path>,
        batch_size: usize,
    ) -> Result<Self> {
        let device = create_device(cpu)?;
        let dtype = device.bf16_default_to_f32();

        let api = if let Some(model_dir) = model_dir {
            hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(model_dir.to_path_buf())
                .build()?
        } else {
            hf_hub::api::sync::Api::new()?
        };
        let repo = api.model(hf_repo.to_string());
        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get("model.safetensors")?;
        let mimi_file = repo.get(&config.mimi_name)?;

        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;
        let vb_lm = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)?
        };
        let batch_size = batch_size.max(1);
        let audio_tokenizer = if batch_size > 1 {
            moshi::mimi::load_b(Some(batch_size), mimi_file.to_str().unwrap(), Some(32), &device)?
        } else {
            moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), &device)?
        };
        let lm = if batch_size > 1 {
            moshi::lm::LmModel::batched(
                batch_size,
                &config.model_config(options.vad),
                moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
            )?
        } else {
            moshi::lm::LmModel::new(
                &config.model_config(options.vad),
                moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
            )?
        };
        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(
            batch_size,
            asr_delay_in_tokens,
            0.,
            audio_tokenizer,
            lm,
        )?;

        // Initialize Whisper model if enabled
        let whisper_model = if options.whisper_enabled {
            // Load app config to get whisper settings
            let app_config = config::AppConfig::load().ok();
            if let Some(app_config) = app_config {
                match whisper::WhisperModel::load(
                    &app_config.whisper,
                    options.whisper_model.as_deref(),
                    options.whisper_quantization.as_deref(),
                    device.clone(),
                    options.whisper_force_lang.as_deref(),
                )
                .await
                {
                    Ok(model) => Some(model),
                    Err(e) => {
                        eprintln!(
                            "Failed to load Whisper model: {}. Continuing without Whisper enhancement.",
                            e
                        );
                        None
                    }
                }
            } else {
                eprintln!("Failed to load config. Continuing without Whisper enhancement.");
                None
            }
        } else {
            None
        };

        Ok(Model {
            state,
            config,
            text_tokenizer,
            timestamps: options.timestamps,
            vad: options.vad,
            dev: device,
            vad_timeout: options.vad_timeout,
            whisper_model: whisper_model.map(|m| Arc::new(m)),
            whisper_enabled: options.whisper_enabled,
            injection_end_time: None,
            verbose_injection: options.verbose_injection,
        })
    }

    pub fn prime_with_audio<P: AsRef<Path>>(&mut self, file_path: P) -> Result<()> {
        self.prime_with_audio_for_slot(file_path, 0)
    }

    pub fn prime_with_audio_for_slot<P: AsRef<Path>>(
        &mut self,
        file_path: P,
        batch_idx: usize,
    ) -> Result<()> {
        let (pcm, sample_rate) = kaudio::pcm_decode(file_path.as_ref())?;
        let pcm = if sample_rate != 24_000 {
            kaudio::resample(&pcm, sample_rate as usize, 24_000)?
        } else {
            pcm
        };

        let audio_duration_secs = pcm.len() as f64 / 24_000.0;
        let discard_window_secs = 1.0;
        if batch_idx >= self.state.batch_size() {
            anyhow::bail!(
                "batch index {} out of range (max {})",
                batch_idx,
                self.state.batch_size()
            );
        }
        let mut mask_vec = vec![false; self.state.batch_size()];
        mask_vec[batch_idx] = true;
        let mask = moshi::StreamMask::new(mask_vec, &self.dev)?;

        for chunk in pcm.chunks(1920) {
            let tensor = Tensor::new(chunk, &self.dev)?.reshape((1, 1, chunk.len()))?;
            let tensor = tensor.broadcast_as((self.state.batch_size(), 1, chunk.len()))?;
            let _ = self.state.step_pcm(tensor, None, &mask, |_, _, _| ())?;
        }

        self.injection_end_time = Some(
            std::time::Instant::now()
                + std::time::Duration::from_secs_f64(audio_duration_secs + discard_window_secs),
        );

        Ok(())
    }

    pub fn reset_stream_state(&mut self) -> Result<()> {
        self.state.reset()?;
        self.injection_end_time = None;
        Ok(())
    }

    fn is_in_injection_window(&self) -> bool {
        if let Some(end_time) = self.injection_end_time {
            std::time::Instant::now() < end_time
        } else {
            false
        }
    }

    pub fn transcribe_file<P: AsRef<Path>>(
        &mut self,
        file_path: P,
        save_audio: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let (pcm, sample_rate) = kaudio::pcm_decode(file_path.as_ref())?;

        if let Some(save_path) = save_audio {
            self.save_audio_wav(&pcm, sample_rate, save_path)?;
        }

        let pcm = if sample_rate != 24_000 {
            kaudio::resample(&pcm, sample_rate as usize, 24_000)?
        } else {
            pcm
        };

        self.transcribe_pcm(pcm)
    }

    pub fn transcribe_live(
        &mut self,
        audio_rx: Receiver<Vec<f32>>,
        save_audio: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let mut sink = NullSink;
        self.transcribe_live_with_sink(audio_rx, save_audio, None, &mut sink)
    }

    pub fn transcribe_live_with_sink<S: TranscriptionSink>(
        &mut self,
        audio_rx: Receiver<Vec<f32>>,
        save_audio: Option<&str>,
        lang_cmd_rx: Option<crossbeam_channel::Receiver<String>>,
        sink: &mut S,
    ) -> Result<TranscriptionResult> {
        use std::io::Write;
        use std::time::{Duration, Instant};

        let mut all_audio = Vec::new();
        let mut words = Vec::new();
        let mut current_text = String::new();
        let mut last_word: Option<(String, f64)> = None;
        let mut word_sent = false;
        let mut printed_eot = false;
        let mut last_voice_activity: Option<Instant> = None;

        // Initialize Whisper components if enabled
        let mut sentence_detector = if self.whisper_enabled {
            let config = config::AppConfig::load()
                .ok()
                .and_then(|c| Some(c.whisper.sentence_detection));
            config.map(|c| whisper::SentenceDetector::new(c))
        } else {
            None
        };

        let mut audio_buffer = if self.whisper_enabled {
            Some(whisper::AudioBuffer::new(30.0, 24000))
        } else {
            None
        };

        let mut display_manager = if self.whisper_enabled && !self.timestamps {
            Some(display::DisplayManager::new())
        } else {
            None
        };

        // Background Whisper worker (non-blocking) if model is available
        let (wh_tx, wh_rx) = crossbeam_channel::unbounded::<whisper::SentenceBuffer>();
        let (wh_res_tx, wh_res_rx) = crossbeam_channel::unbounded::<whisper::WhisperMessage>();
        if self.whisper_enabled {
            if let Some(wm) = self.whisper_model.clone() {
                std::thread::spawn(move || {
                    while let Ok(sentence) = wh_rx.recv() {
                        // Notify processing
                        let _ = wh_res_tx.send(whisper::WhisperMessage::Processing {
                            sentence_id: sentence.id.clone(),
                            original_text: sentence.kyutai_text.clone(),
                            start_time: sentence.start_time,
                            end_time: sentence.end_time,
                        });
                        // Run Whisper in this worker thread
                        let result = wm.transcribe_audio(&sentence.audio_samples);
                        let (corrected_text, confidence) = match result {
                            Ok(t) => (t, 0.95),
                            Err(_) => (sentence.kyutai_text.clone(), 0.0),
                        };
                        let changed = corrected_text != sentence.kyutai_text;
                        let _ = wh_res_tx.send(whisper::WhisperMessage::Complete {
                            sentence_id: sentence.id,
                            original_text: sentence.kyutai_text,
                            corrected_text,
                            confidence,
                            changed,
                        });
                    }
                });
            }
        }

        // Buffer to accumulate samples across small incoming chunks
        let mut sample_buffer: Vec<f32> = Vec::new();

        loop {
            if let Some(ref lang_rx) = lang_cmd_rx {
                while let Ok(lang) = lang_rx.try_recv() {
                    eprintln!("Priming model with language: {}", lang);
                    if let Err(e) = self.prime_with_lang_code(&lang) {
                        eprintln!("Failed to prime language {}: {}", lang, e);
                    }
                }
            }

            let pcm_chunk = match audio_rx.recv() {
                Ok(chunk) => chunk,
                Err(_) => {
                    eprintln!("Audio receiver channel closed, finishing transcription");
                    break;
                }
            };
            if save_audio.is_some() {
                all_audio.extend_from_slice(&pcm_chunk);
            }

            let mut has_voice_activity = false;

            // Accumulate samples so we always process in 1920-sample frames (~80ms @ 24kHz)
            sample_buffer.extend_from_slice(&pcm_chunk);

            // Store audio in buffer for Whisper if enabled (based on absolute time)
            if let Some(ref mut buffer) = audio_buffer {
                let sample_offset = all_audio.len().saturating_sub(sample_buffer.len());
                let chunk_start_time = sample_offset as f64 / 24000.0;
                buffer.push_samples(&pcm_chunk, chunk_start_time);
            }

            // Process in fixed-size frames, keep the remainder in sample_buffer
            let mut consumed = 0usize;
            while consumed + 1920 <= sample_buffer.len() {
                let frame = &sample_buffer[consumed..consumed + 1920];
                let pcm_tensor = Tensor::new(frame, &self.dev)?.reshape((1, 1, ()))?;
                let asr_msgs = self
                    .state
                    .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

                for asr_msg in asr_msgs.iter() {
                    match asr_msg {
                        moshi::asr::AsrMsg::Step { prs, .. } => {
                            if self.vad && prs[2][0] > 0.5 && !printed_eot {
                                printed_eot = true;
                                if !self.timestamps {
                                    print!(" <pause>");
                                    std::io::stdout().flush().ok();
                                }
                            }
                        }
                        moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                            printed_eot = false;
                            has_voice_activity = true;
                            if let Some((word, start_time)) = last_word.take() {
                                if self.is_in_injection_window() {
                                    if self.verbose_injection {
                                        eprintln!("[injection] discarded: {word}");
                                    }
                                    word_sent = false;
                                } else {
                                    if self.timestamps {
                                        println!("[{start_time:5.2}-{stop_time:5.2}] {word}");
                                    }
                                    if !self.timestamps && !word_sent {
                                        sink.handle_message(WebSocketMessage::Word {
                                            word: word.clone(),
                                            start_time,
                                            end_time: Some(*stop_time),
                                        });
                                    }
                                    words.push(WordTimestamp {
                                        word,
                                        start_time,
                                        end_time: Some(*stop_time),
                                    });
                                    word_sent = false;
                                }
                            }
                        }
                        moshi::asr::AsrMsg::Word {
                            tokens, start_time, ..
                        } => {
                            printed_eot = false;
                            has_voice_activity = true;
                            let word = self
                                .text_tokenizer
                                .decode_piece_ids(tokens)
                                .unwrap_or_else(|_| String::new());

                            if self.is_in_injection_window() {
                                if self.verbose_injection {
                                    eprintln!("[injection] discarded: {word}");
                                }
                                continue;
                            }

                            current_text.push(' ');
                            current_text.push_str(&word);

                            // Create WordTimestamp for sentence detection
                            let word_ts = WordTimestamp {
                                word: word.clone(),
                                start_time: *start_time,
                                end_time: None,
                            };

                            // Check for sentence boundaries if Whisper is enabled
                            if let Some(ref mut detector) = sentence_detector {
                                // Get VAD confidence from previous Step message
                                let vad_confidence = if self.vad && printed_eot {
                                    Some(0.9)
                                } else {
                                    None
                                };

                                if let Some(mut sentence) =
                                    detector.process_word(&word_ts, vad_confidence)
                                {
                                    // Extract audio for the sentence
                                    if let Some(ref buffer) = audio_buffer {
                                        sentence.audio_samples = buffer.extract_segment(
                                            sentence.start_time,
                                            sentence.end_time,
                                        );
                                    }

                                    // Queue sentence for background Whisper processing
                                    if self.whisper_enabled && self.whisper_model.is_some() {
                                        let _ = wh_tx.send(sentence.clone());
                                    }

                                    // Update display if using display manager
                                    if let Some(ref mut dm) = display_manager {
                                        dm.complete_sentence(sentence.id, sentence.kyutai_text);
                                    }
                                }
                            }

                            if !self.timestamps {
                                sink.handle_message(WebSocketMessage::Word {
                                    word: word.clone(),
                                    start_time: *start_time,
                                    end_time: None,
                                });
                                word_sent = true;

                                // Only show live transcription if we're in an interactive terminal
                                if atty::is(atty::Stream::Stdout) {
                                    if let Some(ref mut dm) = display_manager {
                                        dm.add_live_word(&word);
                                    } else {
                                        print!(" {}", word);
                                        std::io::stdout().flush().ok();
                                    }
                                }
                            } else {
                                if let Some((prev_word, prev_start_time)) = last_word.take() {
                                    println!(
                                        "[{prev_start_time:5.2}-{start_time:5.2}] {prev_word}"
                                    );
                                    sink.handle_message(WebSocketMessage::Word {
                                        word: prev_word.clone(),
                                        start_time: prev_start_time,
                                        end_time: Some(*start_time),
                                    });
                                    words.push(WordTimestamp {
                                        word: prev_word,
                                        start_time: prev_start_time,
                                        end_time: Some(*start_time),
                                    });
                                }
                                word_sent = false;
                            }

                            last_word = Some((word, *start_time));
                        }
                    }
                }
                consumed += 1920;
            }

            // Drop the processed prefix, retain any tail < 1920 for the next loop
            if consumed > 0 {
                sample_buffer.drain(0..consumed);
            }

            // Update voice activity timestamp if we detected voice
            if has_voice_activity {
                last_voice_activity = Some(Instant::now());
            }

            // Drain whisper results and update display
            for msg in wh_res_rx.try_iter() {
                if let Some(ref mut dm) = display_manager {
                    dm.handle_whisper_message(msg.clone());
                } else {
                    display::print_whisper_status(&msg);
                }

                match msg {
                    whisper::WhisperMessage::Processing {
                        sentence_id,
                        original_text,
                        start_time,
                        end_time,
                    } => {
                        sink.handle_message(WebSocketMessage::WhisperProcessing {
                            sentence_id,
                            original_text,
                            start_time,
                            end_time,
                        });
                    }
                    whisper::WhisperMessage::Complete {
                        sentence_id,
                        original_text,
                        corrected_text,
                        confidence,
                        changed,
                    } => {
                        sink.handle_message(WebSocketMessage::WhisperComplete {
                            sentence_id,
                            original_text,
                            corrected_text,
                            confidence,
                            changed,
                        });
                    }
                }
            }

            // Check for timeout
            if let Some(timeout_secs) = self.vad_timeout {
                if let Some(last_activity) = last_voice_activity {
                    if last_activity.elapsed() > Duration::from_secs_f64(timeout_secs) {
                        break;
                    }
                }
            }
        }

        if let Some((word, start_time)) = last_word.take() {
            if self.timestamps {
                println!("[{start_time:5.2}-     ] {word}");
            }
            words.push(WordTimestamp {
                word,
                start_time,
                end_time: None,
            });
        }

        if !self.timestamps && atty::is(atty::Stream::Stdout) {
            if let Some(mut dm) = display_manager {
                dm.finish();
            } else {
                println!();
            }
        }

        if let Some(save_path) = save_audio {
            self.save_audio_wav(&all_audio, 24000, save_path)?;
        }

        let result = TranscriptionResult {
            text: current_text.trim().to_string(),
            words,
        };

        eprintln!("Sending Final message with {} words", result.words.len());
        sink.handle_message(WebSocketMessage::Final {
            text: result.text.clone(),
            words: result.words.clone(),
        });
        eprintln!("Final message sent");

        Ok(result)
    }

    pub async fn transcribe_live_ws(
        &mut self,
        audio_rx: Receiver<Vec<f32>>,
        save_audio: Option<&str>,
        ws_port: u16,
    ) -> Result<TranscriptionResult> {
        use futures::{SinkExt, StreamExt};
        use std::io::Write;
        use std::sync::Arc;
        use tokio::sync::{broadcast, mpsc, watch};
        use tokio_tungstenite::{accept_async, tungstenite::Message};

        // WebSocket broadcast channel
        let (ws_tx, _ws_rx) = broadcast::channel(100);
        let ws_tx = Arc::new(ws_tx);

        // Channel used to request a restart of the transcription session
        let (restart_tx, mut restart_rx) = mpsc::unbounded_channel();
        let restart_tx = Arc::new(restart_tx);

        // Watch channel used to pause or resume transcription
        let (pause_tx, _pause_rx) = watch::channel(true);
        let pause_tx = Arc::new(pause_tx);
        let auto_resume_pending = Arc::new(AtomicBool::new(true));

        // Track language and settings
        let (lang_tx, lang_rx_watch) = watch::channel::<Option<String>>(None);
        let lang_tx = Arc::new(lang_tx);

        // Spawn WebSocket server
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", ws_port)).await?;
        let ws_tx_clone = ws_tx.clone();
        let restart_tx_clone = restart_tx.clone();
        let pause_tx_clone = pause_tx.clone();
        let auto_resume_clone = auto_resume_pending.clone();
        let lang_tx_clone = lang_tx.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let ws_tx = ws_tx_clone.clone();
                let restart_tx = restart_tx_clone.clone();
                let pause_tx = pause_tx_clone.clone();
                let auto_resume_pending = auto_resume_clone.clone();
                let lang_tx = lang_tx_clone.clone();
                tokio::spawn(async move {
                    let ws_stream = match accept_async(stream).await {
                        Ok(ws) => ws,
                        Err(e) => {
                            eprintln!("WebSocket handshake error: {}", e);
                            return;
                        }
                    };

                    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
                    let mut ws_rx = ws_tx.subscribe();

                    let receive_task = tokio::spawn(async move {
                        while let Some(msg) = ws_receiver.next().await {
                            match msg {
                                Ok(Message::Close(_)) => break,
                                Ok(Message::Text(text)) => {
                                    if let Ok(cmd) = serde_json::from_str::<WebSocketCommand>(&text)
                                    {
                                        match cmd {
                                            WebSocketCommand::Restart => {
                                                let _ = restart_tx.send(());
                                            }
                                            WebSocketCommand::Pause => {
                                                auto_resume_pending.store(false, Ordering::SeqCst);
                                                let _ = pause_tx.send(true);
                                            }
                                            WebSocketCommand::Resume => {
                                                auto_resume_pending.store(false, Ordering::SeqCst);
                                                let _ = pause_tx.send(false);
                                            }
                                            WebSocketCommand::SetLanguage { lang } => {
                                                let _ = lang_tx.send(Some(lang.clone()));
                                                let _ =
                                                    ws_tx.send(WebSocketMessage::LanguageChanged {
                                                        lang,
                                                    });
                                            }
                                            WebSocketCommand::GetStatus => {
                                                // Broadcast current status
                                                let status = WebSocketMessage::Status {
                                                    paused: *pause_tx.borrow(),
                                                    vad: false,
                                                    timestamps: false,
                                                    vad_timeout: None,
                                                    lang: lang_tx.borrow().clone(),
                                                    engine: None,
                                                };
                                                let _ = ws_tx.send(status);
                                            }
                                            WebSocketCommand::SetEngine { .. } => {}
                                            WebSocketCommand::SetVadTimeout { seconds } => {
                                                // No-op here: handled in processing loop via vad_timeout
                                                let status = WebSocketMessage::Status {
                                                    paused: *pause_tx.borrow(),
                                                    vad: false,
                                                    timestamps: false,
                                                    vad_timeout: Some(seconds),
                                                    lang: lang_tx.borrow().clone(),
                                                    engine: None,
                                                };
                                                let _ = ws_tx.send(status);
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("WebSocket receive error: {}", e);
                                    break;
                                }
                                _ => {}
                            }
                        }
                    });

                    let send_task = tokio::spawn(async move {
                        while let Ok(ws_msg) = ws_rx.recv().await {
                            let json_msg = serde_json::to_string(&ws_msg).unwrap_or_default();
                            if ws_sender.send(Message::text(json_msg)).await.is_err() {
                                break;
                            }
                        }
                    });

                    let _ = tokio::join!(receive_task, send_task);
                });
            }
        });

        // Bridge blocking audio receiver to async channel
        let (pcm_tx, mut pcm_rx) = mpsc::unbounded_channel();
        let mut pause_rx = pause_tx.subscribe();
        std::thread::spawn(move || {
            while let Ok(chunk) = audio_rx.recv() {
                if pcm_tx.send(chunk).is_err() {
                    break;
                }
            }
        });

        let mut all_audio = Vec::new();
        let mut overall_words = Vec::new();
        let mut overall_text = String::new();

        // Subscribe to language changes
        let mut lang_rx = lang_rx_watch.clone();
        let mut current_lang: Option<String> = None;

        // Whisper async worker for WS mode
        let (wh_tx, mut wh_rx) = tokio::sync::mpsc::unbounded_channel::<whisper::SentenceBuffer>();
        if self.whisper_enabled {
            if let Some(wm) = self.whisper_model.clone() {
                let ws_tx_bg = ws_tx.clone();
                tokio::spawn(async move {
                    while let Some(sentence) = wh_rx.recv().await {
                        let _ = ws_tx_bg.send(WebSocketMessage::WhisperProcessing {
                            sentence_id: sentence.id.clone(),
                            original_text: sentence.kyutai_text.clone(),
                            start_time: sentence.start_time,
                            end_time: sentence.end_time,
                        });
                        let wm_cl = wm.clone();
                        let sentence_cl = sentence.clone();
                        let res = tokio::task::spawn_blocking(move || {
                            wm_cl.transcribe_audio(&sentence_cl.audio_samples)
                        })
                        .await;
                        let corrected_text = match res {
                            Ok(Ok(t)) => t,
                            _ => sentence.kyutai_text.clone(),
                        };
                        let changed = corrected_text != sentence.kyutai_text;
                        let _ = ws_tx_bg.send(WebSocketMessage::WhisperComplete {
                            sentence_id: sentence.id,
                            original_text: sentence.kyutai_text,
                            corrected_text,
                            confidence: 0.95,
                            changed,
                        });
                    }
                });
            }
        }

        loop {
            auto_resume_pending.store(true, Ordering::SeqCst);
            let mut words = Vec::new();
            let mut current_text = String::new();
            let mut last_word: Option<(String, f64)> = None;
            let mut printed_eot = false;
            let mut last_voice_activity: Option<std::time::Instant> = None;
            let mut restart = false;
            let mut paused = *pause_rx.borrow();

            eprintln!(
                "Starting transcription session (paused - will auto-resume on incoming audio)..."
            );

            // Apply initial language priming if any
            if current_lang.is_some() {
                if let Some(lang) = &current_lang {
                    if let Err(e) = self.prime_with_lang_code(lang) {
                        eprintln!("Failed to prime language {}: {}", lang, e);
                    }
                }
            }

            // Initialize Whisper sentence detection for WS
            let mut sentence_detector = if self.whisper_enabled {
                let cfg = config::AppConfig::load()
                    .ok()
                    .map(|c| c.whisper.sentence_detection);
                cfg.map(|c| whisper::SentenceDetector::new(c))
            } else {
                None
            };
            let mut audio_buffer = if self.whisper_enabled {
                Some(whisper::AudioBuffer::new(30.0, 24000))
            } else {
                None
            };

            loop {
                tokio::select! {
                    _ = restart_rx.recv() => {
                        eprintln!("Received restart command");
                        restart = true;
                        break;
                    }
                    _ = pause_rx.changed() => {
                        paused = *pause_rx.borrow();
                        if paused {
                            eprintln!("Transcription paused");
                        } else {
                            eprintln!("Transcription resumed");
                        }
                    }
                    _ = lang_rx.changed() => {
                        if let Some(lang) = lang_rx.borrow().clone() {
                            current_lang = Some(lang.clone());
                            if let Err(e) = self.prime_with_lang_code(&lang) {
                                eprintln!("Failed to prime language {}: {}", lang, e);
                            }
                            let _ = ws_tx.send(WebSocketMessage::LanguageChanged { lang });
                        }
                    }
                    Some(pcm_chunk) = pcm_rx.recv() => {
                        if paused {
                            if auto_resume_pending.load(Ordering::SeqCst) {
                                eprintln!(
                                    "Audio received while paused - automatically resuming session"
                                );
                                auto_resume_pending.store(false, Ordering::SeqCst);
                                let _ = pause_tx.send(false);
                                paused = false;
                            } else {
                                continue;
                            }
                        }
                        if save_audio.is_some() {
                            all_audio.extend_from_slice(&pcm_chunk);
                        }

                        let mut has_voice_activity = false;

                        // Store audio in buffer for Whisper if enabled
                        if let Some(ref mut buffer) = audio_buffer {
                            let sample_offset = all_audio.len().saturating_sub(pcm_chunk.len());
                            let chunk_start_time = sample_offset as f64 / 24000.0;
                            buffer.push_samples(&pcm_chunk, chunk_start_time);
                        }

                        for pcm in pcm_chunk.chunks(1920) {
                            let pcm_tensor = Tensor::new(pcm, &self.dev)?.reshape((1, 1, ()))?;
                            let asr_msgs = self.state.step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

                            for asr_msg in asr_msgs.iter() {
                                match asr_msg {
                                    moshi::asr::AsrMsg::Step { prs, .. } => {
                                        if self.vad && prs[2][0] > 0.5 && !printed_eot {
                                            printed_eot = true;
                                            let pause_msg = WebSocketMessage::Pause {
                                                timestamp: std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_secs_f64(),
                                            };
                                            let _ = ws_tx.send(pause_msg);

                                            if !self.timestamps {
                                                print!(" <pause>");
                                                std::io::stdout().flush().ok();
                                            }
                                        }
                                    }
                                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                                        printed_eot = false;
                                        has_voice_activity = true;
                                        if self.timestamps {
                                            if let Some((word, start_time)) = last_word.take() {
                                                if self.is_in_injection_window() {
                                                    if self.verbose_injection {
                                                        eprintln!("[injection] discarded: {word}");
                                                    }
                                                } else {
                                                    println!("[{start_time:5.2}-{stop_time:5.2}] {word}");
                                                    let word_ts = WordTimestamp {
                                                        word: word.clone(),
                                                        start_time,
                                                        end_time: Some(*stop_time),
                                                    };
                                                    words.push(word_ts.clone());
                                                    let ws_msg = WebSocketMessage::Word {
                                                        word: word_ts.word,
                                                        start_time: word_ts.start_time,
                                                        end_time: word_ts.end_time,
                                                    };
                                                    let _ = ws_tx.send(ws_msg);
                                                }
                                            }
                                        }
                                    }
                                    moshi::asr::AsrMsg::Word { tokens, start_time, .. } => {
                                        printed_eot = false;
                                        has_voice_activity = true;
                                        let word = self.text_tokenizer
                                            .decode_piece_ids(tokens)
                                            .unwrap_or_else(|_| String::new());

                                        if self.is_in_injection_window() {
                                            if self.verbose_injection {
                                                eprintln!("[injection] discarded: {word}");
                                            }
                                            continue;
                                        }

                                        current_text.push(' ');
                                        current_text.push_str(&word);

                                        // Check for sentence boundaries and queue Whisper processing
                                        if let Some(ref mut detector) = sentence_detector {
                                            let vad_confidence = if self.vad && printed_eot { Some(0.9) } else { None };
                                            if let Some(mut sentence) = detector.process_word(&WordTimestamp { word: word.clone(), start_time: *start_time, end_time: None }, vad_confidence) {
                                                if let Some(ref buffer) = audio_buffer {
                                                    sentence.audio_samples = buffer.extract_segment(sentence.start_time, sentence.end_time);
                                                }
                                                if self.whisper_enabled && self.whisper_model.is_some() {
                                                    let _ = wh_tx.send(sentence);
                                                }
                                            }
                                        }

                                        if !self.timestamps {
                                            print!(" {}", word);
                                            std::io::stdout().flush().ok();

                                            let ws_msg = WebSocketMessage::Word {
                                                word: word.clone(),
                                                start_time: *start_time,
                                                end_time: None,
                                            };
                                            let _ = ws_tx.send(ws_msg);
                                        } else {
                                            if let Some((prev_word, prev_start_time)) = last_word.take() {
                                                println!("[{prev_start_time:5.2}-{start_time:5.2}] {prev_word}");
                                                let word_ts = WordTimestamp {
                                                    word: prev_word.clone(),
                                                    start_time: prev_start_time,
                                                    end_time: Some(*start_time),
                                                };
                                                words.push(word_ts.clone());
                                                let ws_msg = WebSocketMessage::Word {
                                                    word: word_ts.word,
                                                    start_time: word_ts.start_time,
                                                    end_time: word_ts.end_time,
                                                };
                                                let _ = ws_tx.send(ws_msg);
                                            }
                                            last_word = Some((word, *start_time));
                                        }
                                    }
                                }
                            }
                        }

                        if has_voice_activity {
                            last_voice_activity = Some(std::time::Instant::now());
                        }

                        if let Some(timeout_secs) = self.vad_timeout {
                            if let Some(last_activity) = last_voice_activity {
                                if last_activity.elapsed() > std::time::Duration::from_secs_f64(timeout_secs) {
                                    eprintln!("Voice activity timeout reached");
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if let Some((word, start_time)) = last_word.take() {
                if self.timestamps {
                    println!("[{start_time:5.2}-     ] {word}");
                }
                let word_ts = WordTimestamp {
                    word: word.clone(),
                    start_time,
                    end_time: None,
                };
                words.push(word_ts.clone());
                let ws_msg = WebSocketMessage::Word {
                    word: word_ts.word,
                    start_time: word_ts.start_time,
                    end_time: word_ts.end_time,
                };
                let _ = ws_tx.send(ws_msg);
            }

            if !self.timestamps {
                println!();
            }

            overall_words.extend(words.clone());
            if !current_text.is_empty() {
                if !overall_text.is_empty() {
                    overall_text.push(' ');
                }
                overall_text.push_str(current_text.trim());
            }

            let session_result = TranscriptionResult {
                text: current_text.trim().to_string(),
                words: words.clone(),
            };

            let final_msg = WebSocketMessage::Final {
                text: session_result.text.clone(),
                words: session_result.words.clone(),
            };
            let _ = ws_tx.send(final_msg);

            if !restart {
                break;
            }
        }

        if let Some(save_path) = save_audio {
            self.save_audio_wav(&all_audio, 24000, save_path)?;
        }

        Ok(TranscriptionResult {
            text: overall_text,
            words: overall_words,
        })
    }

    pub fn prime_with_lang_code(&mut self, iso_lang: &str) -> Result<()> {
        let ref_code = match iso_lang {
            "de" => "ger",
            "ja" => "jap",
            "es" => "esp",
            "it" => "ita",
            "pt" => "por",
            other => {
                eprintln!(
                    "Warning: Unknown language code '{}'. Supported codes: de, ja, es, it, pt",
                    other
                );
                eprintln!("Skipping language priming.");
                return Ok(());
            }
        };
        let config = config::AppConfig::load()?;
        let path = config.ref_audio_path().join(format!("{}.mp3", ref_code));
        self.prime_with_audio_for_slot(path, 0)
    }

    pub fn prime_with_lang_code_for_slot(
        &mut self,
        iso_lang: &str,
        batch_idx: usize,
    ) -> Result<()> {
        let ref_code = match iso_lang {
            "de" => "ger",
            "ja" => "jap",
            "es" => "esp",
            "it" => "ita",
            "pt" => "por",
            other => {
                eprintln!(
                    "Warning: Unknown language code '{}'. Supported codes: de, ja, es, it, pt",
                    other
                );
                eprintln!("Skipping language priming.");
                return Ok(());
            }
        };
        let config = config::AppConfig::load()?;
        let path = config.ref_audio_path().join(format!("{}.mp3", ref_code));
        self.prime_with_audio_for_slot(path, batch_idx)
    }

    fn transcribe_pcm(&mut self, mut pcm: Vec<f32>) -> Result<TranscriptionResult> {
        if self.config.stt_config.audio_silence_prefix_seconds > 0.0 {
            let silence_len =
                (self.config.stt_config.audio_silence_prefix_seconds * 24000.0) as usize;
            pcm.splice(0..0, vec![0.0; silence_len]);
        }

        let suffix = (self.config.stt_config.audio_delay_seconds * 24000.0) as usize;
        pcm.resize(pcm.len() + suffix + 24000, 0.0);

        let mut words = Vec::new();
        let mut current_text = String::new();
        let mut last_word: Option<(String, f64)> = None;

        for pcm_chunk in pcm.chunks(1920) {
            let pcm_tensor = Tensor::new(pcm_chunk, &self.dev)?.reshape((1, 1, ()))?;
            let asr_msgs = self
                .state
                .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { .. } => {
                        // Handle step messages if needed for VAD or other processing
                    }
                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                        if self.timestamps {
                            if let Some((word, start_time)) = last_word.take() {
                                words.push(WordTimestamp {
                                    word: word.clone(),
                                    start_time,
                                    end_time: Some(*stop_time),
                                });
                            }
                        }
                    }
                    moshi::asr::AsrMsg::Word {
                        tokens, start_time, ..
                    } => {
                        let word = self
                            .text_tokenizer
                            .decode_piece_ids(tokens)
                            .unwrap_or_else(|_| String::new());

                        current_text.push(' ');
                        current_text.push_str(&word);

                        if self.timestamps {
                            if let Some((prev_word, prev_start_time)) = last_word.take() {
                                words.push(WordTimestamp {
                                    word: prev_word,
                                    start_time: prev_start_time,
                                    end_time: Some(*start_time),
                                });
                            }
                            last_word = Some((word, *start_time));
                        }
                    }
                }
            }
        }

        if let Some((word, start_time)) = last_word.take() {
            words.push(WordTimestamp {
                word,
                start_time,
                end_time: None,
            });
        }

        Ok(TranscriptionResult {
            text: current_text.trim().to_string(),
            words,
        })
    }

    fn save_audio_wav(&self, pcm: &[f32], sample_rate: u32, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let mut file = BufWriter::new(File::create(path)?);

        // WAV header
        let data_size = (pcm.len() * 2) as u32; // 16-bit samples
        let file_size = data_size + 36;

        // RIFF header
        file.write_all(b"RIFF")?;
        file.write_all(&file_size.to_le_bytes())?;
        file.write_all(b"WAVE")?;

        // fmt chunk
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?; // chunk size
        file.write_all(&1u16.to_le_bytes())?; // audio format (PCM)
        file.write_all(&1u16.to_le_bytes())?; // num channels
        file.write_all(&sample_rate.to_le_bytes())?; // sample rate
        file.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
        file.write_all(&2u16.to_le_bytes())?; // block align
        file.write_all(&16u16.to_le_bytes())?; // bits per sample

        // data chunk
        file.write_all(b"data")?;
        file.write_all(&data_size.to_le_bytes())?;

        // audio data (convert f32 to i16)
        for &sample in pcm {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            file.write_all(&sample_i16.to_le_bytes())?;
        }

        file.flush()?;
        Ok(())
    }
}

pub fn create_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub mod audio {
    use anyhow::Result;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use crossbeam_channel::Sender;
    use std::thread;

    pub fn list_audio_devices() -> Result<()> {
        let host = cpal::default_host();

        println!("Available input devices:");
        for (i, device) in host.input_devices()?.enumerate() {
            let name = device.name().unwrap_or("Unknown".to_string());
            println!("  {}: {}", i, name);
        }

        Ok(())
    }

    pub fn start_audio_capture(
        audio_tx: Sender<Vec<f32>>,
        device_index: Option<usize>,
    ) -> Result<()> {
        let max_retries = 3;
        let mut retry_count = 0;

        loop {
            match start_audio_capture_internal(audio_tx.clone(), device_index) {
                Ok(()) => break,
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(e);
                    }
                    eprintln!(
                        "Audio capture failed (attempt {}/{}): {}",
                        retry_count, max_retries, e
                    );
                    eprintln!("Retrying audio capture in 1 second...");
                    thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }
        Ok(())
    }

    fn start_audio_capture_internal(
        audio_tx: Sender<Vec<f32>>,
        device_index: Option<usize>,
    ) -> Result<()> {
        let host = cpal::default_host();
        let device = if let Some(index) = device_index {
            host.input_devices()?
                .nth(index)
                .ok_or_else(|| anyhow::anyhow!("Device index {} not found", index))?
        } else {
            host.default_input_device()
                .ok_or_else(|| anyhow::anyhow!("No input device available"))?
        };

        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;

        eprintln!(
            "Using input device: {}",
            device.name().unwrap_or("Unknown".to_string())
        );
        eprintln!("Sample rate: {}", sample_rate);
        eprintln!("Sample format: {:?}", config.sample_format());
        eprintln!("Channels: {}", config.channels());

        let audio_tx_clone = audio_tx.clone();
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let config = config.into();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        // Convert stereo to mono by averaging channels
                        let mono_data = if config.channels == 2 {
                            data.chunks(2)
                                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                                .collect()
                        } else {
                            data.to_vec()
                        };

                        let resampled = if sample_rate != 24000 {
                            let ratio = 24000.0 / sample_rate as f32;
                            let new_len = (mono_data.len() as f32 * ratio) as usize;
                            let mut resampled = Vec::with_capacity(new_len);

                            for i in 0..new_len {
                                let pos = i as f32 / ratio;
                                let idx = pos as usize;
                                let frac = pos - idx as f32;

                                if idx + 1 < mono_data.len() {
                                    let sample =
                                        mono_data[idx] * (1.0 - frac) + mono_data[idx + 1] * frac;
                                    resampled.push(sample);
                                } else if idx < mono_data.len() {
                                    resampled.push(mono_data[idx]);
                                }
                            }
                            resampled
                        } else {
                            mono_data
                        };

                        if audio_tx_clone.send(resampled).is_err() {
                            eprintln!("Audio receiver disconnected");
                            return;
                        }
                    },
                    |err| {
                        eprintln!("Audio stream error: {}", err);
                        std::process::exit(1);
                    },
                    None,
                )?
            }
            cpal::SampleFormat::I16 => {
                let config = config.into();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        // Convert I16 to F32
                        let f32_data: Vec<f32> =
                            data.iter().map(|&sample| sample as f32 / 32768.0).collect();

                        // Convert stereo to mono by averaging channels
                        let mono_data = if config.channels == 2 {
                            f32_data
                                .chunks(2)
                                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                                .collect()
                        } else {
                            f32_data
                        };

                        let resampled = if sample_rate != 24000 {
                            let ratio = 24000.0 / sample_rate as f32;
                            let new_len = (mono_data.len() as f32 * ratio) as usize;
                            let mut resampled = Vec::with_capacity(new_len);

                            for i in 0..new_len {
                                let pos = i as f32 / ratio;
                                let idx = pos as usize;
                                let frac = pos - idx as f32;

                                if idx + 1 < mono_data.len() {
                                    let sample =
                                        mono_data[idx] * (1.0 - frac) + mono_data[idx + 1] * frac;
                                    resampled.push(sample);
                                } else if idx < mono_data.len() {
                                    resampled.push(mono_data[idx]);
                                }
                            }
                            resampled
                        } else {
                            mono_data
                        };

                        if audio_tx_clone.send(resampled).is_err() {
                            eprintln!("Audio receiver disconnected");
                            return;
                        }
                    },
                    |err| {
                        eprintln!("Audio stream error: {}", err);
                        std::process::exit(1);
                    },
                    None,
                )?
            }
            cpal::SampleFormat::U16 => {
                let config = config.into();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        // Convert U16 to F32
                        let f32_data: Vec<f32> = data
                            .iter()
                            .map(|&sample| (sample as f32 - 32768.0) / 32768.0)
                            .collect();

                        // Convert stereo to mono by averaging channels
                        let mono_data = if config.channels == 2 {
                            f32_data
                                .chunks(2)
                                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                                .collect()
                        } else {
                            f32_data
                        };

                        let resampled = if sample_rate != 24000 {
                            let ratio = 24000.0 / sample_rate as f32;
                            let new_len = (mono_data.len() as f32 * ratio) as usize;
                            let mut resampled = Vec::with_capacity(new_len);

                            for i in 0..new_len {
                                let pos = i as f32 / ratio;
                                let idx = pos as usize;
                                let frac = pos - idx as f32;

                                if idx + 1 < mono_data.len() {
                                    let sample =
                                        mono_data[idx] * (1.0 - frac) + mono_data[idx + 1] * frac;
                                    resampled.push(sample);
                                } else if idx < mono_data.len() {
                                    resampled.push(mono_data[idx]);
                                }
                            }
                            resampled
                        } else {
                            mono_data
                        };

                        if audio_tx_clone.send(resampled).is_err() {
                            eprintln!("Audio receiver disconnected");
                            return;
                        }
                    },
                    |err| {
                        eprintln!("Audio stream error: {}", err);
                        std::process::exit(1);
                    },
                    None,
                )?
            }
            format => {
                return Err(anyhow::anyhow!(
                    "Unsupported sample format: {:?}. Supported formats: F32, I16, U16",
                    format
                ));
            }
        };

        stream.play()?;

        loop {
            thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}
