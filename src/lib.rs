use anyhow::Result;
use candle::{Device, Tensor};
use crossbeam_channel::Receiver;
use std::path::Path;

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
/// All messages are JSON objects with a `type` field indicating the message type:
///
/// ### Word Message
/// Sent for each transcribed word as it's recognized:
/// ```json
/// {
///   "type": "word",
///   "word": "hello",
///   "start_time": 1.23,
///   "end_time": 1.45  // null for real-time words without end time yet
/// }
/// ```
///
/// ### Pause Message
/// Sent when voice activity detection detects a pause (requires --vad flag):
/// ```json
/// {
///   "type": "pause",
///   "timestamp": 1234567890.123
/// }
/// ```
///
/// ### Final Message
/// Sent at the end of transcription with complete results:
/// ```json
/// {
///   "type": "final",
///   "text": "complete transcribed text",
///   "words": [
///     {"word": "hello", "start_time": 1.23, "end_time": 1.45},
///     {"word": "world", "start_time": 1.46, "end_time": null}
///   ]
/// }
/// ```
///
/// ### Restart Command
/// Send from client to restart transcription after timeout or final message:
/// ```json
/// {
///   "type": "restart"
/// }
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
#[derive(Debug, Clone, serde::Serialize)]
pub enum WebSocketMessage {
    Word {
        word: String,
        start_time: f64,
        end_time: Option<f64>,
    },
    Pause { timestamp: f64 },
    Final {
        text: String,
        words: Vec<WordTimestamp>,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
pub enum WebSocketCommand {
    Restart,
    Pause,
    Resume,
}

pub struct TranscriptionOptions {
    pub timestamps: bool,
    pub vad: bool,
    pub save_audio: Option<String>,
    pub vad_timeout: Option<f64>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            timestamps: false,
            vad: false,
            save_audio: None,
            vad_timeout: None,
        }
    }
}

impl Model {
    pub fn load_from_hf(hf_repo: &str, cpu: bool, options: TranscriptionOptions) -> Result<Self> {
        let device = create_device(cpu)?;
        let dtype = device.bf16_default_to_f32();

        let api = hf_hub::api::sync::Api::new()?;
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
        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), &device)?;
        let lm = moshi::lm::LmModel::new(
            &config.model_config(options.vad),
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;

        Ok(Model {
            state,
            config,
            text_tokenizer,
            timestamps: options.timestamps,
            vad: options.vad,
            dev: device,
            vad_timeout: options.vad_timeout,
        })
    }

    pub fn prime_with_audio<P: AsRef<Path>>(&mut self, file_path: P) -> Result<()> {
        let (pcm, sample_rate) = kaudio::pcm_decode(file_path.as_ref())?;
        let pcm = if sample_rate != 24_000 {
            kaudio::resample(&pcm, sample_rate as usize, 24_000)?
        } else {
            pcm
        };

        for chunk in pcm.chunks(1920) {
            let tensor = Tensor::new(chunk, &self.dev)?.reshape((1, 1, chunk.len()))?;
            let _ = self
                .state
                .step_pcm(tensor, None, &().into(), |_, _, _| ())?;
        }
        Ok(())
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
        use std::io::Write;
        use std::time::{Duration, Instant};

        let mut all_audio = Vec::new();
        let mut words = Vec::new();
        let mut current_text = String::new();
        let mut last_word: Option<(String, f64)> = None;
        let mut printed_eot = false;
        let mut last_voice_activity: Option<Instant> = None;

        loop {
            let pcm_chunk = match audio_rx.recv() {
                Ok(chunk) => chunk,
                Err(_) => {
                    eprintln!("Audio receiver channel closed");
                    return Err(anyhow::anyhow!("Audio receiver disconnected"));
                }
            };
            if save_audio.is_some() {
                all_audio.extend_from_slice(&pcm_chunk);
            }

            let mut has_voice_activity = false;

            for pcm in pcm_chunk.chunks(1920) {
                let pcm_tensor = Tensor::new(pcm, &self.dev)?.reshape((1, 1, ()))?;
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
                            if self.timestamps {
                                if let Some((word, start_time)) = last_word.take() {
                                    println!("[{start_time:5.2}-{stop_time:5.2}] {word}");
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
                            printed_eot = false;
                            has_voice_activity = true;
                            let word = self
                                .text_tokenizer
                                .decode_piece_ids(tokens)
                                .unwrap_or_else(|_| String::new());

                            current_text.push(' ');
                            current_text.push_str(&word);

                            if !self.timestamps {
                                // Only show live transcription if we're in an interactive terminal
                                if atty::is(atty::Stream::Stdout) {
                                    print!(" {}", word);
                                    std::io::stdout().flush().ok();
                                }
                            } else {
                                if let Some((prev_word, prev_start_time)) = last_word.take() {
                                    println!(
                                        "[{prev_start_time:5.2}-{start_time:5.2}] {prev_word}"
                                    );
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

            // Update voice activity timestamp if we detected voice
            if has_voice_activity {
                last_voice_activity = Some(Instant::now());
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
            println!();
        }

        if let Some(save_path) = save_audio {
            self.save_audio_wav(&all_audio, 24000, save_path)?;
        }

        Ok(TranscriptionResult {
            text: current_text.trim().to_string(),
            words,
        })
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

        // Spawn WebSocket server
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", ws_port)).await?;
        let ws_tx_clone = ws_tx.clone();
        let restart_tx_clone = restart_tx.clone();
        let pause_tx_clone = pause_tx.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let ws_tx = ws_tx_clone.clone();
                let restart_tx = restart_tx_clone.clone();
                let pause_tx = pause_tx_clone.clone();
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
                                                let _ = pause_tx.send(true);
                                            }
                                            WebSocketCommand::Resume => {
                                                let _ = pause_tx.send(false);
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
                            if ws_sender.send(Message::Text(json_msg)).await.is_err() {
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

        loop {
            let mut words = Vec::new();
            let mut current_text = String::new();
            let mut last_word: Option<(String, f64)> = None;
            let mut printed_eot = false;
            let mut last_voice_activity: Option<std::time::Instant> = None;
            let mut restart = false;
            let mut paused = *pause_rx.borrow();

            eprintln!("Starting transcription session (paused - send Resume command to begin)...");

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
                    Some(pcm_chunk) = pcm_rx.recv() => {
                        if paused {
                            continue;
                        }
                        if save_audio.is_some() {
                            all_audio.extend_from_slice(&pcm_chunk);
                        }

                        let mut has_voice_activity = false;

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
                                    moshi::asr::AsrMsg::Word { tokens, start_time, .. } => {
                                        printed_eot = false;
                                        has_voice_activity = true;
                                        let word = self.text_tokenizer
                                            .decode_piece_ids(tokens)
                                            .unwrap_or_else(|_| String::new());

                                        current_text.push(' ');
                                        current_text.push_str(&word);

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
                            data.chunks(2).map(|chunk| (chunk[0] + chunk[1]) / 2.0).collect()
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
                                    let sample = mono_data[idx] * (1.0 - frac) + mono_data[idx + 1] * frac;
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
                        let f32_data: Vec<f32> = data.iter().map(|&sample| sample as f32 / 32768.0).collect();
                        
                        // Convert stereo to mono by averaging channels
                        let mono_data = if config.channels == 2 {
                            f32_data.chunks(2).map(|chunk| (chunk[0] + chunk[1]) / 2.0).collect()
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
                                    let sample = mono_data[idx] * (1.0 - frac) + mono_data[idx + 1] * frac;
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
                        let f32_data: Vec<f32> = data.iter().map(|&sample| (sample as f32 - 32768.0) / 32768.0).collect();
                        
                        // Convert stereo to mono by averaging channels
                        let mono_data = if config.channels == 2 {
                            f32_data.chunks(2).map(|chunk| (chunk[0] + chunk[1]) / 2.0).collect()
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
                                    let sample = mono_data[idx] * (1.0 - frac) + mono_data[idx + 1] * frac;
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
                    "Unsupported sample format: {:?}. Supported formats: F32, I16, U16", format
                ));
            }
        };

        stream.play()?;

        loop {
            thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}
