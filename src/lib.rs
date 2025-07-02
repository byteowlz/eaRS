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
/// ## Usage Example
/// ```bash
/// # Start transcription with WebSocket on port 8080
/// ears --live --ws 8080
/// 
/// # With timestamps and VAD
/// ears --live --ws 8080 --timestamps --vad
/// ```
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    #[serde(rename = "word")]
    Word {
        word: String,
        start_time: f64,
        end_time: Option<f64>,
    },
    #[serde(rename = "pause")]
    Pause {
        timestamp: f64,
    },
    #[serde(rename = "final")]
    Final {
        text: String,
        words: Vec<WordTimestamp>,
    },
}

pub struct TranscriptionOptions {
    pub timestamps: bool,
    pub vad: bool,
    pub save_audio: Option<String>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            timestamps: false,
            vad: false,
            save_audio: None,
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
            let _ = self.state.step_pcm(tensor, None, &().into(), |_, _, _| ())?;
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

        let mut all_audio = Vec::new();
        let mut words = Vec::new();
        let mut current_text = String::new();
        let mut last_word: Option<(String, f64)> = None;
        let mut printed_eot = false;

        for pcm_chunk in audio_rx {
            if save_audio.is_some() {
                all_audio.extend_from_slice(&pcm_chunk);
            }

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
                            let word = self
                                .text_tokenizer
                                .decode_piece_ids(tokens)
                                .unwrap_or_else(|_| String::new());

                            current_text.push(' ');
                            current_text.push_str(&word);

                            if !self.timestamps {
                                print!(" {}", word);
                                std::io::stdout().flush().ok();
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

        if !self.timestamps {
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
        use std::sync::Arc;
        use tokio::sync::broadcast;
        use tokio_tungstenite::{accept_async, tungstenite::Message};
        use std::io::Write;

        // Create broadcast channel for WebSocket messages
        let (ws_tx, _ws_rx) = broadcast::channel(100);
        let ws_tx = Arc::new(ws_tx);

        // Start WebSocket server
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", ws_port)).await?;
        let ws_tx_clone = ws_tx.clone();

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let ws_tx = ws_tx_clone.clone();
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

                    // Handle incoming WebSocket messages (if any)
                    let receive_task = tokio::spawn(async move {
                        while let Some(msg) = ws_receiver.next().await {
                            match msg {
                                Ok(Message::Close(_)) => break,
                                Err(e) => {
                                    eprintln!("WebSocket receive error: {}", e);
                                    break;
                                }
                                _ => {} // Ignore other message types for now
                            }
                        }
                    });

                    // Forward broadcast messages to WebSocket
                    let send_task = tokio::spawn(async move {
                        while let Ok(ws_msg) = ws_rx.recv().await {
                            let json_msg = serde_json::to_string(&ws_msg).unwrap_or_default();
                            if ws_sender.send(Message::Text(json_msg)).await.is_err() {
                                break;
                            }
                        }
                    });

                    tokio::select! {
                        _ = receive_task => {},
                        _ = send_task => {},
                    }
                });
            }
        });

        // Process audio synchronously with WebSocket streaming
        let mut all_audio = Vec::new();
        let mut words = Vec::new();
        let mut current_text = String::new();
        let mut last_word: Option<(String, f64)> = None;
        let mut printed_eot = false;

        for pcm_chunk in audio_rx {
            if save_audio.is_some() {
                all_audio.extend_from_slice(&pcm_chunk);
            }

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
                            if self.timestamps {
                                if let Some((word, start_time)) = last_word.take() {
                                    println!("[{start_time:5.2}-{stop_time:5.2}] {word}");
                                    
                                    let word_timestamp = WordTimestamp {
                                        word: word.clone(),
                                        start_time,
                                        end_time: Some(*stop_time),
                                    };
                                    words.push(word_timestamp.clone());

                                    let ws_msg = WebSocketMessage::Word {
                                        word: word_timestamp.word,
                                        start_time: word_timestamp.start_time,
                                        end_time: word_timestamp.end_time,
                                    };
                                    let _ = ws_tx.send(ws_msg);
                                }
                            }
                        }
                        moshi::asr::AsrMsg::Word {
                            tokens, start_time, ..
                        } => {
                            printed_eot = false;
                            let word = self
                                .text_tokenizer
                                .decode_piece_ids(tokens)
                                .unwrap_or_else(|_| String::new());

                            current_text.push(' ');
                            current_text.push_str(&word);

                            if !self.timestamps {
                                print!(" {}", word);
                                std::io::stdout().flush().ok();

                                // Send word without end time for real-time streaming
                                let ws_msg = WebSocketMessage::Word {
                                    word: word.clone(),
                                    start_time: *start_time,
                                    end_time: None,
                                };
                                let _ = ws_tx.send(ws_msg);
                            } else {
                                if let Some((prev_word, prev_start_time)) = last_word.take() {
                                    println!(
                                        "[{prev_start_time:5.2}-{start_time:5.2}] {prev_word}"
                                    );
                                    
                                    let word_timestamp = WordTimestamp {
                                        word: prev_word.clone(),
                                        start_time: prev_start_time,
                                        end_time: Some(*start_time),
                                    };
                                    words.push(word_timestamp.clone());

                                    let ws_msg = WebSocketMessage::Word {
                                        word: word_timestamp.word,
                                        start_time: word_timestamp.start_time,
                                        end_time: word_timestamp.end_time,
                                    };
                                    let _ = ws_tx.send(ws_msg);
                                }
                                last_word = Some((word, *start_time));
                            }
                        }
                    }
                }
            }
        }

        // Handle final word
        if let Some((word, start_time)) = last_word.take() {
            if self.timestamps {
                println!("[{start_time:5.2}-     ] {word}");
            }
            
            let word_timestamp = WordTimestamp {
                word: word.clone(),
                start_time,
                end_time: None,
            };
            words.push(word_timestamp.clone());

            let ws_msg = WebSocketMessage::Word {
                word: word_timestamp.word,
                start_time: word_timestamp.start_time,
                end_time: word_timestamp.end_time,
            };
            let _ = ws_tx.send(ws_msg);
        }

        if !self.timestamps {
            println!();
        }

        let final_result = TranscriptionResult {
            text: current_text.trim().to_string(),
            words: words.clone(),
        };

        // Send final result
        let final_msg = WebSocketMessage::Final {
            text: final_result.text.clone(),
            words: final_result.words.clone(),
        };
        let _ = ws_tx.send(final_msg);

        if let Some(save_path) = save_audio {
            self.save_audio_wav(&all_audio, 24000, save_path)?;
        }

        Ok(final_result)
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

    pub fn start_audio_capture(audio_tx: Sender<Vec<f32>>, device_index: Option<usize>) -> Result<()> {
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

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let config = config.into();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let resampled = if sample_rate != 24000 {
                            let ratio = 24000.0 / sample_rate as f32;
                            let new_len = (data.len() as f32 * ratio) as usize;
                            let mut resampled = Vec::with_capacity(new_len);

                            for i in 0..new_len {
                                let pos = i as f32 / ratio;
                                let idx = pos as usize;
                                let frac = pos - idx as f32;

                                if idx + 1 < data.len() {
                                    let sample = data[idx] * (1.0 - frac) + data[idx + 1] * frac;
                                    resampled.push(sample);
                                } else if idx < data.len() {
                                    resampled.push(data[idx]);
                                }
                            }
                            resampled
                        } else {
                            data.to_vec()
                        };

                        if audio_tx.send(resampled).is_err() {
                            eprintln!("Audio receiver disconnected");
                        }
                    },
                    |err| eprintln!("Audio stream error: {}", err),
                    None,
                )?
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported sample format. Only F32 is supported."
                ));
            }
        };

        stream.play()?;

        loop {
            thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

