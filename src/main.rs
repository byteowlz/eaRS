// Copyright (c) Kyutai and Tommy Falkowski, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use candle::{Device, Tensor};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::thread;

#[derive(Debug, Parser)]
struct Args {
    /// The audio input file, in wav/mp3/ogg/... format. If not provided, uses microphone.
    in_file: Option<String>,

    /// Use live microphone input instead of file.
    #[arg(long, short)]
    live: bool,

    /// List available audio devices.
    #[arg(long)]
    list_devices: bool,

    /// The repo where to get the model from.
    #[arg(long, default_value = "kyutai/stt-1b-en_fr-candle")]
    hf_repo: String,

    /// Run the model on cpu.
    #[arg(long)]
    cpu: bool,

    /// Display word level timestamps.
    #[arg(long)]
    timestamps: bool,

    /// Display the level of voice activity detection (VAD).
    #[arg(long)]
    vad: bool,
}

fn device(cpu: bool) -> Result<Device> {
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

#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self, vad: bool) -> moshi::lm::Config {
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

struct Model {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    timestamps: bool,
    vad: bool,
    config: Config,
    dev: Device,
}

impl Model {
    fn load_from_hf(args: &Args, dev: &Device) -> Result<Self> {
        let dtype = dev.bf16_default_to_f32();

        // Retrieve the model files from the Hugging Face Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(args.hf_repo.to_string());
        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get("model.safetensors")?;
        let mimi_file = repo.get(&config.mimi_name)?;

        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;
        let vb_lm =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, dev)? };
        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), dev)?;
        let lm = moshi::lm::LmModel::new(
            &config.model_config(args.vad),
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;
        Ok(Model {
            state,
            config,
            text_tokenizer,
            timestamps: args.timestamps,
            vad: args.vad,
            dev: dev.clone(),
        })
    }

    fn run(&mut self, mut pcm: Vec<f32>) -> Result<()> {
        use std::io::Write;

        // Add the silence prefix to the audio.
        if self.config.stt_config.audio_silence_prefix_seconds > 0.0 {
            let silence_len =
                (self.config.stt_config.audio_silence_prefix_seconds * 24000.0) as usize;
            pcm.splice(0..0, vec![0.0; silence_len]);
        }
        // Add some silence at the end to ensure all the audio is processed.
        let suffix = (self.config.stt_config.audio_delay_seconds * 24000.0) as usize;
        pcm.resize(pcm.len() + suffix + 24000, 0.0);

        let mut last_word = None;
        let mut printed_eot = false;
        for pcm in pcm.chunks(1920) {
            let pcm = Tensor::new(pcm, &self.dev)?.reshape((1, 1, ()))?;
            let asr_msgs = self.state.step_pcm(pcm, None, &().into(), |_, _, _| ())?;
            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        // prs is the probability of having no voice activity for different time
                        // horizons.
                        // In kyutai/stt-1b-en_fr-candle, these horizons are 0.5s, 1s, 2s, and 3s.
                        if self.vad && prs[2][0] > 0.5 && !printed_eot {
                            printed_eot = true;
                            if !self.timestamps {
                                print!(" <endofturn pr={}>", prs[2][0]);
                            } else {
                                println!("<endofturn pr={}>", prs[2][0]);
                            }
                        }
                    }
                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                        printed_eot = false;
                        if self.timestamps {
                            if let Some((word, start_time)) = last_word.take() {
                                println!("[{start_time:5.2}-{stop_time:5.2}] {word}");
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
                        if !self.timestamps {
                            print!(" {word}");
                            std::io::stdout().flush()?
                        } else {
                            if let Some((word, prev_start_time)) = last_word.take() {
                                println!("[{prev_start_time:5.2}-{start_time:5.2}] {word}");
                            }
                            last_word = Some((word, *start_time));
                        }
                    }
                }
            }
        }
        if let Some((word, start_time)) = last_word.take() {
            println!("[{start_time:5.2}-     ] {word}");
        }
        println!();
        Ok(())
    }

    fn run_live(&mut self, audio_rx: Receiver<Vec<f32>>) -> Result<()> {
        use std::io::Write;

        eprintln!("Starting live transcription. Press Ctrl+C to stop.");
        eprintln!("Transcription output:");
        eprintln!("{}", "-".repeat(50));

        let mut last_word = None;
        let mut printed_eot = false;

        for pcm_chunk in audio_rx {
            for pcm in pcm_chunk.chunks(1920) {
                let pcm = Tensor::new(pcm, &self.dev)?.reshape((1, 1, ()))?;
                let asr_msgs = self.state.step_pcm(pcm, None, &().into(), |_, _, _| ())?;
                for asr_msg in asr_msgs.iter() {
                    match asr_msg {
                        moshi::asr::AsrMsg::Step { prs, .. } => {
                            if self.vad && prs[2][0] > 0.5 && !printed_eot {
                                printed_eot = true;
                                if !self.timestamps {
                                    print!(" <pause>");
                                    std::io::stdout().flush()?;
                                }
                            }
                        }
                        moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                            printed_eot = false;
                            if self.timestamps {
                                if let Some((word, start_time)) = last_word.take() {
                                    println!("[{start_time:5.2}-{stop_time:5.2}] {word}");
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
                            if !self.timestamps {
                                print!(" {}", word);
                                std::io::stdout().flush()?;
                            } else {
                                if let Some((word, prev_start_time)) = last_word.take() {
                                    println!("[{prev_start_time:5.2}-{start_time:5.2}] {word}");
                                }
                                last_word = Some((word, *start_time));
                            }
                        }
                    }
                }
            }
        }

        if let Some((word, start_time)) = last_word.take() {
            println!("[{start_time:5.2}-     ] {word}");
        }
        println!();
        Ok(())
    }
}

fn list_audio_devices() -> Result<()> {
    let host = cpal::default_host();

    println!("Available input devices:");
    for (i, device) in host.input_devices()?.enumerate() {
        let name = device.name().unwrap_or("Unknown".to_string());
        println!("  {}: {}", i, name);
    }

    Ok(())
}

fn start_audio_capture(audio_tx: Sender<Vec<f32>>) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

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
                    // Resample to 24kHz if necessary
                    let resampled = if sample_rate != 24000 {
                        // Simple linear interpolation resampling
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

    // Keep the stream alive
    loop {
        thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_devices {
        return list_audio_devices();
    }

    let device = device(args.cpu)?;
    eprintln!("Using device: {:?}", device);

    if args.live || args.in_file.is_none() {
        // Live microphone mode
        eprintln!("Loading model from repository: {}", args.hf_repo);
        let mut model = Model::load_from_hf(&args, &device)?;

        let (audio_tx, audio_rx) = unbounded();

        // Start audio capture in a separate thread
        let audio_handle = thread::spawn(move || {
            if let Err(e) = start_audio_capture(audio_tx) {
                eprintln!("Audio capture error: {}", e);
            }
        });

        // Run live transcription
        model.run_live(audio_rx)?;

        audio_handle.join().unwrap();
    } else if let Some(ref in_file) = args.in_file {
        // File mode
        eprintln!("Loading audio file from: {}", in_file);
        let (pcm, sample_rate) = kaudio::pcm_decode(in_file)?;
        let pcm = if sample_rate != 24_000 {
            kaudio::resample(&pcm, sample_rate as usize, 24_000)?
        } else {
            pcm
        };
        eprintln!("Loading model from repository: {}", args.hf_repo);
        let mut model = Model::load_from_hf(&args, &device)?;
        eprintln!("Running inference");
        model.run(pcm)?;
    } else {
        eprintln!("Either provide a file or use --live for microphone input");
        std::process::exit(1);
    }

    Ok(())
}
