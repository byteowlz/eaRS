use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use crossbeam_channel::unbounded;
use prkt::{Device, ParakeetModel};
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "coreml")]
    Coreml,
    #[cfg(feature = "directml")]
    Directml,
}

impl From<DeviceArg> for Device {
    fn from(arg: DeviceArg) -> Self {
        match arg {
            DeviceArg::Cpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            DeviceArg::Cuda => Device::Cuda,
            #[cfg(feature = "coreml")]
            DeviceArg::Coreml => Device::CoreML,
            #[cfg(feature = "directml")]
            DeviceArg::Directml => Device::DirectML,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "prkt")]
#[command(author, version, about = "Parakeet TDT Speech Recognition CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(about = "Transcribe an audio file or stdin")]
    File {
        #[arg(help = "Path to the audio file (WAV format), or '-' for stdin")]
        audio_file: String,

        #[arg(
            long,
            default_value = "istupakov/parakeet-tdt-0.6b-v3-onnx",
            help = "Hugging Face model repository"
        )]
        model: String,

        #[arg(long, help = "Local model directory (overrides --model)")]
        model_dir: Option<PathBuf>,

        #[arg(
            long,
            short = 'd',
            value_enum,
            help = "Execution device (defaults to GPU if available, otherwise CPU)"
        )]
        device: Option<DeviceArg>,

        #[arg(
            long,
            help = "Sample rate of stdin audio (required when using stdin)",
            default_value = "16000"
        )]
        sample_rate: u32,
    },

    #[command(about = "Live transcription from microphone")]
    Live {
        #[arg(
            long,
            default_value = "istupakov/parakeet-tdt-0.6b-v3-onnx",
            help = "Hugging Face model repository"
        )]
        model: String,

        #[arg(long, help = "Local model directory (overrides --model)")]
        model_dir: Option<PathBuf>,

        #[arg(
            long,
            short = 'd',
            value_enum,
            help = "Execution device (defaults to GPU if available, otherwise CPU)"
        )]
        device: Option<DeviceArg>,

        #[arg(
            long,
            default_value = "3",
            help = "Audio chunk size in seconds"
        )]
        chunk_seconds: u32,

        #[arg(
            long,
            default_value = "1",
            help = "Overlap between chunks in seconds"
        )]
        overlap_seconds: u32,

        #[arg(
            long,
            help = "Disable filtering of redundant transcriptions from overlapping audio"
        )]
        no_filter: bool,
    },
}

fn get_default_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        eprintln!("No device specified, defaulting to CUDA");
        return Device::Cuda;
    }

    #[cfg(all(feature = "coreml", not(feature = "cuda")))]
    {
        eprintln!("No device specified, defaulting to CoreML");
        return Device::CoreML;
    }

    #[cfg(all(feature = "directml", not(feature = "cuda"), not(feature = "coreml")))]
    {
        eprintln!("No device specified, defaulting to DirectML");
        return Device::DirectML;
    }

    #[cfg(not(any(feature = "cuda", feature = "coreml", feature = "directml")))]
    {
        eprintln!("No device specified, defaulting to CPU");
        Device::Cpu
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::File {
            audio_file,
            model,
            model_dir,
            device,
            sample_rate,
        } => transcribe_file(audio_file, model, model_dir, device, sample_rate)?,

        Commands::Live {
            model,
            model_dir,
            device,
            chunk_seconds,
            overlap_seconds,
            no_filter,
        } => transcribe_live(model, model_dir, device, chunk_seconds, overlap_seconds, no_filter)?,
    }

    Ok(())
}

fn transcribe_file(
    audio_file: String,
    model_repo: String,
    model_dir: Option<PathBuf>,
    device: Option<DeviceArg>,
    stdin_sample_rate: u32,
) -> Result<()> {
    let device = device.map(|d| d.into()).unwrap_or_else(get_default_device);

    eprintln!("Loading Parakeet model...");
    let mut model = ParakeetModel::load_from_hf(&model_repo, model_dir.as_deref(), device)?;

    let (waveform, sample_rate) = if audio_file == "-" {
        eprintln!("\nReading audio from stdin...");
        let waveform = read_audio_from_stdin()?;
        eprintln!(
            "Audio loaded: {} samples at {} Hz ({:.2} seconds)",
            waveform.len(),
            stdin_sample_rate,
            waveform.len() as f32 / stdin_sample_rate as f32
        );
        (waveform, stdin_sample_rate)
    } else {
        eprintln!("\nLoading audio file: {:?}", audio_file);
        let (waveform, sr) = prkt::load_wav_file(&PathBuf::from(&audio_file))?;
        eprintln!(
            "Audio loaded: {} samples at {} Hz ({:.2} seconds)",
            waveform.len(),
            sr,
            waveform.len() as f32 / sr as f32
        );
        (waveform, sr)
    };

    eprintln!("\nTranscribing...");
    let start = Instant::now();
    let transcript = model.transcribe(&waveform, sample_rate as usize)?;
    let elapsed = start.elapsed();

    eprintln!("\nTranscription completed in {:.2}s", elapsed.as_secs_f32());
    eprintln!(
        "RTF: {:.2}x",
        (waveform.len() as f32 / sample_rate as f32) / elapsed.as_secs_f32()
    );

    println!("\n{}", transcript);

    Ok(())
}

fn read_audio_from_stdin() -> Result<Vec<f32>> {
    use std::io::{stdin, Read};
    
    let mut buffer = Vec::new();
    stdin().read_to_end(&mut buffer)?;
    
    // Convert raw bytes to f32 samples
    // Assume input is raw f32 PCM data (little-endian)
    if buffer.len() % 4 != 0 {
        return Err(anyhow::anyhow!(
            "Invalid audio data: length must be multiple of 4 bytes for f32 samples"
        ));
    }
    
    let mut samples = Vec::with_capacity(buffer.len() / 4);
    for chunk in buffer.chunks_exact(4) {
        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        samples.push(sample);
    }
    
    Ok(samples)
}

fn transcribe_live(
    model_repo: String,
    model_dir: Option<PathBuf>,
    device: Option<DeviceArg>,
    chunk_seconds: u32,
    overlap_seconds: u32,
    no_filter: bool,
) -> Result<()> {
    let device = device.map(|d| d.into()).unwrap_or_else(get_default_device);

    println!("Loading Parakeet model...");
    let mut model = ParakeetModel::load_from_hf(&model_repo, model_dir.as_deref(), device)?;
    println!("Model loaded!\n");

    let (audio_tx, audio_rx) = unbounded();

    // Start audio capture in separate thread
    let capture_handle = thread::spawn(move || {
        if let Err(e) = start_audio_capture(audio_tx) {
            eprintln!("Audio capture error: {}", e);
        }
    });

    println!("🎤 Live transcription started. Speak into your microphone...");
    println!("Press Ctrl+C to stop.\n");

    // Accumulate audio chunks
    let mut audio_buffer = Vec::new();
    let chunk_size = 16000 * chunk_seconds as usize;
    let overlap = 16000 * overlap_seconds as usize;
    let mut last_transcription_time = Instant::now();
    let mut previous_text = String::new();

    loop {
        match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(chunk) => {
                audio_buffer.extend(chunk);

                // Transcribe when we have enough audio
                if audio_buffer.len() >= chunk_size {
                    let duration = last_transcription_time.elapsed();
                    last_transcription_time = Instant::now();

                    // Transcribe the accumulated audio
                    let start = Instant::now();
                    match model.transcribe(&audio_buffer, 16000) {
                        Ok(text) => {
                            let transcription_time = start.elapsed();
                            let audio_duration = audio_buffer.len() as f32 / 16000.0;
                            let rtf = transcription_time.as_secs_f32() / audio_duration;

                            if !text.trim().is_empty() {
                                let output_text = if no_filter {
                                    text.clone()
                                } else {
                                    filter_redundant_text(&previous_text, &text)
                                };

                                if !output_text.trim().is_empty() {
                                    println!(
                                        "[{:.1}s] {} (RTF: {:.2}x, latency: {:.0}ms)",
                                        duration.as_secs_f32(),
                                        output_text,
                                        rtf,
                                        duration.as_millis()
                                    );
                                }

                                previous_text = text;
                            }
                        }
                        Err(e) => {
                            eprintln!("Transcription error: {}", e);
                        }
                    }

                    // Keep overlap for context
                    if audio_buffer.len() > overlap {
                        audio_buffer.drain(0..(audio_buffer.len() - overlap));
                    } else {
                        audio_buffer.clear();
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // No audio received, continue
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                println!("\nAudio capture stopped.");
                break;
            }
        }
    }

    capture_handle.join().unwrap();
    Ok(())
}

fn start_audio_capture(audio_tx: crossbeam_channel::Sender<Vec<f32>>) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

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
    eprintln!("Channels: {}\n", config.channels());

    let audio_tx_clone = audio_tx.clone();
    let channels = config.channels();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert to mono by averaging channels
                let mono_data: Vec<f32> = if channels == 2 {
                    data.chunks(2)
                        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                        .collect()
                } else {
                    data.to_vec()
                };

                // Resample to 16kHz if needed
                let resampled = if sample_rate != 16000 {
                    simple_resample(&mono_data, sample_rate, 16000)
                } else {
                    mono_data
                };

                if audio_tx_clone.send(resampled).is_err() {
                    eprintln!("Audio receiver disconnected");
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                // Convert I16 to F32
                let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();

                // Convert to mono
                let mono_data: Vec<f32> = if channels == 2 {
                    f32_data
                        .chunks(2)
                        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                        .collect()
                } else {
                    f32_data
                };

                // Resample to 16kHz if needed
                let resampled = if sample_rate != 16000 {
                    simple_resample(&mono_data, sample_rate, 16000)
                } else {
                    mono_data
                };

                if audio_tx_clone.send(resampled).is_err() {
                    eprintln!("Audio receiver disconnected");
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config.into(),
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                // Convert U16 to F32
                let f32_data: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f32 - 32768.0) / 32768.0)
                    .collect();

                // Convert to mono
                let mono_data: Vec<f32> = if channels == 2 {
                    f32_data
                        .chunks(2)
                        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                        .collect()
                } else {
                    f32_data
                };

                // Resample to 16kHz if needed
                let resampled = if sample_rate != 16000 {
                    simple_resample(&mono_data, sample_rate, 16000)
                } else {
                    mono_data
                };

                if audio_tx_clone.send(resampled).is_err() {
                    eprintln!("Audio receiver disconnected");
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;

    // Keep the stream alive
    std::thread::park();

    Ok(())
}

fn simple_resample(data: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = to_rate as f32 / from_rate as f32;
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
}

fn normalize_word(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

fn filter_redundant_text(previous: &str, current: &str) -> String {
    if previous.is_empty() {
        return current.to_string();
    }

    let prev_words: Vec<&str> = previous.split_whitespace().collect();
    let curr_words: Vec<&str> = current.split_whitespace().collect();

    if curr_words.is_empty() {
        return String::new();
    }

    let min_overlap_threshold = 1;
    let max_overlap = prev_words.len().min(curr_words.len());
    let mut best_overlap = 0;

    for overlap_len in (min_overlap_threshold..=max_overlap).rev() {
        let prev_suffix = &prev_words[prev_words.len() - overlap_len..];
        let curr_prefix = &curr_words[..overlap_len];

        let matches = prev_suffix.iter().zip(curr_prefix.iter()).all(|(a, b)| {
            normalize_word(a) == normalize_word(b)
        });

        if matches {
            best_overlap = overlap_len;
            break;
        }
    }

    if best_overlap > 0 && best_overlap == curr_words.len() {
        return String::new();
    }

    if best_overlap >= min_overlap_threshold {
        curr_words[best_overlap..].join(" ")
    } else {
        current.to_string()
    }
}
