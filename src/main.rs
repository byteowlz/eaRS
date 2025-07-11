use anyhow::Result;
use clap::Parser;
use crossbeam_channel::unbounded;
use ears::{Model, TranscriptionOptions, audio};
use std::thread;

#[derive(Debug, Parser)]
struct Args {
    /// The audio input file, in wav/mp3/ogg/... format. If not provided, uses microphone.
    in_file: Option<String>,

    /// Use live microphone input instead of file.
    #[arg(long)]
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

    /// Save the audio recording to a file (WAV format).
    #[arg(long)]
    save_audio: Option<String>,

    /// Select audio input device by index. Use --list-devices to see available devices.
    #[arg(long)]
    device: Option<usize>,

    /// Inject reference audio for language priming (esp, ger, jap)
    #[arg(long, short = 'l', value_parser = ["esp", "ger", "jap"])]
    lang: Option<String>,

    /// Start WebSocket server on specified port to stream transcription results
    #[arg(long)]
    ws: Option<u16>,

    /// Automatically terminate after no voice activity for specified seconds
    #[arg(long)]
    vad_timeout: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_devices {
        return audio::list_audio_devices();
    }

    let options = TranscriptionOptions {
        timestamps: args.timestamps,
        vad: args.vad,
        save_audio: args.save_audio.clone(),
        vad_timeout: args.vad_timeout,
    };

    if args.live || args.in_file.is_none() {
        // Live microphone mode
        eprintln!("Loading model from repository: {}", args.hf_repo);
        let mut model = Model::load_from_hf(&args.hf_repo, args.cpu, options)?;

        if let Some(ref lang) = args.lang {
            let path = format!("ref_audio/{}.mp3", lang);
            if let Err(e) = model.prime_with_audio(&path) {
                eprintln!("Warning: failed to process reference audio {}: {}", path, e);
            }
        }

        let device_index = args.device;
        let save_audio_path = args.save_audio.as_deref();
        let ws_port = args.ws;
        
        let result = loop {
            let (audio_tx, audio_rx) = unbounded();

            // Start audio capture in a separate thread
            let _audio_handle = thread::spawn(move || {
                if let Err(e) = audio::start_audio_capture(audio_tx, device_index) {
                    eprintln!("Audio capture error: {}", e);
                }
            });

            let transcription_result = if let Some(ws_port) = ws_port {
                eprintln!("Starting WebSocket server on port {}", ws_port);
                eprintln!("Starting live transcription with WebSocket streaming. Press Ctrl+C to stop.");
                eprintln!("WebSocket endpoint: ws://localhost:{}/", ws_port);
                
                // Run live transcription with WebSocket streaming
                model.transcribe_live_ws(audio_rx, save_audio_path, ws_port).await
            } else {
                eprintln!("Starting live transcription. Press Ctrl+C to stop.");
                eprintln!("Transcription output:");
                eprintln!("{}", "-".repeat(50));

                // Run live transcription
                model.transcribe_live(audio_rx, save_audio_path)
            };

            match transcription_result {
                Ok(result) => break result,
                Err(e) => {
                    eprintln!("Transcription error: {}", e);
                    eprintln!("Attempting to restart audio capture...");
                    thread::sleep(std::time::Duration::from_secs(2));
                    continue;
                }
            }
        };

        if args.timestamps {
            for word in result.words {
                if let Some(end_time) = word.end_time {
                    println!("[{:5.2}-{:5.2}] {}", word.start_time, end_time, word.word);
                } else {
                    println!("[{:5.2}-     ] {}", word.start_time, word.word);
                }
            }
        } else {
            println!("{}", result.text);
        }

        // Audio handle cleanup is managed by the reconnection loop
    } else if let Some(ref in_file) = args.in_file {
        // File mode
        eprintln!("Loading audio file from: {}", in_file);
        eprintln!("Loading model from repository: {}", args.hf_repo);
        let mut model = Model::load_from_hf(&args.hf_repo, args.cpu, options)?;
        eprintln!("Running inference");

        let result = model.transcribe_file(in_file, args.save_audio.as_deref())?;

        if args.timestamps {
            for word in result.words {
                if let Some(end_time) = word.end_time {
                    println!("[{:5.2}-{:5.2}] {}", word.start_time, end_time, word.word);
                } else {
                    println!("[{:5.2}-     ] {}", word.start_time, word.word);
                }
            }
        } else {
            println!("{}", result.text);
        }
    } else {
        eprintln!("Either provide a file or use --live for microphone input");
        std::process::exit(1);
    }

    Ok(())
}

