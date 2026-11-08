use std::env;
use std::path::PathBuf;

use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <MODEL_DIR> <AUDIO_FILE>", args[0]);
        eprintln!();
        eprintln!("  MODEL_DIR   Directory containing a tiny-en model");
        eprintln!("              (encoder_model.ort, decoder_model_merged.ort, tokenizer.bin).");
        eprintln!("              See the User Guide for how to download models, or run the");
        eprintln!("              `download_model` example to fetch tiny-en automatically.");
        eprintln!("  AUDIO_FILE  Any audio file (WAV, MP3, AAC, FLAC, OGG, M4A).");
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  cargo run --example transcribe_file -p moonshine-rs -- ./models/tiny-en ./speech.wav"
        );
        std::process::exit(2);
    }

    let model_dir = PathBuf::from(&args[1]);
    let audio_file = PathBuf::from(&args[2]);

    println!("Moonshine version: {}", moonshine_rs::get_version());
    println!("Loading model from: {}", model_dir.display());
    println!("Loading audio from: {}", audio_file.display());

    if !model_dir.exists() {
        eprintln!(
            "Model directory does not exist: {}\nPlease download tiny-en or specify path.",
            model_dir.display()
        );
        std::process::exit(1);
    }

    if !audio_file.exists() {
        eprintln!("Audio file does not exist: {}", audio_file.display());
        std::process::exit(1);
    }

    // Decode any audio format (WAV, MP3, AAC, FLAC, OGG, etc.) and resample to 16kHz mono PCM
    let pcm_data = moonshine_rs::audio::load_audio_for_transcription(&audio_file)?;
    let sample_rate = 16000;

    println!(
        "Loaded and normalized {} audio samples ({:.2}s at 16kHz)",
        pcm_data.len(),
        pcm_data.len() as f32 / sample_rate as f32
    );

    let options = TranscriberOptions::new();
    let transcriber = Transcriber::from_files(&model_dir, ModelArch::Tiny, Some(&options))?;

    println!(
        "Transcriber loaded successfully (handle: {})",
        transcriber.handle()
    );

    let start_time = std::time::Instant::now();
    let transcript = transcriber.transcribe(&pcm_data, sample_rate)?;
    let duration = start_time.elapsed();

    println!("\n--- TRANSCRIPT (took {:?}) ---", duration);
    for (i, line) in transcript.lines.iter().enumerate() {
        println!(
            "Line {}: [{:.2}s - {:.2}s] {}",
            i + 1,
            line.start_time,
            line.start_time + line.duration,
            line.text
        );
    }
    println!("----------------------------");

    Ok(())
}
