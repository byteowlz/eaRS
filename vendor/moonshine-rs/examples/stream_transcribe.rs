//! Real-time incremental streaming transcription example.
//!
//! Demonstrates feeding PCM audio chunks into a [`TranscriberStream`] in real time
//! and printing interim line updates as they arrive, using streaming flags
//! (`is_updated`, `is_new`, `is_complete`) to minimize UI updates.
//!
//! Run:
//!
//! ```bash
//! cargo run --example stream_transcribe -p moonshine-rs -- ./models/tiny-streaming ./speech.wav
//! ```

use std::env;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use moonshine_rs::audio::load_audio_for_transcription;
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <STREAMING_MODEL_DIR> <AUDIO_FILE>", args[0]);
        eprintln!();
        eprintln!(
            "  STREAMING_MODEL_DIR  Directory containing a streaming model (e.g. tiny-streaming)."
        );
        eprintln!("  AUDIO_FILE           Audio file to simulate streaming input from.");
        std::process::exit(2);
    }

    let model_dir = PathBuf::from(&args[1]);
    let audio_file = PathBuf::from(&args[2]);

    println!("Loading audio from: {}", audio_file.display());
    let pcm = load_audio_for_transcription(&audio_file)?;

    println!(
        "Loading streaming transcriber from: {}",
        model_dir.display()
    );
    let options = TranscriberOptions::new();
    let transcriber =
        Transcriber::from_files(&model_dir, ModelArch::TinyStreaming, Some(&options))?;

    println!("Creating streaming session...");
    let mut stream = transcriber.create_stream()?;

    // Simulate streaming PCM in 100ms chunks (1600 samples at 16kHz)
    let chunk_size = 1600;
    let sample_rate = 16_000;

    println!("--- STREAMING TRANSCRIPTION START ---");

    for (chunk_idx, chunk) in pcm.chunks(chunk_size).enumerate() {
        stream.add_audio(chunk, sample_rate)?;

        // Poll every ~200ms (every 2 chunks of 100ms)
        if chunk_idx % 2 == 0 {
            let transcript = stream.poll(false)?;
            for line in &transcript.lines {
                if line.is_updated || line.is_new {
                    let status = if line.is_complete { "FINAL" } else { "PARTIAL" };
                    println!(
                        "[{status}] line {}: [{:.2}s - {:.2}s] {}",
                        line.id,
                        line.start_time,
                        line.start_time + line.duration,
                        line.text
                    );
                }
            }
        }

        // Simulate real-time delay
        thread::sleep(Duration::from_millis(20));
    }

    println!("--- FINALIZING STREAM ---");
    let final_transcript = stream.finalize()?;
    for line in &final_transcript.lines {
        println!(
            "[FINAL] line {}: [{:.2}s - {:.2}s] {}",
            line.id,
            line.start_time,
            line.start_time + line.duration,
            line.text
        );
    }

    Ok(())
}
