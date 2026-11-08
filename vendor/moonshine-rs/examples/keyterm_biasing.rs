//! Domain customization and keyterm biasing example.
//!
//! Demonstrates biasing streaming transcription towards uncommon words,
//! technical jargon, and product/contact names using `TranscriberOptions::with_keyterms`,
//! `TranscriberOptions::with_context`, and runtime `set_keyterms` mid-stream.
//!
//! Run:
//!
//! ```bash
//! cargo run --example keyterm_biasing -p moonshine-rs -- ./models/tiny-streaming ./speech.wav "Kubernetes,Ceph,etcd"
//! ```

use std::env;
use std::path::PathBuf;

use moonshine_rs::audio::load_audio_for_transcription;
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <STREAMING_MODEL_DIR> <AUDIO_FILE> [KEYTERMS]",
            args[0]
        );
        eprintln!();
        eprintln!(
            "  STREAMING_MODEL_DIR  Directory containing a streaming model (e.g. tiny-streaming)."
        );
        eprintln!("  AUDIO_FILE           Audio file to transcribe.");
        eprintln!(
            "  KEYTERMS             Optional comma-separated key terms (default: 'Kubernetes,Ceph,etcd')."
        );
        std::process::exit(2);
    }

    let model_dir = PathBuf::from(&args[1]);
    let audio_file = PathBuf::from(&args[2]);
    let keyterms = if args.len() > 3 {
        &args[3]
    } else {
        "Kubernetes,Ceph,etcd"
    };

    println!("Loading audio from: {}", audio_file.display());
    let pcm = load_audio_for_transcription(&audio_file)?;

    println!(
        "Loading streaming transcriber from: {}",
        model_dir.display()
    );
    println!("Configuring initial key terms: {keyterms}");

    let options = TranscriberOptions::new()
        .with_keyterms(keyterms)
        .with_keyterm_boost(2.5);

    let transcriber =
        Transcriber::from_files(&model_dir, ModelArch::TinyStreaming, Some(&options))?;

    println!("Creating streaming session...");
    let mut stream = transcriber.create_stream()?;

    // Simulate streaming in 100ms chunks
    let chunk_size = 1600;
    let sample_rate = 16_000;

    println!("--- STREAMING TRANSCRIPTION (with keyterm biasing) ---");

    for (chunk_idx, chunk) in pcm.chunks(chunk_size).enumerate() {
        stream.add_audio(chunk, sample_rate)?;

        // Mid-stream keyterm switch demonstration halfway through
        if chunk_idx == pcm.len() / (chunk_size * 2) {
            println!("\n>>> Dynamically switching key terms mid-stream to: 'Rust,Tokio,Tauri'");
            stream.set_keyterms("Rust,Tokio,Tauri")?;
        }

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
