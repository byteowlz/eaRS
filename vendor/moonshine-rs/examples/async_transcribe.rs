//! Transcribe from an async runtime without blocking the executor.
//!
//! [`Transcriber::transcribe`] is synchronous and CPU-bound. Inside a Tokio (or
//! Tauri / Axum) runtime, run it on the blocking pool via
//! [`tokio::task::spawn_blocking`] so the async executor stays responsive. The
//! [`Transcriber`] is `Send + Sync`, so an `Arc<Transcriber>` can be shared
//! across tasks and cloned cheaply.
//!
//! Run:
//!
//! ```bash
//! cargo run --example async_transcribe -p moonshine-rs -- ./models/tiny-en ./speech.wav
//! ```
//!
//! Dependency-free variant: this pattern does not actually require Tokio — you
//! could offload to a plain `std::thread` and receive the result over a channel.
//! Tokio is used here only because it is the most common async runtime; wiring
//! up the `std::thread` version is left as an exercise.
//!
//! [`Transcriber::transcribe`]: moonshine_rs::Transcriber::transcribe
//! [`Transcriber`]: moonshine_rs::Transcriber

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use moonshine_rs::audio::load_audio_for_transcription;
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <MODEL_DIR> <AUDIO_FILE>", args[0]);
        std::process::exit(2);
    }
    let model_dir = PathBuf::from(&args[1]);
    let audio_file = PathBuf::from(&args[2]);

    // Load once; share across tasks for the lifetime of the app.
    let options = TranscriberOptions::new();
    let transcriber = Arc::new(Transcriber::from_files(
        &model_dir,
        ModelArch::Tiny,
        Some(&options),
    )?);

    let pcm = load_audio_for_transcription(&audio_file)?;

    // Offload the CPU-bound inference so the async runtime is not blocked.
    let worker = transcriber.clone();
    let transcript = tokio::task::spawn_blocking(move || worker.transcribe(&pcm, 16_000)).await??;

    println!("{}", transcript.text());
    Ok(())
}
