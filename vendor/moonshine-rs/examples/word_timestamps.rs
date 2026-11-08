//! Print word-level timestamps and confidence for a transcription.
//!
//! Each [`TranscriptLine`] carries a `words: Vec<TranscriptWord>` with per-word
//! `start` / `end` (seconds) and a `confidence` score. Word timestamps require
//! a model that ships the attention decoder; fetch it with the
//! `word_timestamps` dependency option (see the `download_model` example and
//! [`moonshine_rs::SttDependenciesOptions::with_word_timestamps`]).
//!
//! Run:
//!
//! ```bash
//! cargo run --example word_timestamps -p moonshine-rs -- ./models/tiny-en ./speech.wav
//! ```
//!
//! [`TranscriptLine`]: moonshine_rs::TranscriptLine
//! [`TranscriptWord`]: moonshine_rs::TranscriptWord

use std::env;
use std::path::PathBuf;

use moonshine_rs::audio::load_audio_for_transcription;
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <MODEL_DIR> <AUDIO_FILE>", args[0]);
        std::process::exit(2);
    }
    let model_dir = PathBuf::from(&args[1]);
    let audio_file = PathBuf::from(&args[2]);

    let pcm = load_audio_for_transcription(&audio_file)?;

    let options = TranscriberOptions::new();
    let transcriber = Transcriber::from_files(&model_dir, ModelArch::Tiny, Some(&options))?;
    let transcript = transcriber.transcribe(&pcm, 16_000)?;

    let mut any_words = false;
    for line in &transcript.lines {
        println!("[{:.2}s] {}", line.start_time, line.text);
        for word in &line.words {
            any_words = true;
            println!(
                "    {:>7.2}s - {:<7.2}s  conf={:.2}  {}",
                word.start, word.end, word.confidence, word.text
            );
        }
    }

    if !any_words {
        eprintln!(
            "\nNote: no per-word timestamps were returned. This model likely lacks the\n\
             attention decoder. Re-download with word timestamps enabled, e.g.:\n\
             \n\
             \tuse moonshine_rs::SttDependenciesOptions;\n\
             \tlet opts = SttDependenciesOptions::new()\n\
             \t    .with_arch(ModelArch::Tiny)\n\
             \t    .with_word_timestamps(true);\n"
        );
    }

    Ok(())
}
