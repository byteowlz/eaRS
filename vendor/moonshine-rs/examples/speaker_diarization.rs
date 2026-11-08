//! Transcribe with speaker identification (diarization) enabled.
//!
//! Passing `identify_speakers = true` via [`TranscriberOptions::with_identify_speakers`]
//! asks Moonshine to attribute spans of each line to distinct speakers.
//!
//! Note: Speaker identification requires two diarization models (`segmentation.ort`
//! and `embedding.ort`, ~8.2 MB total). `moonshine-rs` automatically fetches and
//! caches these models on first use into your local OS cache directory. The results
//! appear as [`SpeakerSpan`] entries on every [`TranscriptLine`], carrying a `speaker_index`,
//! timing, and character ranges.
//!
//! Run:
//!
//! ```bash
//! cargo run --example speaker_diarization -p moonshine-rs -- ./models/tiny-en ./two_speakers.wav
//! ```
//!
//! [`TranscriberOptions::with_identify_speakers`]: moonshine_rs::TranscriberOptions::with_identify_speakers
//! [`SpeakerSpan`]: moonshine_rs::SpeakerSpan
//! [`TranscriptLine`]: moonshine_rs::TranscriptLine

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

    // Enable speaker identification.
    let options = TranscriberOptions::new().with_identify_speakers(true);
    let transcriber = Transcriber::from_files(&model_dir, ModelArch::Tiny, Some(&options))?;
    let transcript = transcriber.transcribe(&pcm, 16_000)?;

    let mut any_spans = false;
    for line in &transcript.lines {
        if line.speaker_spans.is_empty() {
            println!("[speaker ?] {}", line.text);
            continue;
        }
        for span in &line.speaker_spans {
            any_spans = true;
            let start = span.start_char as usize;
            let end = (span.end_char as usize).min(line.text.len());
            let snippet = line.text.get(start..end).unwrap_or(&line.text);
            println!(
                "[speaker {}] {:.2}s (+{:.2}s)  {}",
                span.speaker_index, span.start_time, span.duration, snippet
            );
        }
    }

    if !any_spans {
        eprintln!(
            "\nNote: no speaker spans were returned. Speaker identification depends on\n\
             model support and multi-speaker audio. Try audio with distinct speakers."
        );
    }

    Ok(())
}
