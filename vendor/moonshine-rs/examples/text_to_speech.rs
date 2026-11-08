//! Text-to-Speech (TTS) synthesis and streaming speech generation example.
//!
//! Demonstrates:
//! 1. Discovering available voices via [`get_tts_voices`].
//! 2. Splitting text into natural utterance units with [`split_utterances`].
//! 3. Synthesizing one-shot text to PCM audio with [`TtsSynthesizer::synthesize`].
//! 4. Streaming chunked synthesis from an LLM-style token stream with [`TtsSynthesizer::push_text`]
//!    and [`TtsSynthesizer::next_chunk`].
//!
//! Run:
//!
//! ```bash
//! # List voices and demonstrate utterance splitting:
//! cargo run --example text_to_speech -p moonshine-rs
//!
//! # Or synthesize audio with a local TTS model directory:
//! cargo run --example text_to_speech -p moonshine-rs -- /path/to/tts_model_dir "kokoro_af_heart" "Hello from Moonshine Voice!"
//! ```

use std::env;
use std::path::PathBuf;

use moonshine_rs::{
    get_tts_dependencies, get_tts_voices, split_utterances, TtsOptions, TtsStreamStatus,
    TtsSynthesizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Moonshine Voice Text-To-Speech (TTS) Demo ===\n");

    // 1. Discover registered TTS voices
    println!("--- 1. Registered Voices (English) ---");
    let voices_json = get_tts_voices(Some("en"), None)?;
    println!("{voices_json}\n");

    // 2. Discover TTS asset dependencies
    println!("--- 2. TTS Dependencies (English) ---");
    let deps_json = get_tts_dependencies(Some("en"), None)?;
    let parsed: serde_json::Value = serde_json::from_str(&deps_json)?;
    if let Some(groups) = parsed["groups"].as_array() {
        for group in groups {
            let base_url = group["base_url"].as_str().unwrap_or("");
            println!("Group base URL: {base_url}");
            if let Some(files) = group["files"].as_array() {
                for file in files {
                    let name = file["name"].as_str().unwrap_or("");
                    let size = file["size"].as_u64().unwrap_or(0);
                    println!("  - {name} ({size} bytes)");
                }
            }
        }
    }
    println!();

    // 3. Demonstrate sentence and utterance splitting
    println!("--- 3. Utterance Splitting ---");
    let sample_passage = "Welcome to Moonshine Voice! This is an on-device text-to-speech engine. Dr. Smith said: 'Latency is under 100 milliseconds.' Would you like to try it?";
    println!("Original passage:\n  \"{sample_passage}\"\n");

    let units = split_utterances(Some("en"), sample_passage, None)?;
    println!("Split into {} utterance units:", units.len());
    for (i, unit) in units.iter().enumerate() {
        println!("  [{}] \"{}\"", i + 1, unit);
    }
    println!();

    // 4. Model Synthesis (if model directory is supplied as argument)
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("To perform actual audio synthesis, first download a TTS model:\n");
        println!("  cargo run --example download_tts_model -p moonshine-rs -- ./models/tts/kokoro");
        println!("  # or: just download-tts\n");
        println!("Then run this example with the downloaded directory:\n");
        println!(
            "  cargo run --example text_to_speech -p moonshine-rs -- ./models/tts/kokoro kokoro_af_heart \"Hello from Moonshine Voice!\"\n"
        );
        return Ok(());
    }

    let model_dir = PathBuf::from(&args[1]);
    let voice = if args.len() > 2 {
        &args[2]
    } else {
        "kokoro_af_heart"
    };
    let text = if args.len() > 3 {
        &args[3]
    } else {
        "Hello from Moonshine Voice! On-device speech synthesis in Rust."
    };

    println!("--- 4. Synthesizing Audio ---");
    println!("Model dir: {}", model_dir.display());
    println!("Voice:     {voice}");
    println!("Text:      \"{text}\"");

    let options = TtsOptions::new().with_voice(voice).with_speed(1.0);

    // Load synthesizer with model directory
    let synth = TtsSynthesizer::from_files("en", &model_dir, Some(&options))?;

    // A. One-shot synthesis
    println!("\nA. Performing one-shot synthesis...");
    let audio = synth.synthesize(text, None)?;
    println!(
        "Synthesized {} samples at {} Hz ({:.2}s of audio)",
        audio.pcm.len(),
        audio.sample_rate,
        audio.duration_seconds()
    );

    // B. Streaming chunked synthesis (simulating tokens from an LLM)
    println!("\nB. Performing streaming chunked synthesis...");
    let tokens = [
        "Streaming ",
        "synthesis ",
        "speaks ",
        "as ",
        "text ",
        "arrives. ",
        "No ",
        "waiting ",
        "for ",
        "the ",
        "entire ",
        "reply!",
    ];

    for token in tokens {
        print!("{token}");
        synth.push_text(token)?;

        // Poll for any newly ready audio chunks
        while let TtsStreamStatus::Chunk(chunk) = synth.next_chunk()? {
            println!(
                "\n  [Audio Chunk] {} samples (utterance {}, final={})",
                chunk.pcm.len(),
                chunk.utterance_id,
                chunk.is_final
            );
        }
    }
    println!();

    // Signal end of input and drain remaining audio
    synth.end_input()?;
    while let TtsStreamStatus::Chunk(chunk) = synth.next_chunk()? {
        println!(
            "  [Final Audio Chunk] {} samples (utterance {}, final={})",
            chunk.pcm.len(),
            chunk.utterance_id,
            chunk.is_final
        );
    }

    println!("\nStreaming synthesis complete!");
    Ok(())
}
