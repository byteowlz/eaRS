//! Download Moonshine Text-To-Speech (TTS) models using the native dependency manifest API.
//!
//! Resolves official CDN URLs for a language + voice via [`moonshine_rs::get_tts_dependencies`],
//! creating all required subdirectories and downloading each file into a target directory
//! laid out exactly how [`TtsSynthesizer::from_files`] expects.
//!
//! Run:
//!
//! ```bash
//! # Download Kokoro default voice (kokoro_af_heart) for English:
//! cargo run --example download_tts_model -p moonshine-rs -- ./models/tts/kokoro
//!
//! # Or specify language and voice:
//! cargo run --example download_tts_model -p moonshine-rs -- ./models/tts/piper en piper_en_US-lessac-medium
//!
//! # Then synthesize speech:
//! cargo run --example text_to_speech -p moonshine-rs -- ./models/tts/kokoro kokoro_af_heart "Hello from Moonshine Voice!"
//! ```
//!
//! [`TtsSynthesizer::from_files`]: moonshine_rs::TtsSynthesizer::from_files

use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use moonshine_rs::{get_tts_dependencies, TtsOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <OUTPUT_DIR> [LANGUAGE] [VOICE]", args[0]);
        eprintln!();
        eprintln!("  OUTPUT_DIR  Directory to download the TTS model files into.");
        eprintln!("  LANGUAGE    TTS language code (default: en).");
        eprintln!("  VOICE       Voice identifier: kokoro_af_heart (default), piper_<stem>, etc.");
        std::process::exit(2);
    }

    let out_dir = PathBuf::from(&args[1]);
    let language = args.get(2).map(String::as_str).unwrap_or("en");
    let voice = args.get(3).map(String::as_str).unwrap_or("kokoro_af_heart");

    println!("=== Downloading Moonshine TTS Model Assets ===");
    println!("Destination: {}", out_dir.display());
    println!("Language:    {language}");
    println!("Voice:       {voice}\n");

    fs::create_dir_all(&out_dir)?;

    // 1. Resolve the download manifest (URLs, sizes, checksums) in native Rust.
    let options = TtsOptions::new().with_voice(voice);
    let manifest_json = get_tts_dependencies(Some(language), Some(&options))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json)?;

    let groups = manifest["groups"]
        .as_array()
        .ok_or("manifest missing `groups` array")?;

    let mut total_downloaded = 0;
    let mut total_skipped = 0;

    // 2. Download each file listed in the manifest.
    for group in groups {
        let files = group["files"].as_array().ok_or("group missing `files`")?;
        for file in files {
            let name = file["name"].as_str().ok_or("file missing `name`")?;
            let url = file["url"].as_str().ok_or("file missing `url`")?;
            let expected_size = file["size"].as_u64().unwrap_or(0);

            let dest = out_dir.join(name);

            // Create subdirectories if the asset filename contains nested paths
            if let Some(parent) = dest.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent)?;
                }
            }

            if dest.exists() {
                let have = fs::metadata(&dest)?.len();
                if expected_size == 0 || have == expected_size {
                    println!("skip  {name} (already present, {have} bytes)");
                    total_skipped += 1;
                    continue;
                }
            }

            println!("fetch {name} <- {url}");
            let resp = ureq::get(url).call()?;
            let mut bytes = Vec::new();
            resp.into_reader().read_to_end(&mut bytes)?;
            fs::write(&dest, &bytes)?;
            println!("      wrote {} bytes to {}", bytes.len(), dest.display());
            total_downloaded += 1;
        }
    }

    println!("\n=== TTS Model Ready ===");
    println!("Downloaded: {total_downloaded} files, Skipped: {total_skipped} files");
    println!("Directory:  {}", out_dir.display());
    println!("\nSynthesize speech with:");
    println!(
        "  cargo run --example text_to_speech -p moonshine-rs -- {} {} \"Hello from Moonshine Voice!\"",
        out_dir.display(),
        voice
    );

    Ok(())
}
