//! Download a Moonshine STT model using the native dependency manifest API.
//!
//! This resolves the official CDN URLs for a language + architecture via
//! [`moonshine_rs::get_stt_dependencies`], then downloads each file into a
//! target directory laid out exactly how [`Transcriber::from_files`] expects.
//!
//! Run:
//!
//! ```bash
//! cargo run --example download_model -p moonshine-rs -- ./models/tiny-en
//! # then:
//! cargo run --example transcribe_file -p moonshine-rs -- ./models/tiny-en ./speech.wav
//! ```
//!
//! [`Transcriber::from_files`]: moonshine_rs::Transcriber::from_files

use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use moonshine_rs::{get_stt_dependencies_with_options, ModelArch, SttDependenciesOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <OUTPUT_DIR> [LANGUAGE] [ARCH]", args[0]);
        eprintln!();
        eprintln!("  OUTPUT_DIR  Directory to download the model files into.");
        eprintln!("  LANGUAGE    STT language code (default: en).");
        eprintln!("  ARCH        Model architecture: tiny, base, tiny-streaming, small-streaming,");
        eprintln!("              medium-streaming, base-streaming (default: tiny).");
        eprintln!("              NOTE: base-streaming is not yet published in the \"en\" catalog upstream");
        eprintln!("              (see https://github.com/moonshine-ai/moonshine/issues/214).");
        std::process::exit(2);
    }

    let out_dir = PathBuf::from(&args[1]);
    let language = args.get(2).map(String::as_str).unwrap_or("en");
    let arch = match args.get(3).map(String::as_str) {
        None => ModelArch::Tiny,
        Some("tiny") => ModelArch::Tiny,
        Some("base") => ModelArch::Base,
        Some("tiny-streaming") | Some("tinystreaming") => ModelArch::TinyStreaming,
        Some("small-streaming") | Some("smallstreaming") => ModelArch::SmallStreaming,
        Some("medium-streaming") | Some("mediumstreaming") => ModelArch::MediumStreaming,
        Some("base-streaming") | Some("basestreaming") => ModelArch::BaseStreaming,
        Some(other) => {
            eprintln!(
                "Error: Unknown ARCH \"{other}\". Valid values: tiny, base, tiny-streaming, small-streaming, medium-streaming, base-streaming."
            );
            std::process::exit(2);
        }
    };

    fs::create_dir_all(&out_dir)?;

    // 1. Resolve the download manifest (URLs, sizes, checksums) in native Rust.
    let manifest_json = get_stt_dependencies_with_options(
        language,
        &SttDependenciesOptions::new().with_arch(arch),
    )?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json)?;

    let groups = manifest["groups"]
        .as_array()
        .ok_or("manifest missing `groups` array")?;

    // 2. Download each file listed in the manifest.
    for group in groups {
        let files = group["files"].as_array().ok_or("group missing `files`")?;
        for file in files {
            let name = file["name"].as_str().ok_or("file missing `name`")?;
            let url = file["url"].as_str().ok_or("file missing `url`")?;
            let expected_size = file["size"].as_u64().unwrap_or(0);

            let dest = out_dir.join(name);
            if dest.exists() {
                let have = fs::metadata(&dest)?.len();
                if expected_size == 0 || have == expected_size {
                    println!("skip  {name} (already present, {have} bytes)");
                    continue;
                }
            }

            println!("fetch {name} <- {url}");
            let resp = ureq::get(url).call()?;
            let mut bytes = Vec::new();
            resp.into_reader().read_to_end(&mut bytes)?;
            fs::write(&dest, &bytes)?;
            println!("      wrote {} bytes to {}", bytes.len(), dest.display());
        }
    }

    println!("\nModel ready at: {}", out_dir.display());
    println!(
        "Transcribe with:\n  cargo run --example transcribe_file -p moonshine-rs -- {} <AUDIO_FILE>",
        out_dir.display()
    );
    Ok(())
}
