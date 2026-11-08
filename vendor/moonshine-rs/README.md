# moonshine-rs

Idiomatic Rust wrapper for [Moonshine Voice](https://github.com/moonshine-ai/moonshine) — fast, on-device speech-to-text powered by ONNX Runtime.

[![crates.io](https://img.shields.io/crates/v/moonshine-rs.svg)](https://crates.io/crates/moonshine-rs)
[![docs.rs](https://docs.rs/moonshine-rs/badge.svg)](https://docs.rs/moonshine-rs)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

`moonshine-rs` provides high-level, memory-safe Rust bindings to the official `libmoonshine` C API. It statically embeds ONNX Runtime and includes multi-format audio decoding (MP3, WAV, AAC, FLAC, OGG) with automatic 16kHz resampling.

## Features

- **On-Device STT**: Transcribe audio locally using quantized Moonshine models (`tiny-en`, `base-en`, etc.).
- **Real-Time Streaming**: Low-latency incremental speech recognition with `TranscriberStream` & `OwnedTranscriberStream`.
- **Domain Customization**: Dynamically bias recognition towards specialized vocabularies, technical jargon, and contact/product names without model retraining (see [Moonshine Domain Customization Guide](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/domain-customization.md)).
- **Multi-Format Audio**: Direct decoding and 16kHz resampling for MP3, WAV, AAC, FLAC, OGG, and M4A audio files via `moonshine_rs::audio`.
- **Text-to-Speech (TTS)**: On-device voice synthesis (Kokoro, Piper, ZipVoice) with one-shot and streaming chunked synthesis.
- **Zero Runtime `.dylib` Dependencies**: Statically links `libmoonshine` and ONNX Runtime.
- **Safe API**: Typed errors, automatic resource management (`Drop`), and thread-safe transcriber handles.

## Installation

Add `moonshine-rs` to your `Cargo.toml`:

```toml
[dependencies]
moonshine-rs = "0.1"
```

### Building

By default, `moonshine-rs` downloads official prebuilt `libmoonshine` binaries
from [moonshine-ai/moonshine GitHub Releases](https://github.com/moonshine-ai/moonshine/releases)
during `cargo build` — **no C++ toolchain, CMake, or source checkout required**:

```bash
cargo add moonshine-rs
cargo build
```

Platform notes for the prebuilt path:

- **macOS (arm64)**: `libmoonshine.a` with ONNX Runtime statically merged in — the resulting binary is fully self-contained.
- **Linux / Windows**: dynamically linked; the `.so` / `onnxruntime.dll` ships alongside your binary.

To build from source instead (fully static Linux/Windows binaries, or a custom
C++ tree), set `MOONSHINE_DIR` to a local checkout — this requires CMake ≥ 3.22
and a C++20 compiler:

```bash
git clone https://github.com/moonshine-ai/moonshine.git
export MOONSHINE_DIR=/path/to/moonshine
cargo build
```

See the [User Guide](https://github.com/ghchinoy/moonshine-rs/blob/main/docs/user-guide.md)
for the full build matrix and troubleshooting.

## Quick Start Example

```rust
use std::path::Path;
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions};
use moonshine_rs::audio::load_audio_for_transcription;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Decode and resample any audio file (MP3, WAV, AAC, FLAC, etc.) to 16kHz mono PCM
    let pcm_data = load_audio_for_transcription("audio.mp3")?;

    // 2. Load transcriber with model files
    let model_dir = Path::new("path/to/tiny-en/quantized/tiny-en");
    let options = TranscriberOptions::new();
    let transcriber = Transcriber::from_files(model_dir, ModelArch::Tiny, Some(&options))?;

    // 3. Transcribe audio
    let transcript = transcriber.transcribe(&pcm_data, 16000)?;

    // 4. Print transcript
    for line in &transcript.lines {
        println!("[{:.2}s - {:.2}s] {}", line.start_time, line.start_time + line.duration, line.text);
    }

    Ok(())
}
```

### Domain Customization & Keyterm Biasing

Bias streaming recognition towards specialized terms and proper nouns at runtime:

```rust
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions};

// 1. Configure initial keyterms and boost
let options = TranscriberOptions::new()
    .with_keyterms("Kubernetes,Ceph,etcd")
    .with_keyterm_boost(2.5);

let transcriber = Transcriber::from_files(
    "./models/tiny-streaming",
    ModelArch::TinyStreaming,
    Some(&options),
)?;

let mut stream = transcriber.create_stream()?;

// 2. Or switch terms mid-stream without reloading models:
stream.set_keyterms("Rust,Tokio,Tauri")?;
```

### Text-to-Speech (TTS) Synthesis

Synthesize text to audio on-device using Kokoro or Piper:

```rust
use moonshine_rs::{TtsOptions, TtsSynthesizer};

let options = TtsOptions::new().with_voice("kokoro_af_heart");
let synth = TtsSynthesizer::from_files("en", "./models/tts/kokoro", Some(&options))?;

// One-shot synthesis
let audio = synth.synthesize("Hello from Moonshine Voice!", None)?;
println!("Generated {} audio samples at {} Hz", audio.pcm.len(), audio.sample_rate);
```

## Moonshine Models & Assets

- **Hugging Face Model Hub**: [https://huggingface.co/UsefulSensors](https://huggingface.co/UsefulSensors)
- **Model CDN & Catalog Manifests**: [download.moonshine.ai](https://download.moonshine.ai)

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/ghchinoy/moonshine-rs/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](https://github.com/ghchinoy/moonshine-rs/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Documentation

For an in-depth developer guide, see the [User Guide](https://github.com/ghchinoy/moonshine-rs/blob/main/docs/user-guide.md).
