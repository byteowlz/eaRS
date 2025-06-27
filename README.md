# eaRS

A Rust-based speech-to-text transcription tool using Kyutai's STT models.

## Features

- Real-time transcription from microphone
- File-based audio transcription
- Word-level timestamps
- Voice activity detection (VAD)
- Audio device selection
- Audio recording to WAV files
- Multiple audio format support (WAV, MP3, OGG)

## Installation

```bash
cargo build --release
```

## Usage

### Live Transcription

```bash
# Use default microphone
./target/release/ears --live

# Select specific audio device
./target/release/ears --live --device 1

# With timestamps and VAD
./target/release/ears --live --timestamps --vad

# Save audio while transcribing
./target/release/ears --live --save-audio recording.wav
```

### File Transcription

```bash
# Transcribe audio file
./target/release/ears audio.wav

# With timestamps
./target/release/ears audio.mp3 --timestamps
```

### Device Management

```bash
# List available audio devices
./target/release/ears --list-devices
```

## Options

- `--live` - Use live microphone input
- `--device <INDEX>` - Select audio input device by index
- `--timestamps` - Display word-level timestamps
- `--vad` - Show voice activity detection
- `--save-audio <FILE>` - Save audio to WAV file
- `--cpu` - Force CPU inference (disable GPU)
- `--hf-repo <REPO>` - Specify Hugging Face model repository
- `--list-devices` - List available audio devices

## Model

Default model: `kyutai/stt-1b-en_fr-candle`

Supports English and French transcription with 24kHz audio processing. German also seems to work quite nicely.

## Requirements

- Rust 1.70+
- Audio input device for live transcription
- GPU support (CUDA/Metal) optional, falls back to CPU

