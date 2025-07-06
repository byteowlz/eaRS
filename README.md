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
- WebSocket streaming for real-time applications
- Automatic termination after voice activity timeout
- WebSocket session restart capability

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

# Prime the model with reference audio in another language
./target/release/ears --live -l ger

# Automatically terminate after 5 seconds of no voice activity
./target/release/ears --live --vad-timeout 5.0
```

### WebSocket Streaming

```bash
# Start WebSocket server on port 8080
./target/release/ears --live --ws 8080

# With timestamps and VAD
./target/release/ears --live --ws 8080 --timestamps --vad

# With automatic timeout after 3 seconds of silence
./target/release/ears --live --ws 8080 --vad-timeout 3.0
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
- `-l, --lang <LANG>` - Prime language using audio snippet (esp, ger, jap)
- `--ws <PORT>` - Start WebSocket server on specified port
- `--vad-timeout <SECONDS>` - Automatically terminate after no voice activity

## WebSocket API

When using the `--ws` option, eaRS starts a WebSocket server that streams real-time transcription results.

### Connection

Connect to `ws://localhost:<port>/` where `<port>` is specified via the `--ws` option.

### Message Types

#### Word Messages
Sent for each transcribed word as it's recognized:
```json
{
  "type": "word",
  "word": "hello",
  "start_time": 1.23,
  "end_time": 1.45
}
```

#### Pause Messages
Sent when voice activity detection detects a pause (requires `--vad` flag):
```json
{
  "type": "pause",
  "timestamp": 1234567890.123
}
```

#### Final Messages
Sent at the end of each transcription session:
```json
{
  "type": "final",
  "text": "complete transcribed text",
  "words": [
    {"word": "hello", "start_time": 1.23, "end_time": 1.45},
    {"word": "world", "start_time": 1.46, "end_time": null}
  ]
}
```

### Client Commands

#### Restart Transcription
Send from client to restart transcription after timeout or final message:
```json
{
  "type": "restart"
}
```

#### Pause / Resume Transcription
Temporarily halt or continue audio processing without closing the WebSocket:
```json
{ "type": "pause" }
```
Resume with:
```json
{ "type": "resume" }
```

### Usage Pattern

1. Connect to WebSocket endpoint
2. Receive real-time word messages during transcription
3. Receive final message when session ends (timeout or silence)
4. Send restart command to begin new transcription session
5. Repeat as needed

## Model

Default model: `kyutai/stt-1b-en_fr-candle`

Supports English and French transcription with 24kHz audio processing. German also seems to work quite nicely.

## Requirements

- Rust 1.70+
- Audio input device for live transcription
- GPU support (CUDA/Metal) optional, falls back to CPU

