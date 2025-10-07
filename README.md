![ears_banner](banner.png)

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
- Optional termination after voice activity timeout
- WebSocket session restart capability
- Dictation mode with keyboard input simulation
- Hotkey daemon to toggle dictation and switch languages

## Installation

```bash
cargo build --release
```

## Usage

### Unified Mode (one binary)

Start everything in one go: WebSocket server, hotkeys, and dictation typing.

```
./target/release/ears --live --ws 8765 --hotkeys --dictation
```

- Ctrl+Shift+V toggles pause/resume
- Ctrl+Shift+L cycles language (en, de, fr, es, ja)
- --tray shows a system tray icon with the same controls

Or use the zero-setup binary:

```
./target/release/ears-auto
```

This auto-starts the server (WS using config port), dictation, hotkeys, and tray. Each transcribed word is typed immediately to the focused text field.

Note: On macOS, you must grant permissions:
- System Settings → Privacy & Security → Accessibility (add your terminal or ears-auto binary)
- System Settings → Privacy & Security → Input Monitoring (add your terminal or ears-auto binary)

Tray icon uses the bundled logo; macOS template mode can be added later if needed for auto light/dark.

### Configurable Hotkeys

Edit `~/.config/ears/config.toml`:

```
[hotkeys]
# Disable internal hotkeys if your WM binds keys
enable_internal = true
# Toggle dictation
toggle = "ctrl+shift+v"
# Cycle language
language_cycle = "ctrl+shift+l"
```

### Hyprland Integration (IPC)

Use the CLI to control eaRS from your WM binds:

```
# hyprland.conf
bind=SUPER,V,exec,ears-ctl toggle
bind=SUPER,L,exec,ears-ctl lang de
```

Commands: `ears-ctl toggle|pause|resume|lang <code>|status`.

ears-ctl connects with a 2s timeout and waits up to 800ms for status responses, so it won't hang indefinitely if the server is unreachable.

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

# Prime the model with reference audio (ISO 639-1)
./target/release/ears --live -l de

# Automatically terminate after 5 seconds of no voice activity
./target/release/ears --live --vad-timeout 5.0
```

### WebSocket Streaming

```
# Start WebSocket server on port 8080 (starts paused)
./target/release/ears --live --ws 8080

# With timestamps and VAD
./target/release/ears --live --ws 8080 --timestamps --vad

# With automatic timeout after 3 seconds of silence
./target/release/ears --live --ws 8080 --vad-timeout 3.0
```

Control it via JSON commands sent over WebSocket:

- `{ "type": "resume" }` to start
- `{ "type": "pause" }` to pause
- `{ "type": "restart" }` to start a new session
- `{ "type": "set_language", "lang": "de" }` to switch language (primes model)
- `{ "type": "get_status" }` to query state

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
- `-l, --lang <LANG>` - Prime language using audio snippet (ISO 639-1: de, ja, es, it)
- `--ws <PORT>` - Start WebSocket server on specified port
- `--vad-timeout <SECONDS>` - Automatically terminate after no voice activity (not used in ears-auto)

## WebSocket API

When using the `--ws` option, eaRS starts a WebSocket server that streams real-time transcription results with a simple tagged schema.

### Connection

Connect to `ws://localhost:<port>/` where `<port>` is specified via the `--ws` option.

### Message Types

- `word`: `{ "type": "word", "word": "...", "start_time": 0.0, "end_time": 0.1 }`
- `pause`: `{ "type": "pause", "timestamp": 123.45 }`
- `final`: `{ "type": "final", "text": "...", "words": [ ... ] }`
- `language_changed`: `{ "type": "language_changed", "lang": "de" }`
- `status`: `{ "type": "status", "paused": true, "vad": false, "timestamps": false, "vad_timeout": 2.0, "lang": "de" }`

### Control Commands

Send JSON to control the server:

- Resume: `{ "type": "resume" }`
- Pause: `{ "type": "pause" }`
- Restart: `{ "type": "restart" }`
- Set Language: `{ "type": "set_language", "lang": "de" }`
- Get Status: `{ "type": "get_status" }`

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

#### Pause/Resume Transcription

Toggle live inference without disconnecting:

```json
{ "type": "pause" }
{ "type": "resume" }
```

### Usage Pattern

1. Connect to WebSocket endpoint
2. Receive real-time word messages during transcription (clients can type each word immediately)
3. Receive final message when session ends (timeout or silence)
4. Send restart command to begin new transcription session
5. Optionally send pause/resume commands to temporarily stop inference
6. Repeat as needed

See [WEBSOCKET.md](WEBSOCKET.md) for full protocol details and examples.

## Daemon (Hotkey Control)

Start the server, then run the daemon to control it via global hotkeys:

```
# Start the WebSocket server
./target/release/ears --live --ws 8765 --vad-timeout 2.0

# In another terminal, run daemon
./target/release/ears-daemon
```

Default hotkeys:

- Ctrl+Shift+V: toggle pause/resume
- Ctrl+Shift+L: cycle language (en, de, fr, es, ja)

Alternatively, use `ears-auto` for a single-binary experience with hotkeys + dictation + WebSocket all in one.

Note: On macOS, grant Accessibility and Input Monitoring permissions to the terminal/binary.

## Dictation Mode

The `ears-dictation` binary provides keyboard input simulation, allowing transcribed speech to be typed directly into any text field. Alternatively, use `ears-auto` for a zero-setup experience that includes dictation, hotkeys, and WebSocket in a single binary.

### Setup

1. Start the eaRS WebSocket server:

```bash
./target/release/ears --live --ws 8765
```

2. Run the dictation client:

```bash
./target/release/ears-dictation
```

3. Focus your target text field and start speaking. Transcribed text will be typed automatically as each word arrives (immediate typing, not batched).

### Options

- `--host <HOST>` - WebSocket server host (default: localhost)
- `--port <PORT>` - WebSocket server port (default: 8765)
- `--only-final` - Only type final transcriptions, skip word updates

### macOS Permissions

On macOS, you'll need to grant accessibility and input monitoring permissions:

1. System Settings → Privacy & Security → Accessibility
2. System Settings → Privacy & Security → Input Monitoring
3. Add the `ears-dictation` binary (or your terminal app) to both allowed apps

### Example

```bash
# Start transcription server
./target/release/ears --live --ws 8765

# In another terminal, start dictation client
./target/release/ears-dictation
```

## Model

Default model: `kyutai/stt-1b-en_fr-candle`

Supports English and French transcription with 24kHz audio processing. German also seems to work quite nicely.

Whisper integration is optional and behind the `whisper` feature flag. It is experimental and disabled by default.

## Troubleshooting

### ears-ctl hangs or times out
- Ensure `ears-auto` or `ears --live --ws <port>` is running and printed "WS on <port>".
- Check the config port in `~/.config/ears/config.toml` matches the server.
- ears-ctl uses 2s connection timeout and 800ms response timeout; if it fails, the server may not be reachable.

### Hotkeys don't work on macOS
- Grant Accessibility and Input Monitoring permissions to your terminal app or the `ears-auto` / `ears-daemon` binary in System Settings → Privacy & Security.
- Restart the app after granting permissions.

### Dictation doesn't type
- Verify permissions (Accessibility + Input Monitoring) are granted.
- Ensure the text field is focused and accepts keyboard input.
- Check that ears-auto or ears-dictation is receiving Word messages (stdout shows transcription).

### Process exits unexpectedly
- If using `ears --live --vad-timeout <sec>`, the session ends after silence. Use `ears-auto` (no timeout) or omit `--vad-timeout`.

## Requirements

- Rust 1.70+
- Audio input device for live transcription
- GPU support (CUDA/Metal) optional, falls back to CPU
- macOS: Accessibility and Input Monitoring permissions for hotkeys and dictation
