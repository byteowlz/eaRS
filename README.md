![ears_banner](banner.png)

# eaRS

`eaRS` is a Rust-based streaming speech-to-text stack built on Kyutai's models. The tool is now delivered as a single CLI with two responsibilities:

- **Server management**: `ears server start|stop` launches and controls the inference backend.
- **Client capture**: Running `ears` without subcommands streams microphone audio to the server and prints live transcripts.

## Installation

```bash
cargo build --release
```

All binaries are emitted into `./target/release/`.

## Quick start

```bash
# 1. Start the transcription server (runs in the background)
./target/release/ears server start

# 2. Stream your microphone to the server and print live text
./target/release/ears
```

Press `Ctrl+C` in the client to stop streaming. When you are done with the backend:

```bash
./target/release/ears server stop
```

## Server commands

```
./target/release/ears server start \
    [--bind 0.0.0.0:8765] \
    [--hf-repo kyutai/stt-1b-en_fr-candle] \
    [--cpu] \
    [--timestamps] \
    [--vad] \
    [--whisper]   # requires --features whisper
```

- `--bind`: Override the default bind address (`0.0.0.0:<port-from-config>`).
- `--hf-repo`: Choose a different Kyutai Speech repo hosted on Hugging Face.
- `--cpu`: Force CPU execution (otherwise CUDA/Metal is used when available).
- `--timestamps`: Include word timestamps in the server stream.
- `--vad`: Enable voice-activity detection for automatic sentence segmentation.
- `--whisper`: Force-enable Whisper post-processing (only when compiled with the `whisper` feature).

The server writes a PID file to `$XDG_STATE_HOME/ears/server.pid` (or `~/.local/state/ears/server.pid`) so subsequent `start` commands will refuse to launch if an instance is already running. `ears server stop` sends a SIGTERM to the stored PID and removes the PID file; stale files are cleaned up automatically.

## Client options

```
./target/release/ears [--device 1] [--server ws://host:port/] [--timestamps] [--list-devices]
```

- `--list-devices`: Print available input devices and exit.
- `--device`: Select a specific capture device by index.
- `--server`: Point the client at a remote server (`ws://127.0.0.1:<config-port>/` by default).
- `--timestamps`: Print the final transcript with per-word timing instead of live text.

The client streams raw 24 kHz mono PCM to the server and displays each live word as it appears. When the backend signals completion, the final transcript (and optional timestamps) is printed.

## Configuration

Runtime configuration lives at:

```
$XDG_CONFIG_HOME/ears/config.toml
# or ~/.config/ears/config.toml
```

Key sections:

- `[storage]`: Override model cache directories and reference audio location.
- `[whisper]`: Configure optional Whisper enhancement defaults (model, quantization, languages, sentence detection thresholds).
- `[server]`: Default WebSocket port used by `ears server start` and the capture client.

If the file does not exist, it is created on first run together with the reference audio bundle.

## WebSocket protocol

The server emits JSON events:

- `{"type":"word","word":"hello","start_time":1.23,"end_time":null}` – live word updates.
- `{"type":"final","text":"…","words":[…]}` – final transcript with timestamp list.
- `{"type":"whisper_processing"|"whisper_complete",…}` – optional Whisper status messages when Whisper is enabled.

Clients may send `{"type":"stop"}` to end the current session (the capture client does this automatically when interrupted).

## Troubleshooting

- **`ears server start` reports "already running"** – Use `ears server stop` to terminate the existing instance. If the PID no longer exists, `stop` will clean up the stale PID file.
- **Client prints "failed to connect"** – Ensure the server is running and reachable at the URL passed via `--server` (check the configured port).
- **High latency** – Run the server on the same machine as the client or enable GPU acceleration (`--features cuda` or `--features metal`).

Happy transcribing!
