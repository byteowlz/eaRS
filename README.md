![ears_banner](banner.png)

# eaRS

`eaRS` is a Rust-based streaming speech-to-text stack built on Kyutai's models. The tool is now delivered as a single CLI with two responsibilities:

- **Server management**: `ears server start|stop` launches and controls the inference backend.
- **Client capture**: Running `ears` without subcommands streams microphone audio to the server and prints live transcripts.

## System Dependencies

### Linux

On Linux, eaRS uses the system `sentencepiece` library to avoid protobuf conflicts with ONNX Runtime.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libsentencepiece-dev sentencepiece
```

**Fedora/RHEL:**
```bash
sudo dnf install -y sentencepiece-devel
```

**Arch Linux:**
```bash
sudo pacman -S sentencepiece
```

#### Additional Setup for Dictation on Wayland/Linux

eaRS dictation uses the Linux `uinput` interface for reliable keyboard input on both Wayland and X11.

**Quick Setup (Recommended):**
```bash
# Run the automated setup script
./scripts/setup-dictation-linux.sh
```

The script will:
- Load the uinput kernel module
- Configure uinput to load on boot
- Create udev rule for proper `/dev/uinput` permissions
- Add your user to the input group
- Verify permissions

**Manual Setup:**

If you prefer to set up manually:

```bash
# 1. Load the uinput kernel module
sudo modprobe uinput

# 2. Make it permanent (load on boot)
echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf

# 3. Create udev rule for proper permissions
echo 'KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/99-uinput.conf
sudo udevadm control --reload-rules
sudo udevadm trigger

# 4. Reload uinput module to apply new permissions
sudo rmmod uinput
sudo modprobe uinput

# 5. Add your user to the input group
sudo usermod -a -G input $USER

# 6. Log out and log back in (or reboot)

# 7. Verify your setup
groups | grep input
ls -l /dev/uinput
```

You should see something like:
```
crw-rw---- 1 root input 10, 223 Dec 12 10:00 /dev/uinput
```

**Troubleshooting dictation:**
- **Error: "Failed to open /dev/uinput"** – Check: `ls -l /dev/uinput` (should show `crw-rw---- 1 root input`)
- **Wrong permissions?** – Create udev rule (see manual setup step 3 above)
- **Still not working?** – Verify: `lsmod | grep uinput` and `groups | grep input`
- **After adding to group** – You MUST log out and back in for group changes to take effect

### macOS and Windows

On macOS and Windows, `sentencepiece` is compiled from source and statically linked during the build process. No manual installation required.

### Automated Installation (Recommended)

Use the provided `just` recipes which automatically check and install dependencies:

```bash
# Interactive installation with feature selection
just install-ears

# Or use specific presets:
just install-ears-metal      # macOS with Metal acceleration
just install-ears-cuda       # NVIDIA GPU with CUDA
just install-ears-parakeet   # Enable Parakeet engine
just install-ears-default    # CPU only
```

## Build

```bash
cargo build --release

cargo build --release --features apple # For Apple silicon

cargo build --release --features nvidia # For NVIDIA GPU

cargo build --release --features parakeet             # Enable Parakeet (ONNX) engine (CPU)
cargo build --release --features "parakeet nvidia"    # Parakeet + CUDA (Kyutai uses CUDA too)
cargo build --release --features "parakeet apple"     # Parakeet + CoreML, Kyutai + Metal (Apple Silicon)
cargo build --release --features "parakeet amd"       # Parakeet + ROCm (Kyutai stays CPU)
cargo build --release --features "parakeet directml"  # Parakeet + DirectML (Kyutai stays CPU)
```

All binaries are emitted into `./target/release/`.

## Installation

The recommended way to install is using the `just` recipes (see System Dependencies above).

For manual installation:

```bash
cargo install --path .

cargo install --path . --features apple # For Apple silicon

cargo install --path . --features nvidia # For NVIDIA GPU
```

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
    [--engine kyutai|parakeet] \
    [--hf-repo kyutai/stt-1b-en_fr-candle] \
    [--parakeet-repo istupakov/parakeet-tdt-0.6b-v3-onnx] \
    [--parakeet-device cpu|nvidia|apple|amd|directml] \
    [--parakeet-chunk-seconds 3.0] \
    [--parakeet-overlap-seconds 1.0] \
    [--cpu] \
    [--timestamps] \
    [--vad] \
    [--whisper]   # requires --features whisper
```

- `--bind`: Override the default bind address (`0.0.0.0:<port-from-config>`).
- `--engine`: Choose the default engine; when compiled with `parakeet`, both engines load and you can switch via WebSocket `{"type":"setengine","engine":"parakeet"}`.
- `--hf-repo`: Choose a different Kyutai Speech repo hosted on Hugging Face.
- `--parakeet-*`: Configure the Parakeet ONNX engine (defaults are multilingual, no language selection needed). Parakeet weights are CC-BY and are downloaded at runtime; nothing is redistributed.
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
- `[dictation]`: Enable live typing and configure in-app hotkeys.
- `[dictation.notifications]`: Toggle desktop popups and customise start/pause/stop messages shown for dictation state changes.
- `[dictation.hooks]` (requires `cargo build --features hooks`): Run shell commands on start, pause, or stop transitions (e.g., change colours in status bars).

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
