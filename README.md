![ears_banner](banner.png)

# eaRS

`eaRS` is a Rust-based streaming speech-to-text stack built on Kyutai's models. The tool is now delivered as a single CLI with three responsibilities:

- **Server management**: `ears server start|stop` launches and controls the inference backend.
- **Client capture**: Running `ears` without subcommands streams microphone audio to the server and prints live transcripts.
- **Dictation**: `ears dictation start|stop` enables system-wide dictation with hotkey control.

## Installation

### Using just (Recommended)

eaRS uses [just](https://github.com/casey/just) as a command runner. Install it first:

```bash
# macOS
brew install just

# Arch Linux
sudo pacman -S just

# Ubuntu/Debian
sudo apt install just

# Or via cargo
cargo install just
```

Then install eaRS with automatic hardware detection:

```bash

# Install with all features and auto-detected acceleration
just install-all

# Interactive installation - detects your hardware and offers feature selection
just install-ears

# Or use presets for specific configurations:
just install-ears-metal         # macOS with Metal acceleration (Apple Silicon)
just install-ears-cuda          # Linux/Windows with NVIDIA GPU (CUDA)
just install-ears-cuda-parakeet # NVIDIA GPU with Parakeet engine
just install-ears-parakeet      # Parakeet engine (CPU)
just install-ears-default       # CPU only, no acceleration

# Install with custom feature combinations
just install-ears-features "nvidia,parakeet,whisper,hooks"
```

The `just install-ears` command will:

1. Check and install system dependencies (sentencepiece on Linux)
2. Detect your hardware (GPU type, Wayland session, etc.)
3. Set appropriate environment variables (CUDA paths, Wayland compat, etc.)
4. Let you choose which features to enable
5. Build and install the binaries

**Available features:**

| Feature    | Description                                            |
| ---------- | ------------------------------------------------------ |
| `nvidia`   | CUDA acceleration for NVIDIA GPUs                      |
| `apple`    | Metal/CoreML acceleration for Apple Silicon            |
| `amd`      | ROCm acceleration for AMD GPUs                         |
| `directml` | DirectML acceleration for Windows                      |
| `parakeet` | Enable Parakeet ONNX engine                            |
| `whisper`  | Enable Whisper post-processing                         |
| `hooks`    | Enable shell command hooks for dictation state changes |

### Manual Installation

If you prefer not to use just:

```bash
cargo install --path .                        # CPU only
cargo install --path . --features apple       # Apple Silicon
cargo install --path . --features nvidia      # NVIDIA GPU
cargo install --path . --features parakeet    # With Parakeet engine
```

## System Dependencies

### Linux

On Linux, eaRS uses the system `sentencepiece` library to avoid protobuf conflicts with ONNX Runtime.

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y pkg-config build-essential libsentencepiece-dev sentencepiece \
    libudev-dev xorg-dev libxkbcommon-dev
```

**Fedora/RHEL:**

```bash
sudo dnf install -y sentencepiece-devel
```

**Arch Linux:**

```bash
sudo pacman -S sentencepiece
```

> **Note:** The `just install-*` commands automatically check for and install sentencepiece.

### macOS and Windows

On macOS and Windows, `sentencepiece` is compiled from source and statically linked during the build process. No manual installation required.

## Wayland/Linux Keyboard Emulation

eaRS dictation uses the Linux **uinput** kernel interface for reliable keyboard input. This approach works on both Wayland and X11, unlike X11-only tools like `xdotool`.

### Why uinput?

Wayland's security model prevents applications from injecting keystrokes into other windows using traditional X11 methods. eaRS solves this by creating a virtual keyboard device at the kernel level via `/dev/uinput`, which:

- Works on **all** Wayland compositors (Sway, Hyprland, KWin, GNOME, etc.)
- Works on X11 as well (unified codebase)
- Is compositor-agnostic (no protocol negotiation needed)
- Provides reliable, low-latency input

### Quick Setup

Run the automated setup script:

```bash
./scripts/setup-dictation-linux.sh
```

The script will:

1. Load the uinput kernel module
2. Configure uinput to load on boot
3. Create a udev rule for proper `/dev/uinput` permissions
4. Add your user to the `input` group
5. Verify the configuration

**You must log out and log back in after setup for group changes to take effect.**

### Manual Setup

If you prefer to configure manually:

```bash
# 1. Load the uinput kernel module
sudo modprobe uinput

# 2. Make it load on boot
echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf

# 3. Create udev rule for proper permissions
echo 'KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"' | \
    sudo tee /etc/udev/rules.d/99-uinput.conf
sudo udevadm control --reload-rules
sudo udevadm trigger

# 4. Reload uinput to apply permissions
sudo rmmod uinput && sudo modprobe uinput

# 5. Add your user to the input group
sudo usermod -a -G input $USER

# 6. Log out and log back in (required!)

# 7. Verify setup
groups | grep input
ls -l /dev/uinput
```

Expected output:

```
crw-rw---- 1 root input 10, 223 Dec 12 10:00 /dev/uinput
```

### Troubleshooting Dictation

| Issue                         | Solution                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------ |
| "Failed to open /dev/uinput"  | Check permissions: `ls -l /dev/uinput` (should show `crw-rw---- 1 root input`) |
| Wrong group on /dev/uinput    | Create/update udev rule (step 3 above), then reload uinput                     |
| uinput module not loaded      | Run `sudo modprobe uinput` and verify with `lsmod \| grep uinput`              |
| Not in input group            | Run `groups \| grep input` - if missing, add yourself and **log out/in**       |
| Still not working after setup | Reboot to ensure all changes take effect                                       |

### How It Works

```
[Speech] -> [ears server] -> [WebSocket] -> [ears-dictation]
                                                   |
                                                   v
                                          [uinput device]
                                                   |
                                                   v
                                          [/dev/uinput]
                                                   |
                                                   v
                                          [Linux kernel]
                                                   |
                                                   v
                                        [Focused application]
```

The `src/virtual_keyboard.rs` module provides a cross-platform abstraction:

- **Linux**: Uses uinput directly (works on Wayland and X11)
- **macOS/Windows**: Falls back to enigo library

## Build

For development builds:

```bash
cargo build --release                                 # CPU only
cargo build --release --features apple                # Apple Silicon (Metal)
cargo build --release --features nvidia               # NVIDIA GPU (CUDA)
cargo build --release --features parakeet             # Parakeet ONNX engine
cargo build --release --features "parakeet nvidia"    # Parakeet + CUDA
cargo build --release --features "parakeet apple"     # Parakeet + CoreML/Metal
cargo build --release --features "parakeet amd"       # Parakeet + ROCm
cargo build --release --features "parakeet directml"  # Parakeet + DirectML
```

All binaries are emitted into `./target/release/`.

## Just Commands Reference

| Command                              | Description                                            |
| ------------------------------------ | ------------------------------------------------------ |
| `just`                               | Show all available commands                            |
| `just install`                       | Prepare environment (detect hardware, fetch deps)      |
| `just install-ears`                  | Interactive installation with feature selection        |
| `just install-all`                   | Install with all features + auto-detected acceleration |
| `just install-ears-features "f1,f2"` | Install with specific features                         |
| `just check-deps`                    | Check/install system dependencies only                 |
| `just build`                         | Debug build                                            |
| `just build-release`                 | Release build                                          |
| `just check`                         | Fast compile check (no build)                          |
| `just test`                          | Run tests (uses cargo-nextest)                         |
| `just fmt`                           | Format code                                            |
| `just clippy` / `just lint`          | Run linter                                             |
| `just fix`                           | Auto-fix lint warnings                                 |
| `just clean`                         | Clean build artifacts                                  |
| `just update`                        | Update dependencies                                    |
| `just docs`                          | Generate and open documentation                        |

## Quick start

```bash
# 1. Start the transcription server (runs in the background)
./target/release/ears server start

# 2. Stream your microphone to the server and print live text
./target/release/ears

# 3. (Optional) Enable system-wide dictation
./target/release/ears dictation start
```

Press `Ctrl+C` in the client to stop streaming. When you are done with the backend:

If the local server is not running, `ears` will start it automatically before connecting.

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
