![ears_banner](banner.png)

# eaRS

`eaRS` is a Rust-based streaming speech-to-text stack built on Kyutai's models. The tool is now delivered as a single CLI with three responsibilities:

- **Server management**: `ears server start|stop` launches and controls the inference backend.
- **Client capture**: Running `ears` without subcommands streams microphone audio to the server and prints live transcripts.
- **Dictation**: `ears dictation start|stop` enables system-wide dictation with hotkey control, dictionary replacements, and transcript history.

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
just install-ears-default       # CPU only, no acceleration

# Install with custom feature combinations
just install-ears-features "apple,parakeet-rs,hooks"
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
| `parakeet-rs` | Enable Nemotron cache-aware streaming through parakeet-rs |
| `transcribe-cpp` | Enable ggml GGUF streaming models via transcribe.cpp (built from source, needs cmake; `apple`/`nvidia` add Metal/CUDA compute) |
| `hooks`       | Enable shell command hooks for dictation state changes    |

### Manual Installation

If you prefer not to use just:

```bash
cargo install --path .                               # CPU only
cargo install --path . --features apple              # Apple Silicon
cargo install --path . --features nvidia             # NVIDIA GPU
cargo install --path . --features parakeet-rs        # Nemotron streaming
cargo install --path . --features apple,parakeet-rs  # Kyutai Metal + Nemotron
cargo install --path . --features apple,transcribe-cpp # Kyutai Metal + transcribe.cpp (GGUF, Metal)
```

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

> **Note:** The `just install-*` commands automatically check for and install sentencepiece.

### macOS

On macOS, `sentencepiece` is compiled from source and statically linked during the build process. eaRS also depends on Opus through `kaudio`; install the system Opus package so `audiopus_sys` does not fall back to its bundled CMake build:

```bash
brew install pkg-config opus cmake
```

If you see an `audiopus_sys` error mentioning `Compatibility with CMake < 3.5 has been removed from CMake`, Opus was not found by `pkg-config`. Install the packages above and rebuild. The `just check-deps` and `just install-*` recipes install these Homebrew dependencies automatically.

### Windows

On Windows, `sentencepiece` is compiled from source and statically linked during the build process. No manual installation required.

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
| Dictation startup fails       | Check logs at `$XDG_STATE_HOME/ears/dictation.log` (default: `~/.local/state/ears/dictation.log`) |
| Server unreachable on start   | `ears dictation start` now fails fast if the endpoint is not stably reachable; verify configured `dictation.servers` host/port (or `--server` URL) |
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
cargo build --release                                      # CPU only
cargo build --release --features apple                     # Apple Silicon (Metal)
cargo build --release --features nvidia                    # NVIDIA GPU (CUDA)
cargo build --release --features parakeet-rs               # Nemotron streaming
cargo build --release --features "apple,parakeet-rs,hooks" # Typical macOS build
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

# 4. (Optional) Add speech-specific replacements
./target/release/ears dictionary add \
  --replacement "Kyutai" \
  --phrase "cute eye" \
  --phrase "q tie"
```

Press `Ctrl+C` in the client to stop streaming. When you are done with the backend:

If the local server is not running, `ears` will start it automatically before connecting.

```bash
./target/release/ears server stop
```

## Server commands

```bash
ears server start \
    [--bind 0.0.0.0:8765] \
    [--engine kyutai|parakeet-rs|transcribe-cpp] \
    [--hf-repo kyutai/stt-1b-en_fr-candle] \
    [--parakeet-rs-model /path/to/nemotron-model] \
    [--parakeet-rs-lang auto|en|de|es|ja] \
    [--transcribe-cpp-model /path/to/model.gguf] \
    [--transcribe-cpp-lang auto|en|de|...] \
    [--cpu] [--timestamps] [--vad]

# Restart accepts the same arguments as start.
ears server restart --engine parakeet-rs --parakeet-rs-lang de
```

- `--bind`: Override the default bind address (`0.0.0.0:<port-from-config>`).
- `--engine`: Choose the default engine (`kyutai` or `parakeet-rs` when compiled with that feature).
- `--hf-repo`: Choose a different Kyutai Speech repository.
- `--parakeet-rs-model`: Override `[parakeet_rs].model_dir`.
- `--parakeet-rs-lang`: Override `[parakeet_rs].language`; a dictation client `--lang` command overrides it per session.
- `--transcribe-cpp-model`: Streaming-capable transcribe.cpp GGUF model (overrides `[transcribe_cpp].model_path`). Accepts either an existing `.gguf` path or a known model slug, which is auto-downloaded from the `handy-computer` Hugging Face org into the shared HF cache on first use. Append `@QUANT` (e.g. `nemotron-3.5-asr-streaming-0.6b@Q5_K_M`) to pick a non-default quant. One session at a time: the model's compute lease is held per active stream.

Browse and pre-download models (requires the `transcribe-cpp` feature):

```bash
ears models list                 # show known streaming models
ears models pull                 # fzf-pick a model and download it
ears models pull nemotron        # seed the fuzzy picker with a query
ears models pull nemotron-3.5-asr-streaming-0.6b   # exact slug: download directly
ears models refresh              # update the local catalog to the latest list
```

`models pull` prints the cached `.gguf` path to stdout (so `--transcribe-cpp-model "$(ears models pull ...)"` works) and falls back to a numbered prompt when `fzf` is not installed.

The catalog is data, not code: `src/server/catalog.json` is baked into the binary (so `models list` works offline) and generated from the `handy-computer` Hugging Face org by `scripts/gen_catalog.py`. A weekly GitHub Action regenerates it and opens a PR, so newly published streaming models get picked up automatically. `ears models refresh` fetches the maintained catalog into the platform config dir (`~/.config/ears/models.json` on Linux, `~/Library/Application Support/ears/models.json` on macOS), which then overrides the baked copy — so you get new models without reinstalling. Set `EARS_CATALOG_URL` to point at a different source. Any `handy-computer/*-gguf` slug also works directly via `--transcribe-cpp-model`, listed or not.
- `--transcribe-cpp-lang`: Override `[transcribe_cpp].language` for multilingual GGUF models.
- `--cpu`: Force Kyutai CPU execution (otherwise CUDA/Metal is used when available).
- `--timestamps`: Include word timestamps in the server stream.
- `--vad`: Enable Kyutai voice-activity detection.

The server writes a PID file to `$XDG_STATE_HOME/ears/server.pid` (or `~/.local/state/ears/server.pid`). `ears server stop` stops the local server and associated local dictation. `ears server restart` keeps dictation alive so it reconnects to the new server automatically.

## Client options

```
./target/release/ears [--device 1] [--server ws://host:port/] [--timestamps] [--list-devices]
```

- `--list-devices`: Print available input devices and exit.
- `--device`: Select a specific capture device by index.
- `--server`: Point the client at a remote server (`ws://127.0.0.1:<config-port>/` by default).
- `ears-dictation --codec opus`: Use the bandwidth-saving Opus transport for a remote dictation server. It overrides `[dictation].codec`; PCM remains the accuracy-first default.
- `--timestamps`: Print the final transcript with per-word timing instead of live text.
- `--codec`: Audio transport codec, `pcm` (default) or `opus`. Opus cuts upstream
  bandwidth roughly 30x (~768 kbps raw f32 vs ~25 kbps) with no practical accuracy
  loss — useful when the server is remote or the link is slow. The client sends a
  `{"type":"setcodec","codec":"opus"}` command before streaming ogg-opus frames;
  third-party clients can do the same.

The client streams raw 24 kHz mono PCM to the server (or ogg-opus with `--codec opus`) and displays each live word as it appears. When the backend signals completion, the final transcript (and optional timestamps) is printed.

## Dictionary replacements

eaRS can apply a fast dictionary pass to dictation output before typing. This is useful for project names, product names, acronyms, and words that speech-to-text engines consistently mishear.

Dictionary entries map multiple observed phrases to one canonical replacement:

```toml
version = 1

[[entries]]
replace = "Kyutai"
phrases = ["cute eye", "q tie", "kyu tai"]

[[entries]]
replace = "byteowlz"
phrases = ["byte owls", "bite owls", "by alts"]
```

The default global dictionary lives at:

```text
$XDG_CONFIG_HOME/ears/dictionaries/global.toml
# or ~/.config/ears/dictionaries/global.toml
```

Manage it with:

```bash
ears dictionary list
ears dictionary add --replacement "oqto" --phrase "octo" --phrase "O Q T O"
ears dictionary remove --phrase "octo"
ears dictionary remove --replacement "oqto"
ears dictionary test "cute eye inside pie agent"
```

`dictionary add` is flag-based on purpose: `--replacement` is the canonical text, and each `--phrase` is an observed phrase to replace. Repeated adds for the same replacement merge into the existing entry. If a phrase already belongs to another entry, it is moved so each phrase has one owner. Use `dictionary remove --phrase "..."` to remove observed phrases, or `dictionary remove --replacement "..."` to remove an entire canonical entry.

`ears-dictation` hot-reloads dictionary files when they change. Multi-word phrases may span separate streaming events: ordinary text is typed immediately, while a possible dictionary-phrase prefix is held for up to 1.5 seconds awaiting the next word. After installing a new eaRS binary you still need to restart `ears-dictation` once; after that, dictionary edits are picked up automatically.

## Transcript history

Dictation transcript history is enabled by default. Live word events and final events are buffered in memory and periodically appended to JSONL files:

```text
$XDG_STATE_HOME/ears/transcripts/YYYY-MM-DD.jsonl
# or ~/.local/state/ears/transcripts/YYYY-MM-DD.jsonl
```

Unchanged live words omit the `replaced` field by default:

```json
{"ts":1782720576,"type":"word","raw":"and","changed":false}
{"ts":1782720576,"type":"word","raw":"cute eye","replaced":"Kyutai","changed":true}
```

Buffers flush every few seconds, after enough events, on disconnect, and on shutdown. This avoids writing to disk for every live word while keeping recent dictation persisted.

## Configuration

Runtime configuration lives at:

```
$XDG_CONFIG_HOME/ears/config.toml
# or ~/.config/ears/config.toml
```

Key sections:

- `[storage]`: Override model cache directories and reference audio location.
- `[parakeet_rs]`: Set the default Nemotron model directory and multilingual language prompt. CLI server flags override these values; client `--lang` overrides the language per session.
- `[transcribe_cpp]`: Set the default transcribe.cpp GGUF model path and language hint (requires the `transcribe-cpp` feature).
- `[dictation].codec`: Set `"pcm"` (default, accuracy-first) or `"opus"` (bandwidth-saving for remote servers). `ears-dictation --codec` overrides the config value.
- `[server]`: Default WebSocket port used by `ears server start` and the capture client.
- `[dictation]`: Enable live typing and configure in-app hotkeys.
- `[hotkeys]`: Set hotkey bindings and mode (`toggle`, `push_to_talk`, or `hybrid`).
- `[dictation]`: `start_paused = true|false` controls whether dictation starts paused or live.
- `[dictation.notifications]`: Toggle desktop popups and customise start/pause/stop messages shown for dictation state changes.
- `[dictation.hooks]` (requires `cargo build --features hooks`): Run shell commands on start, pause, or stop transitions (e.g., change colours in status bars).
- `[replacement]`: Enable dictionary replacement, configure dictionary paths, and choose case-sensitive matching.
- `[transcripts]`: Enable transcript JSONL history and configure buffering.

Example engine, replacement, and transcript settings:

```toml
[parakeet_rs]
model_dir = "~/Models/parakeet-rs/nemotron-3.5-asr-streaming-0.6b-onnx"
language = "auto"

[replacement]
enabled = true
dictionary_paths = ["~/.config/ears/dictionaries/global.toml"]
case_sensitive = false

[transcripts]
enabled = true
path = "default"
store_raw = true
store_replaced = true
store_unchanged_replaced = false
flush_interval_ms = 2000
flush_max_events = 50
```

If the file does not exist, it is created on first run together with the reference audio bundle.

## WebSocket protocol

The server emits JSON events:

- `{"type":"word","word":"hello","start_time":1.23,"end_time":null}` – live word updates.
- `{"type":"final","text":"…","words":[…]}` – final transcript with timestamp list.

Clients may send `{"type":"stop"}` to end the current session (the capture client does this automatically when interrupted).

## Troubleshooting

- **`ears server start` reports "already running"** – Use `ears server stop` to terminate the existing instance. If the PID no longer exists, `stop` will clean up the stale PID file.
- **Client prints "failed to connect"** – Ensure the server is running and reachable at the URL passed via `--server` (check the configured port).
- **High latency** – Run the server on the same machine as the client or enable GPU acceleration (`--features cuda` or `--features metal`).
