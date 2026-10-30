set positional-arguments

# Display help
help:
    just -l

# === Build ===

# Debug build
build:
    cargo build

# Release build
build-release:
    cargo build --release

# Check compilation without building
check:
    cargo check --all-features

# eaRS companion UI unit tests
ui-test:
    cargo test -p ears-ui --locked

# Headless font/CLI contract checks; no engine, event loop or native windows
ui-presentation-check:
    cargo build --locked -p ears-ui
    bash scripts/check_overlay_presentation.sh target/debug/ears-ui

# macOS synthetic native animation and reduced-motion regression
ui-preview-test:
    apps/ears-ui/scripts/check-parametric-preview.sh

# macOS native pointer routing; build first, use an unlocked desktop with existing event permission
ui-input-test binary="target/debug/ears-ui":
    swift scripts/probe_overlay_hit_test.swift "{{binary}}"
    swift scripts/probe_overlay_hit_test.swift "{{binary}}" --click-through
    swift scripts/probe_overlay_hit_test.swift "{{binary}}" --expanded
    swift scripts/probe_overlay_hit_test.swift "{{binary}}" --word
    swift scripts/probe_overlay_hit_test.swift "{{binary}}" --word --floating
    swift scripts/probe_overlay_hit_test.swift "{{binary}}" --word --floating --click-through

# === Code Quality ===

# Format code
fmt:
    cargo fmt -- --config imports_granularity=Item

# Auto-fix lint warnings
fix *args:
    cargo clippy --fix --all-features --tests --allow-dirty "$@"

# Run linter
clippy:
    cargo clippy --all-features --tests "$@"

# Alias for clippy
lint:
    just clippy

# === Installation ===

install:
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "eaRS Smart Installation"
    echo "======================="
    echo ""
    
    # Show active toolchain if rustup is available
    if command -v rustup &>/dev/null; then
        rustup show active-toolchain
    else
        echo "rustup not found, checking cargo..."
        cargo --version || { echo "Error: cargo not found"; exit 1; }
    fi
    
    # Detect OS
    OS_TYPE="unknown"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        echo "Detected OS: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        echo "Detected OS: macOS"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS_TYPE="windows"
        echo "Detected OS: Windows"
    else
        echo "Warning: Unknown OS type: $OSTYPE"
    fi
    
    # For Linux, detect distribution and set environment variables
    if [[ "$OS_TYPE" == "linux" ]]; then
        # Check if Arch Linux
        if command -v pacman &>/dev/null; then
            echo "Detected: Arch Linux"
            IS_ARCH=true
        else
            IS_ARCH=false
        fi
        
        # Check if Wayland/Hyprland is running
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            echo "Detected: Wayland session"
            IS_WAYLAND=true
        else
            IS_WAYLAND=false
        fi
        
        # Set CUDA environment variables if on Arch Linux with NVIDIA
        if [[ "$IS_ARCH" == true ]] && command -v nvidia-smi &>/dev/null; then
            echo "Detected: NVIDIA GPU on Arch Linux"
            echo "Setting CUDA environment variables..."
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            
            # Auto-detect GPU compute capability
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            echo "GPU: $GPU_NAME"
            
            case "$GPU_NAME" in
                *"RTX 40"*|*"RTX 4"*)
                    export CMAKE_CUDA_ARCHITECTURES=89
                    echo "CUDA architecture: 8.9 (Ada Lovelace)"
                    ;;
                *"RTX 30"*|*"RTX 3"*)
                    export CMAKE_CUDA_ARCHITECTURES=86
                    echo "CUDA architecture: 8.6 (Ampere)"
                    ;;
                *"RTX 20"*|*"RTX 2"*|*"GTX 16"*)
                    export CMAKE_CUDA_ARCHITECTURES=75
                    echo "CUDA architecture: 7.5 (Turing)"
                    ;;
                *)
                    echo "Using default CUDA architecture"
                    ;;
            esac
        fi
        
        # Set Wayland/Hyprland environment variables if needed
        if [[ "$IS_WAYLAND" == true ]]; then
            echo "Setting Wayland-compatible environment variables..."
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        
        # Use system sentencepiece on Linux
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    fi
    
    echo ""
    echo "Fetching dependencies..."
    cargo fetch
    
    echo ""
    echo "Ready to install. Run 'just install-ears' for interactive installation."
    echo "Or use specific install commands:"
    echo "  - just install-ears-cuda    (NVIDIA GPU)"
    echo "  - just install-ears-metal   (macOS Apple Silicon)"
    echo "  - just install-ears-default (CPU only)"

# Check and install system dependencies
check-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "Checking system dependencies..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux: requires system sentencepiece to avoid protobuf conflicts
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            if ! pkg-config --exists sentencepiece; then
                echo "Installing sentencepiece..."
                sudo apt-get update
                sudo apt-get install -y libsentencepiece-dev sentencepiece
            else
                echo "✓ sentencepiece already installed"
            fi
        elif command -v dnf &> /dev/null; then
            # Fedora/RHEL
            if ! pkg-config --exists sentencepiece; then
                echo "Installing sentencepiece..."
                sudo dnf install -y sentencepiece-devel
            else
                echo "✓ sentencepiece already installed"
            fi
        elif command -v pacman &> /dev/null; then
            # Arch Linux
            if ! pkg-config --exists sentencepiece; then
                echo "Installing sentencepiece..."
                sudo pacman -S --noconfirm sentencepiece
            else
                echo "✓ sentencepiece already installed"
            fi
        else
            echo "⚠ Unsupported Linux distribution. Please install sentencepiece manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: sentencepiece is compiled from source (static linking), but
        # kaudio's opus dependency should use Homebrew's libopus. If opus is
        # missing, audiopus_sys falls back to its bundled Opus CMake project,
        # which fails with CMake 4 because that upstream project still declares
        # an ancient cmake_minimum_required version.
        if ! command -v brew &> /dev/null; then
            echo "⚠ Homebrew not found. Please install Homebrew, then run: brew install pkg-config opus cmake"
            exit 1
        fi
        missing=()
        for dep in pkg-config opus cmake; do
            if ! brew list "$dep" &> /dev/null; then
                missing+=("$dep")
            fi
        done
        if ((${#missing[@]} > 0)); then
            echo "Installing macOS build dependencies: ${missing[*]}"
            brew install "${missing[@]}"
        else
            echo "✓ macOS build dependencies already installed"
        fi
        echo "✓ macOS detected - sentencepiece will be compiled from source"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows: sentencepiece is compiled from source (static linking)
        echo "✓ Windows detected - sentencepiece will be compiled from source"
    else
        echo "⚠ Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    echo "✓ All system dependencies checked"

# Run `cargo nextest` since it's faster than `cargo test`, though including
# --no-fail-fast is important to ensure all tests are run.
#
# Run `cargo install cargo-nextest` if you don't have it installed.
test:
    cargo nextest run --no-fail-fast

# Install ears with all features and auto-detected acceleration
install-all:
    #!/usr/bin/env bash
    set -euo pipefail
    
    # First check dependencies
    just check-deps
    
    echo "eaRS Full Installation (All Features)"
    echo "======================================"
    echo ""
    
    # Detect compatible acceleration
    ACCEL_FEATURE=""
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux: Check for GPU
        if command -v nvidia-smi &>/dev/null; then
            echo "✓ NVIDIA GPU detected"
            ACCEL_FEATURE="nvidia"
            
            # Set CUDA environment variables for Arch
            if command -v pacman &>/dev/null; then
                export PATH=/opt/cuda/bin:$PATH
                export CUDA_ROOT=/opt/cuda
                
                GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
                if [[ "$GPU_NAME" != "unknown" ]]; then
                    echo "GPU: $GPU_NAME"
                    case "$GPU_NAME" in
                        *"RTX 40"*|*"RTX 4"*)
                            export CMAKE_CUDA_ARCHITECTURES=89
                            echo "CUDA architecture: 8.9 (Ada Lovelace)"
                            ;;
                        *"RTX 30"*|*"RTX 3"*)
                            export CMAKE_CUDA_ARCHITECTURES=86
                            echo "CUDA architecture: 8.6 (Ampere)"
                            ;;
                        *"RTX 20"*|*"RTX 2"*|*"GTX 16"*)
                            export CMAKE_CUDA_ARCHITECTURES=75
                            echo "CUDA architecture: 7.5 (Turing)"
                            ;;
                    esac
                fi
            fi
        elif lspci 2>/dev/null | grep -i amd | grep -iq vga; then
            echo "✓ AMD GPU detected"
            ACCEL_FEATURE="amd"
        else
            echo "ℹ No GPU acceleration detected, using CPU"
        fi
        
        # Set Wayland environment variables if running Wayland
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            echo "✓ Wayland session detected"
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: Check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo "✓ Apple Silicon detected"
            ACCEL_FEATURE="apple"
        else
            echo "ℹ Intel Mac detected, using CPU"
        fi
        
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows: Check for NVIDIA or suggest DirectML
        if command -v nvidia-smi &>/dev/null; then
            echo "✓ NVIDIA GPU detected"
            ACCEL_FEATURE="nvidia"
        else
            echo "ℹ Using DirectML (Windows GPU acceleration)"
            ACCEL_FEATURE="directml"
        fi
    fi
    
    # Moonshine publishes native bundles for Apple Silicon and Linux x86_64/arm64.
    # Avoid breaking existing install-all on unqualified targets (notably Intel Mac).
    MOONSHINE_SUPPORTED=false
    case "$OSTYPE:$(uname -m)" in
        darwin*:arm64|linux-gnu*:x86_64|linux-gnu*:aarch64|linux-gnu*:arm64) MOONSHINE_SUPPORTED=true ;;
        *) echo "ℹ Moonshine is unavailable for this install-all target; skipping it" ;;
    esac

    # cargo install otherwise builds in a temporary directory; retain the SDK
    # where the Linux runtime helper can find it. Respect an explicit target dir.
    if [[ "$MOONSHINE_SUPPORTED" == true ]]; then
        export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
    fi

    # Build feature list with all supported engines and hooks
    FEATURES=("parakeet-rs" "transcribe-cpp" "hooks")
    if [[ "$MOONSHINE_SUPPORTED" == true ]]; then FEATURES+=("moonshine"); fi

    if [[ -n "$ACCEL_FEATURE" ]]; then
        FEATURES=("$ACCEL_FEATURE" "${FEATURES[@]}")
    fi
    
    FEATURE_STRING=$(IFS=,; echo "${FEATURES[*]}")
    
    echo ""
    echo "Installing with features: $FEATURE_STRING"
    echo ""
    
    # Allow user to confirm or override
    read -rp "Proceed with installation? [Y/n/custom]: " choice
    case "${choice:-y}" in
        [Yy]|"")
            cargo install --locked --path . --force --features "$FEATURE_STRING"
            ;;
        [Cc]*)
            echo ""
            echo "Available acceleration options:"
            echo "  1) nvidia    (CUDA)"
            echo "  2) apple     (Metal/CoreML)"
            echo "  3) amd       (ROCm)"
            echo "  4) directml  (Windows)"
            echo "  5) none      (CPU only)"
            read -rp "Select acceleration [1-5]: " accel_choice
            
            case "${accel_choice:-1}" in
                1) ACCEL_FEATURE="nvidia" ;;
                2) ACCEL_FEATURE="apple" ;;
                3) ACCEL_FEATURE="amd" ;;
                4) ACCEL_FEATURE="directml" ;;
                5) ACCEL_FEATURE="" ;;
                *) echo "Invalid choice, using auto-detected"; ;;
            esac
            
            FEATURES=("parakeet-rs" "transcribe-cpp" "hooks")
            if [[ "$MOONSHINE_SUPPORTED" == true ]]; then FEATURES+=("moonshine"); fi
            if [[ -n "$ACCEL_FEATURE" ]]; then
                FEATURES=("$ACCEL_FEATURE" "${FEATURES[@]}")
            fi
            FEATURE_STRING=$(IFS=,; echo "${FEATURES[*]}")
            
            echo "Installing with features: $FEATURE_STRING"
            cargo install --locked --path . --force --features "$FEATURE_STRING"
            ;;
        *)
            echo "Installation cancelled"
            exit 0
            ;;
    esac

    # On Linux the Moonshine SDK needs its shared libraries beside the binary;
    # the helper is a no-op on macOS.
    if [[ "$MOONSHINE_SUPPORTED" == true ]]; then
        bash ./scripts/install-moonshine-runtime.sh
    fi

    # Companion UI (skipped gracefully when only a headless build is possible)
    if command -v ears-ui >/dev/null 2>&1 || [[ "$OSTYPE" == "darwin"* ]] || [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
        just install-ui
    else
        echo "ℹ No desktop session detected, skipping ears-ui (run 'just install-ui' later)"
    fi

    echo ""
    echo "✓ Installation complete!"
    echo "Installed binaries: ears, ears-server, ears-dictation, ears-ui"

# Install the ears-ui companion app (dictation island, tray, settings; macOS/Win no extra deps, Linux needs Wayland/X11; first GPUI build is slow)
install-ui:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "Installing ears-ui (GPUI companion UI)…"
    cargo install --locked --path apps/ears-ui --force
    echo "✓ ears-ui installed."
    echo "  On macOS, grant the binary Accessibility + Input Monitoring (typing)" 
    echo "  and Microphone (capture) under System Settings → Privacy & Security."
    echo "  Run it with: ears-ui [--settings] [--no-overlay]"

# Install ears with interactive feature selection (checks dependencies first)
install-ears:
    #!/usr/bin/env bash
    set -euo pipefail
    
    # First check dependencies
    just check-deps
    
    # Detect OS and set environment variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Check if Arch Linux with NVIDIA
        if command -v pacman &>/dev/null && command -v nvidia-smi &>/dev/null; then
            echo "Setting CUDA environment variables for Arch Linux..."
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            
            # Auto-detect GPU compute capability
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
            if [[ "$GPU_NAME" != "unknown" ]]; then
                case "$GPU_NAME" in
                    *"RTX 40"*|*"RTX 4"*)
                        export CMAKE_CUDA_ARCHITECTURES=89
                        ;;
                    *"RTX 30"*|*"RTX 3"*)
                        export CMAKE_CUDA_ARCHITECTURES=86
                        ;;
                    *"RTX 20"*|*"RTX 2"*|*"GTX 16"*)
                        export CMAKE_CUDA_ARCHITECTURES=75
                        ;;
                esac
            fi
        fi
        
        # Set Wayland environment variables if running Wayland
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            echo "Setting Wayland-compatible environment variables..."
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        
        # Use system sentencepiece on Linux
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    fi
    
    # Run the interactive install script
    ./scripts/install-ears.sh

# Install ears with specific features (e.g., just install-ears-features "nvidia,parakeet-rs")
# Available features: nvidia, apple, amd, directml, parakeet-rs, transcribe-cpp, moonshine, hooks
# Combine multiple features with commas: "nvidia,parakeet-rs" or "apple,parakeet-rs,transcribe-cpp,hooks"
install-ears-features features:
    #!/usr/bin/env bash
    set -euo pipefail
    
    # First check dependencies
    just check-deps
    
    # Detect OS and set environment variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Check if Arch Linux with NVIDIA
        if command -v pacman &>/dev/null && command -v nvidia-smi &>/dev/null; then
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
            if [[ "$GPU_NAME" != "unknown" ]]; then
                case "$GPU_NAME" in
                    *"RTX 40"*|*"RTX 4"*) export CMAKE_CUDA_ARCHITECTURES=89 ;;
                    *"RTX 30"*|*"RTX 3"*) export CMAKE_CUDA_ARCHITECTURES=86 ;;
                    *"RTX 20"*|*"RTX 2"*|*"GTX 16"*) export CMAKE_CUDA_ARCHITECTURES=75 ;;
                esac
            fi
        fi
        
        # Set Wayland environment variables if running Wayland
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    fi
    
    cargo install --locked --path . --force --features {{features}}
    case ",{{features}}," in
        *,moonshine,*) bash ./scripts/install-moonshine-runtime.sh ;;
    esac

# Install ears with default features (CPU only)
install-ears-default:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    
    # Set Linux-specific environment variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    fi
    
    cargo install --locked --path . --force

# Install ears with Metal acceleration (macOS)
install-ears-metal:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    cargo install --locked --path . --force --features apple

# Install ears with CUDA acceleration (NVIDIA)
install-ears-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    
    # Set CUDA environment variables for Arch Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v pacman &>/dev/null && command -v nvidia-smi &>/dev/null; then
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
            if [[ "$GPU_NAME" != "unknown" ]]; then
                case "$GPU_NAME" in
                    *"RTX 40"*|*"RTX 4"*) export CMAKE_CUDA_ARCHITECTURES=89 ;;
                    *"RTX 30"*|*"RTX 3"*) export CMAKE_CUDA_ARCHITECTURES=86 ;;
                    *"RTX 20"*|*"RTX 2"*|*"GTX 16"*) export CMAKE_CUDA_ARCHITECTURES=75 ;;
                esac
            fi
        fi
        
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    fi
    
    cargo install --locked --path . --force --features nvidia

# Install ears with Kyutai CUDA + parakeet-rs Nemotron
# (kept under the old recipe name for compatibility)
install-ears-cuda-parakeet:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    
    # Set Linux-specific environment variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v pacman &>/dev/null && command -v nvidia-smi &>/dev/null; then
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
            if [[ "$GPU_NAME" != "unknown" ]]; then
                case "$GPU_NAME" in
                    *"RTX 40"*|*"RTX 4"*) export CMAKE_CUDA_ARCHITECTURES=89 ;;
                    *"RTX 30"*|*"RTX 3"*) export CMAKE_CUDA_ARCHITECTURES=86 ;;
                    *"RTX 20"*|*"RTX 2"*|*"GTX 16"*) export CMAKE_CUDA_ARCHITECTURES=75 ;;
                esac
            fi
        fi
        
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    fi
    
    cargo install --locked --path . --force --features nvidia,parakeet-rs

# Install ears with parakeet-rs Nemotron streaming engine
# (kept under the old recipe name for compatibility)
install-ears-parakeet:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    
    # Set Linux-specific environment variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v pacman &>/dev/null && command -v nvidia-smi &>/dev/null; then
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
            if [[ "$GPU_NAME" != "unknown" ]]; then
                case "$GPU_NAME" in
                    *"RTX 40"*|*"RTX 4"*) export CMAKE_CUDA_ARCHITECTURES=89 ;;
                    *"RTX 30"*|*"RTX 3"*) export CMAKE_CUDA_ARCHITECTURES=86 ;;
                    *"RTX 20"*|*"RTX 2"*|*"GTX 16"*) export CMAKE_CUDA_ARCHITECTURES=75 ;;
                esac
            fi
        fi
        
        if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
            export WEBKIT_DISABLE_DMABUF_RENDERER=1
            export GDK_BACKEND=x11
            export WEBKIT_DISABLE_COMPOSITING_MODE=0
            export WEBKIT_ENABLE_MEDIA_STREAM=1
        fi
        
        export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    fi
    
    cargo install --locked --path . --force --features parakeet-rs

# === Offline ModernBERT adapter experiment (not connected to dictation) ===

modernbert-proof-train data="../slopbert/dataset_example.csv":
    uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/train_demo.py --data "{{data}}" --steps 4

modernbert-proof-test:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --project experiments/modernbert-hotswap --no-dev python -m unittest discover -s experiments/modernbert-hotswap -p 'test_*.py' -v

modernbert-proof-bench:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/bench_swaps.py

# Synthetic voice-intent proof (remote generator only; local training/inference)
modernbert-intent-generate:
    uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/generate_intent.py

modernbert-intent-train:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/train_intent.py

modernbert-intent-eval:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/eval_intent.py

modernbert-intent-compare-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for name in intent_adapter intent_ettin32 intent_ettin68 intent_minilm33_tuned intent_minilm33; do
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --project experiments/modernbert-hotswap --no-dev \
            experiments/modernbert-hotswap/compare_intent.py \
            --adapter "experiments/modernbert-hotswap/artifacts/$name" \
            --output "experiments/modernbert-hotswap/artifacts/compare-$name.json"
    done

modernbert-dgemma-eval:
    uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/compare_dgemma.py

modernbert-sys1-eval:
    uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/compare_sys1.py

modernbert-clm-eval:
    uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/compare_clm.py

# Bounded speaker-gate software proof; no microphone/model inference.
test-speaker-gate:
    cargo nextest run --no-fail-fast -p eaRS --lib --no-default-features -E 'test(speaker_gate::) | test(dictation::) | test(config::)'

# Compile the optional prepared-voqx adapter without running/loading a model.
check-speaker-gate:
    cargo check -p eaRS --no-default-features
    cargo check -p eaRS --no-default-features --features speaker-voqx

# Client-side VAD mechanism; no microphone or model required.
test-client-vad:
    cargo check --locked --all-targets --features client-vad
    cargo test --locked --lib --features client-vad

# Explicit-artifact CPU smoke test; caller supplies ORT_DYLIB_PATH.
test-client-vad-model model:
    EARS_TEST_VAD_MODEL="{{model}}" cargo test --locked --lib --features client-vad silero_silence_reset_and_bad_frames -- --ignored

# Read-only offline segmentation preview; no recordings are changed.
preview-client-vad model audio:
    cargo run --locked --example client_vad_segments --features client-vad -- "{{model}}" "{{audio}}"

# Capture inactivity, client loopback WS and ordered server finalization.
# Native acoustic probes stay opt-in; this never opens the microphone.
test-auto-pause:
    cargo check --locked --all-targets --features client-vad,moonshine,parakeet-rs
    cargo test --locked --lib --features client-vad,moonshine,parakeet-rs
    cargo test --locked -p ears-ui --features ears/client-vad,ears/moonshine,ears/parakeet-rs

# Private decision-trace envelope, privacy admission and exact request bytes.
test-command-traces:
    bun test scripts/command_trace.test.ts

# === Maintenance ===

# Clean build artifacts
clean:
    cargo clean

# Update dependencies
update:
    cargo update

# Generate documentation
docs:
    cargo doc --no-deps --open
