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
            export CUDARC_CUDA_VERSION=12060
            
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
            PKGS_TO_INSTALL=()
            if ! command -v pkg-config &> /dev/null; then
                PKGS_TO_INSTALL+=(pkg-config)
            fi
            if ! pkg-config --exists sentencepiece 2>/dev/null; then
                PKGS_TO_INSTALL+=(libsentencepiece-dev sentencepiece)
            else
                echo "✓ sentencepiece already installed"
            fi
            if ! pkg-config --exists libudev 2>/dev/null; then
                PKGS_TO_INSTALL+=(libudev-dev)
            else
                echo "✓ libudev already installed"
            fi
            if ! pkg-config --exists x11 2>/dev/null || ! pkg-config --exists xi 2>/dev/null; then
                PKGS_TO_INSTALL+=(xorg-dev libxkbcommon-dev)
            else
                echo "✓ x11/xorg already installed"
            fi
            if [ ${#PKGS_TO_INSTALL[@]} -gt 0 ]; then
                echo "Installing: ${PKGS_TO_INSTALL[*]}..."
                sudo apt-get update
                sudo apt-get install -y "${PKGS_TO_INSTALL[@]}"
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
        # macOS: sentencepiece is compiled from source (static linking)
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
                export CUDARC_CUDA_VERSION=12060
                
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
    
    # Build feature list with all engines and hooks
    FEATURES=("whisper" "parakeet" "hooks")
    
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
            cargo install --path . --force --features "$FEATURE_STRING"
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
            
            FEATURES=("whisper" "parakeet" "hooks")
            if [[ -n "$ACCEL_FEATURE" ]]; then
                FEATURES=("$ACCEL_FEATURE" "${FEATURES[@]}")
            fi
            FEATURE_STRING=$(IFS=,; echo "${FEATURES[*]}")
            
            echo "Installing with features: $FEATURE_STRING"
            cargo install --path . --force --features "$FEATURE_STRING"
            ;;
        *)
            echo "Installation cancelled"
            exit 0
            ;;
    esac
    
    echo ""
    echo "✓ Installation complete!"
    echo "Installed binaries: ears, ears-server, ears-dictation"

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
            export CUDARC_CUDA_VERSION=12060
            
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

# Install ears with specific features (e.g., just install-ears-features "nvidia,parakeet")
# Available features: nvidia, apple, amd, directml, whisper, parakeet, hooks
# Combine multiple features with commas: "nvidia,parakeet" or "apple,whisper,hooks"
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
            export CUDARC_CUDA_VERSION=12060
            
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
    
    cargo install --path . --force --features {{features}}

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
    
    cargo install --path . --force

# Install ears with Metal acceleration (macOS)
install-ears-metal:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    cargo install --path . --force --features apple

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
            export CUDARC_CUDA_VERSION=12060
            
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
    
    cargo install --path . --force --features nvidia

# Install ears with CUDA + Parakeet (best for NVIDIA GPUs)
install-ears-cuda-parakeet:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    
    # Set Linux-specific environment variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v pacman &>/dev/null && command -v nvidia-smi &>/dev/null; then
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            export CUDARC_CUDA_VERSION=12060
            
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
    
    cargo install --path . --force --features nvidia,parakeet

# Install ears with Parakeet engine
install-ears-parakeet:
    #!/usr/bin/env bash
    set -euo pipefail
    
    just check-deps
    
    # Set Linux-specific environment variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v pacman &>/dev/null && command -v nvidia-smi &>/dev/null; then
            export PATH=/opt/cuda/bin:$PATH
            export CUDA_ROOT=/opt/cuda
            export CUDARC_CUDA_VERSION=12060
            
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
    
    cargo install --path . --force --features parakeet

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
