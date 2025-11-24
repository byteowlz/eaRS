set positional-arguments

# Display help
help:
    just -l

# format code
fmt:
    cargo fmt -- --config imports_granularity=Item

fix *args:
    cargo clippy --fix --all-features --tests --allow-dirty "$@"

clippy:
    cargo clippy --all-features --tests "$@"

install:
    rustup show active-toolchain
    cargo fetch

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

# Install ears with interactive feature selection (checks dependencies first)
install-ears:
    just check-deps
    ./scripts/install-ears.sh

# Install ears with specific features (e.g., just install-ears-features nvidia)
install-ears-features features:
    just check-deps
    cargo install --path . --force --features {{features}}

# Install ears with default features (CPU only)
install-ears-default:
    just check-deps
    cargo install --path . --force

# Install ears with Metal acceleration (macOS)
install-ears-metal:
    just check-deps
    cargo install --path . --force --features apple

# Install ears with CUDA acceleration (NVIDIA)
install-ears-cuda:
    just check-deps
    cargo install --path . --force --features nvidia

# Install ears with Parakeet engine
install-ears-parakeet:
    just check-deps
    cargo install --path . --force --features parakeet
