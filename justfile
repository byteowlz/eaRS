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

# Run `cargo nextest` since it's faster than `cargo test`, though including
# --no-fail-fast is important to ensure all tests are run.
#
# Run `cargo install cargo-nextest` if you don't have it installed.
test:
    cargo nextest run --no-fail-fast

# Install ears with interactive feature selection
install-ears:
    ./scripts/install-ears.sh

# Install ears with specific features (e.g., just install-ears-features nvidia)
install-ears-features features:
    cargo install --path . --force --features {{features}}

# Install ears with default features (CPU only)
install-ears-default:
    cargo install --path . --force

# Install ears with Metal acceleration (macOS)
install-ears-metal:
    cargo install --path . --force --features apple

# Install ears with CUDA acceleration (NVIDIA)
install-ears-cuda:
    cargo install --path . --force --features nvidia

# Install ears with Parakeet engine
install-ears-parakeet:
    cargo install --path . --force --features parakeet
