#!/bin/bash
set -e

# Add targets
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-pc-windows-gnu  
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
rustup target add aarch64-unknown-linux-gnu

# Build for all targets
echo "Building for Linux x86_64..."
cargo build --release --target x86_64-unknown-linux-gnu

echo "Building for Windows x86_64..."
cargo build --release --target x86_64-pc-windows-gnu

echo "Building for macOS x86_64..."
cargo build --release --target x86_64-apple-darwin

echo "Building for macOS ARM64..."
cargo build --release --target aarch64-apple-darwin

echo "Building for Linux ARM64..."
cargo build --release --target aarch64-unknown-linux-gnu

# Create dist directory structure expected by GoReleaser
mkdir -p dist

# Copy binaries to expected locations
cp target/x86_64-unknown-linux-gnu/release/ears dist/ears_Linux_x86_64
cp target/x86_64-pc-windows-gnu/release/ears.exe dist/ears_Windows_x86_64.exe
cp target/x86_64-apple-darwin/release/ears dist/ears_Darwin_x86_64
cp target/aarch64-apple-darwin/release/ears dist/ears_Darwin_arm64
cp target/aarch64-unknown-linux-gnu/release/ears dist/ears_Linux_arm64

echo "All builds completed successfully!"