#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FEATURES=()

# Platform/Acceleration selection
echo "Select hardware acceleration:"
echo "  1) none (CPU only)"
echo "  2) nvidia (CUDA)"
echo "  3) apple (Metal/CoreML)"
echo "  4) amd (ROCm)"
echo "  5) directml (Windows)"
read -rp "Enter choice [1-5]: " platform_choice

case "${platform_choice:-1}" in
2) FEATURES+=("nvidia") ;;
3) FEATURES+=("apple") ;;
4) FEATURES+=("amd") ;;
5) FEATURES+=("directml") ;;
*) ;;
esac

# Engine selection
echo ""
echo "Select additional speech recognition engine:"
echo "(Kyutai moshi is always enabled)"
echo "  1) none"
echo "  2) whisper"
echo "  3) parakeet"
read -rp "Enter choice [1-3]: " engine_choice

case "${engine_choice:-1}" in
2) FEATURES+=("whisper") ;;
3) FEATURES+=("parakeet") ;;
*) ;;
esac

# Hooks selection
echo ""
read -rp "Enable script execution hooks for dictation? [y/N]: " hooks_choice
if [[ "${hooks_choice:-n}" =~ ^[Yy]$ ]]; then
    FEATURES+=("hooks")
fi

# Build feature flag
if [ ${#FEATURES[@]} -eq 0 ]; then
    FEATURE_FLAG=()
    echo ""
    echo "Building with default features (CPU, moshi)..."
else
    FEATURE_STRING=$(IFS=,; echo "${FEATURES[*]}")
    FEATURE_FLAG=(--features "$FEATURE_STRING")
    echo ""
    echo "Building with features: $FEATURE_STRING"
fi

cargo build --release "${FEATURE_FLAG[@]}"

echo "Installing ears..."
cargo install --path "${ROOT_DIR}" --force "${FEATURE_FLAG[@]}"

echo ""
echo "Installation complete. Binaries are available in \$HOME/.cargo/bin."
echo "Installed binaries: ears, ears-server, ears-dictation"
