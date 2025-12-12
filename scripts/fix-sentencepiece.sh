#!/usr/bin/env bash
set -euo pipefail

echo "Fixing sentencepiece protobuf dependency"
echo "========================================="
echo ""

# Check if sentencepiece is installed
if ! pacman -Qi sentencepiece &>/dev/null; then
    echo "❌ sentencepiece is not installed"
    exit 1
fi

# Check if yay is available
if ! command -v yay &>/dev/null; then
    echo "❌ yay is not available. Please install yay first."
    exit 1
fi

echo "Rebuilding sentencepiece against current protobuf..."
yay -S sentencepiece --rebuild

echo ""
echo "✓ sentencepiece has been rebuilt"
echo "You can now run: just install-all"
