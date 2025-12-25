#!/bin/bash
# Rebuild sentencepiece against current protobuf
# Run this script in your terminal (not through an IDE/agent)

set -e

echo "Rebuilding sentencepiece package..."
echo "===================================="
echo ""

cd /tmp
rm -rf sentencepiece-build
mkdir sentencepiece-build
cd sentencepiece-build

echo "Downloading PKGBUILD..."
yay -G sentencepiece

cd sentencepiece

echo ""
echo "Building and installing sentencepiece..."
echo "You will be prompted for your sudo password."
makepkg -si

echo ""
echo "✓ Done! sentencepiece has been rebuilt against protobuf 33"
echo ""
echo "Now you can run: cd ~/byteowlz/eaRS && just install-all"
