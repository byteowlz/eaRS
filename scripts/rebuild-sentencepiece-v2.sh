#!/bin/bash
# Rebuild sentencepiece against current protobuf
# This version installs dependencies first

set -e

echo "Rebuilding sentencepiece package..."
echo "===================================="
echo ""

echo "Step 1: Installing build dependencies..."
sudo pacman -S --needed python-build python-setuptools python-wheel python-installer python-pytest

echo ""
echo "Step 2: Downloading PKGBUILD..."
cd /tmp
rm -rf sentencepiece-build
mkdir sentencepiece-build
cd sentencepiece-build
yay -G sentencepiece

cd sentencepiece

echo ""
echo "Step 3: Building package (no sudo needed for this step)..."
makepkg -s --noconfirm

echo ""
echo "Step 4: Installing the built package..."
sudo pacman -U --noconfirm sentencepiece-*.pkg.tar.zst

echo ""
echo "✓ Done! sentencepiece has been rebuilt against protobuf 33"
echo ""
echo "Now you can run: cd ~/byteowlz/eaRS && just install-all"
