#!/usr/bin/env bash
set -e

echo "eaRS Installation Script for Omarchy (Arch Linux + CUDA)"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Arch Linux
if ! command -v pacman &>/dev/null; then
  echo -e "${RED}Error: This script is designed for Arch Linux (pacman not found)${NC}"
  exit 1
fi

# Check if yay is installed
if ! command -v yay &>/dev/null; then
  echo -e "${YELLOW}yay AUR helper not found. Please install yay first:${NC}"
  echo "  git clone https://aur.archlinux.org/yay.git"
  echo "  cd yay && makepkg -si"
  exit 1
fi

echo -e "${GREEN}Step 1: Installing system dependencies...${NC}"
# Install required system packages
sudo pacman -S --needed cuda cmake alsa-lib opus xdotool

echo "System dependencies installed"

# Install sentencepiece from AUR
echo -e "${GREEN}Step 2: Installing sentencepiece from AUR...${NC}"
if ! pacman -Qi sentencepiece &>/dev/null; then
  yay -S --needed sentencepiece
else
  echo "sentencepiece already installed"
fi

# Check for NVIDIA GPU
echo -e "${GREEN}Step 3: Checking for NVIDIA GPU...${NC}"
if ! command -v nvidia-smi &>/dev/null; then
  echo -e "${YELLOW}Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed.${NC}"
else
  echo "NVIDIA GPU detected:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# Set up environment variables
echo -e "${GREEN}Step 4: Setting up environment variables...${NC}"
export PATH=/opt/cuda/bin:$PATH
export CUDA_ROOT=/opt/cuda
export CUDARC_CUDA_VERSION=12060
export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1

# Detect GPU compute capability
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  echo "Detected GPU: $GPU_NAME"

  # Set appropriate CUDA architecture based on common GPUs
  case "$GPU_NAME" in
  *"RTX 40"* | *"RTX 4"*)
    export CMAKE_CUDA_ARCHITECTURES=89
    echo "Setting CUDA architecture to 8.9 (Ada Lovelace)"
    ;;
  *"RTX 30"* | *"RTX 3"*)
    export CMAKE_CUDA_ARCHITECTURES=86
    echo "Setting CUDA architecture to 8.6 (Ampere)"
    ;;
  *"RTX 20"* | *"RTX 2"* | *"GTX 16"*)
    export CMAKE_CUDA_ARCHITECTURES=75
    echo "Setting CUDA architecture to 7.5 (Turing)"
    ;;
  *)
    echo -e "${YELLOW}Unknown GPU, using default CUDA architecture${NC}"
    ;;
  esac
fi

# Build with CUDA support
echo -e "${GREEN}Step 5: Building eaRS with CUDA support...${NC}"
cargo install path . --features cuda

# Check if build was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Build successful!${NC}"

  # Install binary
  echo -e "${GREEN}Step 6: Installing binary...${NC}"
  sudo cp target/release/ears /usr/local/bin/
  sudo cp target/release/ears-server /usr/local/bin/
  sudo cp target/release/ears-dictation /usr/local/bin/

  echo -e "${GREEN}Installation complete!${NC}"
  echo ""
  echo "Binaries installed to:"
  echo "  - /usr/local/bin/ears"
  echo "  - /usr/local/bin/ears-server"
  echo "  - /usr/local/bin/ears-dictation"
  echo ""
  echo "To use eaRS, run: ears --help"
  echo ""
  echo -e "${YELLOW}Note: To build again in the future, set these environment variables:${NC}"
  echo "  export PATH=/opt/cuda/bin:\$PATH"
  echo "  export CUDA_ROOT=/opt/cuda"
  echo "  export CUDARC_CUDA_VERSION=12060"
  echo "  export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1"
else
  echo -e "${RED}Build failed. Please check the error messages above.${NC}"
  exit 1
fi
