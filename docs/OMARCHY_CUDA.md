# Installing eaRS on Omarchy (Arch Linux)

This guide is specifically for building eaRS on Arch Linux with NVIDIA CUDA support.

## Prerequisites

- Arch Linux (omarchy)
- NVIDIA GPU with drivers installed
- `yay` AUR helper

## Quick Install

Run the automated install script:

```bash
./scripts/install-omarchy.sh
```

This script will:
1. Install CUDA toolkit (if needed)
2. Install cmake (if needed)
3. Install sentencepiece from AUR
4. Detect your GPU and set appropriate CUDA architecture
5. Build eaRS with CUDA support
6. Install binaries to `/usr/local/bin/`

## Manual Installation

If you prefer to install manually:

### 1. Install Dependencies

```bash
# Install CUDA toolkit
sudo pacman -S cuda cmake

# Install sentencepiece from AUR
yay -S sentencepiece
```

### 2. Set Environment Variables

```bash
export PATH=/opt/cuda/bin:$PATH
export CUDA_ROOT=/opt/cuda
export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1

# Optional: Set CUDA architecture for your GPU
# RTX 40 series (Ada Lovelace):
export CMAKE_CUDA_ARCHITECTURES=89

# RTX 30 series (Ampere):
export CMAKE_CUDA_ARCHITECTURES=86

# RTX 20 series / GTX 16 series (Turing):
export CMAKE_CUDA_ARCHITECTURES=75
```

### 3. Build

```bash
cargo build --release --features cuda
```

### 4. Install

```bash
sudo cp target/release/ears /usr/local/bin/
sudo cp target/release/ears-server /usr/local/bin/
sudo cp target/release/ears-dictation /usr/local/bin/
```

## Troubleshooting

### "nvcc not found" Error

Make sure CUDA is in your PATH:
```bash
export PATH=/opt/cuda/bin:$PATH
which nvcc  # Should show /opt/cuda/bin/nvcc
```

### "Unsupported gpu architecture" Error

This means your GPU architecture isn't compatible with the default settings. Set the correct architecture:

```bash
# Check your GPU compute capability at: https://developer.nvidia.com/cuda-gpus
export CMAKE_CUDA_ARCHITECTURES=<your_compute_capability>
```

### GCC 15 Compatibility Issues

If you encounter compilation errors with sentencepiece, make sure you have:
1. Installed sentencepiece from AUR: `yay -S sentencepiece`
2. Set the environment variable: `export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1`

### Build Environment Persistence

To make these environment variables permanent, add them to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export PATH=/opt/cuda/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_ROOT=/opt/cuda' >> ~/.bashrc
echo 'export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1' >> ~/.bashrc
```

### "Unsupported cuda toolkit version" Error

cudarc's build script recognizes a fixed list of CUDA toolkit versions, and each
cudarc release only knows about toolkits that existed when it shipped. eaRS
requires cudarc >= 0.19.9 (the first release supporting CUDA 13.3) and pins it to
match candle-core's own `cudarc ^0.19` requirement, so a current toolkit is
detected automatically.

If you hit this error on a very new toolkit, upgrade cudarc rather than forcing an
older ABI:

```bash
cargo update -p cudarc
```

As a last resort you can still pin the version cudarc should target with
`CUDARC_CUDA_VERSION` (e.g. `13030` for CUDA 13.3), but note this selects the
binding set compiled in, not a compatibility shim -- prefer letting it auto-detect.

## Usage

After installation, you can use eaRS:

```bash
# Basic usage
ears

# Run the WebSocket server
ears-server

# Run dictation mode
ears-dictation
```

For more information, see the main [README.md](README.md).
