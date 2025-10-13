# prkt - Parakeet Speech Recognition CLI

A fast, simple command-line tool for speech-to-text transcription using NVIDIA's Parakeet TDT model.

## Installation

```bash
# Build release version
cargo build --release --bin prkt

# Copy to PATH (optional)
cp target/release/prkt /usr/local/bin/

# With GPU support
cargo build --release --bin prkt --features coreml  # macOS
cargo build --release --bin prkt --features cuda    # NVIDIA GPU
```

## Quick Start

```bash
# Transcribe a file
prkt file audio.wav

# Live transcription from microphone
prkt live

# Pipe audio from ffmpeg
ffmpeg -i input.mp3 -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | prkt file -
```

## Commands

### `prkt file` - File Transcription

Transcribe audio files or piped audio data.

**Basic usage:**
```bash
prkt file audio.wav
```

**From stdin (pipe):**
```bash
# From ffmpeg
ffmpeg -i input.mp3 -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | prkt file -

# From sox
sox input.mp3 -t f32 -r 16000 -c 1 - | prkt file -

# From recorded audio
arecord -f FLOAT_LE -r 16000 -c 1 | prkt file -
```

**Options:**
- `<AUDIO_FILE>` - Path to WAV file, or `-` for stdin
- `--sample-rate <RATE>` - Sample rate of stdin audio (default: 16000)
- `--device <DEVICE>` - Use specific device: cpu, cuda, coreml, directml
- `--model <REPO>` - HuggingFace model repo (default: istupakov/parakeet-tdt-0.6b-v3-onnx)
- `--model-dir <PATH>` - Use local model directory

**Examples:**
```bash
# Transcribe with GPU
prkt file audio.wav --device coreml

# Use custom model
prkt file audio.wav --model my-org/my-parakeet-model

# Pipe from 48kHz audio
ffmpeg -i input.wav -f f32le -acodec pcm_f32le -ar 48000 -ac 1 - | \
  prkt file - --sample-rate 48000
```

### `prkt live` - Live Transcription

Real-time transcription from your microphone.

**Basic usage:**
```bash
prkt live
```

**Options:**
- `--device <DEVICE>` - Use specific device: cpu, cuda, coreml, directml
- `--chunk-seconds <N>` - Audio chunk size in seconds (default: 3)
- `--overlap-seconds <N>` - Overlap between chunks in seconds (default: 1)
- `--no-filter` - Disable filtering of redundant transcriptions from overlapping audio
- `--model <REPO>` - HuggingFace model repo
- `--model-dir <PATH>` - Use local model directory

**Examples:**
```bash
# With GPU for faster processing
prkt live --device coreml

# Larger chunks for better context
prkt live --chunk-seconds 5 --overlap-seconds 2

# Lower latency (but less context)
prkt live --chunk-seconds 2 --overlap-seconds 0

# Disable redundancy filtering (show all raw transcriptions)
prkt live --no-filter
```

**Redundancy Filtering:**

By default, `prkt live` filters out redundant text caused by overlapping audio chunks. This prevents repeated phrases from appearing in the output. Use `--no-filter` to disable this behavior and see all raw transcriptions.

**Output format:**
```
🎤 Live transcription started. Speak into your microphone...
[5.2s] This is a test transcription. (RTF: 25.4x, latency: 320ms)
[8.1s] Speaking more words now. (RTF: 27.1x, latency: 290ms)
```

## Piping Examples

### From FFmpeg

Convert any audio/video format to speech-to-text:

```bash
# From MP3
ffmpeg -i song.mp3 -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | prkt file -

# From video file
ffmpeg -i video.mp4 -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | prkt file -

# From YouTube (requires yt-dlp)
yt-dlp -f bestaudio -o - "https://youtube.com/watch?v=..." | \
  ffmpeg -i - -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
  prkt file -

# Extract specific time range
ffmpeg -ss 00:01:30 -t 00:00:30 -i input.mp3 \
  -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | prkt file -
```

### From SoX

```bash
# Convert any format
sox input.flac -t f32 -r 16000 -c 1 - | prkt file -

# From microphone (real-time)
rec -t f32 -r 16000 -c 1 - | prkt file -

# Apply noise reduction
sox input.wav -t f32 -r 16000 -c 1 - noisered noise-profile.txt 0.21 | \
  prkt file -
```

### From Recording Tools

```bash
# ALSA (Linux)
arecord -f FLOAT_LE -r 16000 -c 1 -d 10 | prkt file -

# macOS (requires sox)
rec -t f32 -r 16000 -c 1 - | prkt file -

# PulseAudio (Linux)
parec --format=float32le --rate=16000 --channels=1 | prkt file -
```

### Batch Processing

```bash
# Transcribe all WAV files in directory
for file in *.wav; do
  echo "Processing $file..."
  prkt file "$file" > "${file%.wav}.txt"
done

# Parallel processing (requires GNU parallel)
ls *.mp3 | parallel -j 4 \
  'ffmpeg -i {} -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
   prkt file - > {.}.txt'
```

## Audio Format Requirements

### For WAV files:
- Format: WAV (auto-detected)
- Sample rate: Any (auto-resampled to 16kHz)
- Channels: Mono or Stereo (auto-converted to mono)
- Bit depth: 16-bit or 32-bit float

### For piped input (stdin):
- Format: Raw f32le (32-bit float, little-endian)
- Sample rate: Specify with `--sample-rate` (default: 16000)
- Channels: Mono only
- **Important**: Must use `-f f32le` in ffmpeg/sox

## Performance Tips

### GPU Acceleration

```bash
# macOS (CoreML)
cargo build --release --bin prkt --features coreml
prkt file audio.wav --device coreml  # 50-100x RTF

# NVIDIA GPU (CUDA)
cargo build --release --bin prkt --features cuda
prkt file audio.wav --device cuda    # 80-120x RTF

# Windows (DirectML)
cargo build --release --bin prkt --features directml
prkt file audio.wav --device directml
```

### Optimize for Latency (Live)

```bash
# Smaller chunks = lower latency, less context
prkt live --chunk-seconds 2 --overlap-seconds 0

# Larger chunks = more context, higher latency
prkt live --chunk-seconds 5 --overlap-seconds 2
```

### Batch Processing

For processing multiple files, reuse the model loading:
```bash
# Instead of running prkt multiple times (slow)
for f in *.wav; do prkt file "$f"; done

# Better: use parallel processing
ls *.wav | parallel -j $(nproc) 'prkt file {}'
```

## Troubleshooting

### "No input device available" (live mode)
- Check system microphone permissions
- Verify microphone is connected and enabled
- On macOS: System Settings > Privacy & Security > Microphone

### "Invalid audio data" (stdin)
- Ensure you're using `-f f32le` in ffmpeg
- Check sample rate matches `--sample-rate` parameter
- Verify mono audio with `-ac 1`

### Poor transcription quality
- Use higher quality audio (lower noise)
- Increase chunk size for more context
- Use GPU for faster, more stable processing
- Ensure proper sample rate conversion

### Slow performance
- Use GPU acceleration (--device coreml/cuda)
- First run downloads ~2.5GB model (cached after)
- For live: reduce chunk size for lower latency

## Advanced Usage

### Custom Preprocessing Pipeline

```bash
# Noise reduction + normalization
ffmpeg -i noisy.mp3 \
  -af "highpass=f=200,lowpass=f=3000,volume=2" \
  -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
  prkt file -
```

### Multi-Language Support

The model supports 25 languages automatically:
```bash
# Works with any supported language
prkt file spanish.wav
prkt file german.wav
prkt file french.wav
```

### Streaming from Network

```bash
# From HTTP stream
curl -s http://example.com/stream.mp3 | \
  ffmpeg -i - -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
  prkt file -

# From RTSP stream
ffmpeg -rtsp_transport tcp -i rtsp://camera/stream \
  -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
  prkt file -
```

### Integration with Other Tools

```bash
# Save both audio and transcript
ffmpeg -i input.mp3 -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
  tee >(prkt file - > transcript.txt) | \
  ffmpeg -f f32le -ar 16000 -ac 1 -i - output.wav

# Live transcription with VAD (voice activity detection)
# Requires `silenceremove` filter
ffmpeg -f avfoundation -i ":0" \
  -af "silenceremove=start_periods=1:stop_periods=-1:detection=peak" \
  -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
  prkt file -
```

## Output Format

**File mode:**
```
Loading Parakeet model...
Loading audio file: "test.wav"
Audio loaded: 53278 samples at 16000 Hz (3.33 seconds)

Transcribing...

Transcription completed in 0.13s
RTF: 25.54x

This is the transcribed text.
```

**Live mode:**
```
Loading Parakeet model...
Using input device: MacBook Pro Microphone
Sample rate: 48000
Channels: 1

🎤 Live transcription started. Speak into your microphone...
Press Ctrl+C to stop.

[3.2s] Hello, this is a test. (RTF: 28.3x, latency: 120ms)
[6.8s] The quick brown fox jumps. (RTF: 26.1x, latency: 150ms)
```

## System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: ~2.5GB for model cache
- **CPU**: Any modern x86-64 or ARM64
- **GPU** (optional):
  - macOS: M1/M2/M3 (CoreML)
  - NVIDIA: GTX 1060 or newer (CUDA)
  - Windows: DirectML-compatible GPU

## Model Information

- **Model**: NVIDIA Parakeet TDT 0.6B v3
- **Languages**: 25 European languages
- **Vocabulary**: 8193 tokens (SentencePiece)
- **Architecture**: Conformer encoder + RNN-T decoder
- **License**: CC-BY-4.0

## See Also

- Main README: [README.md](README.md)
- Examples: [examples/README.md](examples/README.md)
- Implementation details: [SUMMARY.md](SUMMARY.md)
