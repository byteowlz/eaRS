#!/usr/bin/env bash
# Generate longer audio files using Kokoro TTS for parallel transcription demo
# Creates 9 English audio files with varied voices at 1.3x speed

set -e

OUTPUT_DIR="test_audio/generated"
mkdir -p "$OUTPUT_DIR"

# Create temp directory for input files
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "=== Generating English Audio Files with Kokoro TTS ==="
echo "Output directory: $OUTPUT_DIR"
echo "Speed: 1.3x"
echo ""

# Text passages for each file (all English, longer for better demo)
TEXTS=(
    "Welcome to the parallel transcription demonstration. This is a test of the ears server with multiple concurrent audio streams. Each pane will show real time speech recognition processing different audio sources simultaneously. The kyutai model handles batched inference efficiently."
    "Hello and welcome to today's broadcast. We are testing a sophisticated speech recognition system that can process many audio streams at once. This technology uses advanced machine learning to transcribe speech in real time with high accuracy."
    "This is another test audio file for the transcription system. In production environments you might have multiple users streaming audio from different microphones. The server handles each stream independently while maximizing GPU utilization through batched processing."
    "Thank you for joining us on this demonstration of automatic speech recognition. Our system demonstrates the power of modern neural networks for converting spoken words into accurate text transcriptions."
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and serves as a classic pangram for testing typewriters and speech systems alike."
    "Artificial intelligence has revolutionized the field of speech recognition. Deep learning models can now understand spoken language with near human level accuracy across multiple languages and accents."
    "In this extended test passage we demonstrate the transcription capabilities of our system. The audio contains natural speech patterns with various sentence structures, pauses and intonations that challenge even the most advanced speech models."
    "Streaming audio in real time requires efficient processing and low latency communication protocols. Websockets provide a robust mechanism for bidirectional data exchange between clients and servers."
    "The server architecture uses batched inference to process multiple audio streams simultaneously. This approach maximizes GPU utilization and enables cost effective transcription services at scale."
)

# Voice styles for each file (varied English voices)
VOICES=(
    "af_sarah"
    "am_michael"
    "af_nicole"
    "af_bella"
    "am_eric"
    "af_aoede"
    "am_liam"
    "af_river"
    "am_onyx"
)

# Output filenames
OUTPUTS=(
    "demo_en_welcome.wav"
    "demo_en_broadcast.wav"
    "demo_en_production.wav"
    "demo_en_thankyou.wav"
    "demo_en_pangram.wav"
    "demo_en_ai.wav"
    "demo_en_extended.wav"
    "demo_en_streaming.wav"
    "demo_en_server.wav"
)

# Generate each audio file
for i in {0..8}; do
    text="${TEXTS[$i]}"
    voice="${VOICES[$i]}"
    output="${OUTPUTS[$i]}"
    output_path="$OUTPUT_DIR/$output"
    input_file="$TMP_DIR/input_$i.txt"

    echo "Generating: $output"
    echo "  Voice: $voice"

    # Write text to temp file
    echo "$text" > "$input_file"

    # Generate audio with koko at 1.3x speed
    koko -l en-us -s "$voice" --mono true -p 1.3 file --merge -o "$output_path" "$input_file"

    # Verify file exists
    if [[ -f "$output_path" ]]; then
        size=$(du -h "$output_path" | cut -f1)
        echo "  Created: $output_path ($size)"
    else
        echo "  ERROR: Failed to create $output_path"
    fi
    echo ""
done

echo "=== Generated Audio Files ==="
ls -lh "$OUTPUT_DIR"