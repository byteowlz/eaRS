#!/usr/bin/env bash
# Demo script to showcase parallel transcription with ears server
# Creates 9 tmux panes (3x3 grid) each streaming different audio to the server

set -e

SERVER_URL="${1:-ws://localhost:8765/}"
SESSION_NAME="${2:-ears-parallel-demo}"

# Audio files to use (9 different clips)
AUDIO_FILES=(
    "test_audio/english_short_24k.wav"
    "test_audio/english_samantha_24k.wav"
    "test_audio/english_voxcpm2_24k.wav"
    "test_audio/erwin_24k.wav"
    "test_audio/sattelberger_24k.wav"
    "test_audio/english_voxcpm2_long_24k.wav"
    "ref_audio/esp.mp3"
    "ref_audio/ger.mp3"
    "ref_audio/ita.mp3"
)

# Labels for each pane
PANE_LABELS=(
    "English Short"
    "English Samantha"
    "English VoxCPM"
    "Erwin German"
    "Sattelberger German"
    "English Long"
    "Spanish"
    "German"
    "Italian"
)

echo "=== Parallel Transcription Demo ==="
echo "Server: $SERVER_URL"
echo "Session: $SESSION_NAME"
echo ""

# Check if audio files exist
for audio in "${AUDIO_FILES[@]}"; do
    if [[ ! -f "$audio" ]]; then
        echo "Error: Audio file not found: $audio"
        exit 1
    fi
done

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing session..."
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    sleep 1
fi

# Create new session with 3x3 grid layout
echo "Creating tmux session with 3x3 grid layout..."
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 60

# Split into 3x3 grid
# First split vertically into 3 panes
tmux split-window -h -t "$SESSION_NAME"
tmux split-window -h -t "$SESSION_NAME"

# Select left column and split horizontally into 3
tmux select-pane -t "$SESSION_NAME:0.0"
tmux split-window -v -t "$SESSION_NAME:0.0"
tmux split-window -v -t "$SESSION_NAME:0.0"

# Select middle column and split horizontally into 3
tmux select-pane -t "$SESSION_NAME:0.1"
tmux split-window -v -t "$SESSION_NAME:0.1"
tmux split-window -v -t "$SESSION_NAME:0.1"

# Select right column and split horizontally into 3
tmux select-pane -t "$SESSION_NAME:0.2"
tmux split-window -v -t "$SESSION_NAME:0.2"
tmux split-window -v -t "$SESSION_NAME:0.2"

# Set uniform layout
tmux select-layout -t "$SESSION_NAME" tiled

# Give panes meaningful names
for i in {0..8}; do
    tmux select-pane -t "$SESSION_NAME:0.$i" -T "${PANE_LABELS[$i]}"
done

# Now send the streaming commands to each pane
# We need to send audio in chunks because WebSocket expects streaming

echo "Starting parallel streams..."

# Helper function to stream audio to WebSocket
stream_audio() {
    local pane_idx=$1
    local audio_file=$2
    local label=$3
    local pane="$SESSION_NAME:0.$pane_idx"

    # Convert to 24kHz mono WAV if needed, then stream raw PCM
    echo "Pane $pane_idx: Streaming $audio_file ($label)"

    # Use ffmpeg to convert to raw PCM and stream via websocat
    # The server expects float32 samples at 24kHz
    tmux send-keys -t "$pane" "echo '=== $label ==='" C-m
    tmux send-keys -t "$pane" "ffmpeg -i '$audio_file' -ar 24000 -ac 1 -c:a pcm_s16le -f wav - 2>/dev/null | websocat -B 65536 '$SERVER_URL'" C-m
}

# Start streaming in each pane with slight delays to prevent overload
for i in {0..8}; do
    stream_audio "$i" "${AUDIO_FILES[$i]}" "${PANE_LABELS[$i]}"
    sleep 0.5
done

# Attach to the session
echo ""
echo "Starting tmux session. Press Ctrl+C to detach."
echo "You should see all 9 panes streaming audio and receiving transcriptions!"
echo ""
echo "To monitor the server logs separately, run:"
echo "  tail -f /tmp/ears-server.log"
echo ""

# Attach to session
tmux attach-session -t "$SESSION_NAME"