#!/usr/bin/env bash
# Demo script to showcase parallel transcription with ears server
# Creates 9 tmux panes (3x3 grid) each streaming different audio to the server
#
# Usage:
#   ./parallel_tmux_demo.sh              # Default: ws://localhost:8765
#   ./parallel_tmux_demo.sh ws://host:port
#
# Prerequisites:
#   1. ears server must be running
#   2. Python with websockets: pip install websockets
#   3. ffmpeg installed
#   4. tmux installed

set -e

SERVER_URL="${1:-ws://localhost:8765/}"
SESSION_NAME="ears-parallel-demo"

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
    "EN Short"
    "EN Samantha"
    "EN VoxCPM"
    "DE Erwin"
    "DE Sattelberger"
    "EN Long"
    "Spanish"
    "German"
    "Italian"
)

echo "=== Parallel Transcription Demo ==="
echo "Server: $SERVER_URL"
echo "Session: $SESSION_NAME"
echo ""

# Check Python with websockets
if ! python3 -c "import websockets" 2>/dev/null; then
    echo "Installing websockets..."
    pip install websockets
fi

# Check if audio files exist
for i in "${!AUDIO_FILES[@]}"; do
    audio="${AUDIO_FILES[$i]}"
    if [[ ! -f "$audio" ]]; then
        echo "Warning: Audio file not found: $audio"
        unset 'AUDIO_FILES[i]'
        unset 'PANE_LABELS[i]'
    fi
done

# Reindex arrays
AUDIO_FILES=("${AUDIO_FILES[@]}")
PANE_LABELS=("${PANE_LABELS[@]}")

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing session..."
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    sleep 0.5
fi

# Create new session
echo "Creating tmux session with ${#AUDIO_FILES[@]} panes..."
tmux new-session -d -s "$SESSION_NAME" -x 180 -y 50

# Calculate grid dimensions
num_panes=${#AUDIO_FILES[@]}
cols=3
rows=$(( (num_panes + cols - 1) / cols ))

# Split into grid
for ((i=1; i<cols; i++)); do
    tmux split-window -h -t "$SESSION_NAME"
done

for ((i=0; i<cols; i++)); do
    for ((j=1; j<rows; j++)); do
        tmux select-pane -t "$SESSION_NAME:0.$i"
        tmux split-window -v -t "$SESSION_NAME:0.$i"
    done
done

# Set uniform layout
tmux select-layout -t "$SESSION_NAME" tiled

# Send streaming command to each pane
for i in "${!AUDIO_FILES[@]}"; do
    audio="${AUDIO_FILES[$i]}"
    label="${PANE_LABELS[$i]}"
    pane="$SESSION_NAME:0.$i"

    echo "Starting pane $i: $label -> $audio"

    # Send the Python streaming script
    tmux send-keys -t "$pane" "cd /Users/tommyfalkowski/byteowlz/eaRS" C-m
    tmux send-keys -t "$pane" "python3 scripts/parallel_stream.py -s '$SERVER_URL' '$audio'" C-m

    # Small delay between panes
    sleep 0.3
done

# Give panes meaningful titles
for i in "${!PANE_LABELS[@]}"; do
    tmux select-pane -t "$SESSION_NAME:0.$i" -T "${PANE_LABELS[$i]}"
done

# Set window title
tmux rename-window -t "$SESSION_NAME" "Parallel Transcription Demo"

echo ""
echo "=== Demo Started ==="
echo "Session: $SESSION_NAME"
echo "Panes: ${#AUDIO_FILES[@]}"
echo ""
echo "To attach and watch: tmux attach -t $SESSION_NAME"
echo "To detach: Ctrl-b d"
echo "To kill: tmux kill-session -t $SESSION_NAME"
echo ""

# Attach to session
tmux attach-session -t "$SESSION_NAME"