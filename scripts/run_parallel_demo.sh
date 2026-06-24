#!/usr/bin/env bash
# Parallel transcription demo for ears server with kyutai model
# Creates a 3x3 tmux grid showing 9 concurrent transcription streams
#
# Usage:
#   ./scripts/run_parallel_demo.sh              # Default ws://localhost:8765
#   ./scripts/run_parallel_demo.sh ws://host:port
#
# Prerequisites:
#   - ears server running ( ears server start )
#   - uv installed
#   - tmux installed

set -e

SERVER_URL="${1:-ws://localhost:8765}"
SESSION="ears-parallel"

# Get project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Parallel Transcription Demo ==="
echo "Server: $SERVER_URL"
echo "Project: $PROJECT_ROOT"
echo ""

# Check dependencies
command -v tmux >/dev/null 2>&1 || { echo "tmux not found"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv not found"; exit 1; }

# Audio files and labels (9 panes) - English only
declare -A AUDIO_FILES=(
    [0]="test_audio/generated/demo_en_welcome.wav"
    [1]="test_audio/generated/demo_en_broadcast.wav"
    [2]="test_audio/generated/demo_en_production.wav"
    [3]="test_audio/generated/demo_en_thankyou.wav"
    [4]="test_audio/generated/demo_en_pangram.wav"
    [5]="test_audio/generated/demo_en_ai.wav"
    [6]="test_audio/generated/demo_en_extended.wav"
    [7]="test_audio/generated/demo_en_streaming.wav"
    [8]="test_audio/generated/demo_en_server.wav"
)

declare -A LABELS=(
    [0]="Welcome"
    [1]="Broadcast"
    [2]="Production"
    [3]="Thank You"
    [4]="Pangram"
    [5]="AI"
    [6]="Extended"
    [7]="Streaming"
    [8]="Server"
)

# Verify audio files exist
for i in {0..8}; do
    if [[ ! -f "${AUDIO_FILES[$i]}" ]]; then
        echo "ERROR: Missing audio: ${AUDIO_FILES[$i]}"
        echo "Run: ./scripts/generate_demo_audio.sh"
        exit 1
    fi
done

# Kill existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 0.3

# Create session with 9 panes (3x3 grid)
echo "Creating 3x3 tmux grid..."

tmux new-session -d -s "$SESSION" -n "Parallel Transcription" -x 240 -y 72
WINDOW_TARGET="$SESSION:$(tmux display-message -p -t "$SESSION:" '#I')"

# Force a usable detached window size so 9 panes can fit
# (especially important when the current client is small)
tmux set-option -t "$SESSION" window-size manual >/dev/null 2>&1 || true
tmux resize-window -t "$WINDOW_TARGET" -x 240 -y 72 >/dev/null 2>&1 || true

# Create 9 panes independent of base-index/pane-base-index settings
for _ in {1..8}; do
    if ! tmux split-window -t "$WINDOW_TARGET" >/dev/null 2>&1; then
        tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
        echo "ERROR: tmux could not create 9 panes (no space for new pane)."
        echo "Try one of these:"
        echo "  1) Enlarge your terminal, then rerun."
        echo "  2) Set smaller pane minimums: tmux set -g pane-border-status off"
        echo "  3) Run in a fresh terminal window with larger dimensions."
        exit 1
    fi
done
tmux select-layout -t "$WINDOW_TARGET" tiled

# Collect pane ids so indexing is robust
mapfile -t PANE_IDS < <(tmux list-panes -t "$WINDOW_TARGET" -F '#{pane_id}')

if [[ ${#PANE_IDS[@]} -ne 9 ]]; then
    echo "ERROR: expected 9 panes, got ${#PANE_IDS[@]}"
    exit 1
fi

# Send commands to each pane
for i in {0..8}; do
    audio="${AUDIO_FILES[$i]}"
    label="${LABELS[$i]}"
    pane="${PANE_IDS[$i]}"

    tmux select-pane -t "$pane" -T "$label"
    tmux send-keys -t "$pane" "cd $PROJECT_ROOT" C-m

    tmux send-keys -t "$pane" "uv run --with numpy --with websockets python -c \"
import asyncio, json, wave, numpy as np, websockets
async def stream():
    path = '$audio'
    with wave.open(path, 'r') as f:
        frames = f.getnframes()
        data = np.frombuffer(f.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0
    async with websockets.connect('$SERVER_URL') as ws:
        chunk = 4096
        for i in range(0, len(data), chunk):
            await ws.send(data[i:i+chunk].astype(np.float32).tobytes())
        await ws.send('stop')
        while True:
            msg = await ws.recv()
            j = json.loads(msg)
            if j.get('type') == 'word':
                print(j.get('word', ''), end=' ', flush=True)
            elif j.get('type') == 'final':
                print(f'\\nFINAL: {j.get(\"text\", \"\")}')
                break
            elif j.get('type') == 'error':
                print(f'ERROR: {j.get(\"message\")}')
                break
asyncio.run(stream())
\"" C-m

    sleep 0.2
done

echo ""
echo "=== Demo Ready ==="
echo "Session: $SESSION"
echo "Panes: 9 (3x3 grid)"
echo "Audio: 9 English files at 1.3x speed"
echo ""
echo "Commands:"
echo "  Attach:   tmux attach -t $SESSION"
echo "  Detach:   Ctrl-b d"
echo "  Kill:     tmux kill-session -t $SESSION"
echo ""

# Enter session (works both inside and outside tmux)
if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION"
else
    tmux attach -t "$SESSION"
fi