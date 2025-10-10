# LiveSubs

An always-on-top floating subtitle bar for live transcription from eaRS. Displays real-time subtitles from any system audio source via WebSocket connection.

## Features

- Always-on-top floating window for subtitles
- Transparent, compact subtitle bar design
- Draggable window positioning
- Auto-connects to eaRS WebSocket backend
- Minimal, distraction-free interface
- Settings panel for configuration
- System audio source support (via eaRS backend)

## Prerequisites

- bun package manager
- Rust toolchain
- eaRS daemon running with WebSocket server enabled

## Installation

```bash
bun install
```

## Usage

### 1. Start the eaRS WebSocket Server

First, start the eaRS WebSocket server:

```bash
ears start server
```

### 2. Run the Application

For development:
```bash
bun run tauri dev
```

For production build:
```bash
bun run tauri build
```

### 3. Using LiveSubs

1. The application auto-connects to the eaRS WebSocket server on startup
2. Position the subtitle bar anywhere on your screen by clicking and dragging
3. Click the microphone icon to start/stop audio capture
4. Click the settings icon to:
   - Select audio input device (microphone or system audio loopback)
   - Adjust WebSocket port
   - Toggle always-on-top behavior
   - Check connection status
5. Subtitles appear in real-time as audio is transcribed

## Configuration

Click the settings icon (⚙️) in the top-right corner to access:

- **Audio Input**: Select microphone or system audio device for capture
- **WebSocket Port**: Configure the eaRS server port (default: 8765)
- **Always on Top**: Toggle whether the window stays above other applications
- **Connection Status**: View real-time connection and capture status

### System Audio Capture

To capture system audio (desktop audio, application audio, etc.):

- **macOS**: Use BlackHole or similar virtual audio device to route system audio
- **Linux**: Use PulseAudio monitor devices or JACK audio routing
- **Windows**: Use stereo mix or virtual audio cable

Select the appropriate loopback device in the Audio Input dropdown.

## Window Controls

- **Drag**: Click and hold anywhere on the subtitle bar to reposition
- **Settings**: Click the gear icon to open the settings panel
- **Resize**: Window is resizable via standard OS window controls

## WebSocket Messages

The application receives transcription data from eaRS:

### Incoming (Server → Client)

- `{ "type": "word", "word": "...", "start_time": ... }` - Individual words as they're transcribed
- `{ "type": "final", "text": "...", ... }` - Complete utterances (displayed for 3 seconds)

## Technology Stack

- Tauri 2.0 (Desktop application framework)
- React 19
- TypeScript
- Vite
- Tailwind CSS 4
- shadcn/ui components
- Lucide React icons

## Building

Development:
```bash
bun run tauri dev
```

Production build:
```bash
bun run tauri build
```

Built applications will be in `src-tauri/target/release/`.

## License

Same as eaRS project
