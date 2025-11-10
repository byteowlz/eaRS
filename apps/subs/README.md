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
- **Listener Mode**: Monitor transcriptions from other clients without sending audio

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

#### Active Mode (Default)

1. The application auto-connects to the eaRS WebSocket server on startup
2. Position the subtitle bar anywhere on your screen by clicking and dragging
3. Click the microphone icon to start/stop audio capture
4. Click the settings icon to:
   - Select audio input device (microphone or system audio loopback)
   - Adjust WebSocket port
   - Toggle always-on-top behavior
   - Check connection status
5. Subtitles appear in real-time as audio is transcribed

#### Listener Mode

To monitor transcriptions from another client without sending audio:

1. Click the settings icon to open configuration
2. Enable "Listener Mode" checkbox
3. Enter your authentication token (provided by server administrator)
4. Save and reconnect
5. Once authenticated, available streams will be listed
6. Select a stream to monitor
7. Transcriptions from the selected stream will appear in your subtitle bar

Note: Listener mode must be enabled on the eaRS server with valid tokens configured.

## Configuration

Click the settings icon (⚙️) in the top-right corner to access:

- **Connection Mode**: Toggle between Active (send audio) and Listener (receive only) modes
- **Listener Mode Settings** (when enabled):
  - Authentication token for secure access
  - Stream selection (choose which active stream to monitor)
- **Audio Input** (Active mode only): Select microphone or system audio device for capture
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
