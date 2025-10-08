# eaRS Live Transcription UI

A modern, real-time speech-to-text web interface for eaRS, featuring live audio visualization and transcription display.

## Features

- Real-time waveform visualization that responds to transcription activity
- Live transcription display with WebSocket streaming
- Clean, responsive UI built with shadcn/ui components
- Dark mode support with theme toggle (Light/Dark/System)
- Pause/Resume/Restart controls
- Transcript history with automatic scrolling
- Connection status indicators
- No microphone access required (backend handles audio capture)

## Prerequisites

- Node.js 18 or later
- bun package manager
- eaRS daemon running with WebSocket server enabled

## Installation

```bash
bun install
```

## Usage

### 1. Start the eaRS WebSocket Server

First, start the eaRS WebSocket server:

```bash
ears start sever
```

### 2. Run the Development Server

```bash
bun dev
```

The application will be available at `http://localhost:5173`

### 3. Connect and Transcribe

1. Open the application in your browser
2. Enter the WebSocket port (default: 8765)
3. Click "Connect"
4. Click "Start Listening" to begin transcription
5. Speak into your microphone to see live transcription

## Building for Production

```bash
bun run build
```

The built files will be in the `dist/` directory.

## Preview Production Build

```bash
bun run preview
```

## Components

### Transcription Waveform

Real-time visualization that responds to transcription activity from the backend. Shows different states:

- Active: Dynamic waveform that pulses when words are being transcribed
- Processing: Gentle animated wave pattern when paused but connected
- Idle: Dotted line when disconnected

The waveform visualizes transcription activity without accessing the microphone (the backend eaRS server handles audio capture). The animation intensity increases when words are being transcribed and gradually fades during silence.

### Transcription Display

Two sections:

- Live Transcription: Shows the current utterance being transcribed
- Transcript History: Shows completed transcriptions

### Controls

- Connect/Disconnect: Manage WebSocket connection
- Start/Stop Listening: Control transcription state
- Restart: Clear session and restart transcription
- Clear: Clear transcript history

## WebSocket Messages

The UI communicates with eaRS using these message types:

### Outgoing (Client → Server)

- `{ "type": "resume" }` - Start transcription
- `{ "type": "pause" }` - Pause transcription
- `{ "type": "restart" }` - Restart session

### Incoming (Server → Client)

- `{ "type": "word", "word": "...", "start_time": ... }` - Individual words
- `{ "type": "final", "text": "...", "words": [...] }` - Complete utterances
- `{ "type": "status", "paused": true/false, ... }` - Status updates

## Technology Stack

- React 19
- TypeScript
- Vite
- Tailwind CSS 4
- shadcn/ui components
- ElevenLabs UI (LiveWaveform component)
- Lucide React icons

## License

Same as eaRS project
