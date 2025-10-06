# WebSocket Implementation Guide

This guide explains how to integrate eaRS's real-time speech-to-text WebSocket implementation into your applications.

## Overview

eaRS provides a WebSocket server that streams real-time transcription results, allowing you to build applications that need live speech recognition capabilities. The implementation supports multiple concurrent connections, real-time word streaming, and session control commands.

Important: The WebSocket server starts in paused mode by default. Send `{ "type": "resume" }` to begin transcription. Language can be changed at runtime with `{ "type":"set_language", "lang":"de" }`.

## Quick Start

### Starting the WebSocket Server

```bash
# Basic WebSocket server on port 8080
ears --live --ws 8080

# With additional features
ears --live --ws 8080 --timestamps --vad --vad-timeout 3.0
```

### Connecting from JavaScript

```javascript
const ws = new WebSocket('ws://localhost:8080/');

ws.onopen = () => {
    console.log('Connected to eaRS WebSocket');
    
    // Resume transcription to start receiving messages
    ws.send(JSON.stringify({ "Resume": {} }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleTranscriptionMessage(message);
};

ws.onclose = () => {
    console.log('WebSocket connection closed');
};
```

## Message Types

### Incoming Messages (Server → Client)

Schema uses a tagged format: `{ "type": "...", ... }`.

#### 1. Word Messages
Sent for each transcribed word in real-time:

```
{
  "type": "word",
  "word": "hello",
  "start_time": 1.234,
  "end_time": 1.567
}
```

#### 2. Pause Messages
Sent when voice activity detection detects silence:

```
{
  "type": "pause",
  "timestamp": 2.345
}
```

#### 3. Final Messages
Sent at the end of utterances with complete text and word timings:

```
{
  "type": "final",
  "text": "hello world how are you",
  "words": [
    { "word": "hello", "start_time": 1.234, "end_time": 1.567 },
    { "word": "world", "start_time": 1.678, "end_time": 1.890 }
  ]
}
```

### Outgoing Commands (Client → Server)

Use the same tagged schema: `{ "type": "...", ... }`.

#### 1. Restart
```
{ "type": "restart" }
```

#### 2. Pause
```
{ "type": "pause" }
```

#### 3. Resume
```
{ "type": "resume" }
```

#### 4. Set Language
Prime model with language reference audio (ISO 639-1):
```
{ "type": "set_language", "lang": "de" }
```

#### 5. Get Status
```
{ "type": "get_status" }
```

#### 6. Set VAD Timeout
```
{ "type": "set_vad_timeout", "seconds": 3.0 }
```

#### 2. Pause Command
Temporarily pauses transcription:

```json
{
    "Pause": {}
}
```

#### 3. Resume Command
Resumes paused transcription (required to start transcription as server starts in paused mode):

```json
{
    "Resume": {}
}
```

## Implementation Examples

### JavaScript/TypeScript Client

```
class EarsWebSocketClient {
  constructor(port = 8080) {
    this.ws = new WebSocket(`ws://localhost:${port}/`);
    this.setupEventHandlers();
  }

  setupEventHandlers() {
    this.ws.onopen = () => {
      console.log('Connected to eaRS WebSocket');
      this.resume();
    };

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case 'word': this.onWord(msg); break;
        case 'pause': this.onPause(msg); break;
        case 'final': this.onFinal(msg); break;
        case 'language_changed': this.onLanguage(msg.lang); break;
        case 'status': this.onStatus(msg); break;
      }
    };
  }

  onWord({ word, start_time }) {
    console.log(`Word: ${word} (${start_time}s)`);
  }

  onPause({ timestamp }) {
    console.log(`Paused at ${timestamp}s`);
  }

  onFinal({ text }) {
    console.log(`Final: ${text}`);
  }

  restart() { this.ws.send(JSON.stringify({ type: 'restart' })); }
  pause() { this.ws.send(JSON.stringify({ type: 'pause' })); }
  resume() { this.ws.send(JSON.stringify({ type: 'resume' })); }
  setLanguage(lang) { this.ws.send(JSON.stringify({ type: 'set_language', lang })); }
  getStatus() { this.ws.send(JSON.stringify({ type: 'get_status' })); }
}

// Usage
const client = new EarsWebSocketClient(8080);
```

### Python Client

```python
import json
import asyncio
import websockets

class EarsWebSocketClient:
    def __init__(self, port=8080):
        self.uri = f"ws://localhost:{port}/"
        
    async def connect_and_listen(self):
        async with websockets.connect(self.uri) as websocket:
            # Start transcription by sending Resume command
            await websocket.send(json.dumps({"Resume": {}}))
            
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(data)
    
    async def handle_message(self, message):
        if "Word" in message:
            word_data = message["Word"]
            print(f"Word: {word_data['word']} ({word_data['start_time']}s)")
        elif "Pause" in message:
            pause_data = message["Pause"]
            print(f"Paused at {pause_data['timestamp']}s")
        elif "Final" in message:
            final_data = message["Final"]
            print(f"Final: {final_data['text']}")
    
    async def send_command(self, websocket, command):
        await websocket.send(json.dumps(command))

# Usage
async def main():
    client = EarsWebSocketClient(8080)
    await client.connect_and_listen()

asyncio.run(main())
```

### React Component Example

```tsx
import React, { useState, useEffect, useRef } from 'react';

interface WordData {
    word: string;
    start_time: number;
    end_time?: number;
}

interface TranscriptionMessage {
    Word?: WordData;
    Pause?: { timestamp: number };
    Final?: { text: string; words: WordData[] };
}

export const LiveTranscription: React.FC = () => {
    const [currentText, setCurrentText] = useState('');
    const [finalTexts, setFinalTexts] = useState<string[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8080/');
        wsRef.current = ws;

        ws.onopen = () => {
            setIsConnected(true);
            // Start transcription by sending Resume command
            ws.send(JSON.stringify({ Resume: {} }));
        };
        ws.onclose = () => setIsConnected(false);
        
        ws.onmessage = (event) => {
            const message: TranscriptionMessage = JSON.parse(event.data);
            
            if (message.Word) {
                setCurrentText(prev => prev + message.Word!.word + ' ');
            } else if (message.Pause) {
                // Handle pause if needed
            } else if (message.Final) {
                setFinalTexts(prev => [...prev, message.Final!.text]);
                setCurrentText('');
            }
        };

        return () => ws.close();
    }, []);

    const sendCommand = (command: object) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(command));
        }
    };

    return (
        <div>
            <div>Status: {isConnected ? 'Connected' : 'Disconnected'}</div>
            
            <div>
                <button onClick={() => sendCommand({ Restart: {} })}>Restart</button>
                <button onClick={() => sendCommand({ Pause: {} })}>Pause</button>
                <button onClick={() => sendCommand({ Resume: {} })}>Resume</button>
            </div>
            
            <div>
                <h3>Live Transcription:</h3>
                <p>{currentText}</p>
            </div>
            
            <div>
                <h3>Final Texts:</h3>
                {finalTexts.map((text, index) => (
                    <p key={index}>{text}</p>
                ))}
            </div>
        </div>
    );
};
```

## Server Configuration Options

When starting the eaRS WebSocket server, you can use these additional flags:

- `--timestamps`: Include word-level timestamps
- `--vad`: Enable Voice Activity Detection
- `--vad-timeout <seconds>`: Set VAD timeout (default: 2.0s)
- `--model <path>`: Specify custom model path
- `--device <device>`: Set device (auto, cpu, cuda, metal)

Example:
```bash
ears --live --ws 8080 --timestamps --vad --vad-timeout 3.0 --device cuda
```

## Error Handling

### Connection Management

```javascript
class RobustWebSocketClient {
    constructor(port, maxRetries = 5) {
        this.port = port;
        this.maxRetries = maxRetries;
        this.retryCount = 0;
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(`ws://localhost:${this.port}/`);
        
        this.ws.onopen = () => {
            console.log('Connected');
            this.retryCount = 0;
        };
        
        this.ws.onclose = () => {
            if (this.retryCount < this.maxRetries) {
                console.log(`Reconnecting... (${this.retryCount + 1}/${this.maxRetries})`);
                setTimeout(() => {
                    this.retryCount++;
                    this.connect();
                }, 1000 * Math.pow(2, this.retryCount));
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
}
```

## Performance Considerations

1. **Message Frequency**: Word messages are sent in real-time, which can be high-frequency. Consider throttling UI updates if needed.

2. **Connection Limits**: The server supports multiple concurrent connections, but consider your system's limits.

3. **Buffer Management**: For long sessions, implement proper buffer management to prevent memory leaks.

4. **Network Resilience**: Implement reconnection logic for production applications.

## Testing

Use the included `websocket_example.html` file for testing:

```bash
# Start the server
ears --live --ws 8080

# Open websocket_example.html in your browser
# The example provides a complete testing interface
```

## Integration Patterns

### Real-time Captioning
Stream words to display live captions with timing information.

### Voice Commands
Use final text messages to implement voice command recognition.

### Transcription Logging
Capture and store complete transcriptions using final messages.

### Interactive Applications
Use pause/resume commands to control transcription flow based on application state.

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure eaRS server is running with `--ws` flag
2. **No Audio**: Check microphone permissions and audio device settings
3. **High Latency**: Consider adjusting VAD timeout or using faster hardware
4. **Message Parsing Errors**: Ensure you're handling all message types properly

### Debug Mode

Run with verbose logging:
```bash
RUST_LOG=debug ears --live --ws 8080
```

For more information, see the main README.md file and the included `websocket_example.html` for a complete working example.