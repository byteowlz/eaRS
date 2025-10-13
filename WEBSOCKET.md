# WebSocket Implementation Guide

This guide explains how to integrate eaRS's real-time speech-to-text WebSocket server into your applications.

## Overview

The eaRS WebSocket server accepts audio streams and returns real-time transcription results. External clients can connect to stream audio and receive word-by-word transcription with optional timestamps and voice activity detection.

**Important**: The server is designed for single-client audio streaming. While multiple clients can connect for monitoring, only one client should send audio data at a time.

## Quick Start

### Starting the Server

```bash
# Start the eaRS server (runs as a background daemon)
ears server start

# Or start with specific options
ears server start --bind 0.0.0.0:8765 --timestamps --vad
```

The server binds to the WebSocket port specified in your config file (default: 8765).

## Connection Flow

```
Client                          Server
  |                               |
  |-- WebSocket Connect --------->|
  |<-- Connection Accepted -------|
  |                               |
  |-- Binary Audio Frame -------->|
  |-- Binary Audio Frame -------->|
  |                               | [Processing audio...]
  |<-- {"type":"word",...} -------|
  |-- Binary Audio Frame -------->|
  |<-- {"type":"word",...} -------|
  |-- Binary Audio Frame -------->|
  |<-- {"type":"final",...} ------|
  |                               |
  |-- "stop" -------------------->|
  |<-- Connection Close ----------|
```

1. **Connect**: Client establishes WebSocket connection
2. **Stream**: Client continuously sends binary audio frames
3. **Receive**: Server sends transcription messages as JSON
4. **Stop**: Client sends stop command (optional)
5. **Close**: Connection closes naturally or via stop command

### Connecting and Streaming Audio

The server expects:

1. **Binary audio data**: Raw PCM audio as f32 samples (little-endian, 4 bytes per sample)
2. **Audio format**: 24kHz sample rate, mono channel
3. **Text commands**: JSON control messages (optional)

```javascript
const ws = new WebSocket('ws://localhost:8765/');
const audioContext = new AudioContext({ sampleRate: 24000 });

ws.onopen = async () => {
    console.log('Connected to eaRS server');
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    processor.onaudioprocess = (e) => {
        const audioData = e.inputBuffer.getChannelData(0);
        const buffer = new ArrayBuffer(audioData.length * 4);
        const view = new Float32Array(buffer);
        view.set(audioData);
        ws.send(buffer);
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};

ws.onclose = () => {
    console.log('Connection closed');
    audioContext.close();
};
```

## Protocol Specification

### Client → Server Messages

The server accepts two types of WebSocket messages:

#### 1. Binary Audio Data (Primary)

Send raw PCM audio samples as **binary WebSocket frames**:

| Property | Value |
|----------|-------|
| Format | f32 (IEEE 754 32-bit float) |
| Byte Order | Little-endian |
| Sample Rate | 24,000 Hz |
| Channels | Mono (1 channel) |
| Range | -1.0 to 1.0 |

**Binary Encoding Examples:**

JavaScript (Web Audio API):

```javascript
// From AudioBuffer
const samples = audioBuffer.getChannelData(0);  // Float32Array
websocket.send(samples.buffer);  // Send as ArrayBuffer
```

Python (struct):

```python
import struct
samples = [0.1, -0.2, 0.3, ...]  # Your audio samples
audio_bytes = struct.pack(f'<{len(samples)}f', *samples)
await websocket.send(audio_bytes)
```

Python (numpy):

```python
import numpy as np
samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
await websocket.send(samples.tobytes())
```

Rust:

```rust
let samples: Vec<f32> = vec![0.1, -0.2, 0.3];
let bytes: Vec<u8> = samples.iter()
    .flat_map(|&f| f.to_le_bytes())
    .collect();
websocket.send(Message::Binary(bytes)).await?;
```

**Important**: Each f32 sample is exactly 4 bytes. For 1000 samples, send exactly 4000 bytes.

#### 2. Control Commands (Text, Optional)

Send JSON commands as **text WebSocket frames**:

**Stop transcription:**

```json
{ "type": "stop" }
```

Or the legacy plain text format:

```
"stop"
```

Both formats are accepted. The server will close the connection after processing the stop command.

**Change language:**

```json
{ "type": "setlanguage", "lang": "en" }
```

Dynamically switches the transcription language during an active session. Common language codes:
- `en` - English
- `fr` - French
- `de` - German
- `es` - Spanish
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese

The language change takes effect immediately for subsequent audio.

### Server → Client Messages

All messages are JSON with a `type` field:

#### word

Sent for each transcribed word:

```json
{
  "type": "word",
  "word": "hello",
  "start_time": 1.234,
  "end_time": 1.567
}
```

`end_time` may be `null` for words still being processed.

#### final

Sent when transcription completes:

```json
{
  "type": "final",
  "text": "complete transcribed text",
  "words": [
    { "word": "hello", "start_time": 1.234, "end_time": 1.567 },
    { "word": "world", "start_time": 1.678, "end_time": 1.890 }
  ]
}
```

#### error

Sent when an error occurs:

```json
{
  "type": "error",
  "message": "error description"
}
```

Common errors:

- `"server busy"` - Another client is already streaming audio

#### whisper_processing (optional)

Sent when Whisper enhancement starts processing a sentence (requires `--whisper` flag):

```json
{
  "type": "whisper_processing",
  "sentence_id": "uuid",
  "original_text": "text from Kyutai model",
  "start_time": 1.0,
  "end_time": 3.5
}
```

#### whisper_complete (optional)

Sent when Whisper enhancement completes:

```json
{
  "type": "whisper_complete",
  "sentence_id": "uuid",
  "original_text": "original text",
  "corrected_text": "enhanced text",
  "confidence": 0.95,
  "changed": true
}
```

## Implementation Examples

### JavaScript Client with Audio Streaming

```javascript
class EarsClient {
  constructor(serverUrl = 'ws://localhost:8765/') {
    this.serverUrl = serverUrl;
    this.ws = null;
    this.audioContext = null;
    this.processor = null;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.serverUrl);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = () => {
        console.log('Connected to eaRS server');
        resolve();
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
      
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        this.handleMessage(msg);
      };
      
      this.ws.onclose = () => {
        console.log('Connection closed');
        this.stopAudio();
      };
    });
  }

  async startAudio() {
    this.audioContext = new AudioContext({ sampleRate: 24000 });
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        channelCount: 1,
        sampleRate: 24000
      } 
    });
    
    const source = this.audioContext.createMediaStreamSource(stream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    
    this.processor.onaudioprocess = (e) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        const samples = e.inputBuffer.getChannelData(0);
        const buffer = new Float32Array(samples);
        this.ws.send(buffer.buffer);
      }
    };
    
    source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
  }

  handleMessage(msg) {
    switch (msg.type) {
      case 'word':
        console.log(`[${msg.start_time.toFixed(2)}s] ${msg.word}`);
        break;
      case 'final':
        console.log(`\nFinal transcript: ${msg.text}`);
        break;
      case 'error':
        console.error(`Server error: ${msg.message}`);
        break;
      case 'whisper_complete':
        if (msg.changed) {
          console.log(`Whisper: "${msg.original_text}" → "${msg.corrected_text}"`);
        }
        break;
    }
  }

  setLanguage(lang) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "setlanguage", lang }));
    }
  }

  stop() {
    if (this.ws) {
      this.ws.send(JSON.stringify({ type: "stop" }));
      this.ws.close();
    }
    this.stopAudio();
  }

  stopAudio() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// Usage
const client = new EarsClient('ws://localhost:8765/');
await client.connect();
await client.startAudio();

// Change language during transcription:
// client.setLanguage('fr');  // Switch to French
// client.setLanguage('es');  // Switch to Spanish

// When done:
// client.stop();
```

### Python Client with Audio Streaming

```python
import asyncio
import json
import struct
import numpy as np
import sounddevice as sd
import websockets

class EarsClient:
    def __init__(self, server_url='ws://localhost:8765/'):
        self.server_url = server_url
        self.websocket = None
        self.stream = None
        self.sample_rate = 24000
        
    async def connect(self):
        self.websocket = await websockets.connect(self.server_url)
        print('Connected to eaRS server')
        
    async def stream_audio(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(f'Audio status: {status}')
            samples = indata[:, 0]
            audio_bytes = struct.pack(f'<{len(samples)}f', *samples)
            asyncio.create_task(self.websocket.send(audio_bytes))
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=audio_callback
        )
        self.stream.start()
        
    async def listen_for_messages(self):
        try:
            async for message in self.websocket:
                msg = json.loads(message)
                self.handle_message(msg)
        except websockets.exceptions.ConnectionClosed:
            print('Connection closed')
            
    def handle_message(self, msg):
        msg_type = msg.get('type')
        if msg_type == 'word':
            print(f"[{msg['start_time']:.2f}s] {msg['word']}")
        elif msg_type == 'final':
            print(f"\nFinal: {msg['text']}")
        elif msg_type == 'error':
            print(f"Error: {msg['message']}")
        elif msg_type == 'whisper_complete' and msg.get('changed'):
            print(f"Whisper: '{msg['original_text']}' → '{msg['corrected_text']}'")
    
    async def set_language(self, lang):
        if self.websocket:
            await self.websocket.send(json.dumps({'type': 'setlanguage', 'lang': lang}))
    
    async def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.websocket:
            await self.websocket.send('stop')
            await self.websocket.close()

async def main():
    client = EarsClient('ws://localhost:8765/')
    await client.connect()
    await client.stream_audio()
    
    # Change language during transcription:
    # await client.set_language('fr')  # Switch to French
    # await client.set_language('ja')  # Switch to Japanese
    
    try:
        await client.listen_for_messages()
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        await client.stop()

if __name__ == '__main__':
    asyncio.run(main())
```

**Dependencies**: `pip install websockets sounddevice numpy`

### Rust Client Example

```rust
use anyhow::Result;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ServerMessage {
    Word { word: String, start_time: f64, end_time: Option<f64> },
    Final { text: String, words: Vec<WordInfo> },
    Error { message: String },
}

#[derive(Debug, Deserialize, Serialize)]
struct WordInfo {
    word: String,
    start_time: f64,
    end_time: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let (ws_stream, _) = connect_async("ws://localhost:8765/").await?;
    let (mut write, mut read) = ws_stream.split();
    
    let read_task = tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(msg) = serde_json::from_str::<ServerMessage>(&text) {
                        match msg {
                            ServerMessage::Word { word, start_time, .. } => {
                                println!("[{:.2}s] {}", start_time, word);
                            }
                            ServerMessage::Final { text, .. } => {
                                println!("\nFinal: {}", text);
                            }
                            ServerMessage::Error { message } => {
                                eprintln!("Error: {}", message);
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });
    
    let write_task = tokio::spawn(async move {
        let (audio_tx, audio_rx) = crossbeam_channel::unbounded();
        
        std::thread::spawn(move || {
            ears::audio::start_audio_capture(audio_tx, None)
        });
        
        while let Ok(chunk) = audio_rx.recv() {
            let bytes: Vec<u8> = chunk.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            if write.send(Message::Binary(bytes)).await.is_err() {
                break;
            }
        }
    });
    
    tokio::try_join!(read_task, write_task)?;
    Ok(())
}
```

## Server Configuration

The server is configured via `~/.config/ears/config.toml` and command-line flags:

```bash
ears server start [OPTIONS]
```

Options:

- `--bind <addr>`: Bind address (default: `0.0.0.0:<config-port>`)
- `--hf-repo <repo>`: Hugging Face model repository (default: `kyutai/stt-1b-en_fr-candle`)
- `--cpu`: Force CPU execution (otherwise uses CUDA/Metal when available)
- `--timestamps`: Include word-level timestamps
- `--vad`: Enable Voice Activity Detection
- `--whisper`: Enable Whisper post-processing (requires `--features whisper` at build time)

Example:

```bash
ears server start --bind 0.0.0.0:8765 --timestamps --vad
```

Check server status:

```bash
ears server status
```

Stop the server:

```bash
ears server stop
```

## Audio Requirements

### Format Specification

The server expects raw PCM audio with these exact specifications:

| Parameter | Value |
|-----------|-------|
| Format | IEEE 754 32-bit float (f32) |
| Byte Order | Little-endian |
| Sample Rate | 24,000 Hz |
| Channels | Mono (1 channel) |
| Range | -1.0 to 1.0 |

### Resampling

If your audio source uses a different sample rate, you must resample to 24kHz before sending. Most audio libraries provide resampling functions:

**JavaScript (Web Audio API)**:

```javascript
const audioContext = new AudioContext({ sampleRate: 24000 });
```

**Python (librosa)**:

```python
import librosa
audio = librosa.resample(audio, orig_sr=48000, target_sr=24000)
```

**FFmpeg**:

```bash
ffmpeg -i input.wav -ar 24000 -ac 1 -f f32le output.raw
```

### Channel Conversion

If your audio has multiple channels, convert to mono by averaging:

```javascript
const mono = [];
for (let i = 0; i < stereo.length; i += 2) {
    mono.push((stereo[i] + stereo[i + 1]) / 2);
}
```

## Error Handling

### Connection Management

```javascript
class RobustEarsClient {
    constructor(serverUrl, maxRetries = 5) {
        this.serverUrl = serverUrl;
        this.maxRetries = maxRetries;
        this.retryCount = 0;
    }

    async connect() {
        try {
            this.ws = new WebSocket(this.serverUrl);
            
            this.ws.onopen = () => {
                console.log('Connected');
                this.retryCount = 0;
            };
            
            this.ws.onclose = () => {
                if (this.retryCount < this.maxRetries) {
                    const delay = 1000 * Math.pow(2, this.retryCount);
                    console.log(`Reconnecting in ${delay}ms...`);
                    setTimeout(() => {
                        this.retryCount++;
                        this.connect();
                    }, delay);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            this.ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'error') {
                    console.error('Server error:', msg.message);
                    if (msg.message === 'server busy') {
                        console.log('Another client is already connected');
                    }
                }
            };
        } catch (error) {
            console.error('Connection failed:', error);
        }
    }
}
```

## Performance Considerations

### Single Client Design

The server is designed for **one active audio streaming client** at a time. While multiple clients can connect to monitor transcription results, only one should send audio data. Attempting to stream from multiple clients simultaneously will result in mixed audio and poor transcription quality.

### Message Frequency

Word messages are sent in real-time as recognition occurs. For UI applications, consider:

- Throttling display updates to reduce rendering overhead
- Using `requestAnimationFrame` for smooth visual updates
- Buffering words before appending to the DOM

### Audio Chunking

Send audio in reasonable chunk sizes:

- **Too small** (< 1KB): High overhead from many WebSocket frames
- **Too large** (> 64KB): Increased latency before server receives data
- **Recommended**: 4-16KB chunks (about 1000-4000 samples at f32)

### Network Considerations

- Use low-latency networks for best results
- On high-latency connections, expect delayed transcription
- Consider local server deployment for real-time applications

### Resource Usage

The server performs inference on each audio chunk:

- GPU acceleration significantly reduces latency
- CPU-only mode may introduce delays
- Each connection requires ~2-4GB RAM depending on the model

## Testing Your Client

### Test with the Built-in Client

The eaRS CLI includes a reference client you can use to verify the server is working:

```bash
ears server start --timestamps
ears  # In another terminal - streams audio and prints results
```

### Simple Test Script

Test the WebSocket connection without audio:

```bash
# Install wscat: npm install -g wscat
wscat -c ws://localhost:8765/

# The connection should open successfully
# Send: "stop"
# Server will close the connection
```

### Audio Format Validation

Verify your audio encoding is correct:

```python
import struct
samples = [0.0, 0.1, -0.1, 0.5]  # 4 samples
audio_bytes = struct.pack(f'<{len(samples)}f', *samples)
print(f"Bytes: {audio_bytes.hex()}")
print(f"Length: {len(audio_bytes)} (should be {len(samples) * 4})")
```

Expected output: 8 bytes per 2 samples (4 bytes per f32)

## Troubleshooting

### Connection Issues

**"Connection refused"**

- Ensure server is running: `ears server status`
- Check the configured port in `~/.config/ears/config.toml`
- Verify firewall settings if connecting remotely

**"Server busy"**

- Another client is already streaming audio
- Only one client can stream at a time
- Stop the other client or wait for the session to end

### Audio Issues

**No transcription output**

- Verify audio format: 24kHz, mono, f32 little-endian
- Check audio is being sent as binary WebSocket frames
- Ensure samples are in range [-1.0, 1.0]
- Test with: `ears` (built-in client) to verify server works

**Poor transcription quality**

- Verify sample rate is exactly 24,000 Hz
- Check for audio clipping (samples outside [-1.0, 1.0])
- Ensure mono conversion is done correctly for stereo sources
- Use a quality microphone with minimal background noise

**High latency**

- Use GPU acceleration (CUDA/Metal) instead of CPU
- Reduce network latency (run server and client on same machine)
- Check system load - high CPU usage degrades performance

### Message Format Issues

**"Message parsing error"**

- Ensure JSON uses lowercase `type` field: `{"type":"word",...}`
- Don't use old PascalCase format: `{"Word":{...}}`
- Validate JSON syntax with a linter

**Unexpected message types**

- Check if Whisper is enabled (`--whisper` flag)
- Whisper adds `whisper_processing` and `whisper_complete` messages
- These are optional and can be ignored if not using Whisper

### Debug Mode

Enable verbose logging:

```bash
RUST_LOG=debug ears server start
```

View server logs:

```bash
journalctl -u ears-server -f  # If using systemd
# or check stderr output when running manually
```

### Getting Help

If issues persist:

1. Verify your audio format with the validation script above
2. Test with the built-in `ears` client to isolate client vs. server issues
3. Check the repository documentation and examples
4. Include server logs and client code when reporting issues

## Quick Reference

### Audio Format

```
Format:       f32 (IEEE 754 32-bit float, little-endian)
Sample Rate:  24,000 Hz
Channels:     Mono (1)
Range:        -1.0 to 1.0
Frame Type:   Binary WebSocket message
```

### Server Message Types

| Type | Description | Fields |
|------|-------------|--------|
| `word` | Real-time word recognition | `word`, `start_time`, `end_time` |
| `final` | Complete transcription | `text`, `words[]` |
| `error` | Error occurred | `message` |
| `whisper_processing` | Whisper enhancement started | `sentence_id`, `original_text`, `start_time`, `end_time` |
| `whisper_complete` | Whisper enhancement finished | `sentence_id`, `original_text`, `corrected_text`, `confidence`, `changed` |

### Client Commands

| Command | Format | Description |
|---------|--------|-------------|
| Stop | `"stop"` or `{"type":"stop"}` | End transcription session |
| Set Language | `{"type":"setlanguage","lang":"en"}` | Change transcription language |

### Server Commands

```bash
ears server start                 # Start server with config defaults
ears server start --bind 0.0.0.0:8765  # Start on specific port
ears server start --timestamps --vad   # Enable timestamps and VAD
ears server stop                  # Stop running server
ears server status                # Check server status
```

### Example URLs

```
Local:     ws://localhost:8765/
LAN:       ws://192.168.1.100:8765/
Remote:    ws://example.com:8765/
```

Remember: Only one client can stream audio at a time!

