# Listener Mode

The eaRS server supports a secure listener mode that allows clients to receive real-time transcriptions without sending audio. This enables multiple clients to monitor transcription streams from active audio sources.

## Overview

There are two types of connections:

1. **Active Clients** - Send audio data and receive their own transcriptions
2. **Listener Clients** - Only receive transcriptions from other active streams using token-based authentication

## Configuration

Enable listener mode in your config.toml:

```toml
[server]
websocket_port = 8765
host = "localhost"
enable_listener_mode = true
listener_tokens = [
    "secure-token-1",
    "secure-token-2",
    "another-secret-token"
]
```

Configuration options:

- `enable_listener_mode`: Set to `true` to enable listener connections
- `listener_tokens`: Array of valid authentication tokens for listeners

## Generating Tokens

You can generate secure tokens using any method you prefer. Here are some examples:

### Using openssl

```bash
openssl rand -hex 32
```

### Using Python

```python
import secrets
print(secrets.token_hex(32))
```

### Using the Rust TokenValidator utility

```rust
use ears::server::listener::TokenValidator;
let token = TokenValidator::generate_token();
println!("Generated token: {}", token);
```

## Usage

### Starting the Server with Listener Mode

```bash
ears server start
```

The server will automatically enable listener mode if configured in config.toml.

### Active Client Connection

Active clients connect normally and send audio data:

```javascript
const ws = new WebSocket('ws://localhost:8765/');

ws.onopen = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // Start sending audio...
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Transcription:', message);
};
```

### Listener Client Connection

Listener clients must authenticate with a valid token before subscribing to streams:

```javascript
const ws = new WebSocket('ws://localhost:8765/');

ws.onopen = () => {
    // Authenticate with token
    ws.send(JSON.stringify({
        type: 'authenticate',
        token: 'secure-token-1'
    }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'authenticated':
            console.log('Authentication successful');
            // List available streams
            ws.send(JSON.stringify({ type: 'liststreams' }));
            break;
            
        case 'streams':
            console.log('Available streams:', message.stream_ids);
            // Subscribe to the first available stream
            if (message.stream_ids.length > 0) {
                ws.send(JSON.stringify({
                    type: 'subscribe',
                    stream_id: message.stream_ids[0]
                }));
            }
            break;
            
        case 'subscribed':
            console.log('Subscribed to stream:', message.stream_id);
            break;
            
        case 'word':
        case 'final':
            console.log('Transcription message:', message);
            break;
            
        case 'error':
            console.error('Error:', message.message);
            break;
    }
};
```

## Protocol

### Listener Commands

All listener commands are sent as JSON text messages.

#### Authenticate

Authenticate with a valid token:

```json
{
  "type": "authenticate",
  "token": "your-secure-token"
}
```

Response on success:

```json
{
  "type": "authenticated",
  "success": true
}
```

Response on failure:

```json
{
  "type": "error",
  "message": "invalid token"
}
```

#### List Streams

List all active transcription streams (requires authentication):

```json
{
  "type": "liststreams"
}
```

Response:

```json
{
  "type": "streams",
  "stream_ids": [1, 2, 3]
}
```

#### Subscribe to Stream

Subscribe to a specific stream (requires authentication):

```json
{
  "type": "subscribe",
  "stream_id": 1
}
```

Response on success:

```json
{
  "type": "subscribed",
  "stream_id": 1
}
```

Response on failure:

```json
{
  "type": "error",
  "message": "subscription failed: Stream 1 not found"
}
```

### Receiving Transcriptions

Once subscribed, listeners receive the same transcription messages as active clients:

#### Word Message

```json
{
  "type": "word",
  "word": "hello",
  "start_time": 1.234,
  "end_time": 1.567
}
```

#### Final Message

```json
{
  "type": "final",
  "text": "complete transcribed text",
  "words": [
    {"word": "hello", "start_time": 1.234, "end_time": 1.567},
    {"word": "world", "start_time": 1.678, "end_time": 1.890}
  ]
}
```

#### Whisper Enhancement Messages (if enabled)

```json
{
  "type": "whisper_processing",
  "sentence_id": "uuid",
  "original_text": "text from Kyutai model",
  "start_time": 1.0,
  "end_time": 3.5
}
```

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

## Complete Example

### Python Listener Client

```python
import asyncio
import json
import websockets

async def listen_to_transcription(token, server_url='ws://localhost:8765/'):
    async with websockets.connect(server_url) as websocket:
        # Authenticate
        await websocket.send(json.dumps({
            'type': 'authenticate',
            'token': token
        }))
        
        response = await websocket.recv()
        msg = json.loads(response)
        
        if msg.get('type') == 'authenticated':
            print('Successfully authenticated')
        else:
            print(f'Authentication failed: {msg}')
            return
        
        # List streams
        await websocket.send(json.dumps({'type': 'liststreams'}))
        response = await websocket.recv()
        msg = json.loads(response)
        
        if msg.get('type') == 'streams':
            streams = msg.get('stream_ids', [])
            print(f'Available streams: {streams}')
            
            if streams:
                # Subscribe to first stream
                await websocket.send(json.dumps({
                    'type': 'subscribe',
                    'stream_id': streams[0]
                }))
                
                # Listen for transcriptions
                async for message in websocket:
                    msg = json.loads(message)
                    
                    if msg.get('type') == 'word':
                        print(f"[{msg['start_time']:.2f}s] {msg['word']}")
                    elif msg.get('type') == 'final':
                        print(f"\nFinal: {msg['text']}")
                    elif msg.get('type') == 'error':
                        print(f"Error: {msg['message']}")

# Run the listener
asyncio.run(listen_to_transcription('secure-token-1'))
```

### JavaScript Listener Client

```javascript
class TranscriptionListener {
    constructor(token, serverUrl = 'ws://localhost:8765/') {
        this.token = token;
        this.serverUrl = serverUrl;
        this.ws = null;
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.serverUrl);
            
            this.ws.onopen = () => {
                console.log('Connected to server');
                this.authenticate();
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('Connection closed');
            };
        });
    }
    
    authenticate() {
        this.ws.send(JSON.stringify({
            type: 'authenticate',
            token: this.token
        }));
    }
    
    listStreams() {
        this.ws.send(JSON.stringify({
            type: 'liststreams'
        }));
    }
    
    subscribe(streamId) {
        this.ws.send(JSON.stringify({
            type: 'subscribe',
            stream_id: streamId
        }));
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'authenticated':
                console.log('Authenticated successfully');
                this.listStreams();
                break;
                
            case 'streams':
                console.log('Available streams:', message.stream_ids);
                if (message.stream_ids.length > 0) {
                    this.subscribe(message.stream_ids[0]);
                }
                break;
                
            case 'subscribed':
                console.log('Subscribed to stream:', message.stream_id);
                break;
                
            case 'word':
                console.log(`[${message.start_time.toFixed(2)}s] ${message.word}`);
                break;
                
            case 'final':
                console.log('\nFinal transcription:', message.text);
                break;
                
            case 'error':
                console.error('Error:', message.message);
                break;
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Usage
const listener = new TranscriptionListener('secure-token-1');
listener.connect();
```

## Security Considerations

1. **Token Management**
   - Store tokens securely (environment variables, secure config files)
   - Use long, random tokens (at least 32 characters)
   - Rotate tokens periodically
   - Never commit tokens to version control

2. **Network Security**
   - Use WSS (WebSocket Secure) in production
   - Deploy behind a reverse proxy with TLS
   - Implement rate limiting
   - Monitor for suspicious connection patterns

3. **Access Control**
   - One token per client or application
   - Revoke compromised tokens immediately
   - Log authentication attempts

## Troubleshooting

### "Listener mode not enabled" Error

Ensure `enable_listener_mode = true` in your config.toml and restart the server.

### "Invalid token" Error

Verify that:
- The token is in the `listener_tokens` array in config.toml
- There are no extra spaces or characters
- The server has been restarted after config changes

### "Stream not found" Error

This happens when:
- The stream has ended
- No active clients are currently streaming
- An invalid stream_id was provided

Use the `liststreams` command to get current active streams.

### Connection Immediately Closes

The first message must be a listener command. If you send binary data or another message type first, the connection will be treated as an active client.

## Performance

- Listener connections have minimal overhead
- Multiple listeners can subscribe to the same stream
- Transcription messages are broadcast efficiently
- No audio processing occurs for listener connections

## Limitations

- Listeners cannot send audio data
- Listeners cannot control the transcription (pause, resume, etc.)
- Stream IDs are session-based and change when streams restart
- Authentication is required for all listener operations

## Future Enhancements

Potential future features:
- Named streams with persistent IDs
- Per-token permissions (read-only, specific streams)
- Stream metadata (language, duration, client info)
- Listener statistics and monitoring
- Token expiration and renewal
