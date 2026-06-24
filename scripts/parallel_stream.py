#!/usr/bin/env python3
"""
Parallel WebSocket audio streaming client for eaRS server.
Streams multiple audio files concurrently to demonstrate parallel transcription.
"""

import asyncio
import argparse
import io
import os
import sys
import wave
import json
import subprocess
import numpy as np
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets


async def stream_audio_file(websocket, audio_path: str, label: str, chunk_size: int = 4096):
    """Stream an audio file to the WebSocket server."""
    print(f"[{label}] Starting stream: {audio_path}")

    # Convert audio to 24kHz mono float32 raw PCM
    # Use ffmpeg to convert to 24kHz mono float32
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-ar", "24000",
        "-ac", "1",
        "-c:a", "pcm_f32le",
        "-f", "wav",
        "-"
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Read the WAV header (44 bytes) and find data size
        header = await proc.stdout.read(44)
        if len(header) < 44:
            print(f"[{label}] Error: Invalid WAV file")
            await proc.kill()
            return

        # Parse WAV header to get sample rate and data size
        with wave.open(io.BytesIO(header)) as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            n_frames = wav.getnframes()

        print(f"[{label}] Audio: {frame_rate}Hz, {n_channels}ch, {n_frames} frames")

        # Stream in chunks
        bytes_sent = 0
        while True:
            chunk = await proc.stdout.read(chunk_size)
            if not chunk:
                break

            # Convert to float32 if needed
            if sample_width == 2:  # int16
                arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                chunk = arr.tobytes()
            elif sample_width == 4:  # int32
                arr = np.frombuffer(chunk, dtype=np.int32).astype(np.float32) / 2147483648.0
                chunk = arr.tobytes()
            # Already float32

            await websocket.send(chunk)
            bytes_sent += len(chunk)

        await proc.wait()
        print(f"[{label}] Sent {bytes_sent} bytes, waiting for final transcription...")

        # Wait a bit for final transcription
        await asyncio.sleep(3)

    except Exception as e:
        print(f"[{label}] Error: {e}")


async def listen_to_transcription(websocket, label: str, done_event: asyncio.Event):
    """Listen for transcription messages from the server."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "word":
                    print(f"[{label}] {data.get('word', '')}", end=" ", flush=True)
                elif msg_type == "final":
                    print(f"\n[{label}] FINAL: {data.get('text', '')}")
                    print(f"[{label}] Words: {data.get('words', [])}")
                    done_event.set()
                    break
                elif msg_type == "error":
                    print(f"\n[{label}] ERROR: {data.get('message', '')}")
                elif msg_type == "whisper_processing":
                    print(f"\n[{label}] Whisper: {data.get('original_text', '')}")
                elif msg_type == "whisper_complete":
                    print(f" -> {data.get('corrected_text', '')}")

            except json.JSONDecodeError:
                print(f"[{label}] Received non-JSON: {message[:100]}")
    except Exception as e:
        print(f"[{label}] WebSocket error: {e}")


async def stream_conversation(audio_path: str, server_url: str, label: str):
    """Handle one audio file streaming conversation."""
    done_event = asyncio.Event()

    try:
        async with websockets.connect(server_url) as websocket:
            # Start listening task
            listen_task = asyncio.create_task(
                listen_to_transcription(websocket, label, done_event)
            )

            # Stream the audio
            await stream_audio_file(websocket, audio_path, label)

            # Wait for transcription to complete
            try:
                await asyncio.wait_for(done_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                print(f"[{label}] Timeout waiting for transcription")

            # Cancel listening task
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"[{label}] Connection error: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Parallel audio streaming to eaRS WebSocket server"
    )
    parser.add_argument(
        "-s", "--server",
        default="ws://localhost:8765/",
        help="WebSocket server URL"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=9,
        help="Number of concurrent streams"
    )
    parser.add_argument(
        "audio_files",
        nargs="*",
        help="Audio files to stream (default: test_audio/*.wav)"
    )
    args = parser.parse_args()

    # Default audio files if none provided
    if not args.audio_files:
        audio_files = [
            "test_audio/english_short_24k.wav",
            "test_audio/english_samantha_24k.wav",
            "test_audio/english_voxcpm2_24k.wav",
            "test_audio/erwin_24k.wav",
            "test_audio/sattelberger_24k.wav",
            "test_audio/english_voxcpm2_long_24k.wav",
            "ref_audio/esp.mp3",
            "ref_audio/ger.mp3",
            "ref_audio/ita.mp3",
        ]
    else:
        audio_files = args.audio_files

    # Limit to concurrency
    audio_files = audio_files[:args.concurrency]

    print(f"=== Parallel Transcription Demo ===")
    print(f"Server: {args.server}")
    print(f"Concurrent streams: {len(audio_files)}")
    print(f"Files: {audio_files}")
    print()

    # Create tasks for all streams
    tasks = []
    for i, audio_file in enumerate(audio_files):
        label = Path(audio_file).stem
        task = stream_conversation(audio_file, args.server, label)
        tasks.append(task)

    # Run all streams concurrently
    await asyncio.gather(*tasks)

    print("\n=== All streams complete ===")


if __name__ == "__main__":
    import io
    asyncio.run(main())