#!/usr/bin/env python3
"""
eaRS Streaming ASR Benchmark Tool

Streams audio files to eaRS server via WebSocket, comparing Kyutai vs Parakeet
transcription quality in real-time. Measures:
- Word Error Rate (WER) with jiwer
- Word duplication count
- Latency (time to first word, time to final)
- Real-time factor (processing speed vs audio duration)

Usage:
    python3 bench_streaming.py --audio audio.wav --reference "ground truth text"
    python3 bench_streaming.py --audio audio.wav --kyutai-only
    python3 bench_streaming.py --audio audio.wav --parakeet-only
    python3 bench_streaming.py --audio audio.wav --both  # default
"""

import argparse
import asyncio
import json
import struct
import time
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from jiwer import wer, cer
except ImportError:
    print("pip install jiwer")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None


@dataclass
class WordEvent:
    word: str
    start_time: float
    end_time: Optional[float] = None
    received_at: float = 0.0


@dataclass
class StreamResult:
    engine: str
    words: List[WordEvent] = field(default_factory=list)
    final_text: str = ""
    final_words: List[WordEvent] = field(default_factory=list)
    first_word_latency: float = 0.0
    final_latency: float = 0.0
    stream_start: float = 0.0
    stream_end: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def live_text(self) -> str:
        return " ".join(w.word for w in self.words)

    @property
    def duplicated_words(self) -> List[Tuple[int, str]]:
        """Find consecutive duplicated words."""
        dups = []
        words = [w.word.lower().strip(".,!?;:") for w in self.final_words or self.words]
        for i in range(1, len(words)):
            if words[i] == words[i-1] and words[i]:
                dups.append((i, words[i]))
        return dups

    @property
    def duplication_count(self) -> int:
        return len(self.duplicated_words)


def load_audio(path: str, target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate as float32."""
    ext = Path(path).suffix.lower()

    if ext == ".wav" and wavfile:
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32:
            pass
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
    else:
        # Use ffmpeg for other formats
        import subprocess
        cmd = [
            "ffmpeg", "-i", path, "-f", "f32le", "-ac", "1",
            "-ar", str(target_sr), "-v", "error", "-"
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        raw = result.stdout
        data = np.frombuffer(raw, dtype=np.float32)
        sr = target_sr

    if sr != target_sr:
        # Simple resample using scipy
        from scipy.signal import resample
        num_samples = int(len(data) * target_sr / sr)
        data = resample(data, num_samples).astype(np.float32)
        sr = target_sr

    return data, sr


async def stream_to_engine(
    audio: np.ndarray,
    sr: int,
    engine: str,
    ws_url: str,
    chunk_duration_ms: int = 80,
    language: Optional[str] = None,
    switch_at: Optional[List[Tuple[float, str]]] = None,
) -> StreamResult:
    """Stream audio to a specific engine and collect results.

    `switch_at` is a list of (audio_seconds, lang_code) — at each timestamp
    (relative to start of audio) the script issues a `set_language` over the WS.
    """

    result = StreamResult(engine=engine)
    chunk_samples = int(sr * chunk_duration_ms / 1000)
    switches = sorted(switch_at or [], key=lambda x: x[0])

    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            # Switch to requested engine BEFORE any audio
            await ws.send(json.dumps({"type": "setengine", "engine": engine}))

            # Set language if specified
            if language:
                await ws.send(json.dumps({"type": "setlanguage", "lang": language}))

            # Start streaming
            result.stream_start = time.time()
            first_word_received = False

            # Send audio and receive transcriptions concurrently
            async def send_audio():
                next_switch_idx = 0
                for i in range(0, len(audio), chunk_samples):
                    audio_t = i / sr
                    while next_switch_idx < len(switches) and switches[next_switch_idx][0] <= audio_t:
                        t, lang = switches[next_switch_idx]
                        print(f"  [bench] switch_language at audio t={audio_t:.2f}s -> {lang}")
                        await ws.send(json.dumps({"type": "setlanguage", "lang": lang}))
                        next_switch_idx += 1
                    chunk = audio[i:i+chunk_samples]
                    if len(chunk) == 0:
                        break
                    await ws.send(chunk.tobytes())
                    # Simulate real-time: wait chunk duration
                    await asyncio.sleep(chunk_duration_ms / 1000.0)
                # Send stop signal
                await ws.send("stop")

            async def recv_transcriptions():
                nonlocal first_word_received
                try:
                    async for message in ws:
                        if isinstance(message, bytes):
                            continue
                        try:
                            msg = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        msg_type = msg.get("type", "")

                        if msg_type == "word":
                            w = WordEvent(
                                word=msg["word"],
                                start_time=msg.get("start_time", 0),
                                end_time=msg.get("end_time"),
                                received_at=time.time(),
                            )
                            result.words.append(w)
                            if not first_word_received:
                                first_word_received = True
                                result.first_word_latency = time.time() - result.stream_start

                        elif msg_type == "final":
                            result.final_text = msg.get("text", "")
                            for w in msg.get("words", []):
                                result.final_words.append(WordEvent(
                                    word=w["word"],
                                    start_time=w.get("start_time", 0),
                                    end_time=w.get("end_time"),
                                ))
                            result.stream_end = time.time()
                            result.final_latency = result.stream_end - result.stream_start
                            break

                        elif msg_type == "pause":
                            pass
                        elif msg_type == "error":
                            result.errors.append(msg.get("message", "unknown error"))
                except websockets.exceptions.ConnectionClosed:
                    pass

            await asyncio.gather(send_audio(), recv_transcriptions())

    except Exception as e:
        result.errors.append(str(e))

    return result


def print_result(result: StreamResult, reference: Optional[str], audio_duration: float):
    """Print formatted results for one engine."""
    print(f"\n{'='*60}")
    print(f"  Engine: {result.engine.upper()}")
    print(f"{'='*60}")

    if result.errors:
        print(f"  ERRORS: {'; '.join(result.errors)}")
        return

    print(f"  Transcript: {result.final_text}")

    # Duplication analysis
    dups = result.duplicated_words
    if dups:
        print(f"\n  DUPLICATED WORDS ({len(dups)}):")
        for idx, word in dups[:10]:
            print(f"    [{idx}] '{word}' repeated")
        if len(dups) > 10:
            print(f"    ... and {len(dups) - 10} more")
    else:
        print(f"  Duplicated words: 0")

    # Latency metrics
    print(f"\n  Latency:")
    print(f"    First word: {result.first_word_latency*1000:.0f}ms")
    print(f"    Final:      {result.final_latency*1000:.0f}ms")
    rtf = result.final_latency / audio_duration if audio_duration > 0 else 0
    print(f"    RTF:        {rtf:.2f}x ({'real-time' if rtf <= 1.0 else 'slower than real-time'})")

    # Word count
    word_count = len(result.final_words) if result.final_words else len(result.words)
    print(f"    Words:      {word_count}")

    # WER if reference provided
    if reference:
        hypothesis = result.final_text
        if hypothesis.strip():
            try:
                w = wer(reference, hypothesis)
                c = cer(reference, hypothesis)
                print(f"\n  Accuracy:")
                print(f"    WER: {w*100:.1f}%")
                print(f"    CER: {c*100:.1f}%")
            except Exception as e:
                print(f"    WER calculation failed: {e}")
        else:
            print(f"\n  WER: N/A (empty transcript)")

    print()


def compare_results(kyutai: StreamResult, parakeet: StreamResult, reference: Optional[str]):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: KYUTAI vs PARAKEET")
    print(f"{'='*60}")

    if reference:
        k_wer = wer(reference, kyutai.final_text) * 100 if kyutai.final_text else 999
        p_wer = wer(reference, parakeet.final_text) * 100 if parakeet.final_text else 999
        k_cer = cer(reference, kyutai.final_text) * 100 if kyutai.final_text else 999
        p_cer = cer(reference, parakeet.final_text) * 100 if parakeet.final_text else 999
        print(f"  {'Metric':<25} {'Kyutai':>10} {'Parakeet':>10} {'Delta':>10}")
        print(f"  {'-'*55}")
        print(f"  {'WER (%)':<25} {k_wer:>10.1f} {p_wer:>10.1f} {p_wer-k_wer:>+10.1f}")
        print(f"  {'CER (%)':<25} {k_cer:>10.1f} {p_cer:>10.1f} {p_cer-k_cer:>+10.1f}")

    print(f"  {'Duplicated words':<25} {kyutai.duplication_count:>10} {parakeet.duplication_count:>10} {parakeet.duplication_count-kyutai.duplication_count:>+10}")

    k_rtf = kyutai.final_latency / (kyutai.stream_end - kyutai.stream_start) if (kyutai.stream_end - kyutai.stream_start) > 0 else 0
    p_rtf = parakeet.final_latency / (parakeet.stream_end - parakeet.stream_start) if (parakeet.stream_end - parakeet.stream_start) > 0 else 0
    print(f"  {'Final latency (ms)':<25} {kyutai.final_latency*1000:>10.0f} {parakeet.final_latency*1000:>10.0f}")
    print(f"  {'First word (ms)':<25} {kyutai.first_word_latency*1000:>10.0f} {parakeet.first_word_latency*1000:>10.0f}")
    print()


async def main():
    parser = argparse.ArgumentParser(description="eaRS Streaming ASR Benchmark")
    parser.add_argument("--audio", required=True, help="Audio file to transcribe")
    parser.add_argument("--reference", default=None, help="Ground truth text (or file containing it)")
    parser.add_argument("--ws", default="ws://[::1]:8765", help="eaRS WebSocket URL")
    parser.add_argument(
        "--engine",
        choices=["both", "kyutai", "parakeet", "sherpa"],
        default="both",
    )
    parser.add_argument("--language", default=None, help="Language code (de, en, fr, etc.)")
    parser.add_argument(
        "--switch-at",
        action="append",
        default=[],
        metavar="T:LANG",
        help="Switch language at audio timestamp T (seconds), e.g. --switch-at 25:de (repeatable)",
    )
    parser.add_argument("--chunk-ms", type=int, default=80, help="Audio chunk size in ms (default 80)")
    args = parser.parse_args()

    # Load reference
    reference = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            reference = ref_path.read_text().strip()
        else:
            reference = args.reference

    # Load audio
    print(f"Loading {args.audio}...")
    audio, sr = load_audio(args.audio)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s, Sample rate: {sr}Hz, Samples: {len(audio)}")

    results = {}

    if args.engine in ("both", "kyutai"):
        print(f"\nStreaming to KYUTAI...")
        r = await stream_to_engine(audio, sr, "kyutai", args.ws, args.chunk_ms, args.language)
        results["kyutai"] = r
        print_result(r, reference, duration)

    if args.engine in ("both", "parakeet"):
        print(f"\nStreaming to PARAKEET...")
        r = await stream_to_engine(audio, sr, "parakeet", args.ws, args.chunk_ms, args.language)
        results["parakeet"] = r
        print_result(r, reference, duration)

    # Parse --switch-at args
    switches: List[Tuple[float, str]] = []
    for s in args.switch_at:
        if ":" not in s:
            print(f"WARN: ignoring --switch-at '{s}' (expected T:LANG)")
            continue
        t, lang = s.split(":", 1)
        switches.append((float(t), lang.strip()))

    if args.engine == "sherpa":
        print(f"\nStreaming to SHERPA...")
        r = await stream_to_engine(
            audio, sr, "sherpa", args.ws, args.chunk_ms, args.language,
            switch_at=switches,
        )
        results["sherpa"] = r
        print_result(r, reference, duration)

    if "kyutai" in results and "parakeet" in results:
        compare_results(results["kyutai"], results["parakeet"], reference)


if __name__ == "__main__":
    asyncio.run(main())
