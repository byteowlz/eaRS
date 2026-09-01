#!/usr/bin/env python3
"""
eaRS Trailing-Word Latency Probe

Uses spqx to synthesize real speech, appends a controlled amount of trailing
silence, streams it to a live eaRS server in real time, and measures how long
after end-of-speech the LAST word is emitted by the server.

Isolates the server-side emission latency (engine + boundary VAD) from the
dictation client's own buffering: everything here is measured at the WebSocket,
so the dictation client's 1.5s phrase-prefix hold is NOT included. Run the
probe per engine (kyutai / parakeet-rs) to compare engines.

Usage:
    python3 scripts/probe_trailing_word.py --cases 300,600,1000
    python3 scripts/probe_trailing_word.py --engine parakeet-rs --cases 300,600,1000
    python3 scripts/probe_trailing_word.py --engine kyutai --text "alpha bravo charlie now"
    python3 scripts/probe_trailing_word.py --server ws://localhost:8765/ --voice ryan

Requirements: websockets, numpy, ffmpeg on PATH, spqx on PATH, ears server running.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("pip install numpy")
    sys.exit(1)

DEFAULT_TEXT = "alpha bravo charlie now"
# Project reference audio for spqx voice cloning (converted to WAV on first use).
DEFAULT_REF_AUDIO = "ref_audio/ger.mp3"
DEFAULT_REF_TEXT = "Schweresprache"
SAMPLE_RATE = 24_000
# 1024 samples @ 24 kHz = ~42.7 ms per chunk; sent in real time so wall-clock
# deltas between speech end and word arrival are meaningful.
CHUNK_SAMPLES = 1024


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def synth_speech(text, out_wav, voice, ref_wav=None, ref_text=None, attempts=5) -> bool:
    args = ["spqx", "say", "--no-play", "--out", str(out_wav)]
    if voice:
        args += ["--voice", voice]
    elif ref_wav:
        # Ad-hoc voice cloning with the project reference audio.
        args += ["--ref-audio", str(ref_wav)]
        if ref_text:
            args += ["--ref-text", ref_text]
    args.append(text)
    # spqx sampling is stochastic (temp 0.9) and can hit the frame cap without
    # EOS; it still exits 0 and writes a truncated/garbled WAV in that case,
    # which the ASR models then cannot transcribe at all. Only accept a run
    # that reports EOS.
    last_err = ""
    for attempt in range(1, attempts + 1):
        result = sh(args)
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0 and out_wav.exists() and "EOS detected" in combined:
            return True
        last_err = combined.strip()[:300]
        print(f"[spqx] attempt {attempt}/{attempts}: no EOS (truncated?), resampling")
    print(f"[spqx] failed after {attempts} attempts: {last_err}")
    return False


def load_pcm_24k(path):
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-f", "f32le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()[:300]}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def speech_pcm_with_silence(speech, silence_ms):
    silence = np.zeros(int(SAMPLE_RATE * silence_ms / 1000.0), dtype=np.float32)
    return np.concatenate([speech, silence])


def speech_duration_secs(pcm, silence_ms):
    return (len(pcm) - int(SAMPLE_RATE * silence_ms / 1000.0)) / SAMPLE_RATE


async def run_case(ws_url, engine, pcm, speech_secs, label) -> dict:
    words: list = []          # (arrival_rel_to_speech_end, word)
    speech_events: list = []  # (arrival_rel_to_speech_end, active)
    final_text = ""
    error = None

    async with websockets.connect(ws_url) as ws:
        if engine:
            await ws.send(json.dumps({"type": "setengine", "engine": engine}))

        t0 = time.monotonic()
        speech_end_t = t0 + speech_secs

        async def listen():
            nonlocal final_text, error
            async for message in ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                mtype = data.get("type", "")
                now_rel = time.monotonic() - speech_end_t
                if mtype == "word":
                    words.append((now_rel, data.get("word", "")))
                elif mtype == "speech":
                    speech_events.append((now_rel, bool(data.get("active"))))
                elif mtype == "final":
                    final_text = data.get("text", "")
                elif mtype == "error":
                    error = data.get("message", "")

        listen_task = asyncio.create_task(listen())

        # Stream in real time so wall-clock deltas are meaningful.
        samples = pcm.tobytes()
        chunk_bytes = CHUNK_SAMPLES * 4
        t_start = time.monotonic()
        offset = 0
        while offset < len(samples):
            target = t_start + (offset // 4) / SAMPLE_RATE
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(samples[offset:offset + chunk_bytes])
            offset += chunk_bytes

        # Wait for the last word / final (bounded), with a grace window after
        # the last word so stragglers are captured.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if words and (time.monotonic() - (t_start + speech_secs + words[-1][0])) > 0.5:
                break
            await asyncio.sleep(0.05)

        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass

    last_rel = words[-1][0] if words else float("nan")
    return {
        "label": label,
        "words": [w for _, w in words],
        "word_rels": words,
        "last_word_latency": last_rel,
        "final_text": final_text,
        "error": error,
        "speech_events": speech_events,
    }


async def main_async(args) -> int:
    ws_url = args.server
    if not ws_url.endswith("/"):
        ws_url += "/"

    # Server reachability check with a helpful hint.
    try:
        async with websockets.connect(ws_url, open_timeout=3):
            pass
    except Exception as e:
        print(f"[probe] server not reachable at {ws_url}: {e}")
        print("[probe] start it with: ears server start   (or ears server run --engine ...)")
        return 2

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Prefer an explicit --audio, then a cached synthesis, then the pinned
    # golden sample (some spqx takes are unintelligible to the ASR models
    # despite reporting EOS, so a known-good sample is checked in).
    wav = None
    if args.audio:
        wav = Path(args.audio).expanduser()
        if not wav.exists():
            print(f"[probe] --audio not found: {wav}")
            return 3
    else:
        golden = Path(__file__).resolve().parent.parent / "test_audio" / "spqx_alpha_bravo.wav"
        if golden.exists():
            wav = golden
        else:
            wav = cache / "speech.wav"
            if not wav.exists():
                ref_wav = None
                if args.ref_audio:
                    # spqx needs a WAV reference; convert mp3 refs on first use.
                    ref_wav = cache / "ref.wav"
                    if not ref_wav.exists():
                        conv = subprocess.run(
                            ["ffmpeg", "-y", "-loglevel", "error", "-i", args.ref_audio,
                             "-ar", str(SAMPLE_RATE), "-ac", "1", str(ref_wav)],
                            capture_output=True, text=True,
                        )
                        if conv.returncode != 0:
                            print(f"[probe] ref-audio conversion failed: {conv.stderr[:200]}")
                            return 3
                if not synth_speech(args.text, wav, args.voice, ref_wav, args.ref_text):
                    return 3
                print(f"[spqx] synthesized speech -> {wav}")
    speech = load_pcm_24k(wav)

    results = []
    for silence_ms in args.cases:
        label = f"sil{silence_ms}"
        pcm = speech_pcm_with_silence(speech, silence_ms)
        speech_secs = speech_duration_secs(speech, silence_ms)
        if not args.quiet:
            print(f"\n=== case {label} (speech {speech_secs:.2f}s "
                  f"+ {silence_ms}ms trailing silence) ===")
        result = await run_case(ws_url, args.engine, pcm, speech_secs, label)
        results.append(result)
        if result["error"]:
            print(f"  ERROR: {result['error']}")
            continue
        if not args.quiet:
            rel = [f"{rel:+.2f}s:{w}" for rel, w in result["word_rels"]]
            print(f"  words (arrival relative to speech end): {' '.join(rel)}")
            print(f"  final: {result['final_text']!r}")
        print(f"  last-word latency after speech end: {result['last_word_latency']:+.3f}s")

    print("\n=== summary ===")
    print(f"{'case':<12} {'last-word latency':>18}  words")
    for r in results:
        lat = f"{r['last_word_latency']:+.3f}s" if r["words"] else "n/a"
        print(f"{r['label']:<12} {lat:>18}  {' '.join(r['words'])}")
        if r["error"]:
            print(f"{'':<12}  ERROR: {r['error']}")

    print("\n[probe] note: server-side emission latency only — the dictation")
    print("[probe] client's 1.5s dictionary phrase-prefix hold is NOT included here.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="eaRS trailing-word latency probe (spqx-generated audio)"
    )
    parser.add_argument("--server", default="ws://localhost:8765/",
                        help="eaRS WebSocket URL (default: ws://localhost:8765/)")
    parser.add_argument("--engine", default=None,
                        help="engine to select per connection (kyutai|parakeet-rs|...)")
    parser.add_argument("--cases", default="300,600,1000",
                        help="comma-separated trailing-silence durations in ms")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help=f"phrase to synthesize (default: {DEFAULT_TEXT!r})")
    parser.add_argument("--voice", default=None,
                        help="spqx voice name (see: spqx voices)")
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO,
                        help="spqx --ref-audio WAV for voice cloning "
                             "(default: project ref_audio/ger.mp3, auto-converted; "
                             "pass empty string to use the spqx default voice)")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT,
                        help="transcript of --ref-audio")
    parser.add_argument("--audio", default=None,
                        help="use this WAV instead of synthesizing with spqx")
    parser.add_argument("--cache-dir", default="/tmp/ears_trailing_probe",
                        help="cache dir for synthesized audio")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="only print the summary")
    args = parser.parse_args()

    try:
        args.cases = [int(c) for c in str(args.cases).split(",") if c.strip()]
    except ValueError:
        print(f"[probe] invalid --cases: {args.cases}")
        return 2
    if not args.cases:
        print("[probe] --cases must list at least one silence duration")
        return 2

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[probe] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
