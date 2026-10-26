#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "websockets"]
# ///
"""
eaRS dictation stall soak test (trx-sm3r).

Streams a long, transcribed speech recording to a live eaRS server as one
continuous session and reports whether word emission ever dies while speech is
still going in -- the "words stop appearing while the engine keeps running"
failure.

Two subcommands:

  build  Concatenate FLEURS-style clips (a dir of WAVs plus a TSV with
         `id<TAB>file<TAB>raw transcript...`) into one 24 kHz mono f32 stream
         with natural pauses, and write a segment manifest (start/end/text).

  run    Stream that recording to a server (optionally faster than real time),
         collect every `word` event and attribute it to segments by the
         server-side audio timestamp. A segment that received zero words is a
         dead segment; a run of dead segments that never recovers is a stall.

Usage:
    uv run scripts/soak_stall.py build --tsv test.tsv --audio-dir test \
        --minutes 20 --out /tmp/soak-de
    uv run scripts/soak_stall.py run --soak /tmp/soak-de \
        --server ws://127.0.0.1:18791/ --speed 4

Exit code 0 when no stall is detected, 2 when the tail of the stream is dead.
"""

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import websockets

SAMPLE_RATE = 24_000
CHUNK_SAMPLES = 1920  # 80 ms, the dictation client's frame size


def decode_to_24k(path: Path) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def cmd_build(args) -> int:
    rng = random.Random(args.seed)
    rows = []
    seen = set()
    for line in Path(args.tsv).read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) < 3 or cols[1] in seen:
            continue
        seen.add(cols[1])
        rows.append((cols[1], cols[2]))
    rng.shuffle(rows)

    target = args.minutes * 60.0
    pieces: list[np.ndarray] = []
    segments = []
    rate = SAMPLE_RATE
    cursor = 0.0
    for fname, text in rows:
        path = Path(args.audio_dir) / fname
        if not path.is_file():
            continue
        pcm = decode_to_24k(path)
        if args.target_rms > 0:
            # Corpus clips vary by >30 dB; level them like a dictation mic so
            # the soak measures the engine, not the recording gain.
            active = pcm[np.abs(pcm) > 1e-4]
            rms = float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0
            if rms > 0:
                pcm = np.clip(pcm * (args.target_rms / rms), -1.0, 1.0)
        gap = rng.uniform(args.min_gap, args.max_gap)
        pieces.append(np.zeros(int(gap * rate), dtype=np.float32))
        cursor += gap
        dur = len(pcm) / rate
        pieces.append(pcm)
        segments.append({"file": fname, "start": cursor, "end": cursor + dur, "text": text})
        cursor += dur
        if cursor >= target:
            break
    pieces.append(np.zeros(int(2.0 * rate), dtype=np.float32))

    pcm24 = np.concatenate(pieces)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio.f32").write_bytes(pcm24.astype(np.float32).tobytes())
    manifest = {
        "sample_rate": SAMPLE_RATE,
        "duration": len(pcm24) / SAMPLE_RATE,
        "seed": args.seed,
        "target_rms": args.target_rms,
        "segments": segments,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"built {len(segments)} segments, {manifest['duration'] / 60:.1f} min -> {out}")
    return 0


async def stream(ws_url, engine, pcm, speed, lang):
    words = []  # dicts: word, start_time, wall, sent_audio_secs
    events = []
    final = None
    async with websockets.connect(ws_url, max_size=None) as ws:
        if engine:
            await ws.send(json.dumps({"type": "setengine", "engine": engine}))
        if lang:
            await ws.send(json.dumps({"type": "setlanguage", "lang": lang}))
        sent = 0
        t_start = time.monotonic()

        async def listen():
            nonlocal final
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                mtype = data.get("type", "")
                wall = time.monotonic() - t_start
                if mtype == "word":
                    words.append({
                        "word": data.get("word", ""),
                        "start_time": data.get("start_time"),
                        "wall": wall,
                        "sent": sent / SAMPLE_RATE,
                    })
                elif mtype == "final":
                    final = data.get("text", "")
                    return
                else:
                    events.append({"type": mtype, "wall": wall, "sent": sent / SAMPLE_RATE, "data": data})

        listener = asyncio.create_task(listen())
        raw = pcm.tobytes()
        step = CHUNK_SAMPLES * 4
        last_report = 0.0
        while sent * 4 < len(raw):
            target = t_start + (sent / SAMPLE_RATE) / speed
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            chunk = raw[sent * 4: sent * 4 + step]
            await ws.send(chunk)
            sent += len(chunk) // 4
            if sent / SAMPLE_RATE - last_report >= 60.0:
                last_report = sent / SAMPLE_RATE
                print(f"  sent {last_report / 60:.0f} min, words so far {len(words)}", flush=True)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(listener, timeout=30)
        except asyncio.TimeoutError:
            listener.cancel()
    return words, events, final


def normalize(text: str) -> list[str]:
    out = []
    for token in text.lower().split():
        token = "".join(ch for ch in token if ch.isalnum())
        if token:
            out.append(token)
    return out


def word_errors(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein alignment over words: (substitutions, deletions, insertions)."""
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]  # (cost, sub, del, ins)
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(sub, dele, ins, key=lambda c: c[0])
            if best is sub:
                cur.append((sub[0] + 1, sub[1] + 1, sub[2], sub[3]))
            elif best is dele:
                cur.append((dele[0] + 1, dele[1], dele[2] + 1, dele[3]))
            else:
                cur.append((ins[0] + 1, ins[1], ins[2], ins[3] + 1))
        prev = cur
    _, s, d, i = prev[-1]
    return s, d, i


def attribute(words, segments, mode, tail):
    """Count words per segment. mode=start_time uses the server's audio
    timestamp; mode=sent uses how much audio had been sent on arrival."""
    counts = [0] * len(segments)
    for w in words:
        t = w["start_time"] if mode == "start_time" else w["sent"]
        if t is None:
            continue
        for i, seg in enumerate(segments):
            if seg["start"] - 0.3 <= t <= seg["end"] + tail:
                counts[i] += 1
                break
    return counts


def cmd_run(args) -> int:
    soak = Path(args.soak)
    manifest = json.loads((soak / "manifest.json").read_text())
    pcm = np.frombuffer((soak / "audio.f32").read_bytes(), dtype=np.float32)
    segments = manifest["segments"]
    print(f"streaming {manifest['duration'] / 60:.1f} min ({len(segments)} segments) at {args.speed}x to {args.server}")
    words, events, final = asyncio.run(stream(args.server, args.engine, pcm, args.speed, args.lang))

    counts = attribute(words, segments, args.attribute, args.tail)
    expected = [len(s["text"].split()) for s in segments]
    dead = [i for i, c in enumerate(counts) if c == 0]
    # Longest trailing run of dead segments: a stall never recovers.
    trailing_dead = 0
    for c in reversed(counts):
        if c:
            break
        trailing_dead += 1
    longest_run, run = 0, 0
    for c in counts:
        run = run + 1 if c == 0 else 0
        longest_run = max(longest_run, run)

    # Longest stretch of streamed audio with no word at all, measured by how
    # much audio had been sent when each word arrived. This survives server
    # backlog and engine timestamp conventions; a stall shows up as a gap far
    # longer than any pause in the recording.
    marks = [0.0] + [w["sent"] for w in words] + [manifest["duration"]]
    gaps = sorted(
        ((b - a, a) for a, b in zip(marks, marks[1:])),
        reverse=True,
    )
    max_pause = max(
        (b["start"] - a["end"] for a, b in zip(segments, segments[1:])),
        default=0.0,
    )

    per_minute = {}
    for seg, c, e in zip(segments, counts, expected):
        m = int(seg["start"] // 60)
        got, exp = per_minute.get(m, (0, 0))
        per_minute[m] = (got + c, exp + e)

    ref = normalize(" ".join(seg["text"] for seg in segments))
    hyp = normalize(" ".join(w["word"] for w in words))
    subs, dels, ins = word_errors(ref, hyp)
    wer = (subs + dels + ins) / max(len(ref), 1)

    print(f"\nwords received: {len(words)}  (reference words: {sum(expected)})")
    print(f"WER {wer:.1%} (sub {subs}, del {dels}, ins {ins}) over {len(ref)} normalized reference words")
    print(f"segments: {len(segments)}, dead: {len(dead)}, longest dead run: {longest_run}, trailing dead: {trailing_dead}")
    print("longest wordless audio stretches (sec @ sent):",
          ", ".join(f"{g:.1f}@{a:.0f}" for g, a in gaps[:5]),
          f"(longest pause in recording {max_pause:.1f}s)")
    print("per-minute received/reference words:")
    for m in sorted(per_minute):
        got, exp = per_minute[m]
        bar = "#" * int(40 * got / exp) if exp else ""
        print(f"  min {m:3d}: {got:4d}/{exp:4d} {bar}")
    if dead:
        print("dead segments (index @ start sec):", ", ".join(f"{i}@{segments[i]['start']:.0f}" for i in dead[:40]))
    if final is None:
        print("no final message received")

    report = {
        "server": args.server,
        "engine": args.engine,
        "speed": args.speed,
        "attribute": args.attribute,
        "words": words,
        "events": events,
        "final": final,
        "counts": counts,
        "expected": expected,
        "dead": dead,
        "trailing_dead": trailing_dead,
        "longest_dead_run": longest_run,
        "longest_wordless": gaps[0][0] if gaps else 0.0,
        "wer": {"wer": wer, "sub": subs, "del": dels, "ins": ins, "ref_words": len(ref)},
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"report -> {args.report}")

    stalled = trailing_dead >= args.stall_segments or (gaps and gaps[0][0] >= args.stall_secs)
    print("RESULT:", "STALL" if stalled else "ok")
    return 2 if stalled else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--tsv", required=True)
    b.add_argument("--audio-dir", required=True)
    b.add_argument("--minutes", type=float, default=20.0)
    b.add_argument("--min-gap", type=float, default=0.4)
    b.add_argument("--max-gap", type=float, default=1.6)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--target-rms", type=float, default=0.05,
                   help="normalize each clip's active-sample RMS to this level (0 keeps corpus levels)")
    b.add_argument("--out", required=True)

    r = sub.add_parser("run")
    r.add_argument("--soak", required=True, help="directory written by `build`")
    r.add_argument("--server", default="ws://127.0.0.1:8765/")
    r.add_argument("--engine", default=None, help="send setengine first (kyutai, parakeet-rs, ...)")
    r.add_argument("--lang", default=None, help="send setlanguage first")
    r.add_argument("--speed", type=float, default=1.0, help="stream faster than real time")
    r.add_argument("--attribute", choices=["start_time", "sent"], default="sent",
                   help="map words to segments by server audio timestamp or by audio sent at arrival")
    r.add_argument("--tail", type=float, default=2.5, help="seconds after a segment that still count for it")
    r.add_argument("--stall-segments", type=int, default=3, help="trailing dead segments that count as a stall")
    r.add_argument("--stall-secs", type=float, default=15.0, help="wordless stretch of sent audio that counts as a stall")
    r.add_argument("--report")

    args = ap.parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
