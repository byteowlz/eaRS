#!/usr/bin/env python3
"""Single-terminal parallel multi-stream showcase for eaRS.

No tmux required. Opens N websocket streams concurrently, sends different audio
files, prints per-stream progress, and can export metrics as JSON/CSV.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed


DEFAULT_FILES = [
    "test_audio/generated/demo_en_welcome.wav",
    "test_audio/generated/demo_en_broadcast.wav",
    "test_audio/generated/demo_en_production.wav",
    "test_audio/generated/demo_en_thankyou.wav",
    "test_audio/generated/demo_en_pangram.wav",
    "test_audio/generated/demo_en_ai.wav",
    "test_audio/generated/demo_en_extended.wav",
    "test_audio/generated/demo_en_streaming.wav",
    "test_audio/generated/demo_en_server.wav",
]


@dataclass
class StreamResult:
    label: str
    file: str
    ok: bool
    words: int
    final_text: str
    elapsed_s: float
    first_word_latency_s: float
    final_latency_s: float
    rtf: float
    audio_duration_s: float
    error: str = ""


class Printer:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def log(self, msg: str) -> None:
        async with self._lock:
            print(msg, flush=True)


def get_wav_duration_seconds(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        return (frames / sr) if sr else 0.0


async def send_audio(ws, audio_path: str, pace: float) -> None:
    """Send audio as 24kHz mono float32 chunks (80ms each)."""
    chunk_size = 7680  # 24k * 0.08s * 4 bytes
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-ar",
        "24000",
        "-ac",
        "1",
        "-f",
        "f32le",
        "-",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdout is not None
    while True:
        chunk = await proc.stdout.read(chunk_size)
        if not chunk:
            break
        await ws.send(chunk)
        if pace > 0:
            await asyncio.sleep(0.08 * pace)

    rc = await proc.wait()
    if rc != 0:
        err = b""
        if proc.stderr is not None:
            err = await proc.stderr.read()
        raise RuntimeError(f"ffmpeg failed ({rc}): {err.decode(errors='ignore').strip()}")


async def run_stream_once(
    idx: int,
    audio_path: str,
    server: str,
    pace: float,
    timeout: float,
    live_words: bool,
    live_chunk_words: int,
    printer: Printer,
) -> StreamResult:
    label = f"S{idx+1}:{Path(audio_path).stem}"
    t0 = time.perf_counter()
    words = 0
    final_text = ""
    first_word_t: Optional[float] = None
    final_t: Optional[float] = None

    if not Path(audio_path).exists():
        return StreamResult(label, audio_path, False, 0, "", 0.0, 0.0, 0.0, 0.0, 0.0, f"missing file: {audio_path}")

    try:
        audio_duration_s = get_wav_duration_seconds(audio_path)
    except Exception as e:
        return StreamResult(
            label,
            audio_path,
            False,
            0,
            "",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            f"failed reading wav duration: {e}",
        )

    recv_task = None
    try:
        async with websockets.connect(server, max_size=None, ping_interval=20, ping_timeout=20) as ws:
            await printer.log(f"[{label}] connected")

            async def recv_loop() -> None:
                nonlocal words, final_text, first_word_t, final_t
                word_buf: list[str] = []
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    msg_type = data.get("type")
                    now = time.perf_counter()

                    if msg_type == "word":
                        w = data.get("word", "")
                        words += 1
                        if first_word_t is None:
                            first_word_t = now
                        if live_words and w:
                            word_buf.append(w)
                            if len(word_buf) >= live_chunk_words:
                                await printer.log(f"[{label}] {' '.join(word_buf)}")
                                word_buf.clear()
                    elif msg_type == "final":
                        if live_words and word_buf:
                            await printer.log(f"[{label}] {' '.join(word_buf)}")
                            word_buf.clear()
                        final_text = data.get("text", "")
                        final_t = now
                        await printer.log(f"[{label}] FINAL: {final_text}")
                        return
                    elif msg_type == "error":
                        raise RuntimeError(f"server error: {data.get('message', 'unknown')}")

            recv_task = asyncio.create_task(recv_loop())

            await send_audio(ws, audio_path, pace)
            await ws.send("stop")
            await asyncio.wait_for(recv_task, timeout=timeout)

        elapsed = time.perf_counter() - t0
        first_word_latency_s = (first_word_t - t0) if first_word_t else 0.0
        final_latency_s = (final_t - t0) if final_t else elapsed
        rtf = (elapsed / audio_duration_s) if audio_duration_s > 0 else 0.0

        await printer.log(f"[{label}] done | words={words} | {elapsed:.1f}s | rtf={rtf:.2f}")
        return StreamResult(
            label,
            audio_path,
            True,
            words,
            final_text,
            elapsed,
            first_word_latency_s,
            final_latency_s,
            rtf,
            audio_duration_s,
            "",
        )

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        return StreamResult(
            label,
            audio_path,
            False,
            words,
            final_text,
            elapsed,
            (first_word_t - t0) if first_word_t else 0.0,
            0.0,
            (elapsed / audio_duration_s) if audio_duration_s > 0 else 0.0,
            audio_duration_s,
            "timeout waiting for final",
        )
    except ConnectionClosed as e:
        elapsed = time.perf_counter() - t0
        return StreamResult(
            label,
            audio_path,
            False,
            words,
            final_text,
            elapsed,
            (first_word_t - t0) if first_word_t else 0.0,
            0.0,
            (elapsed / audio_duration_s) if audio_duration_s > 0 else 0.0,
            audio_duration_s,
            f"connection closed: {e}",
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return StreamResult(
            label,
            audio_path,
            False,
            words,
            final_text,
            elapsed,
            (first_word_t - t0) if first_word_t else 0.0,
            0.0,
            (elapsed / audio_duration_s) if audio_duration_s > 0 else 0.0,
            audio_duration_s,
            str(e),
        )
    finally:
        if recv_task is not None and not recv_task.done():
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)
        elif recv_task is not None and recv_task.done():
            # Consume exception if any to avoid "Task exception was never retrieved"
            _ = recv_task.exception()


async def run_stream_with_retry(
    idx: int,
    audio_path: str,
    server: str,
    pace: float,
    timeout: float,
    retries: int,
    retry_backoff: float,
    printer: Printer,
) -> StreamResult:
    attempt = 0
    while True:
        result = await run_stream_once(idx, audio_path, server, pace, timeout, printer)
        if result.ok or attempt >= retries:
            if not result.ok:
                await printer.log(f"[{result.label}] FAIL: {result.error}")
            return result

        attempt += 1
        await printer.log(f"[{result.label}] retry {attempt}/{retries} after error: {result.error}")
        await asyncio.sleep(retry_backoff * attempt)


async def delayed_run(
    delay_s: float,
    idx: int,
    audio_path: str,
    args,
    printer: Printer,
) -> StreamResult:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    return await run_stream_with_retry(
        idx,
        audio_path,
        args.server,
        args.pace,
        args.timeout,
        args.retries,
        args.retry_backoff,
        printer,
    )


def write_json(path: str, results: list[StreamResult], wall_time_s: float, server: str) -> None:
    payload = {
        "server": server,
        "wall_time_s": wall_time_s,
        "streams": [asdict(r) for r in results],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str, results: list[StreamResult]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Showcase parallel multi-stream transcription (no tmux)")
    parser.add_argument("--server", default="ws://localhost:8765", help="eaRS websocket URL")
    parser.add_argument("--pace", type=float, default=0.35, help="send pace factor per 80ms chunk (0=as fast as possible)")
    parser.add_argument("--timeout", type=float, default=60.0, help="timeout waiting for final per stream")
    parser.add_argument("--streams", type=int, default=9, help="number of streams to run")
    parser.add_argument("--stagger", type=float, default=0.10, help="delay between stream starts in seconds")
    parser.add_argument("--retries", type=int, default=1, help="retries per stream after failure")
    parser.add_argument("--retry-backoff", type=float, default=0.5, help="linear retry backoff base seconds")
    parser.add_argument("--json-out", default="", help="write detailed metrics JSON")
    parser.add_argument("--csv-out", default="", help="write detailed metrics CSV")
    parser.add_argument("files", nargs="*", help="audio files (default: generated English demo set)")
    args = parser.parse_args()

    files = args.files if args.files else DEFAULT_FILES
    files = files[: args.streams]

    printer = Printer()
    await printer.log("=== eaRS Parallel Multi-Stream Showcase ===")
    await printer.log(
        f"server={args.server} | streams={len(files)} | pace={args.pace} | stagger={args.stagger} | retries={args.retries}"
    )

    t0 = time.perf_counter()
    tasks = [
        delayed_run(i * args.stagger, i, file_path, args, printer)
        for i, file_path in enumerate(files)
    ]
    results = await asyncio.gather(*tasks)
    total = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]

    print("\n=== Summary ===")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(
            f"{status:4} {r.label:28} words={r.words:4} "
            f"time={r.elapsed_s:5.1f}s rtf={r.rtf:4.2f} first={r.first_word_latency_s:4.2f}s"
        )

    if ok:
        avg_rtf = sum(r.rtf for r in ok) / len(ok)
        avg_first = sum(r.first_word_latency_s for r in ok) / len(ok)
        print(f"\nAverages (OK only): rtf={avg_rtf:.2f} | first_word_latency={avg_first:.2f}s")

    print(f"Total streams: {len(results)} | OK: {len(ok)} | FAIL: {len(bad)} | wall-time: {total:.1f}s")

    if args.json_out:
        write_json(args.json_out, results, total, args.server)
        print(f"JSON metrics written: {args.json_out}")
    if args.csv_out:
        write_csv(args.csv_out, results)
        print(f"CSV metrics written: {args.csv_out}")


if __name__ == "__main__":
    asyncio.run(main())
