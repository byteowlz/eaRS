#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.4", "soundfile==0.13.1", "rapidfuzz==3.14.1"]
# ///
"""Local, opt-in German bulk comparison. Explicit cached models/binaries only.

No downloads, hosted inference, dictionary, diarization or LLM post-processing.
All artifacts go into a new private directory. Timings include process/model
startup with a warmed filesystem; they are NOT all model-resident timings.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein
import soundfile as sf


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(re.sub(r"[^\w\s]", "", text).split())


def score(text, reference):
    hyp, ref = normalize(text), normalize(reference)
    return {"word_errors": Levenshtein.distance(ref.split(), hyp.split()),
            "reference_words": len(ref.split()), "char_errors": Levenshtein.distance(ref, hyp),
            "reference_chars": len(ref), "output_words": len(hyp.split())}


def fixtures(tsv, audio_dir, out, count):
    """Deduplicate by reference, seeded shuffle; PCM16 + identical 1s tail."""
    with tsv.open() as source:
        rows = list(csv.reader(source, delimiter="\t"))
    random.Random(20261027).shuffle(rows)
    selected, seen, audio = [], set(), []
    for row in rows:
        source = audio_dir / Path(row[1]).name
        identity = normalize(row[3])
        if identity in seen or not source.is_file():
            continue
        pcm, rate = sf.read(source, dtype="float32")
        if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("expected finite 16kHz mono FLEURS")
        if not 3 <= len(pcm) / rate <= 30:
            continue
        seen.add(identity)
        audio.append(pcm)
        padded = np.concatenate([pcm, np.zeros(rate, dtype=np.float32)])
        path = out / f"clip-{len(selected):03d}.wav"
        sf.write(path, padded, rate, subtype="PCM_16")
        selected.append({"id": path.stem, "path": str(path), "reference": row[3],
                         "source_file": source.name, "source_sha256": sha256(source),
                         "sha256": sha256(path), "seconds": len(padded) / rate, "kind": "clip"})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("insufficient unique references")
    joined = np.concatenate([part for pcm in audio for part in (pcm, np.zeros(16000, dtype=np.float32))])
    path = out / "bulk.wav"
    sf.write(path, joined, 16000, subtype="PCM_16")
    selected.append({"id": "bulk", "path": str(path), "reference": " ".join(c["reference"] for c in selected),
                     "sha256": sha256(path), "seconds": len(joined) / 16000, "kind": "bulk"})
    for name, pcm in [("silence", np.zeros(48000)),
                      ("noise", np.random.default_rng(43).normal(0, 0.005, 48000))]:
        path = out / f"{name}.wav"
        sf.write(path, pcm, 16000, subtype="PCM_16")
        selected.append({"id": name, "path": str(path), "reference": "", "sha256": sha256(path),
                         "seconds": 3.0, "kind": "negative"})
    return selected


def execute(command, prefix, env):
    """Owned process group, bounded deadline; macOS time reports bytes."""
    flag = "-l" if sys.platform == "darwin" else "-v"
    started = time.perf_counter()
    with prefix.with_suffix(".stdout").open("x") as stdout, prefix.with_suffix(".stderr").open("x") as stderr:
        process = subprocess.Popen(["/usr/bin/time", flag, *map(str, command)], stdout=stdout,
                                   stderr=stderr, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    wall = time.perf_counter() - started
    if code:
        raise RuntimeError(f"exit {code}; see {prefix}.stderr")
    diagnostic = prefix.with_suffix(".stderr").read_text()
    pattern = r"(\d+)\s+maximum resident set size" if sys.platform == "darwin" else r"Maximum resident set size \(kbytes\):\s*(\d+)"
    match = re.search(pattern, diagnostic)
    if not match:
        raise ValueError("missing RSS evidence")
    return {"wall_s": wall, "peak_rss_bytes": int(match[1]) * (1 if sys.platform == "darwin" else 1024)}


def moonshine_text(trace, cases):
    finals = {case["id"]: {} for case in cases}
    for raw in trace.read_text().splitlines():
        row = json.loads(raw)
        if row["kind"] == "snapshot":
            for line in row["lines"]:
                if line["complete"]:
                    finals[row["clip"]][line["id"]] = line["text"]
    return [" ".join(finals[case["id"]].values()) for case in cases]


def run_suite(engine, cases, out):
    name, kind, binary, model = engine
    out.mkdir(mode=0o700)
    prefix = out / "process"
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if kind == "moonshine":
        inputs, trace = out / "inputs.json", out / "trace.jsonl"
        inputs.write_text(json.dumps([{"id": c["id"], "path": c["path"]} for c in cases]))
        env.update(MOONSHINE_ORT_SINGLE_THREAD="1", EARS_MOONSHINE_TRACE_MODEL=str(model),
                   EARS_MOONSHINE_TRACE_INPUTS=str(inputs), EARS_MOONSHINE_TRACE_OUT=str(trace),
                   EARS_MOONSHINE_TRACE_PACKET_MS="80", EARS_MOONSHINE_TRACE_TAIL_MS="0",
                   EARS_MOONSHINE_TRACE_BOUNDARY_VAD="0" if name.endswith("-native-vad") else "1")
        command = [binary, "moonshine_revision_trace", "--ignored", "--nocapture"]
        metrics = execute(command, prefix, env)
        texts = moonshine_text(trace, cases)
    elif kind == "parakeet":
        if len(cases) == 1:
            command = [binary, "transcribe", "--model", model, "--input", cases[0]["path"],
                       "--lang", "de", "--threads", "4", "--json"]
            metrics = execute(command, prefix, env)
            texts = [json.loads(prefix.with_suffix(".stdout").read_text())["text"]]
        else:
            manifest, output = out / "paths.txt", out / "bench.json"
            manifest.write_text("".join(c["path"] + "\n" for c in cases))
            command = [binary, "bench", "--model", model, "--manifest", manifest,
                       "--lang", "de", "--threads", "4", "--json", output]
            metrics = execute(command, prefix, env)
            files = json.loads(output.read_text())["files"]
            if [r["path"] for r in files] != [c["path"] for c in cases]:
                raise ValueError("parakeet count/order mismatch")
            texts = [r["text"] for r in files]
    else:
        command = [binary, "-m", model, "-l", "de", "-t", "4", "-ojf", "--output-txt"]
        for i, case in enumerate(cases):
            command += ["-f", case["path"], "-of", str(out / f"whisper-{i}")]
        metrics = execute(command, prefix, env)
        texts = [(out / f"whisper-{i}.txt").read_text().strip() for i in range(len(cases))]
    if len(texts) != len(cases):
        raise ValueError("output count mismatch")
    metrics["command"] = list(map(str, command))
    metrics["cases"] = [{"id": c["id"], "text": text, **score(text, c["reference"])}
                        for c, text in zip(cases, texts, strict=True)]
    (out / "result.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def summarize(engine, initial, runs, bulk):
    clips = [r for r in initial["cases"] if r["id"].startswith("clip-")]
    words, chars = sum(r["reference_words"] for r in clips), sum(r["reference_chars"] for r in clips)
    median = statistics.median(r["wall_s"] for r in runs)
    return {"engine": engine[0], "clip_wer_pct": 100 * sum(r["word_errors"] for r in clips) / words,
            "clip_cer_pct": 100 * sum(r["char_errors"] for r in clips) / chars,
            "reference_words": words, "bulk_seconds": bulk["seconds"], "median_wall_s": median,
            "rtfx": bulk["seconds"] / median, "rtf": median / bulk["seconds"],
            "peak_rss_mib": max(r["peak_rss_bytes"] for r in runs) / 1048576,
            "bulk_wer_pct": [100 * r["cases"][0]["word_errors"] / r["cases"][0]["reference_words"] for r in runs],
            "negative_outputs": {r["id"]: r["text"] for r in initial["cases"] if r["id"] in ("silence", "noise")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tsv", "audio-dir", "out", "probe", "moonshine-base", "moonshine-cased",
                 "parakeet-bin", "parakeet-model", "whisper-bin", "whisper-turbo", "whisper-small"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-vad-only", action="store_true", help="Controlled Moonshine run without the outer eaRS boundary VAD; native SDK endpoints remain")
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.clips <= 64 or not 1 <= args.runs <= 5:
        parser.error("bounded clips=2..64 and runs=1..5 required")
    os.umask(0o077)
    args.out = args.out.resolve()
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    engines = [("moonshine-base", "moonshine", args.probe, args.moonshine_base),
               ("moonshine-cased", "moonshine", args.probe, args.moonshine_cased),
               ("parakeet-v3-q8", "parakeet", args.parakeet_bin, args.parakeet_model),
               ("whisper-turbo-q8", "whisper", args.whisper_bin, args.whisper_turbo),
               ("whisper-small", "whisper", args.whisper_bin, args.whisper_small)]
    if args.native_vad_only:
        engines = [(name + "-native-vad", kind, binary, model) for name, kind, binary, model in engines[:2]]
    fingerprints = {}
    for name, _, binary, model in engines:
        files = sorted(p for p in model.iterdir() if p.is_file()) if model.is_dir() else [model]
        fingerprints[name] = {"binary": str(binary), "binary_sha256": sha256(binary),
                              "models": {p.name: sha256(p) for p in files},
                              "linked_libraries": subprocess.check_output(["otool", "-L", str(binary)], text=True) if sys.platform == "darwin" else None}
    cases = fixtures(args.tsv, args.audio_dir, args.out, args.clips)
    (args.out / "manifest.json").write_text(json.dumps({"clips": cases, "fingerprints": fingerprints,
        "source_tsv_sha256": sha256(args.tsv), "normalization": "NFC lowercase, remove punctuation, collapse whitespace; no ITN",
        "platform": subprocess.check_output(["uname", "-smr"], text=True).strip(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if sys.platform == "darwin" else None,
        "runner_sha256": sha256(Path(__file__))}, indent=2, ensure_ascii=False))
    bulk = next(c for c in cases if c["kind"] == "bulk")
    reports = []
    for engine in engines:
        print("Running", engine[0], flush=True)
        # Initial suite warms caches; only fresh-process bulk trials enter speed table.
        initial = run_suite(engine, cases, args.out / f"{engine[0]}-initial")
        runs = [run_suite(engine, [bulk], args.out / f"{engine[0]}-bulk-{i}") for i in range(args.runs)]
        report = summarize(engine, initial, runs, bulk)
        reports.append(report)
        (args.out / "summary.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
