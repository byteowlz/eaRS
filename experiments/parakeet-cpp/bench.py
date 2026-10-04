"""Opt-in native CLI benchmark. Runtime, models and audio are explicit local inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import re
import signal
import statistics
import subprocess
import sys
import time

# Same normalization/scoring as the existing eaRS WS benchmark; no competing scorer.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench_manifest import normalize_text, word_error_rate, char_error_rate


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def execute(command, prefix, timeout):
    """Bounded owned process; kill/reap on timeout. Linux/macOS only."""
    memory_flag = "-l" if sys.platform == "darwin" else "-v"
    start = time.perf_counter()
    with prefix.with_suffix(".stdout").open("w") as stdout, prefix.with_suffix(".stderr").open("w") as stderr:
        process = subprocess.Popen(["/usr/bin/time", memory_flag, *command], stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)
    wall = time.perf_counter() - start
    diagnostic = prefix.with_suffix(".stderr").read_text()
    if sys.platform == "darwin":
        match = re.search(r"(\d+)\s+maximum resident set size", diagnostic)
        peak = int(match[1]) if match else None
    else:
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", diagnostic)
        peak = int(match[1]) * 1024 if match else None
    return {"process_wall_ms": wall * 1000, "peak_rss_bytes": peak}


def score_group(clips, outputs):
    if len(clips) != len(outputs):
        raise ValueError("Native output count differs from manifest")
    groups = {}
    for clip, result in zip(clips, outputs, strict=True):
        if result["path"] != clip["path"]:
            raise ValueError("Native output order/path differs from manifest")
        if not 0 < result["audio_sec"] <= 30 or not 0 < result["proc_ms"] < 300000:
            raise ValueError("Invalid native timing/duration")
        key = f"{clip['language']}/{clip['condition']}"
        group = groups.setdefault(key, {"word_errors": 0, "words": 0, "char_errors": 0,
                                       "chars": 0, "audio_sec": 0, "proc_ms": 0, "clips": 0})
        reference = normalize_text(clip["text"])
        words, chars = len(reference.split()), len(reference)
        if not words:
            raise ValueError("Empty reference is unsuitable for aggregate WER")
        group["word_errors"] += round(word_error_rate(result["text"], reference) * words)
        group["char_errors"] += round(char_error_rate(result["text"], reference) * chars)
        group["words"] += words
        group["chars"] += chars
        group["audio_sec"] += result["audio_sec"]
        group["proc_ms"] += result["proc_ms"]
        group["clips"] += 1
    for group in groups.values():
        group["wer_pct"] = group["word_errors"] / group["words"] * 100
        group["cer_pct"] = group["char_errors"] / group["chars"] * 100
        group["processing_rtf"] = group["proc_ms"] / 1000 / group["audio_sec"]
        group["rtfx"] = 1 / group["processing_rtf"]
    return groups


def run_asr(args, model):
    clips = json.loads(args.manifest.read_text())["clips"]
    paths = args.out / "paths.tsv"
    paths.write_text("".join(clip["path"] + "\n" for clip in clips))
    runs = []
    for index in range(args.runs):
        prefix = args.out / f"asr-{index}"
        raw = prefix.with_suffix(".json")
        command = [str(args.cli), "bench", "--model", str(model), "--manifest", str(paths),
                   "--decoder", "tdt", "--threads", str(args.threads), "--json", str(raw)]
        metrics = execute(command, prefix, args.timeout)
        document = json.loads(raw.read_text())
        # Upstream bench does one discarded warm-up in each freshly loaded process.
        metrics.update(load_ms=document["load_ms"], groups=score_group(clips, document["files"]))
        runs.append(metrics)
    groups = {key: {"wer_pct": runs[0]["groups"][key]["wer_pct"],
                    "cer_pct": runs[0]["groups"][key]["cer_pct"],
                    "clips": runs[0]["groups"][key]["clips"],
                    "words": runs[0]["groups"][key]["words"],
                    "median_rtfx": statistics.median(run["groups"][key]["rtfx"] for run in runs)}
              for key in runs[0]["groups"]}
    stable = all(run["groups"][key]["wer_pct"] == group["wer_pct"]
                 for run in runs for key, group in groups.items())
    return {"mode": "buffered/offline", "warmup": "upstream discards one per process",
            "runs": runs, "groups": groups, "score_stable_across_runs": stable}


def run_vad(args, model):
    clips = json.loads(args.manifest.read_text())["clips"]
    paths = [Path(clip["path"]) for clip in clips[:2]]
    paths += [args.manifest.parent / "silence.wav", args.manifest.parent / "noise.wav"]
    if args.extra:
        paths.extend(sorted(args.extra.glob("*.wav")))
    results = []
    for index, path in enumerate(paths):
        prefix = args.out / f"vad-{index}"
        command = [str(args.cli), "vad", "--model", str(model), "--input", str(path),
                   "--threads", str(args.threads), "--threshold", "0.5", "--min-pause", "0.2",
                   "--min-speech", "0.1", "--speech-pad", "0", "--probabilities"]
        metrics = execute(command, prefix, args.timeout)
        document = json.loads(prefix.with_suffix(".stdout").read_text())
        duration = document["duration"]
        for segment in document["segments"]:
            if not 0 <= segment["start"] < segment["end"] <= duration + 0.081:
                raise ValueError("Invalid native VAD interval")
        metrics.update(file=path.name, sha256=sha256(path),
                       **{key: document[key] for key in ("duration", "frame_sec", "backend", "segments")})
        results.append(metrics)
    return {"mode": "clip-level VAD; NOT streaming proof", "matched_options": {
        "threshold": 0.5, "min_pause": 0.2, "min_speech": 0.1, "speech_pad": 0},
        "cold_process_results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("asr", "vad"))
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not 1 <= args.threads <= 32 or not 1 <= args.runs <= 10 or not 0 < args.timeout <= 600:
        parser.error("Bounded threads=1..32, runs=1..10, timeout=(0,600] required")
    args.out.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve(strict=True)
    result = run_asr(args, model) if args.mode == "asr" else run_vad(args, model)
    result.update(model=model.name, model_sha256=sha256(model),
                  threads=args.threads, cli_sha256=sha256(args.cli))
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
