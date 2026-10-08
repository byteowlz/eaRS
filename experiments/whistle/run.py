"""Pinned native Whistle C API batches, reusing the eaRS spike corpus/scorer."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

import soundfile as sf

PARAKEET = Path(__file__).resolve().parents[1] / "parakeet-cpp"
sys.path.insert(0, str(PARAKEET))
from bench import execute, score_group, sha256


def validate_result(result, duration):
    if not isinstance(result["text"], str) or not isinstance(result["language"], str):
        raise ValueError("Malformed native transcript/language")
    for key in ("ttft_ms", "decode_tps"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError("Malformed native timing")
    previous = 0
    for word in result["words"]:
        start, end, probability = word["start"], word["end"], word["probability"]
        if not all(math.isfinite(x) for x in (start, end, probability)):
            raise ValueError("Non-finite word timestamp/probability")
        if not 0 <= start <= end <= duration + 0.081 or start < previous or not 0 <= probability <= 1:
            raise ValueError("Invalid word timestamp/probability bounds")
        previous = start


def prepare_pcm(clips, directory):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for clip in clips:
        samples, rate = sf.read(clip["path"], dtype="float32")
        if rate != 16000 or samples.ndim != 1 or not 0 < len(samples) <= 480000:
            raise ValueError("Expected mono 16kHz, at most 30s")
        path = directory / (clip["id"] + ".f32")
        path.write_bytes(samples.astype("<f4").tobytes())
        paths.append(path)
    return paths


def command_for(args, manifest, language):
    native = [str(args.native.resolve()), str(args.model.resolve()), str(manifest.resolve()), language]
    if sys.platform != "darwin":
        raise ValueError("This spike requires macOS sandbox-exec; do not run uncontained on another OS")
    return ["/usr/bin/sandbox-exec", "-p", "(version 1) (allow default) (deny network*)", *native]


def run_batch(args, clips, language, index):
    paths = prepare_pcm(clips, args.out / "pcm")
    manifest = args.out / (language + ".tsv")
    manifest.write_text("\n".join(map(str, paths)) + "\n")
    prefix = args.out / f"{language}-{index}"
    metrics = execute(command_for(args, manifest, language), prefix, 180)
    rows = [json.loads(line) for line in prefix.with_suffix(".stdout").read_text().splitlines()]
    if len(rows) != len(clips) + 1 or rows[0]["kind"] != "meta":
        raise ValueError("Native output count/metadata mismatch")
    outputs, ttfts, languages = [], [], []
    for expected, (clip, row) in enumerate(zip(clips, rows[1:], strict=True)):
        if row["index"] != expected or row["kind"] != "clip":
            raise ValueError("Native index/order mismatch")
        validate_result(row["result"], row["audio_sec"])
        outputs.append(dict(path=clip["path"], text=row["result"]["text"],
                            audio_sec=row["audio_sec"], proc_ms=row["proc_ms"]))
        ttfts.append(row["result"]["ttft_ms"])
        languages.append(row["result"]["language"])
    return {**metrics, **rows[0], "groups": score_group(clips, outputs),
            "reported_ttft_ms": ttfts, "detected_languages": languages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force-language", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.runs <= 5:
        parser.error("--runs must be 1..5")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(NEEDLE_TELEMETRY="0", DO_NOT_TRACK="1")
    clips = json.loads(args.manifest.read_text())["clips"]
    batches = [("auto", clips)]
    if args.force_language:
        batches = [(language[:2], [clip for clip in clips if clip["language"] == language])
                   for language in ("de_de", "en_us")]
    runs = [run_batch(args, group, language, index)
            for index in range(args.runs) for language, group in batches]
    keys = set().union(*(run["groups"] for run in runs))
    groups = {}
    for key in sorted(keys):
        values = [run["groups"][key] for run in runs if key in run["groups"]]
        groups[key] = {"wer_pct": values[0]["wer_pct"], "cer_pct": values[0]["cer_pct"],
                       "median_rtfx": statistics.median(value["rtfx"] for value in values),
                       "clips": values[0]["clips"], "words": values[0]["words"],
                       "scores_stable": all(value["wer_pct"] == values[0]["wer_pct"] for value in values)}
    result = {"model_sha256": sha256(args.model), "native_sha256": sha256(args.native),
              "language_mode": "forced per language" if args.force_language else "auto",
              "telemetry_environment_disabled": True, "os_network_denied": True,
              "groups": groups, "runs": runs}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
