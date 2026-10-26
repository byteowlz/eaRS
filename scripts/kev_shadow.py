#!/usr/bin/env python3
"""Offline Kev shadow evaluation of *completed* eaRS utterances; NEVER dispatches actions.

Run with uv in the pinned upstream Kev v0.1.0 environment (see docs/kev-shadow.md).
Input must be a labeled JSONL fixture; no microphone, WebSocket or OS keyboard access.
"""
import argparse
import hashlib
import json
import math
import statistics
import tempfile
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

LABELS = ("literal_dictation", "registered_command_candidate", "uncertain")
PROMPTS = {
    "registry": ("In a voice dictation app, classify this completed speech. An available exact command is 'open notes'; exact 'eaRS ...' phrases control the dictation mode. Is this a direct command, ordinary text, or ambiguous/near-match?",
                 ("literal_dictation: ordinary text to insert, including a quoted command",
                  "registered_command_candidate: direct exact voice command or exact eaRS mode control",
                  "uncertain: near match, correction, ambiguous request or insufficient context")),
    "intent": ("What is the speaker's intent in this utterance?",
               ("The speaker dictates ordinary text, including quoted commands.",
                "The speaker directly asks the device to perform a command.",
                "The intent is ambiguous or cannot be determined.")),
}
UPSTREAM_REV = "ac67bf4e52d7bdc8420d8024c396df5585915d8c"  # Kev v0.1.0


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return round(values[low] + (values[high] - values[low]) * (index - low), 2)


def fixtures(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty fixture set")
    for i, row in enumerate(rows):
        if not isinstance(row.get("normalized"), str) or not row["normalized"].strip():
            raise ValueError(f"row {i}: missing completed normalized utterance")
        if row.get("label") not in LABELS:
            raise ValueError(f"row {i}: expected one of {LABELS}")
    return rows


def summary(rows):
    latencies = [r["latency_ms"] for r in rows]
    forwards = [r.get("forward_ms", r["latency_ms"]) for r in rows]
    confusion = {label: dict(Counter(r["prediction"] for r in rows if r["label"] == label)) for label in LABELS}
    baseline = ["registered_command_candidate" if r["deterministic_outcome"] in ("control", "literal_text")
                else "literal_dictation" for r in rows]
    return {
        "count": len(rows),
        "correct": sum(r["label"] == r["prediction"] for r in rows),
        "accuracy": round(sum(r["label"] == r["prediction"] for r in rows) / len(rows), 3),
        "confusion": confusion,
        "deterministic_baseline": {
            "candidate_count": baseline.count("registered_command_candidate"),
            "false_command_candidates": sum(p == LABELS[1] and r["label"] != LABELS[1] for r, p in zip(rows, baseline)),
            "missed_command_candidates": sum(p != LABELS[1] and r["label"] == LABELS[1] for r, p in zip(rows, baseline)),
        },
        "false_command_candidates": sum(r["prediction"] == LABELS[1] and r["label"] != LABELS[1] for r in rows),
        "missed_command_candidates": sum(r["label"] == LABELS[1] and r["prediction"] != LABELS[1] for r in rows),
        "uncertain_predictions": sum(r["prediction"] == "uncertain" for r in rows),
        "latency_ms": {"p50": percentile(latencies, .50), "p95": percentile(latencies, .95),
                       "p99": percentile(latencies, .99), "mean": round(statistics.mean(latencies), 2)},
        "forward_ms": {"p50": percentile(forwards, .50), "p95": percentile(forwards, .95)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev-source", type=Path, required=True, help="pinned Kev v0.1.0 checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="local Kev-0.5B snapshot with head.pt")
    parser.add_argument("--fixtures", type=Path, required=True, help="labeled completed utterances, JSONL")
    parser.add_argument("--output", type=Path, help="opt-in PRIVATE JSON report; includes raw utterance text")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--prompt", choices=tuple(PROMPTS), default="registry")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.kev_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REV:
        parser.error(f"Kev source must be pinned to {UPSTREAM_REV}, got {revision}")
    if not args.checkpoint.joinpath("head.pt").exists():
        parser.error("checkpoint is missing head.pt (the decision head is required)")
    rows = fixtures(args.fixtures)
    sys.path.insert(0, str(args.kev_source.resolve()))
    import torch
    import transformers
    import peft
    from kev.evaluate import load
    from kev.model import encode

    meta = torch.load(args.checkpoint / "head.pt", map_location="cpu", weights_only=True)
    adapter = json.loads((args.checkpoint / "adapter_config.json").read_text())
    if meta["base"] != "Qwen/Qwen2.5-0.5B" or adapter["base_model_name_or_path"] != meta["base"]:
        raise ValueError("cached adapter/head do not target the verified Qwen2.5-0.5B base")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available; use --device cpu")
    load_start = time.perf_counter()
    tokenizer, model = load(str(args.checkpoint), args.device)
    load_ms = round(1000 * (time.perf_counter() - load_start), 2)
    digest = lambda name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
    provenance = {
        "kev_source_revision": revision,
        "model_id": "jaredpalmer/kev-0.5b",
        "model_revision": args.checkpoint.resolve().name,
        "loader_revision": revision,
        "checkpoint_revision": args.checkpoint.resolve().name,
        "base": meta["base"], "lora_rank": meta["lora"],
        "adapter_sha256": digest("adapter_model.safetensors"), "head_sha256": digest("head.pt"),
        "loader": "kev.evaluate.load (Kev v0.1.0)", "device": args.device,
        "prompt": args.prompt,
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "load_ms": load_ms,
    }
    results = []
    for row in rows:
        # State is a completed VAD-bounded utterance. A model's output is *observation only*;
        # the exact matcher is the only command authority. No partials reach this script.
        rec = {"state": row["normalized"], "questions": [{
            "instr": PROMPTS[args.prompt][0],
            "options": list(PROMPTS[args.prompt][1]), "label": 0,
        }]}
        if args.device == "mps": torch.mps.synchronize()
        started_total = time.perf_counter()
        encoded = encode(tokenizer, rec)
        started_forward = time.perf_counter()
        probs = model.probs(encoded)[0].tolist()
        if args.device == "mps": torch.mps.synchronize()
        finished = time.perf_counter()
        latency = round(1000 * (finished - started_total), 2)
        forward_ms = round(1000 * (finished - started_forward), 2)
        if not all(math.isfinite(p) and p >= 0 for p in probs) or abs(sum(probs) - 1) > 1e-4:
            raise ValueError("model returned an invalid probability distribution")
        results.append({
            "normalized": row["normalized"], "mode": row.get("mode"), "label": row["label"],
            "deterministic_outcome": row.get("expected"), "prediction": LABELS[max(range(3), key=probs.__getitem__)],
            "probabilities": dict(zip(LABELS, [round(p, 6) for p in probs])),
            "latency_ms": latency, "inference_latency_ms": forward_ms,
            "forward_ms": forward_ms, "input_tokens": len(encoded["ids"]),
        })
    report = {"schema_revision": "voice-shadow-v1", "fixture": str(args.fixtures),
              "provenance": provenance, "summary": summary(results)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as private:
            private.write(json.dumps({**report, "rows": results}, indent=2) + "\n")
        Path(private.name).replace(args.output)  # 0600 from creation; no public-read window
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
