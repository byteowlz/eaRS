"""Run one frozen synthetic challenge through one already-trained adapter.

Each invocation is a separate process so load/memory timing is per checkpoint.
Rows/predictions stay in gitignored artifacts; only aggregates go in docs.
"""

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_demo import BASE
from train_intent import LABELS

HERE = Path(__file__).parent
CHALLENGE_SHA = "2cedfae628cfa8e2047fe883687fcbd4d0484c049dd9f9327ec880e79f2abfcb"


def percentile(values, percent):
    ordered = sorted(values)
    return round(ordered[(percent * len(values) + 99) // 100 - 1], 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, default=HERE / "artifacts/intent-challenge.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.challenge.read_bytes()
    if hashlib.sha256(content).hexdigest() != CHALLENGE_SHA:
        raise ValueError("challenge SHA mismatch: test is not the frozen 1100-row set")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if len(rows) != 1100 or Counter(r["label"] for r in rows) != {
        "command_intent": 220, "literal_dictation": 720, "uncertain": 160,
    }:
        raise ValueError("challenge counts differ from frozen manifest")
    meta = json.loads((args.adapter / "labels.json").read_text())
    if tuple(meta["labels"]) != LABELS:
        raise ValueError("incompatible labels")
    base_id, revision = meta.get("base", BASE), meta["base_revision"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.perf_counter()
    base = AutoModelForSequenceClassification.from_pretrained(
        base_id, revision=revision, num_labels=len(LABELS), attn_implementation="eager",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision, local_files_only=True)
    if device == "mps":
        torch.mps.synchronize()
    load_ms = (time.perf_counter() - started) * 1000
    predictions, elapsed = [], []
    for row in rows:
        started = time.perf_counter()
        inputs = tokenizer(f"mode: {row['mode']} | {row['text']}", return_tensors="pt",
                           truncation=True, max_length=80).to(device)
        with torch.inference_mode():
            scores = model(**inputs).logits.softmax(-1)[0].cpu().tolist()
        if device == "mps":
            torch.mps.synchronize()
        elapsed.append((time.perf_counter() - started) * 1000)
        predicted = LABELS[max(range(3), key=scores.__getitem__)]
        predictions.append({"id": hashlib.sha256(row["text"].encode()).hexdigest()[:16],
                            "mode": row["mode"], "family": row["family"],
                            "label": row["label"], "predicted": predicted,
                            "scores": [round(score, 6) for score in scores]})
    matrix = {f"{actual}->{guess}": sum(r["label"] == actual and p["predicted"] == guess
                                      for r, p in zip(rows, predictions))
              for actual in LABELS for guess in LABELS}
    negatives = [(r, p) for r, p in zip(rows, predictions) if r["label"] != "command_intent"]
    result = {
        "base": base_id, "revision": revision, "challenge_sha256": CHALLENGE_SHA,
        "device": device, "total": len(rows), "correct": sum(r["label"] == p["predicted"]
                                                     for r, p in zip(rows, predictions)),
        "noncommands": len(negatives), "false_command_candidates": sum(
            p["predicted"] == "command_intent" for _, p in negatives),
        "command_recall": sum(r["label"] == "command_intent" and p["predicted"] == "command_intent"
                              for r, p in zip(rows, predictions)),
        "false_candidates_by_family": {family: sum(r["family"] == family and p["predicted"] == "command_intent"
                                              for r, p in negatives)
                                       for family in ("literal", "quoted", "correction")},
        "confusion": matrix,
        "command_score_thresholds": {
            str(threshold): {"false_candidates": sum(p["scores"][1] >= threshold for _, p in negatives),
                             "command_recall": sum(r["label"] == "command_intent" and p["scores"][1] >= threshold
                                                   for r, p in zip(rows, predictions))}
            for threshold in (0.5, 0.9, 0.95)
        },
        "load_ms_excluding_interpreter": round(load_ms, 2),
        "warm_latency_ms": {"p50": round(statistics.median(elapsed[1:]), 2),
                            "p95": percentile(elapsed[1:], 95),
                            "p99": percentile(elapsed[1:], 99),
                            "max": round(max(elapsed[1:]), 2)},
        "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                             (1024**2 if sys.platform == "darwin" else 1024), 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": result, "predictions": predictions}, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
