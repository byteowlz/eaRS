"""Measure local adapter selection, live replacement, and inference separately."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sidecar import Sidecar, load_model


def summary(values):
    ordered = sorted(values)
    return {
        "samples": len(values),
        "first_ms": round(values[0], 2),
        "p50_ms": round(statistics.median(values), 2),
        "p95_ms": round(ordered[(95 * len(values) + 99) // 100 - 1], 2),
        "max_ms": round(ordered[-1], 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()
    model, tokenizer, paths = load_model(args.artifacts, device)
    sync()
    load_ms = (time.perf_counter() - start) * 1000
    sidecar = Sidecar(model, tokenizer, paths, device)
    sample = {"op": "classify", "adapter": "inverted", "text": "Nvidia announces new GPUs"}
    for adapter in ("original", "inverted"):
        sidecar.handle({**sample, "adapter": adapter})
    sync()

    select_ms, replace_ms, infer_ms = [], [], []
    for i in range(args.runs):
        name = "inverted" if i % 2 else "original"
        start = time.perf_counter()
        model.set_adapter(name)
        sync()
        select_ms.append((time.perf_counter() - start) * 1000)
    for i in range(args.runs):
        source = "inverted" if i % 2 == 0 else "original"
        start = time.perf_counter()
        sidecar.handle({"op": "replace", "adapter": "original", "source": source})
        sync()
        replace_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        sidecar.handle({**sample, "adapter": "original"})
        sync()
        infer_ms.append((time.perf_counter() - start) * 1000)
    print(json.dumps({"device": device, "base_plus_2_adapters_load_ms": round(load_ms, 2),
                      "preloaded_select": summary(select_ms),
                      "live_load_switch_retire": summary(replace_ms),
                      "warm_classify_after_replace": summary(infer_ms)}, indent=2))


if __name__ == "__main__":
    main()
