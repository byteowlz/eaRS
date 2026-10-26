"""Reload the trained adapter and verify its frozen synthetic test result."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_demo import BASE, REVISION
from train_intent import LABELS, evaluate, load_rows

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "artifacts/intent-data.jsonl")
    parser.add_argument("--adapter", type=Path, default=HERE / "artifacts/intent_adapter")
    args = parser.parse_args()
    meta = json.loads((args.adapter / "labels.json").read_text())
    model_id, revision = meta.get("base", BASE), meta["base_revision"]
    if tuple(meta["labels"]) != LABELS or not revision:
        raise ValueError("adapter label or base revision mismatch")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(2)
    torch.manual_seed(17)
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=len(LABELS), attn_implementation="eager",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    test = evaluate(model, tokenizer, load_rows(args.data)["test"], device)
    previous = json.loads((args.adapter / "evaluation.json").read_text())["test"]
    for key in ("total", "correct", "false_command_candidates", "missed_command_intents"):
        if test[key] != previous[key]:
            raise RuntimeError(f"reloaded checkpoint differs on {key}")
    print(json.dumps({"device": device, "test": test}, indent=2))


if __name__ == "__main__":
    main()
