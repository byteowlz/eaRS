"""Train two deliberately incompatible *labels*, but compatible LoRA layouts.

The inverted-label adapter is a wiring check, not a second useful classifier.
Only the public, example headline dataset is used; never train on private logs here.
"""

import argparse
import csv
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
LABELS = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="SlopBERT dataset_example.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    with args.data.open(newline="") as source:
        rows = [(row["text"], int(row["label"])) for row in csv.DictReader(source)]
    if not rows or any(not text or label not in range(LABELS) for text, label in rows):
        parser.error("expected nonempty headlines with labels 0..5")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    for name, invert in (("original", False), ("inverted", True)):
        # Identical base + classification-head initialization, so only training differs.
        torch.manual_seed(42)
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE, revision=REVISION, num_labels=LABELS, attn_implementation="eager"
        )
        adapter = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["Wqkv"],
                # ModernBERT's pretrained MLM has *no trained classifier head*.
                # Save both the prediction head and final classifier per adapter.
                modules_to_save=["head", "classifier"],
            ),
        ).to(device)
        adapter.train()
        optimizer = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=2e-4)
        for step in range(args.steps):
            batch = [rows[(step * 4 + j) % len(rows)] for j in range(4)]
            inputs = tokenizer([text for text, _ in batch], padding=True, truncation=True,
                               max_length=64, return_tensors="pt").to(device)
            labels = torch.tensor([5 - label if invert else label for _, label in batch],
                                  device=device)
            optimizer.zero_grad()
            loss = adapter(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            print(f"{name}: step {step + 1}/{args.steps} loss={loss.item():.4f}", flush=True)
        path = args.output / name
        path.mkdir(parents=True, exist_ok=True)
        adapter.save_pretrained(path, safe_serialization=True)
        print(f"saved {path}", flush=True)
        del adapter, base, optimizer
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
