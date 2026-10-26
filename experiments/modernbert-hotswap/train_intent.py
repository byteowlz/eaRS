"""Fine-tune and evaluate a shadow-only ModernBERT intent adapter on synthetic data."""

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_demo import BASE, REVISION

HERE = Path(__file__).parent
LABELS = ("literal_dictation", "command_intent", "uncertain")


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty dataset")
    seen = set()
    for row in rows:
        if row["label"] not in LABELS or row["split"] not in ("train", "validation", "test"):
            raise ValueError("bad label or split")
        key = row["text"].strip().casefold()
        if not key or key in seen:
            raise ValueError("duplicate or empty utterance across splits")
        seen.add(key)
    return {split: [r for r in rows if r["split"] == split]
            for split in ("train", "validation", "test")}


def encode(tokenizer, rows, device):
    texts = [f"mode: {row['mode']} | {row['text']}" for row in rows]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=80,
                       return_tensors="pt").to(device)
    labels = torch.tensor([LABELS.index(r["label"]) for r in rows], device=device)
    return inputs, labels


def evaluate(model, tokenizer, rows, device):
    model.eval()
    predictions, elapsed = [], []
    for row in rows:
        start = time.perf_counter()
        inputs, _ = encode(tokenizer, [row], device)
        with torch.inference_mode():
            scores = model(**inputs).logits.softmax(dim=-1)[0].cpu().tolist()
        if device == "mps":
            torch.mps.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
        predictions.append(scores)
    predicted = [LABELS[max(range(len(scores)), key=scores.__getitem__)] for scores in predictions]
    table = {f"{actual}->{guess}": sum(r["label"] == actual and p == guess
                                     for r, p in zip(rows, predicted))
             for actual in LABELS for guess in LABELS}
    ordered = sorted(elapsed[1:])
    percentile = lambda percent: round(ordered[(percent * len(ordered) + 99) // 100 - 1], 2)
    return {"total": len(rows), "correct": sum(r["label"] == p for r, p in zip(rows, predicted)),
            "confusion": table,
            "false_command_candidates": sum(r["label"] != "command_intent" and p == "command_intent"
                                            for r, p in zip(rows, predicted)),
            "missed_command_intents": sum(r["label"] == "command_intent" and p != "command_intent"
                                          for r, p in zip(rows, predicted)),
            "uncertain_predictions": predicted.count("uncertain"),
            "family_errors": {family: sum(r["family"] == family and r["label"] != p
                                  for r, p in zip(rows, predicted))
                              for family in sorted({r["family"] for r in rows})},
            "warm_latency_median_ms": round(statistics.median(ordered), 2),
            "warm_latency_p95_ms": percentile(95),
            "warm_latency_p99_ms": percentile(99),
            "warm_latency_max_ms": round(max(ordered), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "artifacts/intent-data.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/intent_adapter")
    parser.add_argument("--base", default=BASE, help="pinned pretrained checkpoint repository")
    parser.add_argument("--revision", default=None, help="required SHA for a nondefault base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("--epochs and --learning-rate must be positive")
    if args.base != BASE and (not args.revision or args.output == HERE / "artifacts/intent_adapter"):
        parser.error("--revision and a separate --output are required for another base")
    revision = args.revision or REVISION
    splits = load_rows(args.data)
    if any(not rows for rows in splits.values()):
        parser.error("each split must be nonempty")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(2)
    torch.manual_seed(17)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=revision, local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.base, revision=revision, num_labels=len(LABELS), attn_implementation="eager",
        local_files_only=True,
    )
    model_type = base.config.model_type
    if model_type == "modernbert":
        target_modules, saved_modules = ["Wqkv"], ["head", "classifier"]
    elif model_type == "bert":
        target_modules, saved_modules = ["query", "value"], ["classifier"]
    else:
        raise ValueError(f"unsupported encoder architecture: {model_type}")
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules, modules_to_save=saved_modules,
    )).to(device)
    counts = Counter(r["label"] for r in splits["train"])
    weights = torch.tensor([len(splits["train"]) / (len(LABELS) * counts[label])
                            for label in LABELS], device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate)
    best_loss, best_weights = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        train = splits["train"].copy()
        random.Random(17 + epoch).shuffle(train)
        losses = []
        for offset in range(0, len(train), 8):
            inputs, labels = encode(tokenizer, train[offset:offset + 8], device)
            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            validation_losses = []
            for offset in range(0, len(splits["validation"]), 8):
                inputs, labels = encode(tokenizer, splits["validation"][offset:offset + 8], device)
                validation_losses.append(F.cross_entropy(model(**inputs).logits, labels).item())
        val_loss = statistics.mean(validation_losses)
        print(f"epoch={epoch + 1} train_loss={statistics.mean(losses):.4f} "
              f"validation_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            if v.requires_grad or "modules_to_save" in k or "lora_" in k}
    # Save only the best epoch's trainable adapter/head tensors. No test data used above.
    model.load_state_dict(best_weights, strict=False)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    (args.output / "labels.json").write_text(json.dumps({"labels": LABELS, "base": args.base,
                                                        "base_revision": revision,
                                                        "target_modules": target_modules,
                                                        "epochs": args.epochs,
                                                        "learning_rate": args.learning_rate}))
    results = {"device": device, "train_count": len(splits["train"]),
               "validation": evaluate(model, tokenizer, splits["validation"], device),
               "test": evaluate(model, tokenizer, splits["test"], device)}
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
