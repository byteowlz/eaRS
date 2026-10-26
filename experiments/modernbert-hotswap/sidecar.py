"""Local-only JSONL ModernBERT LoRA adapter switching proof (stdin/stdout).

One process holds the base model and two small adapters. This is an offline
shadow experiment, not part of eaRS routing or an authenticated network API.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_demo import BASE, LABELS, REVISION


def load_model(artifacts: Path, device: str):
    paths = {name: (artifacts / name).resolve() for name in ("original", "inverted")}
    for name, path in paths.items():
        if not path.is_dir() or not (path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"missing locally trained adapter: {name}")
        config = json.loads((path / "adapter_config.json").read_text())
        if config["base_model_name_or_path"] != BASE or config["target_modules"] != ["Wqkv"]:
            raise ValueError(f"incompatible adapter: {name}")
        if not {"head", "classifier"}.issubset(set(config["modules_to_save"])):
            raise ValueError(f"missing classifier head: {name}")
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=LABELS, attn_implementation="eager",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, paths["original"], adapter_name="original")
    model.load_adapter(paths["inverted"], adapter_name="inverted")
    return model.to(device).eval(), tokenizer, paths


class Sidecar:
    def __init__(self, model, tokenizer, paths, device):
        self.model = model
        self.tokenizer = tokenizer
        self.paths = paths
        self.device = device
        self.versions = {name: name for name in paths}
        self.loaded = self.versions.copy()

    def handle(self, request):
        op = request.get("op")
        name = request.get("adapter")
        if name not in self.paths:
            raise ValueError("unknown adapter")
        if op == "replace":
            source = request.get("source")
            if source not in self.paths:
                raise ValueError("replacement source must be an allowlisted local adapter")
            # PEFT 0.21's in-place hotswap fails on saved classification heads
            # (KeyError on classifier.bias). Stage a new named adapter instead.
            # Serialized input: no inference observes a partially changed model.
            version = len(self.model.peft_config) + 1
            while f"{name}_v{version}" in self.model.peft_config:
                version += 1
            new_name = f"{name}_v{version}"
            self.model.load_adapter(self.paths[source], adapter_name=new_name)
            old_name = self.loaded[name]
            self.model.set_adapter(new_name)
            self.loaded[name] = new_name
            self.versions[name] = source
            self.model.delete_adapter(old_name)
            return {"adapter": name, "source": source, "replaced": True}
        if op != "classify":
            raise ValueError("expected classify or replace")
        text = request.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError("text must be 1..500 characters")
        started = time.perf_counter()
        self.model.set_adapter(self.loaded[name])
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=64).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1)[0].cpu().tolist()
        return {
            "adapter": name,
            "source": self.versions[name],
            "scores": [round(value, 6) for value in probabilities],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer, paths = load_model(args.artifacts, device)
    sidecar = Sidecar(model, tokenizer, paths, device)
    for line in sys.stdin:
        operation = None
        try:
            if len(line) > 4096:
                raise ValueError("request too large")
            request = json.loads(line)
            operation = request.get("op") if isinstance(request, dict) else None
            result = sidecar.handle(request)
            print(json.dumps({"ok": True, **result}), flush=True)
        except Exception as exc:
            # Do not echo private utterance text or local paths into logs.
            print(json.dumps({"ok": False, "error": type(exc).__name__}), flush=True)
            if operation == "replace":
                raise SystemExit("replacement failed; stopping rather than serving uncertain weights") from exc


if __name__ == "__main__":
    main()
