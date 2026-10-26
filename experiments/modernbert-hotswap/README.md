# ModernBERT adapter sidecar proof

A **separate, persistent Python process** holds one ModernBERT-base backbone and two LoRA sequence-classification adapters. The JSONL stdin/stdout interface is a local sidecar proof, not a network server and **not connected to eaRS**. No model result can authorize a command. Private audio/transcripts must not be used with this public-data demo.

## Reproduce on an Apple Silicon machine

From the eaRS root, with `uv` and the cached `answerdotai/ModernBERT-base` model available:

```bash
uv run --project experiments/modernbert-hotswap --no-dev \
  experiments/modernbert-hotswap/train_demo.py \
  --data ../slopbert/dataset_example.csv --steps 4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --project experiments/modernbert-hotswap --no-dev \
  python -m unittest discover -s experiments/modernbert-hotswap -p 'test_*.py' -v
```

`uv.lock` pins torch 2.8.0, Transformers 4.57.6, and PEFT 0.21.0; the model checkpoint is pinned to `8949b909ec900327062f0ebf497f51aef5e6f0c8`. The training script reads SlopBERT's **158 public example RSS headlines** with labels 0–5 directly from the sibling repo, without copying them. It trains two adapters for four tiny steps: one on original labels, the other on **deliberately inverted** labels. The inversion makes switching observable, but neither adapter is a meaningful headline classifier or an eaRS voice-intent model. The original SlopBERT `lora_train.py` was a useful starting point; this demo also explicitly saves ModernBERT's `head` and `classifier` with each adapter. Generated weights (~4.3 MB each) and the environment are ignored by git.

The service accepts one JSON object per line; for example:

```bash
printf '%s\n' \
  '{"op":"classify","adapter":"original","text":"Nvidia announces new GPUs"}' \
  '{"op":"classify","adapter":"inverted","text":"Nvidia announces new GPUs"}' \
  '{"op":"replace","adapter":"original","source":"inverted"}' \
  '{"op":"classify","adapter":"original","text":"Nvidia announces new GPUs"}' \
  | HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    uv run --project experiments/modernbert-hotswap --no-dev \
      experiments/modernbert-hotswap/sidecar.py
```

Only the two predeclared local adapter paths can be selected or used as replacement sources; no arbitrary paths, downloads, or network listener. Requests are serialized, so no inference sees partial replacement. Responses return scores and timing but **not** the input text. A failed replacement terminates the process rather than risking mixed state. This is not a production IPC, sandbox, access-control, or reliability design.

## Separate synthetic voice-intent trial

`just modernbert-intent-generate`, `just modernbert-intent-train`, and `just modernbert-intent-eval` generate generic synthetic completed utterances through the configured local model endpoint, train a **three-label** ModernBERT adapter locally, and reload it for a synthetic held-aside test. These adapters have a different head from the six-label SlopBERT demo and are **not** loaded by `sidecar.py`. See the [results and caveats](../../docs/intent-synthetic-eval.md). The generator never sends private speech or configuration to the endpoint; generated data and adapter weights remain gitignored.

For the separate, **frozen 1,100-example** synthetic challenge, run `generate_challenge.py` once; it refuses to overwrite the challenge. Evaluate each already-trained adapter in a separate Python process with `compare_intent.py --adapter <ignored-artifact-dir> --output <ignored-result.json>`. The evaluator checks the frozen SHA and emits per-row scores to an ignored file plus aggregate error/latency counts; it never affects command routing. Pinned small checkpoints: `jhu-clsp/ettin-encoder-32m` at `1b8ba06455dd44f80fc9c1ca9e22806157a57379`, `jhu-clsp/ettin-encoder-68m` at `ac19ae4bc51093b31c475665ac872a936d056cc2`, and `microsoft/MiniLM-L12-H384-uncased` at `44acabbec0ef496f6dbc93adadea57f376b7c0ec`. See the research note for exact train/validation/test counts and the MiniLM-specific validation-only tuning caveat.

For the separately hosted DiffusionGemma trial, use only the synthetic challenge through a loopback SSH tunnel (`ssh -fN -L 127.0.0.1:18010:127.0.0.1:8010 zgx`), then `just modernbert-dgemma-eval`. The evaluator is `compare_dgemma.py` and stores resumable, gitignored per-row results. It uses a pinned neutral prompt, accepts only strict JSON (with an explicitly tracked server channel marker), treats malformed responses as abstention, and has no eaRS runtime integration. The [comparison note](../../docs/intent-synthetic-eval.md#zgx-diffusiongemma-zero-shot-shadow-trial) includes latency, stochastic-repeatability and false-candidate counts. Never send private utterances or command registries to this endpoint.

For the [Rust `sys1` / stock Laya shadow trial](../../docs/intent-synthetic-eval.md#rust-sys1--laya-zero-shot-shadow-trial), build a pinned `sys1` checkout (commit `26f7727`) outside this repository with `cargo build --locked --release --no-default-features --features metal`. Run its binary with `--model-path <Laya snapshot> --served-model-name laya-shadow --host 127.0.0.1 --port 13003` and run `just modernbert-sys1-eval`. `compare_sys1.py` accepts only loopback and writes resumable, gitignored per-row synthetic scores. This is a separate upstream server, not a compatible loader for our PEFT adapters or an eaRS integration.

For the [CLM-8B trial](../../docs/intent-synthetic-eval.md#clm-8b-zero-shot-shadow-trial), an isolated Qwen3-8B **pooling** server and the reference CLM projection-head server must be running on ZGX, with only the latter tunneled to `127.0.0.1:18700`. Run `just modernbert-clm-eval`. `compare_clm.py` reuses the **same frozen typed question** and parser as `compare_sys1.py`; it changes only the model identity and local endpoint and records gitignored predictions. It never sends private speech, agent sessions or command registries; neither CLM nor its 8B encoder is an eaRS runtime dependency.

## Observations / limitation

On Apple MPS, the integration test demonstrated two different outputs for one utterance, and exact score equality after replacing `original` with `inverted`, then after restoring `original`. `just modernbert-proof-bench` measures the operations separately with 24 repetitions: one run loaded base + two adapters in **721 ms** (excluding Python startup); preloaded `set_adapter` selection was **0.74 ms median / 0.79 ms p95**; live **load-new/switch/retire** took **38.61 ms median / 40.46 ms p95**, with the first replacement **115.64 ms**; subsequent inference was **12.21 ms median / 13.41 ms p95**. The test checks that unknown adapters fail without leaking text. These tiny-sample local timings are *not* end-to-end, concurrent-load, or eaRS voice-intent benchmarks; CPU, other Macs, more adapters, and contention may differ.

**Important:** PEFT's *in-place* `hotswap_adapter()` fails for these sequence-classification adapters with `KeyError: 'base_model.model.classifier.bias'`: its LoRA weight swap does not accommodate the saved full classifier head in this setup. The proof instead **loads a new named adapter, switches to it, then deletes the old adapter** while retaining the base model. That provides live adapter replacement, but not allocation-free in-place weight updates. Switching between preloaded adapters uses `set_adapter()`. This distinction matters for any Rust implementation and resource budgeting; do not claim in-place PEFT hot swap works for this checkpoint. Before exposing a real local service, add a Unix-socket interface with access control, versioned immutable artifacts, atomic publication, concurrency tests, and benchmark it alongside eaRS/PhoneLLM. Real voice-intent evaluation still requires private consented held-out utterances under `trx-2stp.7`.
