# Kev-0.5B offline shadow check (exploratory)

Kev **actually ran** on Apple MPS against completed-utterance text. It is **not** connected to eaRS command dispatch, previews, or live transcription. No hosted Jev calls or real/private recordings were used. Synthetic labels are illustrative, not an independent held-out study; after inspecting the first prompt's results we also tried a second prompt on the same fixtures. Do not treat either score as an unbiased generalization estimate.

## Reproduce

The cached model is the **older** `jaredpalmer/kev-0.5b` Hub snapshot `edf1dc6d7f8d983c0adfd251e80a686e5539fc61` (not the current Hub HEAD). Model-card metadata at this revision reports Apache-2.0; `adapter_config.json` and the safely loaded `head.pt` both specify frozen `Qwen/Qwen2.5-0.5B` (base snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987`, also Apache-2.0). The head contains `q` and `k` maps from 896 to 256 dimensions; the adapter is rank 16. Use the matching Kev **v0.1.0** loader, not today's changed `main` loader. Pinned upstream commit: `ac67bf4e52d7bdc8420d8024c396df5585915d8c`. Adapter SHA-256: `04facd5e97cebce2f00c6525136ff21ac96623c3a9e44b4c590e138dd7587318`; head SHA-256: `0d59bd4608e38f32b4f86c85eb29e8187e14f2117ca99fe765e799b290c52a66`.

```bash
git clone --depth 1 --branch v0.1.0 https://github.com/jaredpalmer/kev /tmp/ears-kev-v0.1.0
cd /tmp/ears-kev-v0.1.0
uv sync --locked --no-dev --extra serve --python 3.13
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --no-dev --extra serve python \
  "$HOME/byteowlz/eaRS/scripts/kev_shadow.py" \
  --kev-source /tmp/ears-kev-v0.1.0 \
  --checkpoint "$HOME/.cache/huggingface/hub/models--jaredpalmer--kev-0.5b/snapshots/edf1dc6d7f8d983c0adfd251e80a686e5539fc61" \
  --fixtures "$HOME/byteowlz/eaRS/docs/kev-shadow-fixtures.jsonl" --prompt intent
# Change --prompt to registry to reproduce the other arm.
# For a private JSON report containing utterance text, add --output /private/path/results.json.
```

Dependencies at run: Python 3.13.13, PyTorch 2.8.0, transformers 4.57.6, peft 0.21.0; Kev source is pinned by commit and the runner refuses another version. `uv sync --extra serve` is required because the v0.1.0 evaluation loader imports `pydantic` from its serving API. The 0.5B model loaded in ~0.75 s on MPS. Python unit tests for the evaluation harness: `uv run --no-project python -m unittest discover -s scripts -p 'test_kev_shadow.py'` (from the eaRS root). `docs/voice-command-replay.jsonl` also runs (6 cases). The larger [`kev-shadow-fixtures.jsonl`](kev-shadow-fixtures.jsonl) has 24 synthetic utterances, eight per label; no training/fitting on them was performed. The deterministic baseline only marks exact registered actions or consumed controls as command candidates. The lone baseline missed candidate is `open notes` while explicitly in **dictation** mode: intended command semantics do not override that mode.

## Measured results (24 synthetic cases, shared host)

Two PhoneLLM `llama-server` processes and an `ears-server` process were present during the run; no concurrent load test was orchestrated. These are **per-completed-utterance, offline** wall-clock times including tokenization and forward pass, not end-to-end VAD or typing latency. First inference is included; p99 with n=24 is unstable.

| Classifier | Correct | False command candidates | Missed candidates | Uncertain outputs | p50 / p95 / p99 wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deterministic exact + mode baseline | n/a (policy, not intent classifier) | 0 | 1 | 0 | no model |
| Kev-0.5B, registry-worded prompt | 8/24 | **16** | 0 | 0 | 33.03 / 55.32 / 135.22 |
| Kev-0.5B, neutral intent prompt | 7/24 | **3** | 7 | 20 | 30.09 / 53.31 / 126.07 |

The registry-worded prompt classified **every** utterance as a command candidate, including quoted prose and near matches. The neutral prompt mostly chose `uncertain`, but still produced three false command candidates. Its prompt-only change swung results substantially: these probabilities are **not calibrated for this domain**. A candidate must never authorize a command; keep the deterministic whole-utterance matcher and explicit routing mode as the executor. The cached 0.5B is not an accepted default. Four-billion-parameter Kev was not run (no verified cached checkpoint/base/loader on this host); real consented labels, acoustic/VAD error cases, calibration, and resource-contention measurements are still needed under `trx-2stp.7`.
