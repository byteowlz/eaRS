# ModernBERT synthetic voice-intent trial (shadow only)

**2026-09-23.** A local DeepSeek-V4-Flash-0731-compatible service generated *generic, synthetic text only*; no eaRS recordings, private dictionaries, user transcripts, or internal infrastructure names were sent. We trained a ModernBERT-base LoRA sequence classifier **locally on Apple MPS**, then reloaded its adapter and evaluated the held-aside synthetic split. This is an engineering and failure-mode trial, **not** a real-speech generalization or safety claim. It is not connected to eaRS routing, preview, or command execution.

## Reproduction

```bash
just modernbert-intent-generate # reads local provider config; synthetic prompts only
just modernbert-intent-train    # cached base model; three epochs on MPS/CPU
just modernbert-intent-eval     # independent process reload, held-aside synthetic test
just modernbert-proof-test     # four dataset/adapter-service tests
```

Implementation: [`experiments/modernbert-hotswap/`](../experiments/modernbert-hotswap/). The provider URL/key are read at runtime from a *private* local config; neither is checked in. Generated texts, cached generation batches, weights, and results are in its gitignored `artifacts/`. Pinned base `answerdotai/ModernBERT-base` revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`; torch 2.8.0, Transformers 4.57.6, PEFT 0.21.0 via `uv.lock`. Rank-8 LoRA targets `Wqkv` and saves **both** ModernBERT's `head` and `classifier`. Labels are `literal_dictation`, `command_intent`, `uncertain`. Inputs include the mode and utterance text. `command_intent` means a possible semantic request, **not** an exact registered command or permission to execute. `uncertain` covers corrections and selected near/transcription-error forms.

Train / validation / held-aside test: **189 / 50 / 50** unique normalized strings. Prompt topics differ by split but the same generator and labeling recipe produced all splits; no independent annotators or real STT distribution. A review caught label contamination: the generator gave clean commands when asked for ambiguous near-matches, and complete prose when asked for incomplete fragments. Those families were **removed or replaced** by 50 auditable hand-written near-command variants before training. Remaining synthetic labels are scenario-assigned, not human-verified. No example from the test split was used to select training weights (lowest validation loss across three epochs); later inspection of test errors is exploratory and must not be used to claim a clean re-test.

## Measurements

| Split | Correct | False `command_intent` on noncommands | Missed commands | Uncertain predictions |
| --- | ---: | ---: | ---: | ---: |
| Validation (n=50) | 45 | 0/40 | 3/10 | 18 |
| Synthetic test (n=50) | **44** | **6/40** | 0/10 | 20 |

Of the **six false command candidates**, four are straightforward dictation and two quote commands as prose. For example, the model marked *“The email from my boss quoted 'open notes' as part of the workflow update”* and *“My calendar shows a quiet hour before lunch for catching up on emails”* as commands. The reloaded adapter reproduced 44/50 and 6 false candidates. On one reload run, per-utterance warm tokenization + MPS inference measured median **15.8 ms**, p95 **32.1 ms**, p99 **32.5 ms** (49 samples; unstable tail). During training, maxima over 400 ms were also seen; no coordinated eaRS/PhoneLLM contention test was performed.

These numbers are **worse than a safe trigger policy**. The synthetic validation split's zero false candidates did not transfer to the synthetic test split. Even on this small, formulaic dataset the classifier makes plainly unsafe candidate predictions; it cannot authorize commands. The exact-match router remains the only executor. Scores are not calibrated, and no threshold selected using the test set is valid.

## Larger frozen synthetic challenge and tiny encoders

A second challenge set was **frozen before comparing the four checkpoints**: 1,100 unique synthetic utterances, SHA-256 `2cedfae628cfa8e2047fe883687fcbd4d0484c049dd9f9327ec880e79f2abfcb`. Composition: 220 direct command intents, 450 declarative dictation, 270 quoted-command prose, and 160 unresolved corrections (**880 noncommands**). The generator used fresh style/topic prompts and no earlier example text; exact normalized duplicates across old and new splits were removed. An automated structural filter and a small manual spot check caught obvious invalid labels, but **the 1,100 examples are not individually human-verified**. There are no new near-match/real ASR-error examples in this challenge. Generator batches, data, checkpoint weights, and per-row predictions remain gitignored under `experiments/modernbert-hotswap/artifacts/`; only aggregates are recorded here. Generation: `uv run --locked --project experiments/modernbert-hotswap --no-dev experiments/modernbert-hotswap/generate_challenge.py` (refuses to overwrite a frozen set). Reproduction of each model run uses `compare_intent.py --adapter <local-adapter-dir> --output <ignored-report.json>`; it rejects a changed challenge hash.

All adapters trained on the *same original* 189/50 train/validation examples (three labels, rank-8 LoRA, saved classifier head) and evaluated on the same challenge. The three ModernBERT-architecture checkpoints used three epochs at learning rate 1e-4. MiniLM at the same setting collapsed to always predicting dictation; its **separately reported tuned run** used eight epochs at 5e-4, chosen after validation/dev inspection but **before inspecting the larger challenge**. This is a comparison of working recipes, not a controlled causal comparison of encoder architectures.

| Frozen-checkpoint trial | Correct / 1,100 | False command candidates / 880 noncommands | Command intent found / 220 | MPS warm p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| ModernBERT-base 149M | 651 | **160** (18.2%) | 29 | 14.6 / 28.3 |
| Ettin encoder 32M | 654 | **195** (22.2%) | 79 | 8.9 / 22.5 |
| Ettin encoder 68M | 713 | **121** (13.8%) | 32 | 12.9 / 27.5 |
| MiniLM-L12-H384 33M, validation-tuned | 781 | **124** (14.1%) | 109 | 9.8 / 17.5 |
| MiniLM 33M, same initial 3-epoch recipe | 720 | 0 | **0** | 9.7 / 17.4 |

The initial MiniLM predicts **dictation for everything** (720/1,100), so its zero false positives have zero command coverage. In the tuned run, MiniLM has the best command recall in this synthetic trial, but it still marks clear noncommands as commands, e.g. *“The museum shop guide includes a typed note about member discounts”*. Ettin 68M's lower false-candidate count differs from MiniLM's by only three out of 880. For orientation, a binomial Wilson 95% interval on MiniLM's 124/880 is roughly **12.0–16.6%**; generator batches are correlated, so this interval is likely too optimistic. An illustrative whole-utterance string matcher for the five example phrases recognized four of 220 synthetic commands (most are intentionally paraphrases/multi-action requests) and no noncommands; this is **not** a run of eaRS's actual dictionary-normalizing router. Do not mistake semantic-intent recall for registered-command eligibility. A fixed command-score threshold of 0.9 suppressed every false candidate from tuned MiniLM **and every true command**: this is not a calibration win.

**Verdict for the tiny models:** none is close to a safe command trigger. The dataset shift from the 50-example test to this larger *still-synthetic* challenge also shows that accuracy on synthetic held-out data can be brittle. Relative speed/accuracy suggests MiniLM 33M is worth considering for *offline, advisory* classification, not that it is proven better on real eaRS utterances. Measurements were made sequentially on one Apple MPS machine (not under orchestrated eaRS/PhoneLLM contention), in Python, including tokenization but excluding IPC and cold interpreter startup.

## ZGX DiffusionGemma zero-shot shadow trial

The self-hosted ZGX `dgemma` vLLM endpoint serves NVIDIA's **NVFP4 DiffusionGemma 26B-A4B IT** (25.2B total / 3.8B active parameters). This is a *generative diffusion model*, not Jev, not a tiny encoder, and not fine-tuned on the eaRS 189 training examples. We sent only the **same hash-verified, synthetic 1,100-example challenge** over an SSH loopback tunnel. A single neutral three-label prompt was fixed before the full run (SHA-256 `c9610cb64da13e3f2b1dbee676ee879407bffd83bea4eef97d400c7c4fbe868a`); it defines quoted speech as literal and corrections as uncertain without listing registered commands. The endpoint rejects `temperature` and structured `response_format` for diffusion inference. The server usually emits a literal `thought\n` channel marker before otherwise valid JSON; the parser permits *only* that exact marker followed by a one-key JSON label. Other malformed outputs abstain and stay in the denominator. The four-case endpoint smoke test verified the wire format before the full pass; no prompt changes followed.

| Frozen synthetic challenge | DiffusionGemma shadow result |
| --- | ---: |
| Correct three-way labels | **976/1,100** (88.7%) |
| False command candidates among noncommands | **3/880** (0.34%): all declarative dictation; zero quoted/correction cases |
| Commands identified | **129/220** (58.6%) |
| Invalid response / abstention | **32/1,100** (2.9%) |
| Endpoint round-trip, 2 concurrent clients | **703 ms p50 / 1,675 ms p95 / 2,106 ms p99** |

The three false candidates are utterances about needing a reminder, **not** unambiguous instructions to the assistant; some synthetic labels are inherently debatable. The 32 invalid outputs were **not** silently mapped to a safe label or removed. A separate, fixed text-hash sample of 25 rows per family was repeated twice: **4/100 labels changed between repetitions** (and eight differed between the original full pass and the first repeat). For example, two malformed repeat responses contained Markdown JSON fences. Diffusion sampling here offers no supported temperature or seed control, so this is a single-run performance estimate, not a reproducible deterministic classifier. A nominal binomial 95% interval for 3/880 is 0.12–1.00%, but generator-family correlations and weak labels invalidate treating it as a real-world false-positive guarantee.

This zero-shot trial looks more promising **on this synthetic set** than the four locally fine-tuned encoders (best smaller-model false count 121/880), but the comparison is not controlled: different architecture, training, prompt, hardware, network path and inference API. Its ~0.7-second round trip is **not** compatible with word-by-word dictation latency; the encoder timings above exclude this endpoint and ran on a different machine. No private speech, registries, or real transcripts were sent; no model was placed on the eaRS runtime path. No score calibration, Rust parity, real consented held-out speech, or coordinated eaRS/PhoneLLM load test exists. It remains **shadow-only**: even one false candidate forbids model-based authorization, while exact registered matching and static policy still control all actions. `trx-2stp.7` remains open.

Reproduce with the existing frozen `experiments/modernbert-hotswap/artifacts/intent-challenge.jsonl`: establish `ssh -fN -L 127.0.0.1:18010:127.0.0.1:8010 zgx`, then run `just modernbert-dgemma-eval`. `compare_dgemma.py` refuses non-loopback URLs, verifies the challenge hash, and resumes gitignored per-row JSONL only if the prompt, parser and model match. `--smoke` runs one case per family; `--sample-per-family 25 --output <ignored-file>` provides the fixed repeatability sample. No data or results from a private speaker are involved.

## Rust `sys1` / Laya zero-shot shadow trial

The [open-source `sys1` Rust/Candle server](https://github.com/alvarobartt/sys1) is a real System One-compatible typed-decision API with dynamic batching and CPU/Metal/CUDA support. At reviewed source commit `26f7727` (v0.0.2), its **only** supported architecture/model is `convaiinnovations/laya`, a ModernBERT backbone with model-specific *decision heads*. It does not load our PEFT ModernBERT intent LoRA adapters or perform adapter hot swapping. We built this upstream commit with Metal in a **temporary clone**, loaded the locally cached Laya snapshot `7c76b622dfc5cac71b2dc1c29873efe2ce509a05` (~804 MB weights), and ran its `choice` question on the **identical frozen challenge**. The neutral three-way question was fixed before the full run (SHA-256 `c4093cd2350ad65687f2fb82836beb3cea1b4a0801dc2dc39f8f1730da49d5ff`); no private speech or registry was supplied.

| Frozen synthetic challenge | Rust `sys1` / stock Laya |
| --- | ---: |
| Correct three-way labels | **367/1,100** (33.4%) |
| False command candidates / 880 noncommands | **718** (81.6%): 334 declarative, 233 quoted, 151 corrections |
| Commands identified / 220 | **219** (99.5%) |
| Invalid response / abstention | **0** |
| Local Apple Metal endpoint, 2 concurrent clients | **44 ms p50 / 52 ms p95 / 70 ms p99** |

This is **severe overcalling**, not impressive voice-intent recall. A 0.9 command-score cutoff still produces **195 false candidates and misses 112 commands** on the test set; that post-hoc figure is diagnostic, *not* a valid tuned threshold. A fixed 100-case text-hash sample repeated twice and checked against the full pass produced **zero label or score differences**; the server itself is deterministic under this test. The paper/post's ~14 ms on RTX PRO 6000 is **not** an eaRS or Apple Metal measurement. The API and reusable batching are interesting infrastructure, but stock Laya's general typed-decision ability is **not** a pretrained eaRS command-intent detector. It cannot authorize any action, and we did not connect it to dictation.

Reproduce from a locally built `sys1` at commit `26f7727`: serve the pinned snapshot on `127.0.0.1:13003` with `--model-path <snapshot> --served-model-name laya-shadow --no-default-features --features metal` applying to the **build** (not CLI); then run `just modernbert-sys1-eval`. `compare_sys1.py` validates the frozen challenge and question hash, refuses non-loopback endpoints, and writes gitignored per-row results. A `--smoke` mode and fixed `--sample-per-family 25` repeatability mode are available. This trial did not validate other models or fine-tuned heads.

## CLM-8B zero-shot shadow trial

The public [Contrastive Language Model reference implementation](https://github.com/Contrastive-LM/CLM) at commit `7956937c58ed5839c06ddc4dc6b6b61c3a3e4094` serves a Jev-compatible `choice` question by matching separately embedded states and candidate actions. We ran its **unmodified** public head `Contrastive-LM/CLM-v0.1-8B` revision `87655cb835bd76fd66c2da78e1e3709f7fa11a94` with the required **Qwen/Qwen3-8B** last-token-pooled encoder revision `b968826d9c46dd6066d109eabc6255188de91218`. On ZGX, a temporary bounded vLLM `0.29.1rc1.dev347+gdee37d891` pooling container (23% GPU allocation, 48 GiB container memory limit, eager mode) served the 8B encoder; the separate Python head server ran on CPU. The head endpoint was reached through a loopback SSH tunnel. Existing services remained healthy. Only the same SHA-verified **1,100 generic synthetic utterances** were used; no private agent sessions, speech or registry were sent.

We reused the **exact three-way typed question from the Laya trial**, frozen before the CLM run (SHA-256 `c4093cd2350ad65687f2fb82836beb3cea1b4a0801dc2dc39f8f1730da49d5ff`). This is a single zero-shot prompt/checkpoint, not prompt optimization or fine-tuning. Per-row results are gitignored and strictly parsed by the shared `compare_sys1.py` harness, selected by `compare_clm.py`.

| Frozen synthetic challenge | CLM-8B zero-shot |
| --- | ---: |
| Correct three-way labels | **544/1,100** (49.5%) |
| False command candidates / 880 noncommands | **331** (37.6%): 205 declarative, 73 quoted, 53 corrections |
| Commands identified / 220 | **102** (46.4%) |
| `uncertain` predictions / 160 uncertain examples | **0 / 160** |
| Invalid response / abstention | **0** |
| SSH-tunneled endpoint, 2 concurrent clients | **179 ms p50 / 398 ms p95 / 574 ms p99** |

Stock CLM is **not** a usable eaRS voice-intent classifier: it misses over half of synthetic command intents and raises hundreds of false candidates. A fixed text-hash sample of 100 cases repeated twice matched the full pass exactly (zero label or score changes). The warmed repeated sample measured ~94–97 ms p50, not a substitute for the full-pass latency. In two separate diagnostic requests with the same question, fresh synthetic states cost 46 encoder tokens each while repeating one state cost **zero** encoder tokens—evidence of embedding reuse, not an end-to-end latency guarantee. The reference CLM head uses a **frozen 8B encoder** and can fine-tune its projection heads; it is not a tiny standalone Rust runtime. Its published “up to 9× faster than Jev” measurements involve different tasks, hardware and candidate-cache patterns. Here the candidate set has only three labels, so much of its large-option caching advantage is irrelevant.

Reproduce after starting a pinned local Qwen3-8B vLLM **pooling** endpoint and the pinned CLM head on ZGX: bind the embedder to loopback, cap its GPU allocation, serve `clm-serve --emb-url http://127.0.0.1:<embedding-port>/v1/embeddings --emb-model qwen3-8b --ckpt <pinned-head.pt> --no-download --device cpu --action-cache 0 --no-ui --host 127.0.0.1 --port <head-port>`, and tunnel only the head port to `127.0.0.1:18700`; run `just modernbert-clm-eval`. The vLLM pooling revision, GPU limits and upstream source hash above are required for a like-for-like comparison. No training or private-session collection occurred, no eaRS runtime wiring exists, and the deterministic command gate remains the only executor. Real consented held-out speech and coordinated load are still open under `trx-2stp.7`.
