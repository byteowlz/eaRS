# Phonon-2 evaluation for eaRS

Local tests, 2026-10-01; issue `trx-89as`. **Verdict: promising optional endpointed/preview backend, not a drop-in native-streaming replacement.** No eaRS default, runtime, or user configuration was changed. All audio stayed local.

## Pinned runtime and model

- [FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2), model revision `9c7fef3584499a88fe8d394427f45851bbb8b446`; archive `phonon-2.bps.tar.zst`, SHA-256 `98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e`. The official downloader verified this pinned digest and every extracted member.
- Actual inference: PyPI `fermion-research==0.2.4`, MLX 0.32.3, mlx-audio 0.5.7, mlx-lm 0.31.3, NumPy 2.5.3, Python 3.12.13. Default Phonon-2 levers `tdt16,dense16` were applied successfully; no hotwords.
- [Runtime repository](https://github.com/fermionresearch/phonon) inspected at `bc7c1cae7baeab43e4f15c281305e33c8ea9f46e`. Inference used the wheel's `fermion._speech.engine_phonon2`, not the repository's older top-level `stt.py` (which contains Phonon-1 paths).
- Apple M2 Ultra, 192 GiB unified memory, macOS arm64; MLX GPU execution.
- Weights: **CC-BY-4.0**, derived from NVIDIA Parakeet TDT 0.6B v3; runtime: **Apache-2.0**. Distribution must preserve attribution, licence and change notices; do not label the weights Apache-2.0.

## Accuracy and warm speed

English inputs are identical to the [Audio8 spike](audio8-asr-spike.md): the existing synthetic `test_audio/spqx_alpha_bravo.wav` and first three rows of `hf-internal-testing/librispeech_asr_dummy`, revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`. These three public clips share one speaker; this is a smoke test, not representative accuracy coverage.

Each local probe loaded the model once, ran the official live warmup, then decoded each file four times. Below are the ranges across **two probes** of the median of the last three passes. Timings exclude load/resampling and synchronize MLX around inference.

| Input | Audio seconds | Warm decode ms | Warm throughput | WER |
| --- | ---: | ---: | ---: | ---: |
| Synthetic alpha/bravo | 3.200 | 39–44 | 73–81× | 75.00% |
| `1272-128104-0000` | 5.855 | 50–68 | 86–116× | 0.00% |
| `1272-128104-0001` | 4.815 | 44–59 | 82–111× | 10.00% |
| `1272-128104-0002` | 12.485 | 73–95 | 131–171× | 0.00% |

The official `fermion speech bench` independently measured **130.6× realtime** across three passes over those three public English files. Its peak RSS was **3.10 GB** and peak MLX allocation **1.52 GB**; these are different accounting views, not quantities to add together. This did not test an hour-long recording or reproduce the advertised M5 MacBook Air hardware.

Public English aggregate: **1 / 59 words = 1.69% WER**, versus Audio8's 3/59 on the same references. The Phonon error is `manner` -> `matter`, while Audio8's score includes two `Mister` -> `Mr.` formatting substitutions. Do not infer a general accuracy win from these three clips. Both struggle on the accented synthetic sample; Phonon batch output was `Aflor Uchrali now` for intended `alpha bravo charlie now`.

WER uses jiwer lowercasing, punctuation removal, repeated-space collapse and stripping, identically to Audio8. Repeat batch transcripts were deterministic. Model loading was about **10–11 s** after caches/shaders existed; the first CLI invocation loaded in **22.6 s**, with **39.4 s** whole-command time including imports/load/first decode. Keep it resident for dictation.

### German: actually tested, but not advertised support

The card positions Phonon-2 as English-only, but its multilingual Parakeet ancestry still produced German text. No language override was supplied; the current Phonon-2 CLI exposes none.

Three public FLEURS German test clips were read from `google/fleurs`, converted-parquet revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`, `de_de/test/0000.parquet`. Three repeated whole-file decodes each produced identical text.

| FLEURS ID | Audio seconds | Warm throughput | WER |
| --- | ---: | ---: | ---: |
| 2003 | 17.94 | 212× | 33.33% |
| 1738 | 11.16 | 169× | 9.52% |
| 1849 | 12.42 | 160× | 13.04% |

Aggregate: **11 / 62 words = 17.74% WER**. Errors include compound splitting (`gegenüberliegenden` -> `Gegenüber liegen den`), `Wenn` -> `When`, and `Vermächtnis` -> `Vermechtnis`; the simple normalizer also penalizes `T Rex` -> `T-Rex`. This demonstrates some German capability, not reliable German support or superiority to eaRS's existing multilingual engine. A same-fixture current-engine baseline remains necessary.

## Paced live behavior

The wheel's actual `LiveSession` and `run_wav(..., realtime=True)` were exercised on all four English fixtures with **1.2 s appended silence**, after warmup. Callbacks were timestamped; an instrumentation-only subclass recorded the session's acoustic `_last_voice` sample index immediately before finalization.

- First **nonempty provisional partial** arrived around **0.4–1.1 s** from file-stream start. Leading silence and short-prefix recognition affect this; it is not first committed-word latency.
- Final callbacks arrived **0.70–0.83 s after the session energy gate's last voiced block**, including inference. This is a gate-defined endpoint, not a human-annotated speech endpoint.
- All final tail words survived. The longest clip finalized twice at an internal pause; the joined transcript remained correct after normalization.
- Partials revised earlier words: `mister Quilton` -> `mister Quilter`; `middle Age` -> `middle classes`; `He told me` -> `He tells us`. The 12.5 s clip's first partial was the incorrect `Okay.`. **Never type or authorize commands from these hypotheses.**
- Added silence/segmentation can change final spelling: the synthetic padded-live output was `Aflor Uschraali now`, not the batch `Aflor Uchrali now`.
- A 3 s all-zero waveform through the live energy gate emitted no transcript. The unmodified official `fermion listen ... --wav` CLI was also tested and reproduced the expected public-clip final.

This is **rolling whole-segment re-decoding**, not cache-aware streaming: every partial runs batch inference on all audio accumulated in the segment. The implementation uses 50 ms input blocks, a first partial threshold of 350 ms, subsequent partials approximately every 500 ms of audio, an adaptive RMS gate, 700 ms silence closure and a 30 s segment cap. Batch throughput and live responsiveness are different metrics. Long holds, sustained sessions, noisy endpointing and concurrent LLM/UI contention were not validated.

## Size caveat

The verified compressed download is **163,515,201 bytes** and the extracted container approximately **177 MB**. That is not inference RAM. The default `dense16` lever expands the encoder's five-value weights into FP16 matrices at load; the remaining graph uses BF16, with GPU-resident greedy TDT decoding (`tdt16`). The measured ~1.5 GB MLX / 3.1 GB RSS above is more relevant to eaRS contention than the download headline.

## Reproduce locally

Use a scratch directory, not the eaRS source tree:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  fermion-research==0.2.4 mlx==0.32.3 mlx-audio==0.5.7 mlx-lm==0.31.3 \
  soundfile scipy zstandard
export FERMION_CACHE_DIR="$PWD/cache"
uv run --no-project --python .venv/bin/python fermion transcribe \
  phonon-2 /path/to/clip.wav --json --verbose
uv run --no-project --python .venv/bin/python fermion speech bench \
  phonon-2 --audio /path/to/clip.wav --runs 3 --json
uv run --no-project --python .venv/bin/python fermion listen \
  phonon-2 --wav /path/to/clip-with-trailing-silence.wav --verbose
```

The pinned wheel's model downloader validates the archive digest above rather than trusting a mutable Hub file. For an existing verified unpacked model, pass its directory instead of `phonon-2`. The full dependency freeze, temporary probes, fixtures, raw timings/events and results are local under `/tmp/ears-phonon2/`; they are not committed audio or permanent assets.

## Browser/WASM follow-up

Phonon-2's published artifact/runtime is not a drop-in browser model. A browser port would need its container/quantization loader, feature extraction and TDT decoder, plus WASM kernels or compatible WebGPU/ONNX graphs. Native MLX timings do not predict browser performance. The compressed 164 MB download does not resolve the memory issue above.

For browser-local Oqto speech input, the current [Moonshine WASM binding](https://github.com/moonshine-ai/moonshine/tree/main/language-bindings/wasm) is a stronger first candidate: it ships an actual browser CPU/WASM streaming runtime and separate MIT-licensed English and German streaming models. This is **not** the old English-only Moonshine release. The [current catalog](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/available-models.md) includes German Tiny and Small; explicitly select Tiny rather than taking the larger language default.

CDN HEAD checks on 2026-10-01, summing the eight standard manifest files without optional word-timestamp decoders: English Tiny Streaming **45.23 MB**, German Tiny Streaming **32.32 MB**, German Small Streaming **121.80 MB**. These exclude the WASM/JS runtime and are downloads, not runtime RAM or performance measurements. Tiny English plus German needs about 78 MB of model cache, with one language loaded at a time. Models are language-specific: mixed-language utterances need separate evaluation. Default SIMD/multithread WASM requires COOP/COEP cross-origin isolation; the upstream documents a single-thread fallback build. No Moonshine browser inference was tested in this session.

A useful alternative is multilingual Whisper through Transformers.js/ONNX Runtime Web, using WASM or WebGPU: standard INT8 encoder plus merged decoder weight files total approximately **41 MB tiny**, **77 MB base**, **249 MB small**, excluding tokenizer/runtime. It uses one model for English/German, but chunked/endpointed inference rather than native streaming. [Vosk-browser](https://github.com/ccoreilly/vosk-browser) is another CPU/WASM streaming option, with separate ~40 MB English / ~45 MB German model downloads, but its older small models need a strong dictation-quality check.

Keep browser capture/resampling, provisional/final transcript events and model/provider selection separate. Self-host/cache pinned models, keep recognition in a worker, make any server fallback explicit rather than silently uploading audio, and never commit partial recognition to text/commands. Benchmark on actual mobile browsers before choosing defaults.

## Integration recommendation

Worth an **optional resident MLX/local sidecar** trial for endpointed dictation or fast post-utterance ASR. Route revisable hypotheses through eaRS `Interim` only; commit words/commands only from completed segments. Resample 24 kHz eaRS PCM to 16 kHz, use bounded queues, honor cancel/reset/language errors, and give one layer ownership of speech boundaries so the sidecar's 700 ms gate does not duplicate eaRS VAD. Hotword bias is currently accepted but explicitly **not implemented** by this decoder; keep eaRS dictionary correction independent.

Do not switch defaults or claim native streaming/German parity yet. Next proof should compare the current eaRS engine and Phonon on the same consented English/German dictations, including names, code terms, pauses, noise and long utterances, then measure committed-word/final latency under normal machine contention.
