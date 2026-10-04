# parakeet.cpp Ultra/Redux and VAD spike

2026-10-04. Task `trx-7nw3.6`; buffered evaluation `trx-7nw3.3`; no-speech
follow-up `trx-7nw3.7`. [Runnable experiment](../experiments/parakeet-cpp/README.md)
and [aggregate evidence](../experiments/parakeet-cpp/evidence.json).

## Verdict

- **Continue evaluating Ultra Q8 for buffered German ASR.** The native C++ port
  works, with better accuracy than packed Redux in this small smoke subset.
- **Keep Redux as a resource/speed option**, not the German quality default.
- **Do not replace live Silero with either VAD head.** Both falsely open speech
  on silence and noise-only controls. The head API tested is clip-level, not
  streaming. Silero passes the no-speech and partitioned-stream controls.
- **Not a catalog-only addition:** the exact existing eaRS transcribe.cpp Rust
  dependency rejects these GGUF artifacts. A format conversion/adapter decision
  and its own proof are required. No production engine, GPUI, browser, session
  ownership, dependency selection or default was changed by this spike.

## Reproducible scope

CPU source build of `mudler/parakeet.cpp` at
`b3eaa487c72a6b9fb7bcb4915b1395e8950b612e`, ggml at
`e705c5fed490514458bdd2eaddc43bd098fcce9b`; shared library and CLI, optional
speaker/sound/server components disabled. AppleClang 21.0.0.21000101, Apple M2
Ultra, 24 physical cores, 192 GiB RAM, native ternary dispatch **NEON** confirmed
by `test_ternary`. No GPU; four compute threads for CLI measurements.

Weights from `mudler/parakeet-cpp-gguf`, pinned revision
`741158ae71e64ef5c89385862c18f777d07a97a1`; setup verifies the hashes published
in the pinned upstream manifest. Ultra/Redux and their slices: CC BY 4.0,
converted copies of Moondream/NVIDIA models; keep attribution on redistribution.
Silero v6.2.3 F32 GGUF: MIT. Runtime source: MIT. Dependencies/distribution
still need the normal selected-artifact audit; this is not a blanket license
review or packaging approval.

FLEURS test audio from `google/fleurs` at
`d7c758a6dceecd54a98cac43404d3d576e721f07` (CC BY 4.0). First five qualifying
archive entries per language, mono 16kHz, 3–30 seconds, 114 German and 103
English reference words. Same clips for every model, clean and deterministic
whole-clip-RMS **5dB white Gaussian noise**, no gain normalization. Measured
clipping fraction zero. Original/derived WAV hashes, seeds and source filenames
are recorded without audio or reference-text dumps. These are **10 unique
clips**, not 20 independent recordings; noisy versions are paired controls.
The FLEURS preview API returned a scan-limit error, so the runner streams the
original archives rather than downloading a 726MB Parquet row group.

Normalization/scoring imports `scripts/bench_manifest.py`: lowercase, remove
punctuation, collapse whitespace, keep Unicode letters/digits; hyphens are
removed rather than split. Corpus WER/CER are weighted by reference words/chars,
not averaged per clip. CER includes normalized spaces. Numerals are not expanded.

## Measured buffered ASR

Three fresh CLI benchmark processes per model; each loads once and discards an
untimed first-clip warm-up before measuring every clip. Transcription timings
exclude model load and audio file IO. Same hypothesis scores across all three
runs. Values below are corpus WER percent; RTFx is median grouped throughput.

| Model | German clean | German 5dB | English clean | English 5dB | German RTFx clean/noisy | Peak native RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultra Q8 | 2.63 | 14.04 | 2.91 | 8.74 | 13.42 / 13.29 | 1113 MiB |
| Redux packed | 7.02 | 23.68 | 6.80 | 23.30 | 24.51 / 24.41 | 575 MiB |
| Original v3 Q8 baseline | 4.39 | 15.79 | 5.83 | 11.65 | 13.65 / 13.63 | 1106 MiB |

The v3 baseline is an existing local converted GGUF, SHA-256
`4d69a4a6683f4f2d952bad794c1357ca6eb628027695b4699c5a9ad4cd07d757`.
Its download revision was not recovered; setup does not download it. The
comparison is **inside the same C++ runtime**, not against current Foxline/eaRS
server or its deployed recognizer. Existing local Photon findings in `trx-2735`
use different subsets/runners; no cross-port parity claim.

Two explicit reciprocals avoid BENCH_SPEC's conflicting formula/example:
`processing_rtf = processing_seconds / audio_seconds` (lower better),
`rtfx = audio_seconds / processing_seconds` (higher better). No first-word,
streaming-preview, microphone release-to-final or playback latency claim.

Load timings vary with uncontrolled OS caching: Redux 1579/60/60ms; Ultra
3107/150/116ms; v3 137/111/111ms. They precede warm-up, so **load completion is
not verified first-inference readiness**. They are neither controlled cold-start
measurements nor production readiness metrics. Peak native RSS is the maximum
of `/usr/bin/time -l` per fresh process, excluding the Python parent; file sizes
(213MB Redux, 942MB Ultra) are not working-memory figures. No p99 claims on three
runs, no statistical quality ranking on five clips, no battery/GPU/platform
inference from these CPU results.

## VAD: important negative evidence

Matched options: threshold 0.5, min-pause 0.2s, min-speech 0.1s, padding 0.
These are deliberately matched rather than each detector's different defaults.
Pure silence is exactly three seconds of zero PCM16 at 16kHz. Noise-only is
three seconds of seeded Gaussian noise (seed 42, sigma 0.015), no human speech.

| Detector | Silence speech intervals | Noise-only speech | Peak native RSS, 33s clip | Warm C API median, 33s |
| --- | --- | --- | ---: | ---: |
| Ultra Q8 slice | `[0, 3]` | `[0, 3]` | 247 MiB | 246ms |
| Redux slice | `[0.24, 2.88]` | `[0.24, 2.96]` | 249 MiB | 263ms |
| Silero F32 | none | none | 16.4 MiB | 103ms |

Full Ultra/Redux parents give **exactly the same silence JSON/probabilities** as
slices. The files match their published hashes. This narrows away slice-only
extraction; it does not prove whether the root cause is clip-level frontend
normalization, learned head behavior or quantization. No upstream/runtime/model
patch was made. An independently runnable no-speech assertion passes for Silero
and **fails for both heads**. Preserve it under `trx-7nw3.7` before any live
adoption; do not treat upstream higher-recall/F1 headlines as a safety gate.

The 33s control repeats/truncates the first German clip; it is a memory/time
stress input, not independent accuracy data. Three warm C API calls reuse a
loaded context and include file decoding, JSON allocation/parsing/free; native
thread default is **8**, distinct from the four-thread CLI results. The Python
process's memory is not included in the native CLI RSS column. Native model
errors after invalid threshold, missing audio and ASR on VAD-only contexts are
reported, then a successful VAD call on the same context proves recovery.

Both head slices reject `parakeet_capi_vad_stream_begin` with
`context does not hold a Silero VAD model`. Silero supports it; arbitrary
511-sample partitions reproduce one-shot probabilities within 0.0001 and emit
start/end events. Packed Redux also rejects native ASR stream creation. A VAD
cut around an offline ASR call does not establish genuine streaming ASR.

Supplementary synthetic German `Ja`, `Nein`, `Danke` from the existing
Martin/Kokorox worker, resampled to 16kHz, plus one-second silence before/after,
were detected by all three at the matched settings. The heads retain wider
regions; Silero narrows some tails. These are **synthetic, not phone recordings**,
and lack independent human-aligned boundaries, so no recall/F1 or endpoint
quality score is assigned. Raw fixtures stay local; `--extra DIR` can replay
consented human controls later. Initial old `koko` CLI binary failed on the
Martin token input name; the current standalone worker was used instead.

## Integration and proof status

`loader.rs` builds independently with the **exact eaRS transcribe.cpp revision**
`63a44d9239d610b3908e8a66b384924cd4a77217`. All three C++ GGUF artifacts above
are rejected with `gguf load error (status 4)` before capabilities can be queried.
The existing `TranscribeCppEngine::load` streaming-only guard is untouched.
This establishes artifact incompatibility at the tested loader, not a general
claim that transcribe.cpp can never support the model family.

Proof completed: pinned source build/setup/hash verification; **115 upstream
CTest tests passed**; ten harness unit/process/scoring/noise tests pass (one
real-model gate skipped unless configured); real Silero no-speech gate passes;
both head gate failures reproduced; real C API rejection/recovery/stream
partition probes; Rust loader `cargo check`, release build/run and format check.
No full eaRS application build/WS/GPUI/browser test, live capture, barge-in,
production cancellation/playback or packaged-platform verification was performed.

Next decision belongs to `trx-7nw3.3`: evaluate a bounded standalone worker or
an audited conversion/runtime change, measure readiness and actual human replay,
and retain genuine streaming separately. No hard-to-reverse adapter architecture
was selected in this experimental change.
