# Apple CoreML Nemotron 3.5 spike

The proposed eaRS backend is the **model and its Apple Silicon inference path**, not FluidVoice (GPLv3). eaRS already has a separate `parakeet-rs` Nemotron ONNX backend (`src/server/parakeet_rs.rs`, 560 ms chunks) and a transcribe.cpp GGUF backend. A CoreML `.mlpackage` cannot be loaded by either backend as-is.

## Local smoke test (synthetic audio only)

- Model: [`BarathwajAnandan/nemotron-3.5-asr-streaming320-int8-CoreML`](https://huggingface.co/BarathwajAnandan/nemotron-3.5-asr-streaming320-int8-CoreML), revision `d5fa1694dd2a257fdb7027557bccc6364b0c02ba`, approximately 668 MB on disk. The model card refers to NVIDIA's OpenMDW-1.1 terms; verify redistribution terms separately before packaging anything.
- Runtime: [`altic-dev/FluidAudio`](https://github.com/altic-dev/FluidAudio), revision `09c23cce76126920b3ff6710cdb154bdf9c126b8` (the same revision pinned by FluidVoice). Its CLI supports 320 ms chunks with `--model-dir`. FluidAudio is Apache-2.0; no FluidVoice source or model assets were added to eaRS.
- Environment: Apple Silicon macOS, Swift 6.3.3 / Xcode 26.6. Built the CLI in a temporary checkout and loaded the model from a temporary local download. CoreML printed `Failed to PropagateInputTensorShapes ... ios17.slice_by_index: zero shape error` but **loaded and transcribed successfully**; the diagnostic alone is not evidence of a fatal error.
- Input: `test_audio/spqx_alpha_bravo.wav` (3.2 s, 24 kHz, synthetic speech generated from a non-English reference voice). Its generation script specifies `alpha bravo charlie now`. This is one accent-sensitive fixture, not held-out human speech or an accuracy benchmark.
- Warm, whole-file inference: **0.336 s processing** for the 3.2 s file, transcript **“Aflow Utrali now”**, missing/misrecognizing words. Model initialization was not counted; first-time CoreML load/compilation took tens of seconds, while a subsequent launch loaded faster. This is not live microphone latency.
- A temporary CLI instrumentation outside eaRS fed consecutive **320 ms** slices to `NemotronStreamingAsrManager.process(audioBuffer:)`, reading `getPartialTranscript()` each time and calling `finish()`. First nonempty partial (`Af`) appeared after **1.92 s of audio fed**; `Aflow Utra` appeared at 2.24 s; final transcript remained `Aflow Utrali now`. Feeding was faster than real time, so these positions are *acoustic progress*, not measured live first-word latency. No stable-word or end-of-utterance latency has been established.

## Side-by-side: CoreML vs the ONNX backend backend eaRS already ships

The same synthetic sample was also run through eaRS's existing streaming **ONNX** Nemotron path (`parakeet-rs` engine, 560 ms chunks, `nemotron-3.5-asr-streaming-0.6b-onnx`). Both backends mis-transcribed the phrase and both emit their first text at nearly the same acoustic point. Wall-clock was measured from the start of the stream on the same 3.2 s file (which has ~0.47 s of leading silence).

| Backend | First text arrives | Full phrase | Transcript of the fixture |
|---|---|---|---|
| ONNX (`parakeet-rs`, in eaRS) | 2.37 s (committed word, streams before speech end) | last word +0.33 s after speech end | `Aflo Uschali now` |
| CoreML (FluidAudio 320 ms) | 1.65 s (first nonempty provisional partial) | 2.92 s | `Aflow Utrali now` |

Interpretation: both are streaming Nemotron variants that reach their first output at roughly the same acoustic position and produce the same class of errors on this foreign-accented synthetic sample. CoreML did **not** show a meaningful first-word-committed or accuracy advantage here; its sole real differentiator relative to the already-shipped ONNX path is ANE/CPU utilization, which this sample did not measure.

The CoreML streaming plugin also differs architecturally: `NemotronStreamingAsrManager` exposes a *revisable* `getPartialTranscript()` partial rather than committed per-word deltas. For eaRS, provisional CoreML text would have to be routed only through `WebSocketMessage::Interim` (never typed as `Word` events) unless/until a stable-commit boundary is defined.

The SDK release logger writes informational output to unified logging (where dynamic transcript fields may be private)."}] A temporary `print` in the throwaway checkout exposed the transcript and per-chunk partials; none of those changes were copied into eaRS. To reproduce the standard CLI path, download the pinned model to an **outside-repository** directory and run:

```bash
swift build -c release --product fluidaudiocli
.build/release/fluidaudiocli nemotron-transcribe \
  --input /path/to/eaRS/test_audio/spqx_alpha_bravo.wav \
  --chunk 320 --model-dir /path/to/downloaded/CoreML/model --target-lang en-US
```

## Integration boundary and go/no-go

If a broader side-by-side trial justifies it, add an **optional Apple-only local Swift/CoreML adapter** that accepts eaRS session PCM (currently 24 kHz; SDK resamples to 16 kHz), keeps the model loaded, bounds audio queues, and translates SDK output into eaRS `SessionSink` events. `WebSocketMessage::Interim` already supports preview without typing; only demonstrably committed words may become `Word` events. Finish/reset and emit the completed speech boundary once; keep the deterministic voice-command executor unchanged. Test cancellations, partial revisions/token joins, short utterances, tails, language changes, and process failure before any live routing.

Do **not** switch the default on this smoke result. Compare both backends on the same consented held-out completed utterances, measuring transcript errors, first *committed* word and finalization latency, and memory/ANE contention during ordinary dictation. Synthetic data alone cannot establish quality or command safety. Tracked as `trx-2stp.14`.
