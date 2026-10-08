# Whistle: tiny native German/English ASR spike

2026-10-08, `trx-7nw3.8`. [Runnable spike](../experiments/whistle/README.md)
and [evidence](../experiments/whistle/evidence.json). Source branch
`spike/whistle`, stacked on the previous `spike/parakeet-cpp` experiment.

## Verdict

Useful **footprint-constrained buffered ASR candidate**, not a German/default
replacement on this evidence. Whistle is far smaller/faster than Ultra/Redux,
but loses substantial accuracy on the same small public corpus, especially
added noise. Forcing language does not rescue German noise performance.
No production adapter, catalog, GPUI, browser, Foxline protocol or default changed.

## Real native measurements

Apple M2 Ultra, 192 GiB RAM, CPU only. Same pinned FLEURS corpus as
[the Parakeet spike](parakeet-cpp-spike.md): five unique German clips/114 words,
five English clips/103 words, clean and seeded whole-clip-RMS 5dB white Gaussian
noise pairs. Same existing eaRS punctuation/Unicode normalization and
word-weighted WER/CER; no keyword/gold-reference hints. This is a small smoke
subset, not statistical ranking or human phone/microphone quality proof.

Three fresh native processes per configuration. Model loaded once per process,
one discarded warm-up, timed C API per clip excludes PCM file IO and includes
native JSON generation. Whistle: full eight-layer decoder/default five beams,
compiled C API thread policy (CLI documents fast cores capped at four, but the
public C API has no setter; effective thread count not independently measured).
Parakeet CLI: explicitly four threads, fresh same-day reruns, no concurrent
benchmarks launched by this harness. CPU scheduling/background load uncontrolled.

| Model/mode | German clean WER | German 5dB WER | English clean WER | English 5dB WER | German clean RTFx | Peak native RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Whistle auto | 16.67% | 78.95% | 8.74% | 36.89% | 94.11x | 83.4 MiB |
| Whistle forced de/en | 16.67% | 75.44% | 8.74% | 36.89% | 93.83x | 82.8 MiB |
| Redux packed | 7.02% | 23.68% | 6.80% | 23.30% | 23.92x | ~575 MiB |
| Ultra Q8 | 2.63% | 14.04% | 2.91% | 8.74% | 12.02x | ~1108 MiB |

Scores stable across runs. One noisy German clip was detected as English in
auto mode; forcing German improved only four word errors in the 114-word cohort.
This compares native runtime candidates, not the deployed eaRS/Foxline recognizer.
Whistle's file is exactly **16,919,407 bytes**, not its runtime RAM requirement.
RTFx = audio seconds / native processing seconds; reciprocal processing RTF
is separate. Reported first-token timings span 6.3–30.7ms (median 18.5ms) over
these different clip lengths. Those are model-reported, **after the whole clip
is available**, not live microphone first-word/commit latency. Load ~5.1–5.6ms
excludes file read and first inference; neither cold-cache startup nor readiness
was controlled. No M4/browser/mobile/GPU/battery or p99 claim.

## Safety and interface findings

- Zero PCM and stationary noise give empty text/language, offline and buffered
  streaming. But **nonstationary noise with no speech** produces `Vielen Dank.`
  in both modes: seed 43, sigma 0.00005 Gaussian samples times Hann(48000),
  PCM16 mono 16kHz, three seconds. An independent no-speech assertion reproduces
  the failure, preserved under **`trx-7nw3.9`**. A German FLEURS leading-background
  prefix gives the same phrase; its full stream prematurely commits it at two
  received seconds. Do not equate two-pass agreement with acoustic correctness.
- Public stream API appends/re-transcribes audio and commits words agreed across
  two passes. Encoder is noncausal. Real one-second CLI feeds/flush ran on a
  German clip and controls, back-to-back, **not paced live recording**. This is
  reprocessed buffered streaming, not proof of a true incremental encoder,
  indefinite 30s rollover, cancellation fencing or reliable endpointing.
- Native word timestamp/probability schema, finite values and monotonic start
  bounds pass for the measured batches. Attention alignment and probabilities
  are not independently adjudicated boundaries or calibrated correctness scores.
- Two-byte output buffer returns **positive 47 token count**, only `{` plus
  terminator. Eight-byte before/after canaries remain intact. A normal 64KiB
  buffer succeeds; no memory-overflow claim. Caller must validate complete JSON
  and enforce limits, not trust nonnegative status. **`trx-7nw3.10`** tracks the
  native capacity contract and integration implications.
- One process-global, non-thread-safe model per kind; no public unload/thread
  setter in the shipped header. Prototype retains model bytes and serializes
  calls in an owned subprocess, with finite PCM/30s/64-clip/output bounds and
  inherited tested timeout-group cleanup. A future adapter needs explicit
  readiness, ownership, state reset/cancellation and concurrency proof.
- Only the speech model was loaded. No Needle text model, tools, `needle_complete`
  or action execution. Recognition remains evidence; action/turn authority stays
  outside the speech engine. Do not replace existing model capability distinctions.

## Privacy, licensing and provenance

Inspected code repository commit
`af2654dffa4987f7e2cbbacdf59c2f854fd18302`; model HF revision
`b358ddadd89b7a713b5aa131f23032d3cca1b251`; prebuilt runtime/header/archive HF
revision `2ae11323dc000f5e70c49f7403efa6af12ba9e67`. SHA-256 pinned and verified
by setup. Source package, model and runtime license files declare **Apache-2.0**;
not an MIT model/engine claim. Native C++ engine implementation was not found in
that GitHub checkout (Python wrapper/model code is present), so we compiled our
caller against the pinned archive, **not the engine from audited source**.

README says telemetry defaults on, whereas the model card/blog says engine reads
no environment variables. Python telemetry source describes anonymous usage
counts, not audio/text, but this is not proof for an opaque binary. One initial
**public-fixture** CLI smoke ran before opt-out/network containment; no claim
about its egress. Every subsequent measured native invocation used
`NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1` and macOS `deny network*`; same-profile
outbound HTTPS control fails. This proves tested direct-network containment,
not a full security audit or portable no-egress packaging. No private audio used.

## Verification and remaining scope

Pinned artifact setup/hashes, native C++17 caller with `-Wall -Wextra -Werror`,
three-run auto and forced-language measurement, same-day Ultra/Redux runs,
word-schema checks, actual zero/stationary controls, offline/streaming shaped-noise
failure, output canaries, NaN/oversized-input rejection and OS network denial.
Five Whistle tests pass with actual native/model inputs (including three
invalid-field subcases); the sixth is the opt-in **known failing adoption gate**. Shared prior runner tests remain separate.

Native/body timing comparisons still differ in runtime/thread policy and tiny
corpus coverage. No source-engine rebuild/audit, all-app compile, browser, WS,
GPUI, live mic, long-stream rollover, production barge-in or packaged-platform
proof. Results support a bounded optional tiny-runtime evaluation, not adoption.
