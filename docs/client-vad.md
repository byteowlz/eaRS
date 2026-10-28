# Client-side VAD (experimental)

Client-owned PCM, not ASR event arrival time, must determine retained clip
boundaries. Recognition can deliver its first word after the current one-second
history lookback, losing the start. A late commit can also consume audio belonging
to the next utterance. See `trx-63fk.4` and timeout work `trx-xc5e`.

## Implemented

- `dictation::capture_segments`: fixed-frame, sample-clock segmentation driven by
  same-frame speech probabilities. Onset debounce retains the first speech frame
  and pre-roll; short pauses stay inside the clip; silence hangover permits
  bounded post-roll. Consecutive spans do not overlap. Invalid input, discontinuous
  clocks and duration overflow discard the pending clip and require a reset,
  rather than silently returning truncated training audio.
- `dictation::client_vad` (`client-vad` feature): local CPU Silero ONNX scoring,
  16 kHz mono, 512 samples/frame, one recurrent state per stream. It is not an
  RMS threshold. Model paths are explicit; no downloads or microphone ownership.
- Offline `client_vad_segments` example: prints JSONL sample spans from an explicit
  local audio file. It neither edits recordings nor creates training labels.

These are NOT yet wired into live retention or the timeout. Existing live clips
still use ASR arrival-time slicing; the clipping bug remains open. Synthetic
probability tests prove the segmentation mechanism, not Silero recall on quiet
human speech. The artifact smoke test proves inference, silence rejection, reset
and invalid-frame handling, not real-world boundary quality.

## Checks and offline preview

```bash
just test-client-vad
# Explicit local model + ONNX Runtime shared library; no automatic installation:
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib just test-client-vad-model /path/to/silero_vad.onnx
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib just preview-client-vad /path/to/silero_vad.onnx /path/to/mono-16k.wav
```

Preview limits: 32 MiB encoded input and five minutes of decoded audio. The
existing decoder uses the first channel; supply mono 16 kHz PCM. This is an
explicit offline diagnostic, not a general-purpose untrusted upload endpoint.
Output is preview-only, with half-open sample offsets and an end reason;
`EndOfInput` is not an acoustic silence completion. It does not read transcripts.

Pinned smoke-test model source (not bundled or installed):
`snakers4/silero-vad@066409f3d9db45a30d21dfcbf9147826ffdd7052`,
`src/silero_vad/data/silero_vad.onnx`, SHA-256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
Source and wrapper input contract: https://github.com/snakers4/silero-vad/tree/066409f3d9db45a30d21dfcbf9147826ffdd7052
The upstream project/model is MIT-licensed; preserve its notice when distributing
an artifact. No model or runtime binary is included by this change.

## Live integration gates

Keep one microphone owner. Score a continuous, statefully resampled stream and
bind each probability to that exact PCM frame. Reset detector, frame assembler
and segmenter together across pause, cancel, disconnect and privacy transitions.
Do not attach recognition output to a clip by FIFO alone: noise, missing results
and differing endpoint policies can shift associations. Preserve original ASR
output separately and label alignment honestly. Check privacy over the whole
span, not only at commit time. Do not persist uncertain excluded/secure audio.

The idle timeout must stop sending new speech only after an ordered drain, then
accept the pending final before rearming wake. Local silence alone is not a
server acknowledgement. Before activation: native capture/resampling tests,
marker-preservation tests, privacy/race tests, and an authorized live comparison.
Do not treat these modules as speaker-gate or action-authorization evidence.
