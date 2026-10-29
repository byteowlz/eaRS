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

Client-VAD retention is wired into the engine capture consumer as an explicit
opt-in below. It no longer waits for an ASR word/final to cut a clip. The original
24 kHz capture PCM is retained; only detection is downsampled to 16 kHz. Synthetic
marker tests exercise the production retention methods, delayed recognition,
privacy transitions, release, export and purge. They do not prove Silero recall
or boundary quality on quiet human speech. Native/live verification and timeout
drain integration remain separate gates; an old installed binary is not updated
by building this source.

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

## Enable client-owned recording boundaries

Build with `--features client-vad` (plus the usual engine features), then configure
explicit, local, hash-verified artifacts:

```toml
[dictation.audio_history.client_vad]
model_path = "/path/to/silero_vad.onnx"
model_sha256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
runtime_path = "/path/to/libonnxruntime.dylib"
```

A missing feature/model/runtime or wrong hash disables retention with a visible
error, never falls back to known-clipped ASR-arrival slicing. Configuration alone
is not proof of active recording; restart/reinitialize and verify a new record.
Without this optional table, legacy retention remains backward-compatible.

Complete clips retain configured `pre_roll_ms` (default 1000 ms, rounded up to
32 ms frames) measured from the first accepted speech frame, 128 ms onset
debounce, a 512 ms silence decision window, and 512 ms post-roll. Those are provisional operating points,
not measured accuracy guarantees. A manual stop may save available PCM with
`capture_boundary.complete=false`; cancel/disconnect discard the pending span.
Duration overflow or an evicted PCM span fails rather than silently truncating.

Acoustic clips have empty transcript fields and `capture_boundary.source` set to
`client_silero`. Original ASR output is kept in private `recognition.jsonl`, marked
`alignment=unassigned` and `training_eligible=false`, with session ID and capture
sample position. Neither arrival order nor a noise detection silently creates a
training pair. Review or independent transcription must establish a reference.
Export includes relevant unassigned recognition events. Purging a clip also
removes unassigned suggestions from its whole session (a deliberate conservative
privacy rule), and purge-all removes suggestions even if no clips were produced.

One engine-thread consumer orders capture, retention and transport; no detached
receiver competes across reconnections. App identity/Secure Input is polled for
each capture batch without reading window titles on macOS. Observed protected,
unknown or changed contexts discard pending audio and require quiet before
recording resumes; ASR suggestions are suppressed after such transitions, cancel or disconnect until
a new capture session. This is sampled context observation, not proof against a
focus change inside an OS callback or an unobserved queued-audio interval. Native
privacy/race testing remains required; no capture/identity evidence authorizes an
action.

## Remaining qualification

The idle timeout pauses capture and rearms wake locally; old ASR finalization is
separately owned and bounded. Ordered drain is optional stronger completion
proof, never a prerequisite for local pause. Local silence alone is not a server
acknowledgement. See `auto-pause.md` for legacy completion, deferred text and new
capture ownership. Native capture/resampling, marker-preservation, privacy/race
checks and an authorized live comparison remain deployment qualification gates.
Do not treat these modules as speaker-gate or action-authorization evidence.

## Capture quality

`audio_input.rs` now keeps downmix and bandlimited resampling state for the full
microphone stream, not each callback. Synthetic tests reproduce and prevent
callback-dependent sample loss, split-channel failures and high-frequency aliasing.
Native 24 kHz bypasses filtering; other supported rates have at most 128 target
samples (under 6 ms) of lookahead/buffering at stream end. The converter is not
reset at utterance boundaries. Equal-weight multichannel mixing is not
speaker-layout-aware. PCM amplitude is preserved, not silently normalized.

The user's active device log reports 48 kHz F32 mono; that alone says nothing about
microphone placement, input gain or acoustic quality. Quiet recent recordings and
zero full-scale samples are not proof of good SNR. Acoustic comparisons and live
user acceptance remain outstanding.
