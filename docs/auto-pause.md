# Silence-triggered dictation pause

Opt-in client-side VAD inactivity, separate from recording segmentation. Default
off. The trial operating point is three seconds; it is not a forced maximum
session length and does not interrupt an actively held push-to-talk/hybrid key.

```toml
[dictation.auto_pause]
idle_ms = 3000              # 0 disables; otherwise 500..=3600000
drain_timeout_ms = 10000    # failure bound, not a successful finalization
```

The detector uses `[dictation.auto_pause.vad]` when supplied, otherwise the explicit
`[dictation.audio_history.client_vad]` model/hash/runtime configuration. It works
with audio-history retention disabled and does not write recordings itself. A
missing feature/artifact, wrong hash or detector error never becomes a guessed
silence decision. There are no implicit downloads, remote inference or speaker
identification. The detector has independent recurrent/frame state so recording
privacy resets cannot silently change the idle controller.

Build matching components (the dependency alias is `ears`, not the package name):

```bash
cargo build --release -p ears-ui --features ears/client-vad,ears/parakeet-rs,ears/moonshine
cargo build --release --bin ears-server --features parakeet-rs,moonshine
```

A matching standalone server and a supported current engine must advertise
`dictation-drain/1`. Moonshine and Parakeet implement it. Older servers,
unsupported engines and the legacy broadcast-server path never receive an unsafe
timer fallback: manual capture remains active, with an availability error.
Switching engine revokes/restores the advertised capability. A code build is not
an installed or live-qualified deployment.

## Client lifecycle

- Silence is measured in scored 32 ms capture frames, not the arrival of ASR text
  or elapsed wall time. No audio means no invented silence deadline. Any detected
  speech renews the interval; the detector's accuracy is still a qualification
  concern. Physical key holds defer the cut.
- After the idle deadline and queued input has been handled, stop sending new
  transcription audio and enqueue an owned drain request behind all accepted
  audio. Keep the capture generation active while receiving trailing results.
- A bounded in-memory buffer holds post-cut input. If speech is detected during
  finalization (or a physical hold requests continuation), rotate to a fresh
  stream and replay it once without rescoring or recording it twice. Otherwise,
  accept the owned final/ACK and pause; the existing wake tee rearms then.
- The client reconciles the terminal cumulative Final against its bounded stream
  Word prefix. It appends only a genuinely missing suffix, including final-only
  output, never erases/retypes committed words or guesses across revisions.
  A mismatch is a visible failure. Original recognition remains unassigned to
  acoustic training clips, as described in `client-vad.md`.
- Cancel, mode/config changes and capture-generation changes fence old callbacks
  and discard uncertain pending commands. Rapid pause/resume rotates the stream
  rather than letting an old ACK pause a new capture. A normal manual stop also
  rotates an auto-pause-enabled stream; it is not a promise of graceful finalization.
- No matching final/ACK, malformed acknowledgement, disconnect, detector failure
  during drain or replay overflow produces a successful pause claim. The client
  stops that capture with an explicit error, discards unconfirmed pending text,
  closes the old stream and bounds writer shutdown. Already streamed text is not
  erased. Source recordings already committed are not deleted.

Pause happens after the silence interval **plus actual finalization latency**.
During finalization the app is still completing dictation, not listening for a
new wake phrase. Wait for the paused state before saying the wake phrase; the
existing wake detector also has its own warm-up limit (see `wake.md`). No new
visual countdown or sound is introduced by this change.

## Wire contract: `dictation-drain/1`

```json
{"type":"capabilities","features":["dictation-drain/1"]}
{"type":"drain","request_id":1}
{"type":"final","text":"all session words","words":[]}
{"type":"drained","request_id":1}
```

The final example abbreviates word timestamps. The terminal `Final.text` must be
the cumulative join of that engine session's Word history, plus any final-only
tail. The engine's queued drain owns the exact nonzero request ID; it is not an
interception of an unrelated Final. All prior accepted PCM and controls must
precede it. ACK follows final result emission, then the server sends Close. The
reader admits no more audio after accepting the barrier. Each new stream has a
fresh transcript ledger; request IDs are not reused within the client runtime.

The client flushes an Opus encoder's retained partial frame before enqueuing the
barrier, adding at most 40 ms of transport-only zero padding. That padding never
changes recording PCM or the detector clock. Malformed PCM, nonfinite samples or
Opus decoding errors terminate the server stream rather than acknowledging a
stream with dropped input.

Parakeet audio, language, boundary and drain commands share one bounded FIFO.
The worker flushes even a short accepted tail and propagates decoder errors;
failed admission/inference never ACKs. Moonshine uses its ordered native worker.
Server finish/write deadlines close failed streams without fabricating an ACK.
Native in-flight inference cannot be forcibly cancelled; its capacity permit
remains held until the worker returns. These mechanisms apply to the owned
connection, not other clients or sessions, and grant no action authority.

## Verification and remaining native gates

Model-free tests exercise the production client session over real loopback WS,
including delayed/final-only tails, duplicate/stale ACKs, renewed speech replay,
physical holds, unsupported servers, cancellation racing a late word and missing
ACKs. Server loopback tests and the production Parakeet worker with a fake decoder
cover FIFO ordering, short tails, errors, capacity cleanup and final/ACK/Close.
Capture marker/privacy/partition tests continue to cover recording integration.

```bash
just test-auto-pause
```

Native silence/quiet speech, endpoint latency, wake reactivation, hardware hotkeys,
long sessions and actual user acceptance are separate from those fixtures.
Parakeet retains its existing three terminal 560 ms zero chunks even after an
earlier boundary flush: native silence hallucinations are not yet qualified and
were not hidden by weakening the flush. Maximum-session limits are not implemented.
Source audio and private transcripts must remain local during qualification.
