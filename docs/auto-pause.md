# Silence-triggered dictation pause

Opt-in client-side VAD inactivity, separate from recording segmentation. Default
off. The trial operating point is three seconds; it is not a forced maximum
session length and does not interrupt an actively held push-to-talk/hybrid key.

```toml
[dictation.auto_pause]
idle_ms = 3000              # 0 disables; otherwise 500..=3600000
drain_timeout_ms = 10000    # failure bound, not a successful finalization
speech_threshold = 0.5    # finite, strictly between 0 and 1
```

The detector uses `[dictation.auto_pause.vad]` when supplied, otherwise the explicit
`[dictation.audio_history.client_vad]` model/hash/runtime configuration. It works
with audio-history retention disabled and does not write recordings itself. A
missing feature/artifact, wrong hash or detector error never becomes a guessed
silence decision. There are no implicit downloads, remote inference or speaker
identification. The detector has independent recurrent/frame state so recording
privacy resets cannot silently change the idle controller.

### Speech threshold in background noise

Settings → Dictation → Auto-pause exposes a speech-probability slider (0.01–0.99,
0.01 steps; default 0.50). Raise it gradually, for example to 0.65–0.75, if noise
keeps renewing the silence interval. These are trial values, not calibrated noise
presets: higher values can also miss quiet speech. Once locally paused, speech
alone does not resume dictation: use wake or the capture control. A threshold
does not separate background speakers from the owner.

Click/drag or focus the slider and use arrows, Home or End. Tab/Shift-Tab move
focus through Settings. A release or keyboard step saves only this field in the
engine's `config.toml`, preserving other values/comments and intentional dotfile
symlinks via atomic replacement. A save failure is shown inline and does not
apply the edit. Synthetic previews never save core settings. The live command
updates classification without rotating capture, changing its generation,
resetting the detector or cancelling an outstanding Final/ACK drain. It does not
enable auto-pause or provide missing detector/model support. Manual file edits
are loaded when the engine starts again.

This threshold is **only for local capture inactivity**.
Recording onset/offset settings, wake thresholds and the server's RMS endpoint
are unchanged. It does not itself fix the delayed-word endpoint issue.

The UI dependency always includes `client-vad`, including plain `just install-ui`
and `cargo build -p ears-ui`. Compiled support does not enable the feature, load a
model or change saved settings. Other clients must enable `client-vad` explicitly.
Optional local-server builds can still add ASR engines (the alias is `ears`):

```bash
cargo build --release -p ears-ui --features ears/client-vad,ears/parakeet-rs,ears/moonshine
cargo build --release --bin ears-server --features parakeet-rs,moonshine
```

**Local pause does not require a server capability, response or connection.**
`dictation-drain/1` is an optional stronger ASR-completion contract; Moonshine and
Parakeet implement it. Legacy servers receive no unsupported drain command.
A later capability advertisement can upgrade the pending operation once, behind
all pre-cut audio and the Opus flush. Neither a timeout nor a legacy Final is
invented confirmation. A source build is not an installed/live-qualified deployment.

## Client lifecycle

- Silence is measured in scored 32 ms capture frames, not the arrival of ASR text
  or elapsed wall time. No audio means no invented silence deadline. Any detected
  speech renews the interval; the detector's accuracy is still a qualification
  concern. Physical key holds defer the cut.
- After the idle deadline and queued input has been handled, pause local capture
  immediately and rearm the existing wake tee. Seal recording capture once.
  Keep the old ASR stream separately owned and bounded while receiving tails.
  With negotiated support, its owned drain follows all accepted pre-cut audio.
- Connecting handshakes and disconnected retry waits also service the local VAD
  clock and physical-hold guard. Offline audio is not replayed into a later
  connection/target. No available ASR completion is reported as unconfirmed.
- A new wake, capture control or hold may start capture while the old stream
  finishes. Its bounded PCM queue belongs to the new generation: it is not sent,
  scored or retained under the old one. After old completion/failure it replays
  once through the fresh stream's observers. Overflow stops capture explicitly;
  it never dispatches a partial replay.
- The client reconciles the terminal cumulative Final against its bounded stream
  Word prefix. It appends only a genuinely missing suffix, including final-only
  output, never erases/retypes committed words or guesses across revisions.
  A mismatch is a visible failure. Original recognition remains unassigned to
  acoustic training clips, as described in `client-vad.md`.
- Already-received words and final-only suffixes may finish the old insertion
  session only while locally paused, with no newer capture, the same known
  application/window snapshot and Secure Event Input off. Unknown/changed focus
  or a newer generation defers text rather than typing into the new target.
  Once deferred, later words stay deferred even if focus returns. This is not
  authenticated identity or a caret/field-tracking guarantee.
- Deferred text is bounded private evidence, not an executable action. It emits
  `DeferredTranscript { capture_generation, text, confirmed }` with a visible
  nonfatal warning. The UI keeps the latest deferred result in memory separately
  from normal utterances; a dedicated review/reinsert UI is not implemented.
  Only existing explicitly enabled transcript retention may persist it; no
  recording, training or transcript retention setting is enabled by this path.
- Cancel invalidates pending insertion immediately when queued, including while
  capture is already paused. Mode/stream changes and superseded generations fence
  old callbacks and discard uncertain commands. Old control phrases/ACKs cannot
  stop a newer capture. A normal manual stop is not graceful ASR confirmation.
- Missing/invalid final/ACK, disconnect and writer failure produce an explicit
  **unconfirmed ASR result**, never a successful completion claim. Legacy tails
  remain pending only up to `drain_timeout_ms`. Received words are preserved;
  words a legacy backend never emits cannot be reconstructed or promised. Already
  typed text is not erased; committed recordings are not deleted. Failure closes
  the old stream with bounded writer shutdown, without stopping a valid newer
  capture.

`State(Suspended)` now means local capture is paused, independently of
`Finalization { capture_generation, state: Pending | Confirmed | Unconfirmed }`.
Only the owned terminal Final/ACK confirms the stronger completion contract.
Recognition confirmation is not proof of insertion or permission to act. ASR
status events do not resume capture or extend island activity; a completion error
may still surface its normal nonfatal warning. Per-frame VAD/idle/hold diagnostics
and Settings duration controls remain separate work.

Pause no longer includes ASR finalization latency. The wake detector still has
its own warm-up limit (see `wake.md`). Dictation paused does **not** mean the
hardware microphone is off when always-listening wake is enabled. No new visual
countdown, sound, server upgrade, restart or saved-setting change is required.

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
including delayed/final-only tails, duplicate/stale ACKs, new-generation PCM
replay, physical holds, legacy/no-ACK servers, stalled handshakes, disconnected
retry, immediate queued-cancel fencing, changed/unknown targets and deferred text.
These tests use synthetic audio/typing targets, not a live microphone. Server loopback tests and the production Parakeet worker with a fake decoder
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
