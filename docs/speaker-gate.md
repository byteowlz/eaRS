# Optional speaker instruction gate

Status: bounded software slice for `trx-2stp.4`, **not live-filter-ready or
robot-ready**. Default-disabled ordinary dictation is unchanged. No enrollment,
model/profile loading, deployment, microphone, UI redesign or robot effects are
performed by this feature.

## Three distinct capabilities

| Path | Implemented | Qualification |
|---|---|---|
| `ears::speaker_gate::Gate` completed-audio path | Immutable bounded PCM token shared by ASR and evidence; final/dictionary join; asynchronous evidence; scoped single-use dispatch | Generated PCM, synthetic ASR/evidence and canonical synthetic registries tested. Real Kyutai completed-audio adapter compiles; model inference unrun. |
| `speaker_gate::voqx::VoqxProvider` (`speaker-voqx` Cargo feature) | Explicitly supplied `Embedder` + canonical recognition snapshot; resampling and real voqx embedding/matcher calls | Compile-only, **evidence-only**. It cannot permit allowlist instructions: checkpoint attestation and calibrated overlap/speaker-change detector are missing. |
| Native dictation over legacy WebSocket | Configured policy blocks before live typing, mode controls, command matching, `Utterance` or `PromptReady` dispatch | Enabled modes report `unsupported_audio_binding`; provisional `Partial` remains reference-only. No microphone-arrival slicing, speaker cache or fallback. Browser/remote speaker gating is not implemented. |

`Status` distinguishes mode, provider presence and `unavailable`, `evidence_only`
or `provider_contract_unqualified`. There is deliberately **no automatically
ready state**. A provider declaring its contract is not empirical qualification.

## Terms and use

- **Utterance**: one bounded completed PCM input; not an arbitrarily long capture.
- **Evidence**: recognition status and detector/coverage information, not identity
  authentication or execution permission. Canonical `voice_id` is voqx-owned.
- **Scope**: consumer, target, generation, current policy and authoritative
  recognition model/profile snapshot. Not a current-focus-derived grant.
- **Eligibility**: permits one exact normalized instruction to reach an independently
  authorized consumer. It does not create an operator, tool or robot grant.

A native library consumer owns capture, boundaries, model preparation and the
actual permission/effect implementation:

1. Create a `Scope` and prepared local `EvidenceProvider`; no implicit load/download.
2. `begin`, feed mono **24 kHz** chunks with `push_pcm`, then `seal_audio` at the
   completed acoustic boundary. Duplicate boundaries return the same token.
3. On an ASR worker, call the token's `transcribe` with a final-only transcriber,
   applying dictionary normalization before returning the authoritative text.
   `transcribe_kyutai` wires the same immutable PCM into an existing caller-owned
   eaRS `Model::transcribe_pcm`; it loads nothing. This requires an exclusively
   owned, unbatched model, resets state around the input (including prior priming),
   and must not use a shared server model. Existing ASR-only silence finalization
   is not speaker evidence; no silence is added to the recognition input. Its real
   inference is not part of software fixture proof. ASR callbacks are trusted in-process adapters, not
   an untrusted way to attach arbitrary speaker verdicts to another input.
4. `submit_final` and nonblocking `poll`. Newer final corrections invalidate old
   results; same-revision changed payload is terminal conflict. Partial ASR text
   never enters this interface. Acoustic end and final completion are distinct.
5. Match the normalized, complete command exactly using the existing command
   registry; no confidence/model guess routes to effects. Call `dispatch` with
   the exact text, **current** scope, an independent permission check and effect.
   The receipt is consumed before permission/effect, including failure/panic.
   No duplicate, changed-text or annotation receipt can execute.
6. `reconfigure` on cancel, reconnect, target/focus, model, policy/allowlist or
   registry-profile changes. This fences all pending work even if a caller
   accidentally reuses its revision number. `discard` frees a single utterance.
   The current authoritative registry snapshot must be refreshed before effects;
   the module does not watch registry files or notice undeclared external changes.

Keep `transcribe`/`transcribe_kyutai` off capture/render threads. The gate submits
speaker work to one bounded local worker. Do not resubmit detached observations
as instructions. `Mode::Annotate` produces reference outcomes with bounded
uncertainty/failure reasons and can **never** dispatch, regardless of its text.
Actual voqx's unsupported consistency detector remains visible in those outcomes.

## Fail-closed/resource contract

Confident identity must match an allowed canonical voice ID, exact engine/model
and profile revision, full PCM coverage, finite score, and qualified
speaker-consistency evidence. Unknown/tentative/short/invalid audio, uncovered
suffix, possible overlap/switch, unsupported detector, incompatible/missing
provider, provider error/panic, expired evidence, stale scope, payload conflict,
revocation, permission denial and queue overload produce no effects. Two allowed
people are supported; display names and synthesis references never authorize.

Limits: four admitted utterances; each at most 720,000 samples (30 seconds at
24 kHz) and 4096 UTF-8 transcript bytes; one queued job, one running job and one
queued result. Oversized/nonfinite audio poisons the whole utterance. Appending
PCM after sealing invalidates the prefix. Denied entries consume bounded slots
until discarded; there is no unbounded dedupe ledger. Unallocated/consumed IDs
cannot be readmitted. Sealed tokens retained by a consumer are that consumer's
memory responsibility.

The first final starts the evidence deadline (default 2 seconds, maximum
30 seconds); corrections do not extend it. A timeout fences late results even
if the provider ignores cancellation. Provider cancellation is cooperative:
this does not forcibly interrupt native inference, join a hung thread on drop,
or restart workers indefinitely. A blocked provider may deny availability;
it cannot yield approval for a later instruction. Poll regularly to observe
expiry and discard finished entries. An external provider must keep its own
panic hooks/logs privacy-safe; the gate does not log its returned errors.

Status/outcome serialization excludes PCM, raw/normalized text, names, voice IDs,
vectors, scores and target/registry paths. Legacy provisional transcript events
retain their existing local reference semantics; they are not eligible events.
The enabled legacy adapter neither records denied transcript decisions nor emits
them in `VoiceDecision`; its reason has empty raw/normalized fields.

## voqx adapter limits

`PreparedRegistry` consumes an explicitly supplied canonical `VoiceRegistry`,
retains only exact engine/checkpoint recognition profiles, computes a private
revision fingerprint, and calls the canonical voqx matcher. It is an in-memory
snapshot, **not another identity store**. Explicit thresholds and calibration
provenance are required; validation is not empirical calibration. Unit-test
thresholds describe synthetic vector geometry, not a security recommendation.

At the pinned `speakrs 0.4.2`, reject less than 240,000 input samples as
`insufficient_audio`, and more as `unsupported_coverage`, **before inference**.
Only an exact 10-second 24 kHz input is resampled to the required 160,000 samples.
No prefix truncation, caller-added silence, repetition or artificial padding.
Even that exact window reports `SpeakerConsistency::Unsupported`; WeSpeaker
alone is not an overlap/switch detector. The caller-supplied profile model version
is not attestation of the actual embedder checkpoint. This is why actual voqx
remains evidence-only even for a confident identity.

This source slice uses the read-only sibling `../voqx` checkout (reviewed HEAD
`36ca757a7f4f5f28bfc2b4ed97cfe63919e5082e`) and Cargo.lock's `speakrs 0.4.2`.
Standalone packaging without that sibling/pinned dependency has not been qualified.
No real registry, recording, private vector or model was used for the fixture
suite. The ignored `opt_in_local_voqx_adapter_remains_evidence_only` test requires
separately authorized local model inference, explicit
`EARS_SPEAKER_INFERENCE_CONSENT=explicit-local-generated-pcm-only` and an existing
`EARS_SPEAKER_MODEL_DIR`; it uses generated PCM and a synthetic registry and must
not be enabled by ordinary CI. It is unrun in this slice.

## Configuration

Per-consumer policy lives at `[voice_commands.speaker_policy]` in the existing
XDG eaRS `config.toml`. Missing policy means `disabled`. `annotate` and `allowlist`
on the **legacy WS dictation client suppress dispatch**; configuring two voice
IDs does not make that adapter capable. Physical/PTT release and keyboard cancel
are never gated by voice evidence; enabled gating keeps Escape available.

See [`../examples/speaker-policy.toml`](../examples/speaker-policy.toml) and
[`speaker-policy.schema.json`](speaker-policy.schema.json) (policy object only).
No UI control or model configuration is implied. Scope/model/profile revisions
are library consumer lifecycle inputs, not claims supplied by untrusted ASR JSON.

## Proposed upstream contracts (not implemented)

- **eaRS `trx-2stp.15`** owns a versioned, negotiated WS binding: capture/stream
  epoch + generation, utterance ID, exact PCM representation/rate and sample span
  or digest/revision, authoritative final ID/revision/hash, separate acoustic-end
  and committed-final markers. Engines unable to establish that exact join must
  advertise unknown/unsupported; old clients retain ordinary dictation and cannot
  infer instruction capability. Corrections, dropped audio, Opus/resampling,
  cancellation/reconnect and target/policy/profile revocation need explicit
  compatibility semantics and zero-dispatch negative tests.
- **voqx `trx-j7hk`** owns loaded-checkpoint provenance, exact compatible canonical
  profile snapshot/revocation reporting, observed sample/window coverage and a
  discoverable calibrated overlap/speaker-change/uncertainty capability. This
  means detector evidence within documented conditions, **not proof that no other
  speaker exists**. Missing detector/provenance/coverage remains unsupported or
  unknown. Existing **voqx `trx-hyha`** retains window/padding exposure work;
  **`trx-ej3w`** retains speaker-change acoustic scoring. These are linked work,
  not duplicated algorithms or authorization promises.

`trx-2stp.4` remains open/in progress and is blocked by the eaRS WS follow-up for
useful live WS operation. Cross-repo voqx work is text-linked, not an invented
cross-repo dependency. No follow-up public API/protocol/source change is made here.

## Verification and safety

Run `just test-speaker-gate` and `just check-speaker-gate`. On the reviewed
macOS build, optional-feature test linking also emitted duplicate protobuf symbol
warnings from static ORT/SentencePiece (exit 0); this remains a native packaging
qualification gap, not a warning to suppress. No all-features accelerator build
or cross-platform qualification is claimed. The tests cover exact
sealed PCM/final joining, dictionary-first exact mock intents, two allowed IDs,
reference-only annotations, negative statuses/coverage/provider failures, stale
corrections/verdicts, cancellation/reconnect/target/policy/profile changes,
revocation at effect, single-use payload binding, deadlines/resource bounds,
privacy-safe status and enabled legacy suppression. Existing default-disabled
word/command behavior and keyboard/physical cancellation are regression-tested
with a recording keyboard, not an OS input device.

No claim of EN/DE/ES identity accuracy, calibrated false-accept rate, live AEC,
physical microphone/playback operation, native/browser parity, operator/device
security or robot safety follows. A recording, cloned owner TTS or self-echo can
match a voice. Controllers separately require authenticated authority, scoped
operations, confirmations, physical interlocks and an independent emergency stop.
This slice implements no robot motion or actuation.
