# Command decision traces (v1)

Shared speech/ctx interchange: `examples/command-trace.schema.json` defines the
`byteowlz.command-trace/v1` envelope. This is not a second command catalog, action
policy, or execution grant. The user-assigned context/action agent owns its action
and ctx work; this seam records its decisions without prescribing those actions.

## Producer contract

Correlate events with a generated `decision_id` and, when known, `utterance_id`.
Keep them across model comparison, policy, preview, feedback and execution. Never
reconstruct a new instruction from a cumulative ASR session Final.

Suggested event kinds and payloads:

| Kind | Evidence |
| --- | --- |
| `routing_input` | Completed text, mode, language, capture/cancellation generation, scoped ctx snapshot reference and age, action-catalog version/hash |
| `model_request` | Provider, requested model ID, exact ordered request UTF-8 and SHA-256, fixture or other input provenance |
| `model_result` | Correlated request ID, returned model ID, verified revision or null, exact response UTF-8 and SHA-256, HTTP status, wall-clock timings, complete raw probabilities |
| `policy_decision` | Deterministic baseline, allowed/refused/uncertain result and reason; model confidence is not permission |
| `action_proposal` | Stable action ID, typed arguments, stable target reference, risk, context and cancellation generation; producer owns the action schema |
| `owner_feedback` | Accepted/rejected/corrected intent, action and target, explicit label source; distinguish preview acceptance from actual execution success |
| `execution_result` | Dispatch receipt, target revalidation, success/failure/cancelled/unavailable and duration; do not infer success from a button press |
| `failure` | Stage, failure classification, retry/cancellation information and known request/result references |
| `suppressed` | Timestamp and generic suppression reason only, with no text, context or identifying correlation fields |

Store bodies as strings, **not reconstructed JSON objects**: TypeSafe/D3 input
semantics depend on object order. Record hashes of exactly those UTF-8 strings.
Never include authentication headers, cookies, environment dumps or credentials.
A returned model identifier is not a verified checkpoint hash; unknown revisions
remain null. Preserve unsuccessful and uncertain decisions too.

## Privacy and labels

- Admission comes from trusted runtime state, not a model response or ctx payload.
  Record owner content only with consent, Secure Event Input off and a nonexcluded
  app. Context collection must stay within the explicitly consented scope.
- All raw traces stay in private local storage, normally
  `$XDG_STATE_HOME/ears/command-traces` (or the ctx owner's equivalent): directories
  0700, files 0600, outside every git checkout. No hosted web UI, remote logs or
  automatic cloud export. Secret screening is defense in depth, not a substitute
  for the producer's privacy gate; arbitrary secrets cannot be detected reliably.
- Log only scoped routing inputs, not ambient conversations or indiscriminate
  machine activity. Reuse existing consented recording references if useful;
  do not collect audio again or guess ASR-text/acoustic-clip alignment.
- A model prediction, Jev teacher output or successful action is not ground truth
  for user intent. Keep fixture, teacher, owner-review and outcome labels distinct.
  Every event has `training_eligible = false`; a separate reviewed export creates
  an immutable dataset version in the central registry. Live owner/context traces
  are restricted: card only in git, never raw traces, labels or manifests.

## Current implementation boundary

`scripts/command_trace.ts` is a tested **offline-worker** sink: explicit admission,
owner-only files, git-path and common-secret rejection, suppression records,
exact body preservation, unique per-process files and durable append. Run:

```bash
just test-command-traces
```

Frozen experiment scripts in registered data are provenance copies, not production
entry points. Replays must read that immutable version and write a new version or
separate trace store; never rerun an old harness in place over its frozen outputs.
Legacy summaries explicitly mark unavailable raw bytes rather than inventing them.

It performs synchronous disk I/O. Do not call it on the live keystroke/ASR event
loop. It is not yet a deployed eaRS live trace hook. Runtime producers must use a
bounded worker, surface overflow/disk failure as explicit gaps, and not claim an
action has a durable audit receipt before persistence succeeds. Dictation stays
independent; consequential actions must not silently proceed after required audit
failure. This contract does not relax existing speaker-evidence or human-gesture
execution requirements. Training consent and execution consent remain separate.
