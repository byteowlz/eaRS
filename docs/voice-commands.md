# Deterministic voice routing (incremental slice)

This is a **dictation-client** feature. The WebSocket STT server remains generic; no inference decides whether a command executes. In `$XDG_CONFIG_HOME/ears/config.toml` (or the platform config directory's `ears/config.toml` when XDG is unset), set:

```toml
[voice_commands]
mode = "command_aware" # dictation (default) | command_aware | command_only
# path = "/absolute/path/to/commands.toml" # default: $XDG_CONFIG_HOME/ears/commands.toml
```

Command registry (`commands.toml`, [JSON Schema](voice-commands.schema.json)):

```toml
[[command]]
phrase = "insert greeting"
risk = "safe"
action = { type = "literal_text", text = "Hello there" }
```

Only **literal_text** actions are supported in this slice. They type a configuration-authored string (and a trailing space) through the existing insertion mode; no speech captures, arguments, shell invocation, shortcut injection, executable lookup, or control characters (including Enter/newlines). Only explicitly `risk = "safe"` definitions are accepted; consequential/destructive tiers fail closed until a confirmation flow exists. Empty/duplicate/reserved phrases and invalid registries disable **all** registered commands (fail closed). The registry is checked for changes at completed utterance boundaries and on `Reload`. Do not put sensitive phrases in a shared config. A command can only run when capture is active and the completed utterance is an exact match after dictionary replacement and canonical whitespace/case/terminal punctuation normalization. In dictation mode only reserved controls are recognized; all other words stream through the existing word/dictionary buffer immediately. No speculative command text is inserted and later deleted.

In command-aware mode unmatched utterances type at `Speech(false)`; in command-only they are discarded. Partial word events never execute commands. The island shows dictionary-normalized provisional text on *each* word even when no OS keystrokes are inserted; its armed label displays the current mode. A leading possible `eaRS` control prefix is briefly held in live dictation; once impossible it is released. Exact reserved controls: `eaRS dictation mode`, `eaRS command-aware mode`, `eaRS command-only mode`, `eaRS start typing`, `eaRS stop typing`, `eaRS cancel`, `eaRS what mode`. `start typing` temporarily streams in command-only until `stop typing`; `what mode` surfaces the current mode; `cancel` pauses capture. Controls require a raw leading `eaRS` and normalized whole-utterance equality. When internal hotkeys are active, Escape is automatically enabled in command modes: it discards pending words, pauses capture, and restores dictation mode even if `hotkeys.escape_cancels` is false. The existing capture hotkey remains usable. Incomplete phrases, connection loss, and capture release do not dispatch commands. Neither a model nor a dictionary alias can synthesize a reserved leading token from unrelated speech. Dictation may still hold *dictionary phrase prefixes* for the pre-existing bounded phrase timer, independent of command classification.

`VoiceDecision { raw, normalized, outcome }` events are emitted **once at completion** for observability (`streamed` means already typed word-by-word, not discarded). Transcript history writes a `voice_decision_<outcome>` record following its existing privacy settings; text is not persisted if that setting is disabled. The actions and control events are client-local, not sent upstream. Modes restart from config (not persisted automatically).

## Shadow decision-model evaluation (not production authorization)

A UI/CLI subscriber can collect `VoiceDecision` events, with consent, into a private JSONL file and run an **offline** typed classifier on completed VAD-bounded utterances only. No shadow inference on the audio/typing loop and no model result influences routing. Recommended record contract (schema revision `voice-shadow-v1`): `{schema_revision, raw, normalized, deterministic_outcome, label, probabilities: {literal_dictation, registered_command_candidate, uncertain}, model_id, model_revision, loader_revision, inference_latency_ms}`. A human label is one of `literal_dictation`, `registered_command_candidate`, `uncertain`; include a separate `registered_phrase` only if verified against the same registry revision. Store config/dictionary revisions and language when comparing runs. Record p50/p95/p99 latency, false-positive command candidates and uncertain rate versus the deterministic baseline, stratified by mode and speaker/noise. Replay the labeled [fixtures](voice-command-replay.jsonl) and real consented utterances; never treat a classifier candidate as authorization. Preserve raw/normalized strings only in a local, private opt-in store; do not check real speech into git.

A subsequent [offline Kev-0.5B evaluation](kev-shadow.md) used the pinned model/base/loader on synthetic completed utterances. It performed poorly and is not approved for runtime use. Four-billion-parameter Kev still needs verification and measurement. Hosted Jev is a research reference, not a runtime dependency. The replay fixture and deterministic event seam allow opt-in shadow evaluations without increasing ordinary dictation latency. Speaker identity, executable/shortcut/platform adapters, confirmation and richer audit remain open in trx-2stp.3/.4; do not claim these are implemented.
