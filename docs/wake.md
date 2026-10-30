# Wake phrases

Start dictation by saying a wake phrase instead of pressing the toggle hotkey.
Off by default (`[dictation.wake]`, see `examples/config.toml`).

## How it works

- eaRS remains the only microphone owner. While dictation is **paused**, the captured
  24 kHz audio is resampled to 16 kHz and streamed to a detector process. While
  dictation is capturing, nothing is sent; the detector is reset when capture pauses.
- The detector is any command speaking `wake-stream/1`:
  - stdin: binary frames `kind: u8`, `len: u32 LE`, payload. `A` = 16 kHz mono s16le
    audio, `P` = playback gate (1 byte), `R` = reset, `Q` = quit.
  - stdout: JSON lines. First `{"type":"ready","protocol":"wake-stream/1","sample_rate":16000,"phrases":[...]}`,
    then `{"type":"detection","phrase":"...","score":0.91,"at_sample":123456}` or
    `{"type":"error","message":"..."}`. stderr is free-form logging.
- `actions` maps reported phrases to `start_dictation` or `ignore`. Unlisted phrases do
  nothing. `min_score` adds an optional floor.
- The reference detector is `waqy listen --manifest <release-dir>` from the
  [waqy](https://github.com/byteowlz/waqy) repository. A release directory (`manifest.json` with a sha256 per file) holds the
  shared front end and one classifier per phrase, so swapping models means pointing the
  command at another directory.

## After a wake: pre-roll and phrase stripping

While paused, eaRS keeps the last few seconds of captured audio. When a mapped phrase
starts capture, the audio from `preroll_ms` (default 1500) before the detection is sent to
the server first, so "Hey Oqto, schreib bitte ..." keeps its first words. That audio also
contains the wake phrase, so up to three leading words found in `strip_words` are dropped
once (as typed text and at the utterance boundary). A hotkey start sends no pre-roll and
strips nothing. An utterance that is only the wake phrase types nothing.

## Stop phrases

`stop_phrases` (e.g. `"oqto stopp"`) are whole utterances that pause capture, like the
reserved `eaRS cancel`. They are matched after removing punctuation ("Okto, stopp." works),
only at the start of an utterance: a possible stop phrase is held back briefly like the
`eaRS` prefix and released (typed) as soon as it cannot match. Nothing is typed for a stop
phrase. They need no detector: the transcript decides, at the end of the utterance
(about half a second after you stop speaking). A bare "stopp" is deliberately not
supported; it would stop dictation whenever the word is dictated.

## Speaker-tool discovery (read-only)

```sh
ears dictation speaker-probe --json
ears dictation speaker-probe --voqx /path/to/voqx --json
```

The shared discovery operation looks beside the eaRS executable, along PATH, then
in supported per-user installation locations (including Cargo/local bin and macOS
Homebrew prefixes). An explicit override takes precedence and never silently falls
back. It invokes only `voqx capabilities --json`, with bounded output and a two-second
deadline; it does not load eaRS settings, rewrite the configured speaker command,
record, enroll, download models or start a service.

Results distinguish missing/nonexecutable files, probe failure, unsupported capabilities
and `compatible_executable`. Failures still emit structured JSON and exit nonzero. Older
builds without the capability command may report `probe_failed`, not an empty voice list.
`model_state` and `registry_state` remain `not_checked`; executable compatibility is not
model/profile readiness or calibrated filtering. The CLI/API is the nonvisual foundation
for the planned eaRS reference-setup flow (`trx-pbfx`); no settings UI is added here.

## After a wake: the speaker check (Layer 2, annotation only)

A wake can additionally ask a speaker command whether the voice belongs to an enrolled
person. It is off unless `[dictation.wake.speaker]` sets `command`, and it **annotates**: the
verdict is written to the wake log and stderr, and nothing about the dictation changes.

```toml
[dictation.wake.speaker]
command = ["voqx", "identify", "--json", "--mode", "cpu"]
identity = "vox-owner" # expected voice_id; checked locally, never passed to the command
threshold = 0.6      # uncalibrated example, not a recommended filtering threshold
mode = "annotate"    # annotate (default) | filter
min_ms = 3000        # shorter audio is not judged: the verdict is unknown
timeout_ms = 4000
```

Only the WAV path is appended to the configured command arguments. `identity` is a local
comparison target, **not** an extra positional argument to `voqx identify`. The command
must emit exactly one JSON object; compact and pretty-printed multiline JSON both work.
Diagnostics belong on stderr, not before/after the JSON. Custom tools needing a target
argument must put it explicitly in `command`; the adapter does not invent one.

For a configured identity, `voice_id`, `status = "confident"`, and a finite score meeting
the consumer threshold are required. Matching IDs produce *owner*, a different confident
ID produces *other*. Missing IDs/status, tentative/insufficient/incompatible results, and
scores below the consumer threshold produce *unknown*—low similarity is not proof of a
different speaker. A custom score-only verifier is supported only with no configured
identity; it must define its own target. These labels remain uncalibrated evidence.

Stdout and stderr are drained concurrently, each limited to 64 KiB. The deadline covers
process exit and both streams; failures kill/reap the direct child and yield *unknown*.
Raw stderr is not copied into evidence. Trusted wrapper commands must manage their own
subprocesses. Run checks on a blocking worker, never the audio or insertion thread.

For voqx's opt-in masked profiles, add `"--preprocessing", "valid-v1"` to the command and
enroll the matching profile explicitly first; there is no automatic registry migration.

The adapter invokes a command, so a separately installed voqx binary (or compatible tool)
is required at runtime; no voqx daemon is needed. The audio it is given is
the pre-roll plus what follows the wake, up to about 12 s, kept under
`$XDG_STATE_HOME/ears/wake-speaker/` so a logged verdict can be inspected. The log gains lines
like `{"type":"speaker","verdict":"owner","speaker_score":0.71,"audio_ms":3120,...}`.
Snapshot privacy/retention still needs the shared audio-history controls (`trx-hava`);
keep collection disabled for normal use until that follow-up is addressed.

`min_ms = 3000` is the existing conservative buffer default, not measured speech duration
or a fundamental model limit. Earlier duration claims confounded leading silence, padding
and unconfirmed speaker labels. No real-speaker operating point has been qualified here.
`filter` is parsed but **not wired**: it records the verdict and logs that filtering is not
implemented. Every failure resolves to *unknown*, never a filtering decision.

Live words are typed as they arrive. A future filter must hold text **before insertion**;
it cannot safely discard text retroactively. Annotation does not delay or alter insertion.
Unknown reasons in the JSONL log are bounded adapter diagnostics, not raw provider text.

Speaker evidence is not authentication or permission to act. Enrollment of other people
requires its own appropriate consent; it is not a default.

### Verification

`cargo test --lib wake_speaker` covers the argument boundary, multiline JSON, identity and
status checks, concurrent pipe draining, output limits and timeout behavior. With the
updated voqx binary and cached weights available, run the real CLI contract test:

```sh
VOQX_BINARY=/path/to/voqx cargo test --test wake_speaker_voqx -- --ignored
```

That test creates a generated-signal fixture and isolated registry, exercises enrollment
and identification through the actual CLI, and verifies that queries do not mutate the
registry. It does not use a personal registry or claim microphone/speaker accuracy.

## Wake log

With `log = true` (default when wake is enabled) every detection and near miss is appended
to `$XDG_STATE_HOME/ears/wake-log.jsonl` (mode 0600): time, phrase, score and what eaRS did
(`started`, `ignored`, `below_min_score`, `unmapped`, `already_capturing`; near misses
`none`). No audio. Near misses need a detector that reports them (`waqy listen
--report-above 0.3`). The console shows the same as `wake: hey_oqto 0.74 -> started`.

## Boundaries

A detection is evidence that a phrase was probably spoken. Here it can only do what
the toggle hotkey does: start capture (with the same notifications and hooks). It never
authenticates a speaker, selects a session or runs a command. If the detector cannot
start or exits, eaRS reports an error and dictation keeps working with the hotkey.

## Known limits

- After a pause (and after host playback), the detector needs about 2 s of audio before it
  can score again, so a wake phrase said immediately after pausing is missed.
- Pre-roll timing maps detector samples back to capture samples; a few tens of
  milliseconds of offset are possible.
- eaRS does not play audio itself, so the playback gate (`P`) is not driven yet;
  hosts that speak (Foxline, Oqto read-aloud) must gate it.
- Measured false-trigger rates come from the detector's own evaluation, not from eaRS.
