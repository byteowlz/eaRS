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
- The reference detector is `wake listen --manifest <release-dir>` from the `wake`
  repository. A release directory (`manifest.json` with a sha256 per file) holds the
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

## Wake log

With `log = true` (default when wake is enabled) every detection and near miss is appended
to `$XDG_STATE_HOME/ears/wake-log.jsonl` (mode 0600): time, phrase, score and what eaRS did
(`started`, `ignored`, `below_min_score`, `unmapped`, `already_capturing`; near misses
`none`). No audio. Near misses need a detector that reports them (`wake listen
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
