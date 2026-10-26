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

## Boundaries

A detection is evidence that a phrase was probably spoken. Here it can only do what
the toggle hotkey does: start capture (with the same notifications and hooks). It never
authenticates a speaker, selects a session or runs a command. If the detector cannot
start or exits, eaRS reports an error and dictation keeps working with the hotkey.

## Known limits

- The wake phrase itself is not transcribed, and words spoken right after it may be
  clipped while capture and the server catch up (no pre-roll is sent yet).
- eaRS does not play audio itself, so the playback gate (`P`) is not driven yet;
  hosts that speak (Foxline, Oqto read-aloud) must gate it.
- Measured false-trigger rates come from the detector's own evaluation, not from eaRS.
