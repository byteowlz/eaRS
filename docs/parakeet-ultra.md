# Parakeet Ultra as an eaRS engine

**Moondream's Parakeet Ultra** (CC-BY-4.0, 0.6B TDT, 1.2 GB safetensors) is the
same architecture as `parakeet-tdt-0.6b-v3`, which eaRS already streams — but it
is a stronger checkpoint, especially for German and under noise, and it ships
for a Python runtime rather than ONNX. TDT is not cache-aware, so chunked
streaming degrades.

eaRS therefore runs Ultra as an **utterance engine**, off by default:

- The server buffers one utterance (bounded by `max_utterance_secs`), writes a
  temporary 24 kHz mono WAV, and runs your `command` with that path appended.
- The command prints `{"text": "..."}`, optionally with
  `{"words": [{word, start_time, end_time}]}`. Everything else is an error:
  a failed, timed-out or non-JSON command reports a problem and produces no
  text. Nothing is downloaded or bundled.
- Words and one `Final` are emitted when the utterance ends (the server's
  acoustic boundary); there is no interim/streaming text, because a
  non-cache-aware model would have to re-decode the whole utterance anyway.
- Live dictation stays on the streaming `parakeet-rs` engine (v3/Nemotron):
  using Ultra there would trade latency and stability for accuracy.

```toml
[parakeet_ultra]
command = ["python", "/opt/parakeet-ultra/transcribe.py"]
language = "de"
timeout_ms = 60000
max_utterance_secs = 120
```

Build with `--features parakeet-ultra`. The model itself is the operator's: the
worker script loads the safetensors checkpoint (revision
`moondream/parakeet-ultra`, cc-by-4.0 attribution applies) and owns its Python
dependencies, so eaRS gains no torch/NeMo dependency and the production server
is unaffected unless the engine is configured.

Selected locally on 96 reviewed German clips: Ultra 2.98% WER overall / 3.50% on
the holdout, against 5.5-7% for the previous candidates (vendor and local
numbers in the eval notes; re-measure on your own audio before switching).
