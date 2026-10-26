# Audio8 ASR Infinite evaluation

Local smoke test, 2026-09-30; issue `trx-0hf0`. **Verdict: watch / optional English-only experiment, not a default engine replacement.** No eaRS runtime or user configuration was changed. All inference was local; no recordings were uploaded.

## Pinned inputs

- Model: [Edge0/Audio8-ASR-Infinite](https://huggingface.co/Edge0/Audio8-ASR-Infinite), revision `7476824bc222e4ad509d286e8cae8b8d3f371129`; approximately 8.17 GB weights, bfloat16. Model card declares Apache-2.0.
- Runtime: [Edge0-AI/Audio8-ASR-Infinite](https://github.com/Edge0-AI/Audio8-ASR-Infinite), revision `c8ba8eea829be0339e8d7757f8ca52dac06e1e32`; Apache-2.0. Installed with its `client` extra using uv in an isolated temporary environment.
- Machine: Apple M2 Ultra, 192 GiB unified memory, macOS arm64; Python 3.12.13, Torch 2.13.0, Transformers 5.13.0; MPS device, bfloat16.
- Decode: upstream `simulated_streaming_greedy_decode_batch`, English, 80 ms clock, 480 ms configured delay, 18 left-pad tokens, 10 right-pad text tokens, maximum 512 generated tokens; model/cache initialized once.
- Audio: existing `test_audio/spqx_alpha_bravo.wav` and the first three rows of [hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy), revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`, `clean/validation-00000-of-00001.parquet`. The synthetic sample has a non-English reference accent; the public clips all share one speaker. Neither is representative dictation coverage.

## Actual results

Decode time excludes model loading and waveform loading/resampling. MPS was synchronized immediately before and after each timed decode. The synthetic clip was decoded once as warmup and once again for measurement.

| Input | Audio seconds | Decode seconds | Processing / audio | Normalized WER |
| --- | ---: | ---: | ---: | ---: |
| Synthetic alpha/bravo | 3.200 | 2.176 | 0.680 | 75.00% |
| `1272-128104-0000` | 5.855 | 3.679 | 0.628 | 5.88% |
| `1272-128104-0001` | 4.815 | 3.024 | 0.628 | 10.00% |
| `1272-128104-0002` | 12.485 | 9.458 | 0.758 | 3.13% |

A ratio below one means faster-than-real-time processing. These are single measurements, not latency distributions. Model initialization plus transfer to MPS took **5.024 s** with weights already downloaded. The first synthetic decode took **6.684 s**, showing a substantial cold-inference penalty.

WER uses jiwer after lowercasing, removing punctuation, collapsing spaces and stripping. Public clips aggregate to **3 / 59 words = 5.08% WER**; two substitutions are written-form `MISTER` -> `Mr.` rather than acoustic errors. The other substitution is `and` -> `in`. Do not compare this tiny, selected subset with upstream benchmark scores.

Actual transcripts:

- Synthetic reference: `alpha bravo charlie now`; output: `Aflor Uchrali. Now.` The same fixture was also difficult for Nemotron in the [CoreML/ONNX spike](nemotron-coreml-spike.md); this is not evidence Audio8 wins that comparison.
- `0000`: `Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.`
- `0001`: `Nor is Mr. Quilter's manner less interesting than his matter.`
- `0002`: `He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating in its results occur most readily to the mind.`

All four final words survived the simulated-streaming flush. **Live first committed word, VAD endpoint latency, cancellation, concurrent dictation/LLM contention, device memory, and continuous-stream stability were not measured.** This Torch test processes a known waveform through causal windows and returns final text; it is not a paced WebSocket or microphone latency test. The 480 ms delay setting is not an observed end-to-end latency.

## Reproduction and checkpoint loader trap

The unchanged upstream Torch CLI failed before inference:

```text
RuntimeError: Audio8 ASR Infinite checkpoint does not exactly match the current weight format.
incompatible={'missing_keys': {... semantic_vad_heads.0.weight ...}}
```

The published directory contains both `model.safetensors` and `model.safetensors.index.json`. Transformers prefers the single main file, bypassing the index and the eight VAD parameters in `semantic_vad_heads.safetensors`. Both files were downloaded correctly. A local indexed view fixes loading without modifying any tensor, disabling VAD, or relaxing the strict loader: rename the main file in a new directory and update only its index entries. The successful load read **938 weight tensors**, including the eight VAD tensors, versus the main file's 930. Assertions verified that the two files' key union exactly matches the index, the documented CLI reproduces the synthetic transcript, and aggregate public-clip WER is 3/59.

Run outside the eaRS repository:

```bash
git clone https://github.com/Edge0-AI/Audio8-ASR-Infinite.git audio8-runtime
git -C audio8-runtime checkout c8ba8eea829be0339e8d7757f8ca52dac06e1e32
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python './audio8-runtime[client]'
uv run --no-project --python .venv/bin/python hf download \
  Edge0/Audio8-ASR-Infinite \
  --revision 7476824bc222e4ad509d286e8cae8b8d3f371129 \
  --local-dir checkpoint --exclude '*.mp4'
uv run --no-project --python .venv/bin/python python - <<'PY'
import json
from pathlib import Path
src = Path('checkpoint').resolve()
dst = Path('checkpoint-indexed')
dst.mkdir()  # fresh directory; original download remains unchanged
for p in src.iterdir():
    if p.is_file() and p.name not in ('model.safetensors', 'model.safetensors.index.json'):
        (dst / p.name).symlink_to(p)
(dst / 'asr.safetensors').symlink_to(src / 'model.safetensors')
index = json.loads((src / 'model.safetensors.index.json').read_text())
index['weight_map'] = {
    k: ('asr.safetensors' if v == 'model.safetensors' else v)
    for k, v in index['weight_map'].items()
}
(dst / 'model.safetensors.index.json').write_text(json.dumps(index))
PY
uv run --no-project --python .venv/bin/python python \
  -m audio8_asr_infinite.examples.torch_streaming_decode \
  --checkpoint checkpoint-indexed --audio /path/to/sample.wav \
  --language en --device mps --dtype bfloat16 \
  --transcription-delay-ms 480 --output result.json
```

That CLI reproduces transcript behavior; its whole-process elapsed time includes loading, unlike the table's timed decode calls. Raw run artifacts for this session are local in `/tmp/ears-audio8/`: `evaluate.py`, `results.json`, `evaluate-fixed.log`, and `requirements.lock.txt`. They are temporary, not checked-in assets.

## eaRS fit

- **No German:** model/runtime accept only `en` and `zh`. Calling `resolve_qwen_language_token_id(None, 'de')` was tested and raises `ValueError: language must be exactly 'zh' or 'en', got 'de'.` Do not silently map German to English.
- **Native streaming, but a different deployment stack:** bounded 30 s rolling KV and 24/7 operation are upstream claims tied to their CUDA/vLLM 0.27.1 plugin and Docker stack. They were not proven by this short MPS/Torch test. There is no supplied Apple CoreML/MLX or eaRS Rust engine in the evaluated repository.
- **Sidecar boundary if adopted:** eaRS's 24 kHz float PCM would need conversion to the model's 16 kHz input. Upstream realtime uses base64 PCM16 `input_audio_buffer.append`, explicit commits, and `transcription.delta` / `transcription.done`, not eaRS `Word` events. Join subword deltas safely and prove commitment before typing; handle flush, language changes, disconnects, bounded queues and reset. VAD heads existing in the checkpoint do not establish usable eaRS turn-boundary behavior.
- **Quality/size tradeoff:** upstream reports English LibriSpeech clean/other WER 3.042/6.808 versus Voxtral realtime 2.210/5.552. Its large advertised advantage is on Chinese; those are publisher measurements, not this test. The 8.17 GB checkpoint is much heavier than the tested ~668 MB int8 CoreML Nemotron conversion.

Keep current engines/defaults. Reconsider for Chinese dictation or an English-only GPU sidecar after a paced comparison against the current eaRS backend on identical consented utterances, including tails, pauses, silence, noise and a stream exceeding the 30 s cache window.
