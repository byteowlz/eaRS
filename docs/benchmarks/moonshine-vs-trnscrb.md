# German bulk: Moonshine vs trnscrb engines

Local experiment, 2026-10-27 (`trx-7nw3.3.2`). No production defaults or context
bias were changed. This is a small public read-speech benchmark, **not** a claim
about broad German, personal vocabulary, live dictation, or the MacBook.

## Comparison

Apple M2 Ultra, macOS 26.6.2, 192 GiB unified memory. 24 unique reference texts
from FLEURS `de_de/test`, 540 reference words. Each clean 16 kHz mono PCM16 clip
has one second of silence appended. The longer fixture concatenates those exact
padded clips: **321.3 seconds (5m21s)**. The
[corpus manifest](moonshine-trnscrb-corpus.json) includes source and prepared
hashes. Source TSV matches FLEURS revision `d7c758a6dceecd54a98cac43404d3d576e721f07`;
this alone is not an independent proof of every audio file's upstream revision.

| Model / runtime | Device | 24 clips WER | Joined WER | Joined wall time | Speed | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| Moonshine Tiny German base / eaRS-style native stream | CPU, 1 ORT thread | 18.3% | 17.6% | 14.75 s | 21.8× | 252 MiB |
| Moonshine Tiny local cased fine-tune / same | CPU, 1 ORT thread | 17.4% | 16.1% | 15.74 s | 20.4× | 373 MiB |
| Parakeet TDT v3 Q8 / parakeet.cpp v0.4.0 | Metal GPU, 4 CPU threads | 7.4% | 13.5% | 4.77 s | 67.4× | 5015 MiB |
| Whisper large-v3-turbo Q8 / whisper.cpp 1.8.6 | Metal GPU, 4 CPU threads | 7.6% | 8.0% | 7.21 s | 44.6× | 1262 MiB |
| Whisper small / whisper.cpp 1.8.6 | Metal GPU, 4 CPU threads | 10.4% | 13.9% | 6.94 s | 46.3× | 961 MiB |

WER is corpus-weighted Levenshtein word distance after NFC normalization,
lowercasing, punctuation deletion and whitespace collapse; no number/ITN
normalization. German `ß` is not folded into `ss`. Results include substitutions,
deletions and insertions against FLEURS text; references can themselves be
imperfect. Accuracy figures are single outputs; repeated joined runs produced the
same WER in each condition. This is **not** evidence of an accuracy difference
on owner dictation, and potential fine-tune training overlap was not audited.

Speed is 321.3 / median wall time, three sequential **fresh-process** runs after
an initial suite has warmed the file/system caches. Process launch and model load
are included; network download, transcription manager/ffmpeg conversion,
diarization, dictionary and LLM cleanup are excluded. Joined WAV is already
16 kHz PCM; Moonshine additionally decodes/resamples to 24 kHz. The streaming
probe runs the actual native SDK with 80 ms packets, eaRS's 300 ms outer boundary
VAD, and same forced-final/restart policy; it logs all interim snapshots, so
instrumentation overhead is included. It is not the deployed WebSocket path;
therefore these are not end-to-end `ears -f` timings. Whisper uses its normal
5-beam/5-best settings and the same `-ojf --output-txt` outputs as trnscrb.
Parakeet uses the same `transcribe --json` backend and model as trnscrb.
`trnscrbr` is a separate, simpler wrapper whose default is Whisper small.

A one-hour speed projection from the cased Moonshine row is about **2m56s**,
not a measured hour-long run. RSS is the maximum observed by `/usr/bin/time` in
the fresh process, not total system/GPU allocation or peak across subprocesses;
Metal uses unified memory and process RSS is not a precise GPU footprint. The
models and backend types differ; this is a practical configured-runtime
comparison, **not CPU-vs-CPU or inherent model efficiency**. CPU-only Moonshine
was constrained to one ORT thread, while the other two used GPU plus CPU.

The separate-clip and joined WERs differ: notably Parakeet loses accuracy when
fed this joined fixture whole. That is an observation about this particular
segmentation, not proof about other recordings. The final five normalized words
of the joined transcript matched the reference for Parakeet and both Whisper
models; neither Moonshine output matched. Both Moonshine outputs retained speech
near the end but made word errors there. This is a single tail observation, not a
complete cutoff/cancellation audit.

## Negative checks and boundary control

One three-second zero signal and one seeded low-level white-noise signal, both
without a reference transcript, produced **empty text** for both Moonshine models
and Parakeet. Both Whisper models hallucinated nonempty text on each. These two
negative examples per model are not a comprehensive no-speech evaluation; eaRS
already tracks a separate Moonshine speech-to-silence false-positive issue.

To distinguish model decoder behavior from eaRS's outer segment boundaries,
a separate diagnostic reran the same Moonshine models and corpus **without the
outer BoundaryVad**. The SDK's own voice detector still runs:

| Model | 24 clips WER | Joined WER | Joined wall time | Speed |
|---|---:|---:|---:|---:|
| Base, native detector only | 16.3% | 15.7% | 22.54 s | 14.3× |
| Cased, native detector only | 16.3% | 15.9% | 22.22 s | 14.5× |

These timings are particularly probe-dependent: the trace repeatedly serializes
all earlier native lines (up to 57 vs 2 with outer boundaries). They do **not**
prove native-detector-only is intrinsically slower, nor authorize changing
endpointing. Output quality is still limited on this cohort. Both modes returned
empty on the two simple negatives.

## Reproduce

Private raw data, per-run stdout/stderr, trace snapshots, binary/model hashes and
linked-library paths stay in `~/.local/share/ears-bulk-asr/`. The models and
fine-tune are not in git. The local initial run directory also preserves the exact
probe executable and source used for the main table. Generated data directories
are private; choose a new output directory each time.

Prepare the cached FLEURS de test TSV/audio locally; the script does not fetch
models or data. Build the release test executable with:

```bash
cargo test --release --locked --features moonshine --lib --no-run --message-format=json
# Select the compiler-artifact executable whose target name is `ears`.
```

Then substitute explicit local paths (the private model must be a native Tiny
Streaming bundle):

```bash
uv run scripts/bench_bulk_asr.py \
  --tsv /local/fleurs/de_de/test.tsv --audio-dir /local/fleurs/de_de/test \
  --out /private/new-benchmark --probe /local/compiled-test-executable \
  --moonshine-base /local/base-bundle --moonshine-cased /private/cased-bundle \
  --parakeet-bin /local/parakeet-cli --parakeet-model /local/tdt-v3-q8.gguf \
  --whisper-bin /local/whisper-cli --whisper-turbo /local/large-v3-turbo-q8.bin \
  --whisper-small /local/ggml-small.bin --clips 24 --runs 3
```

Add `--native-vad-only` for the separate control. Initial Moonshine suite uses
one persistent native transcriber; initial Parakeet suite uses its `bench`
model-resident mode; Whisper uses a multi-file process. Those initial suites are
for individual-clip accuracy only and their total times are not compared. The
three reported joined runs use fresh processes for every engine. Tests:

```bash
uv run --with numpy==2.3.4 --with soundfile==0.13.1 --with rapidfuzz==3.14.1 \
  --no-project python -m unittest discover -s scripts -p test_bench_bulk_asr.py
cargo check --locked --features moonshine --lib --bins
cargo test --locked --features moonshine --lib
```
