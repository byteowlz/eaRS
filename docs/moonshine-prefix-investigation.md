# Moonshine early-word commit investigation

2026-10-27, `trx-qge5.1`. **Decision: retain finalized-phrase typing.** Earlier
prefix typing is promising for latency, but none of the five tested policies was
revision-free across the tested models and packet sizes. This work changes no
production insertion behavior.

## What the native API guarantees

Pinned runtime: `moonshine-rs`/`moonshine-sys` 0.2.6, official SDK release v0.1.5,
C API version 30000. Source inspected at
[`234f60f`](https://github.com/moonshine-ai/moonshine/tree/234f60faa0eb388b01cdf7e60aca232af37aefda/core).

- `transcriber.cpp` exposes completion per **line**, not an authoritative stable
  text/token prefix. `is_updated` distinguishes actual inference from cached
  poll results. Its update interval is measured in new audio, not wall time.
- `moonshine-streaming-model.cpp` has stable **encoder frames**. That does not
  mean text tokens decoded from them are immutable.
- Speculative decoder verification accepts a prefix for the **current** acoustic
  memory. Later updates may diverge earlier and rebuild the decoder cache.
- No completed line was subsequently revised in these traces. That is bounded
  evidence consistent with the API contract, not a universal quality claim.

## Experiment

28 fixed inputs: 24 public FLEURS German read-speech recordings and four existing
synthetic German commands, 330.368 seconds total. The same recordings were replayed
through the public German Tiny native bundle and the existing local cased
fine-tune, at 40, 80 and 160 ms input packets. These six conditions reuse the same
28 clips; they are **not** six independent corpora.

`moonshine_revision_trace` is an ignored Rust test that invokes the real native
stream API. It uses the production ingress `BoundaryVad` (300 ms hangover),
forced flush and stream restart behavior. It records every snapshot, line ID,
completion/update flag, consumed-audio position, input hash and model-file hashes.
A one-second silence suffix ensures completion. Line IDs are opaque and differ
between captures; normalize them by first occurrence before comparing replays.
Model execution used CPU, one native ORT thread; replay is faster than real time but preserves the native
audio-based polling schedule. No microphone, keyboard insertion or model upload.

The offline analyzer simulates prefix commits only on independent native updates
with additional audio. It never counts repeated cached polls as agreement. Once
a later snapshot contradicts typed text, that line is frozen; the analyzer does
**not** quietly replace already-typed text with its eventual final.

The oracle is the native line's own final text, **not human ground truth**. These
are revision measurements, not WER or a comparison of model accuracy. Exact case
and punctuation are counted, with a separate alphanumeric/lowercase diagnostic.
The actual keyboard formatting layer is not simulated.

## Results at 80 ms packets

Each arm yielded 68 nonempty lines. The mismatch columns count lines whose
simulated early typed prefix disagreed with their eventual final. Earlier transient
contradictions that later returned to the typed prefix are reported separately in
the local JSON.

| Policy | Public base: final mismatch | Cased fine-tune: final mismatch | Fine-tune median lead, early words only |
|---|---:|---:|---:|
| Emit available text except last word | 41/68 | 33/68 | 1.84 s |
| Prefix agreed across 2 updates, hold last word | 21/68 | 20/68 | 2.00 s |
| Prefix agreed across 3 updates, hold last word | 7/68 | 8/68 | 1.68 s |
| 3 updates, hold 2 words, unchanged for 1 s | 7/68 | 7/68 | 1.76 s |
| 3 updates, hold 2 words, unchanged for 2 s | 1/68 | 1/68 | 1.52 s |

Three-update agreement still had **3 base / 4 fine-tune alphanumeric word-prefix
mismatches**, not just capitalization or punctuation differences. The conservative
two-second policy at 80 ms had only case/punctuation mismatches, but its 1.52 s
median applies only to the 160 fine-tune words typed early. It helped 28/68 lines;
mean first-word lead across all lines was 0.83 s.

Changing packet size exposed genuine word revisions even under the two-second
policy:

| Input packets | Public base: exact / alphanumeric final mismatch lines | Cased fine-tune: exact / alphanumeric final mismatch lines |
|---|---:|---:|
| 40 ms | 3/67 / 1/67 | 2/67 / 1/67 |
| 80 ms | 1/68 / 0/68 | 1/68 / 0/68 |
| 160 ms | 4/64 / 1/64 | 1/64 / 0/64 |

Thus, “three agreements plus two seconds” is **not a safe commit rule**. Packet
size changes the evaluation/segmentation schedule; choosing only the favorable
80 ms result would hide an actual failure. Lead times above are consumed-audio
clock differences relative to phrase completion, not measured OS insertion latency.

## Recommendation

Keep finalized-line insertion as the default. Do not add speculative keystrokes,
a rollback timer or automatic backspaces to arbitrary focused applications.

The next bounded step, if lower insertion latency is required, is a decoder-level
commit contract or explicitly constrained append-only decoding. That must prove
which words cannot change and measure the accuracy cost of locking them. A
revisable draft surface is another product option, but requires an explicit UI
choice; it is not an excuse to overwrite text in the focused app.

## Reproduction

Raw traces and the input manifest remain local under
`~/.local/share/ears-moonshine-prefix/round-1/`; no recordings, fine-tune weights or
raw trace text are checked in. Each trace pins the runtime, model-file hashes and
input hashes. Private model paths must remain available on the machine executing
the probe.

Prepare a local input JSON array, for example
`[{"id":"public-001","path":"/path/to/public.wav"}]`, then:

```bash
MOONSHINE_ORT_SINGLE_THREAD=1 \
EARS_MOONSHINE_TRACE_MODEL=/path/to/native-ort-bundle \
EARS_MOONSHINE_TRACE_INPUTS=/private/inputs.json \
EARS_MOONSHINE_TRACE_OUT=/private/new-trace.jsonl \
EARS_MOONSHINE_TRACE_PACKET_MS=80 \
cargo test --release --locked --features moonshine --lib \
  moonshine_revision_trace -- --ignored --nocapture

uv run scripts/analyze_moonshine_prefix.py \
  --trace /private/new-trace.jsonl --out /private/results.json
uv run --no-project python -m unittest discover -s scripts \
  -p test_analyze_moonshine_prefix.py
```

Both tools refuse to overwrite their output; on Unix, transcript-bearing output
files are private (0600). Ordinary tests do not load models or use recordings. Synthetic regression tests cover late lexical changes,
case/punctuation, cached polls, same-audio duplicate updates, short replies and
nonempty partials retracted by empty finals.
