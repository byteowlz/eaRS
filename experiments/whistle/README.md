# Whistle native spike

Task `trx-7nw3.8`. [Measured report](../../docs/whistle-spike.md),
[aggregate evidence](evidence.json). No production engine/default changes.
Stacked on `spike/parakeet-cpp` to reuse its corpus/scoring and uv dependencies.

macOS arm64 only: C++17 compiler, uv, curl, shasum, `/usr/bin/time`, sandbox-exec.
Setup downloads pinned publisher binaries/header/static archive and 16.9MB
weights outside git, verifies every SHA-256 and compiles the small C API caller.
The native engine implementation was not found in the inspected source checkout;
this does not claim an independently reproducible native engine source build.

```bash
export WHISTLE_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/ears/whistle-spike"
just whistle-spike-setup
just parakeet-spike-fixtures "$WHISTLE_CACHE/fixtures" 5
just whistle-spike --native "$WHISTLE_CACHE/native/native-bench" \
  --model "$WHISTLE_CACHE/whistle.cact" --manifest "$WHISTLE_CACHE/fixtures/manifest.json" \
  --out "$WHISTLE_CACHE/results/auto" --runs 3
just whistle-spike --native "$WHISTLE_CACHE/native/native-bench" \
  --model "$WHISTLE_CACHE/whistle.cact" --manifest "$WHISTLE_CACHE/fixtures/manifest.json" \
  --out "$WHISTLE_CACHE/results/forced" --force-language --runs 3
WHISTLE_NATIVE="$WHISTLE_CACHE/native/native-bench" \
WHISTLE_MODEL="$WHISTLE_CACHE/whistle.cact" just whistle-spike-test
```

All measured native batches run with macOS direct-network denial, and
`NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1`. The runtime must be supplied explicitly;
inference does not download or start the publisher Python package/tool agent.
The helper loads only a speech model, serializes requests, retains model bytes,
bounds finite PCM/output and emits checked JSON rows. Its native timings exclude
PCM file reading, discard one warm-up and include native JSON generation.

Zero PCM and stationary noise pass. A no-speech Hann-shaped noise burst currently
hallucinates in offline and reprocessed-stream modes; this gate **fails** until
`trx-7nw3.9` is resolved:

```bash
WHISTLE_PULSE_GATE=1 WHISTLE_NATIVE="$WHISTLE_CACHE/native/native-bench" \
WHISTLE_MODEL="$WHISTLE_CACHE/whistle.cact" just whistle-spike-test
```

`trx-7nw3.10` records insufficient native output capacity returning a positive
token count with truncated JSON (canaries intact). Always validate the payload;
positive return is not proof of a complete transcript. C API is process-global,
non-thread-safe; no public unload or thread setter. Do not infer causal streaming,
word alignment quality, calibrated confidence or action authorization.

Publisher source package, model and engine license files declare Apache-2.0.
Retain applicable notices on distribution; selected-runtime dependencies and
exact prebuilt behavior still require review. Audio/models/binaries/raw logs stay
outside git. Python harness shares the prior spike's uv environment deliberately.
