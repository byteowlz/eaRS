# parakeet.cpp spike

Opt-in, CPU-only evaluation, tracked by `trx-7nw3.6`. No eaRS engine/catalog/UI
changes. Results and adoption limits: [spike report](../../docs/parakeet-cpp-spike.md).

Requirements: git, CMake, C++17 compiler, curl, shasum, `/usr/bin/time`, uv,
Rust/Cargo and network access for setup. Runner supports macOS/Linux; measured
here only on Apple Silicon. About 1.2 GB of model downloads, plus build/cache.

```bash
# All runtime/model/data outputs outside git. setup creates this directory.
export SPIKE_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/ears/parakeet-cpp-spike"
just parakeet-spike-setup
just parakeet-spike-test
just parakeet-spike-fixtures "$SPIKE_CACHE/fixtures" 5
CLI="$SPIKE_CACHE/build/examples/cli/parakeet-cli"

for model in redux-packed ultra-q8_0; do
  just parakeet-spike asr --cli "$CLI" --model "$SPIKE_CACHE/models/$model.gguf" \
    --manifest "$SPIKE_CACHE/fixtures/manifest.json" \
    --out "$SPIKE_CACHE/results/$model" --threads 4 --runs 3
done
for model in redux-vad ultra-vad-q8_0 silero-vad-f32; do
  just parakeet-spike vad --cli "$CLI" --model "$SPIKE_CACHE/models/$model.gguf" \
    --manifest "$SPIKE_CACHE/fixtures/manifest.json" \
    --out "$SPIKE_CACHE/results/$model" --threads 4
  # Linux: use libparakeet.so instead of libparakeet.dylib.
  uv run --project experiments/parakeet-cpp experiments/parakeet-cpp/capi_probe.py \
    --library "$SPIKE_CACHE/build/libparakeet.dylib" \
    --model "$SPIKE_CACHE/models/$model.gguf" \
    --audio "$SPIKE_CACHE/fixtures/silence.wav"
done
just parakeet-spike-loader "$SPIKE_CACHE/models/redux-packed.gguf" \
  "$SPIKE_CACHE/models/ultra-q8_0.gguf"
```

`setup.sh` pins source/submodule/model revisions and verifies every model hash.
It disables Metal/CUDA/server/speaker/sound-event components; it does not fetch
all optional upstream submodules. FLEURS generator streams the original archives
instead of the preview API, selects first qualifying archive entries (3–30s),
and creates seeded 5dB white-noise pairs, silence and noise-only controls.

`bench.py` reuses eaRS text normalization/scoring. ASR warm timing comes from
upstream's one-model CLI benchmark, not the parent process wall clock. Owned
subprocess groups are killed/reaped on timeout. Raw diagnostics/hypotheses stay
in the explicit output directory. `--extra DIR` adds optional VAD WAV controls.
`capi_probe.py` exercises native string/context/stream ownership, rejection and
recovery; it is not a shipping Python ASR adapter.

The no-speech gate intentionally **fails** for both heads at the pinned revision:

```bash
SPIKE_CLI="$CLI" SPIKE_VAD_MODEL="$SPIKE_CACHE/models/ultra-vad-q8_0.gguf" \
SPIKE_SILENCE="$SPIKE_CACHE/fixtures/silence.wav" \
  uv run --project experiments/parakeet-cpp python -m unittest discover \
  -s experiments/parakeet-cpp -p test_spike.py
```

Use `silero-vad-f32.gguf` for the passing control. Keep the failing gate until
`trx-7nw3.7` has evidence for a real correction; do not silently redefine silence
as speech. Code MIT; Ultra/Redux/slices and FLEURS CC BY 4.0 (credit
Moondream/NVIDIA and Google FLEURS respectively); Silero code/model MIT.
Do not commit downloaded models, audio, raw private recordings or local logs.
