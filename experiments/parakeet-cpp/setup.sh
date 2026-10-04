#!/usr/bin/env bash
# CPU-only isolated source build. No runtime assets are placed in the repository.
set -euo pipefail
: "${SPIKE_CACHE:=${XDG_CACHE_HOME:-$HOME/.cache}/ears/parakeet-cpp-spike}"
REV=b3eaa487c72a6b9fb7bcb4915b1395e8950b612e
WEIGHTS_REV=741158ae71e64ef5c89385862c18f777d07a97a1
mkdir -p "$SPIKE_CACHE/models"
if [[ ! -d "$SPIKE_CACHE/source/.git" ]]; then
    git clone --filter=blob:none https://github.com/mudler/parakeet.cpp "$SPIKE_CACHE/source"
fi
git -C "$SPIKE_CACHE/source" checkout --detach "$REV"
git -C "$SPIKE_CACHE/source" submodule update --init third_party/ggml
cmake -S "$SPIKE_CACHE/source" -B "$SPIKE_CACHE/build" -DCMAKE_BUILD_TYPE=Release \
    -DPARAKEET_WITH_CED=OFF -DPARAKEET_WITH_VOICEDETECT=OFF -DPARAKEET_BUILD_SERVER=OFF \
    -DPARAKEET_BUILD_TESTS=ON -DPARAKEET_SHARED=ON -DPARAKEET_GGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build "$SPIKE_CACHE/build" -j "${SPIKE_BUILD_JOBS:-8}"
ctest --test-dir "$SPIKE_CACHE/build" --output-on-failure
while read -r model expected; do
    file="$SPIKE_CACHE/models/$model"
    if [[ ! -f "$file" ]]; then
        curl --fail --location --retry 2 "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/$WEIGHTS_REV/$model" -o "$file.part"
        mv "$file.part" "$file"
    fi
    actual=$(shasum -a 256 "$file" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || { printf 'Hash mismatch: %s\n' "$model" >&2; exit 1; }
done <<'MODELS'
redux-packed.gguf 574614b9a4d9f72ab202877a7ad6a1f4bf819dd1d27b9d42dfe8cd429fcebdd5
ultra-q8_0.gguf c2fb452a9df468a141012b01c8c168a25ce93f710897c7de6e353c6cc250986a
redux-vad.gguf 588e1d6e2ee5b6cdfd9ec5ea98dc0993d5bea498d9cc4ec8d6077041eef8a34f
ultra-vad-q8_0.gguf 8b891a4435e97438104ca07c72530d0c5fe62b986baee48b2dd4e1500c1d4758
silero-vad-f32.gguf 1398e5ce230bd8f20c06f825ed3a41dc4528ef7e6fe8f8ec8a5b7aedefbee943
MODELS
printf 'CLI: %s/build/examples/cli/parakeet-cli\n' "$SPIKE_CACHE"
