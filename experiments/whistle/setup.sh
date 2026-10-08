#!/usr/bin/env bash
# Explicit artifact setup; inference later runs with OS network denial.
set -euo pipefail
: "${WHISTLE_CACHE:=${XDG_CACHE_HOME:-$HOME/.cache}/ears/whistle-spike}"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Measured spike supports macOS arm64 only' >&2; exit 1; }
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$WHISTLE_CACHE/native"
fetch() {
    local repo=$1 revision=$2 remote=$3 path=$4 expected=$5
    if [[ ! -f "$path" ]]; then
        curl --fail --location --retry 2 "https://huggingface.co/$repo/resolve/$revision/$remote" -o "$path.part"
        mv "$path.part" "$path"
    fi
    [[ "$(shasum -a 256 "$path" | awk '{print $1}')" == "$expected" ]] || { echo "Hash mismatch: $remote" >&2; exit 1; }
}
MODEL_REV=b358ddadd89b7a713b5aa131f23032d3cca1b251
ENGINE_REV=2ae11323dc000f5e70c49f7403efa6af12ba9e67
fetch Cactus-Compute/whistle "$MODEL_REV" whistle.cact "$WHISTLE_CACHE/whistle.cact" b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb
fetch Cactus-Compute/whistle "$MODEL_REV" LICENSE "$WHISTLE_CACHE/LICENSE" cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
fetch Cactus-Compute/needle3 "$ENGINE_REV" macos-arm64/needle "$WHISTLE_CACHE/native/needle" 342fa2c6f140e702354a99c4201c9057535ec908eed35c7382e911a19d1d2724
fetch Cactus-Compute/needle3 "$ENGINE_REV" macos-arm64/libneedle.a "$WHISTLE_CACHE/native/libneedle.a" 98da47c15e1065b4cdc7ddc55e825be78414d4586832db3becaeded39a373df4
fetch Cactus-Compute/needle3 "$ENGINE_REV" macos-arm64/needle.h "$WHISTLE_CACHE/native/needle.h" 914bbd00423939b02fc349f91cdb7c12531b3f5bb7105b743e0e93c7e409d7b1
fetch Cactus-Compute/needle3 "$ENGINE_REV" LICENSE "$WHISTLE_CACHE/native/LICENSE" cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
chmod u+x "$WHISTLE_CACHE/native/needle"
clang++ -std=c++17 -O3 -Wall -Wextra -Werror -I "$WHISTLE_CACHE/native" \
    "$HERE/native_bench.cpp" "$WHISTLE_CACHE/native/libneedle.a" -o "$WHISTLE_CACHE/native/native-bench"
printf 'Native caller: %s/native/native-bench\n' "$WHISTLE_CACHE"
