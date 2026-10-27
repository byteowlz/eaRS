#!/usr/bin/env bash
# moonshine-sys links Linux shared libraries, but its rpath does not propagate
# to our executables. Install both SDK libraries beside the binaries, where
# eaRS\x27s existing $ORIGIN rpath finds them. macOS SDK linking is static.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    exit 0
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin_dir="${1:-${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin}"
target_dir="${2:-${CARGO_TARGET_DIR:-$root/target}}"
lib_dir=""
shopt -s nullglob
for candidate in "$target_dir"/release/build/moonshine-sys-*/out/moonshine-prebuilt/moonshine-voice-linux-*/lib; do
    if [[ -f "$candidate/libmoonshine.so" && -f "$candidate/libonnxruntime.so.1" ]]; then
        if [[ -z "$lib_dir" || "$candidate" -nt "$lib_dir" ]]; then
            lib_dir="$candidate"
        fi
    fi
done
if [[ -z "$lib_dir" ]]; then
    printf "Moonshine SDK libraries not found under %s; build the moonshine feature first.\n" "$target_dir/release/build" >&2
    exit 1
fi
mkdir -p "$bin_dir"
for name in libmoonshine.so libonnxruntime.so.1; do
    tmp="$bin_dir/.$name.tmp.$$"
    install -m 755 "$lib_dir/$name" "$tmp"
    mv -f "$tmp" "$bin_dir/$name"
done
printf "Installed Moonshine runtime libraries in %s\n" "$bin_dir"
