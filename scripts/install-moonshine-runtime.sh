#!/usr/bin/env bash
# Install only libraries built from the vendored repaired core. No stock fallback.
set -euo pipefail

case "$(uname -s)" in
    Linux) names=(libmoonshine.so libonnxruntime.so.1) ;;
    Darwin)
        # Apple Silicon uses the self-contained static Moonshine framework.
        [[ "$(uname -m)" != "arm64" ]] || exit 0
        names=(libmoonshine.dylib libonnxruntime.1.23.2.dylib)
        ;;
    *) exit 0 ;;
esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin_dir="${1:-${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin}"
target_dir="${2:-${CARGO_TARGET_DIR:-$root/target}}"
lib_dir=""
shopt -s nullglob
for candidate in "$target_dir"/release/build/moonshine-sys-*/out/runtime; do
    complete=true
    for name in "${names[@]}"; do
        [[ -f "$candidate/$name" ]] || complete=false
    done
    if [[ "$complete" == true && ( -z "$lib_dir" || "$candidate" -nt "$lib_dir" ) ]]; then
        lib_dir="$candidate"
    fi
done
if [[ -z "$lib_dir" ]]; then
    printf 'Vendored Moonshine libraries not found under %s; build the moonshine feature first.\n' "$target_dir/release/build" >&2
    exit 1
fi
mkdir -p "$bin_dir"
for name in "${names[@]}"; do
    tmp="$bin_dir/.$name.tmp.$$"
    install -m 755 "$lib_dir/$name" "$tmp"
    mv -f "$tmp" "$bin_dir/$name"
done
printf 'Installed vendored Moonshine runtime libraries in %s\n' "$bin_dir"
