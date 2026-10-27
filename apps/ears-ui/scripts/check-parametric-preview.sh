#!/usr/bin/env bash
# Synthetic window-only macOS regression: no engine, microphone, tray or config writes.
set -euo pipefail
if [[ "$(uname -s)" != Darwin ]]; then echo 'requires macOS and Screen Recording permission' >&2; exit 2; fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/ears-parametric-check.XXXXXX")
pid=''
cleanup() {
  if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  rm -rf -- "$work"
}
trap cleanup EXIT
cd "$repo"
cargo build -p ears-ui --locked --offline >/dev/null
capture() {
  local flag=$1 name=$2 window
  ./target/debug/ears-ui "$flag" >"$work/$name.log" 2>&1 & pid=$!
  sleep 2
  window=$(swift apps/ears-ui/scripts/preview-window.swift "$pid" | head -1)
  [[ -n "$window" ]] || { echo "synthetic window not found: $name" >&2; return 1; }
  screencapture -x -l "$window" "$work/$name-a.png"
  sleep 0.4
  screencapture -x -l "$window" "$work/$name-b.png"
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; pid=''
}
capture --preview-parametric animated
capture --preview-parametric-reduced reduced
capture --preview-parametric-override override
capture --preview-floating floating
export EARS_OQTOPUS_PACK="$repo/apps/ears-ui/tests/fixtures/oqtopus-synthetic.oqp"
capture --preview-oqtopus oqtopus
capture --preview-oqtopus-reduced oqtopus-reduced
capture --preview-settings settings
swift apps/ears-ui/scripts/check-settings-text.swift "$work/settings-a.png"
uv run --no-project --offline --with pillow python -I - "$work" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageChops
p = Path(sys.argv[1])
def icon(name):
    return Image.open(p / name).convert('RGB').crop((45, 0, 90, 35))
a, b = icon('animated-a.png'), icon('animated-b.png')
r, s = icon('reduced-a.png'), icon('reduced-b.png')
u, v = icon('override-a.png'), icon('override-b.png')
assert a.getbbox() and r.getbbox(), 'synthetic indicator not visible'
assert ImageChops.difference(a, b).getbbox(), 'animated indicator did not change'
assert not ImageChops.difference(r, s).getbbox(), 'reduced-motion indicator moved'
assert ImageChops.difference(u, v).getbbox(), 'explicit reduced-motion override did not animate'
oa, ob = icon('oqtopus-a.png'), icon('oqtopus-b.png')
ra, rb = icon('oqtopus-reduced-a.png'), icon('oqtopus-reduced-b.png')
assert ImageChops.difference(oa, ob).getbbox(), 'synthetic vector Oqtopus did not animate'
assert not ImageChops.difference(ra, rb).getbbox(), 'reduced-motion Oqtopus moved'
assert oa.getpixel((20, 15))[2] > 100, 'synthetic vector artwork not visible'
floating = Image.open(p / 'floating-a.png').convert('RGBA')
assert all(floating.getpixel((x, 0))[3] == 0 for x in (20, 320, 619)), 'floating window chrome is visible'
image = Image.open(p / 'animated-a.png').convert('RGBA')
for x in (35, 604):
    assert image.getpixel((x, 0))[3] > 100 and image.getpixel((x, 8))[3] < 50, 'outward throat missing'
print('native synthetic windows: vector animation, reduced motion, throats and borderless floating panel verified')
PY
