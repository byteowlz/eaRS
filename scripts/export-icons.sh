#!/usr/bin/env bash
# Render committed app icons from the approved eaRS logo SVGs (never edits the artwork).
# Requires librsvg (rsvg-convert), ImageMagick (magick) and macOS iconutil.
# Mirrors byteowlz/xlatch scripts/export-apple-icons.sh; see logo/icons/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in rsvg-convert magick iconutil; do
    command -v "$tool" >/dev/null || { echo "Missing dependency: $tool" >&2; exit 1; }
done
out=logo/icons
mkdir -p "$out/macos" "$out/ios" "$out/linux" "$out/windows"
# No timestamps in PNGs, so reruns are byte-identical.
stamp=(-define png:exclude-chunks=date,time)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# macOS: fit the rounded tile proportionally into a centered 896 px area of a
# transparent 1024 px canvas (64 px optical margin, as in xlatch).
rsvg-convert --keep-aspect-ratio -w 896 -h 896 logo/ears_logo_mac_white_on_black.svg -o "$work/mac.png"
master="$out/macos/ears_1024.png"
magick "$work/mac.png" -background none -gravity center -extent 1024x1024 -colorspace sRGB "${stamp[@]}" "PNG32:$master"
iconset="$work/AppIcon.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
    magick "$master" -resize "${size}x${size}" "${stamp[@]}" "PNG32:$iconset/icon_${size}x${size}.png"
    magick "$master" -resize "$((size * 2))x$((size * 2))" "${stamp[@]}" "PNG32:$iconset/icon_${size}x${size}@2x.png"
done
iconutil -c icns "$iconset" -o "$out/macos/AppIcon.icns"

# iOS / App Store marketing master: opaque full-bleed square, no alpha (the OS masks it).
rsvg-convert -w 1024 -h 1024 logo/ears_logo_white_on_black.svg -o "$work/ios.png"
magick "$work/ios.png" -background black -gravity center -extent 1024x1024 -alpha remove -alpha off \
    -colorspace sRGB "${stamp[@]}" "PNG24:$out/ios/ears_1024.png"

# Linux (hicolor theme sizes) and Windows (.ico) reuse the padded rounded tile.
for size in 16 24 32 48 64 128 256 512; do
    magick "$master" -resize "${size}x${size}" "${stamp[@]}" "PNG32:$out/linux/ears_${size}.png"
done
magick "$master" -define icon:auto-resize=256,64,48,32,24,16 "$out/windows/ears.ico"
echo "Exported macOS ICNS/master, iOS master, Linux PNGs and Windows ICO into $out."
