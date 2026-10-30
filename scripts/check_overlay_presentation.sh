#!/usr/bin/env bash
# Synthetic/read-only CLI contract. Never launches a native window or dictation.
set -euo pipefail
binary=${1:-target/debug/ears-ui}
sandbox=$(mktemp -d "${TMPDIR:-/tmp}/ears-ui-presentation.XXXXXX")
trap 'rm -rf "$sandbox"' EXIT
export XDG_CONFIG_HOME="$sandbox/config"
export XDG_DATA_HOME="$sandbox/data"
export XDG_STATE_HOME="$sandbox/state"
"$binary" --overlay-config > "$sandbox/config.json"
"$binary" --overlay-schema > "$sandbox/schema.json"
"$binary" --overlay-font-check > "$sandbox/fonts.json"
bun -e '
const assert = require("node:assert/strict");
const [directory] = process.argv.slice(1);
const read = async name => await Bun.file(`${directory}/${name}.json`).json();
const config = await read("config"), schema = await read("schema"), fonts = await read("fonts");
assert.equal(config.appearance, "classic");
assert.equal(config.transcript, "current");
assert.equal(config.preview, "inline");
assert.equal(config.font, "system");
for (const key of ["appearance", "transcript", "preview", "font"]) assert(schema.properties.overlay.properties[key].enum.includes(config[key]));
assert.equal(fonts.registered, true);
assert.equal(fonts.departure_mono, true);
assert.equal(fonts.jetbrains_mono, true);
' "$sandbox"
set +e
"$binary" --preview-curser > "$sandbox/unknown.out" 2> "$sandbox/unknown.err"
status=$?
set -e
[[ "$status" == 2 ]]
[[ ! -e "$XDG_CONFIG_HOME" && ! -e "$XDG_DATA_HOME" && ! -e "$XDG_STATE_HOME" ]]
printf '%s\n' 'PASS: legacy defaults, schema, native font registration, fail-closed CLI; no XDG state written'
