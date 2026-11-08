# Vendored Moonshine runtime

Ordinary native `--features moonshine` builds and browser runtime staging use
eaRS's pinned `0.1.5-ears.1` repair. No upstream PR, sibling checkout, manual patch
application or stock-runtime fallback is needed. ASR model selection and weights
are unchanged.

The upstream v0.1.5 runtime omitted final partial 1,280-sample blocks and could
leave encoder lookahead unflushed on an empty final call. Our core preserves all
real samples, trims padding-only features before attention, and makes finalizing
without additional samples and repeated finalization safe. It retains the
aligned frontend ABI required by older English exports.

Remaining endpointing differences can be investigated with the separate
`moonshine-segmentation-compare` offline example. That experiment does not enable
a production policy or change weights.

## Native

```bash
just check-moonshine-vendor
just test-moonshine-vendor
```

The vendored `moonshine-rs` wrapper points directly to `vendor/moonshine-sys`,
so Git/path library consumers also get the fix without inheriting a root patch. Its source archive and the public ONNX Runtime dependencies are
checksum-pinned; missing native dependencies download once to the XDG cache.
For strictly cached builds set `MOONSHINE_OFFLINE=1`. Cargo's `--offline` does not
itself control HTTP in build scripts. Invalid explicit `MOONSHINE_DIR` overrides
and corrupt dependencies fail instead of selecting a different runtime.

The Apple Silicon build embeds a static framework. Shared-library platforms must
package the generated `moonshine-sys` `out/runtime` libraries with executables;
the Unix installer handles Linux and Intel macOS. Windows DLLs are staged there
for packaging. Only Apple Silicon native execution is locally qualified.

## Browser

The browser package carries the patched runtime archive, manifest and licences.
`stageMoonshineRuntime(destination)` verifies and stages it without network
access. It never fetches or relabels the original v0.1.5 artifact. Existing callers
still choose `runtimeBaseUrl`; no live site's assets are replaced automatically.

The packaged WASM was built with Emscripten 4.0.8 and verified through real WASM
inference and the eaRS browser `Recognizer`. No models or recordings are included.

## Maintenance and limits

Source/runtime provenance lives in `vendor/moonshine-sys/provenance.json` and
`packages/browser/scripts/vendor/moonshine/manifest.json`. The source bundle has
two build-only CMake integration changes in addition to the audio repair. All
upstream/third-party source licences are retained; the readable audio-fix diff
and its MIT notice remain in `patches/`.

`uv run --no-project python scripts/vendor_moonshine.py --help` documents the
reproducible update tool. It reads a canonical source checkout and verified WASM
build; it does not clone, patch another checkout, upload data or publish artifacts.
Changed shipped bytes require a new eaRS runtime revision and regression checks.

This fixes actual audio loss, not all HF/native quality differences. VAD crop
context, process-history behavior and quantization qualification remain separate.
Completed-phrase typing and model/default promotion gates are unchanged.
