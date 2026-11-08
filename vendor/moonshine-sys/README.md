# eaRS Moonshine runtime

This `moonshine-sys 0.2.6` Cargo patch builds the repaired Moonshine core from
`core-source.tar.gz`. Ordinary `--features moonshine` builds use it automatically;
there is no published-stock-SDK or sibling-checkout fallback.

- Upstream source: `234f60faa0eb388b01cdf7e60aca232af37aefda` (v0.1.5).
- Final-audio/lookahead repair: `5a1358d3253b81ef24ae2a23cb25fe47bb16b783`.
- Local integration keeps an explicitly pinned ORT path through nested CMake
  includes and preserves relocatable Linux runtime lookup.
- `provenance.json` pins source bytes and critical sources; `ort-artifacts.json`
  pins the required public ONNX Runtime libraries by target, size and SHA-256.
- The source archive retains upstream and third-party licences. It includes the
  SDK's embedded public VAD, but no private data or ASR model bundles.

Missing ORT dependencies download into `$XDG_CACHE_HOME/ears-moonshine/ort/`
(or `~/.cache/ears-moonshine/ort/`). `MOONSHINE_ORT_CACHE` overrides that cache.
Set `MOONSHINE_OFFLINE=1` for strictly cached, network-free native builds.
Corrupt cache entries fail; remove them explicitly rather than accepting repairs
or a different library. Cargo's `--offline` alone does not govern build-script HTTP.

An explicit `MOONSHINE_DIR` must match the pinned source, including integration
changes; an invalid or unpatched override fails, never falls back. Normally leave
it unset. No source tree is edited after archive extraction.

Apple Silicon uses a self-contained static framework. Linux and Intel macOS
need the built shared libraries beside installed executables; the eaRS install
script stages them from `moonshine-sys`'s `out/runtime`. Windows builds stage the
DLLs/import library there as well. Only Apple Silicon native execution has been
qualified locally; other target entries are dependency/build contracts, not a
claim of successful execution on those hosts.

For deliberate updates, use `scripts/vendor_moonshine.py` with the canonical
upstream checkout and a verified WASM build. Compile WASM with
`-ffile-prefix-map=$HOME=/build -fdebug-prefix-map=$HOME=/build` for both C and C++;
the packer rejects binaries embedding the build home. Review source/ORT/runtime hashes,
licences and regression evidence; increment the eaRS runtime revision before
shipping changed bytes. No upstream PR or binary publication is required.
