# Native device proof

Standalone, unpublished Rust workspace implementing the device-scene v1 mechanism.
The normative interface is [the scene spec](../../spec/device-scene-v1.md); concrete
materials are a Look, not a replacement for Scheme/Role. This is **partial native
proof infrastructure**, not a selected design, production settings replacement,
or a visual-parity claim.

## Modules

- `device-scene`: closed serde model, semantic validation, fixed viewport budget,
  camera projection/inverse and ray intersections. No I/O or graphics dependencies.
- `device-render-wgpu`: actual 3D chassis, vents, feet, perforated plates, bevelled
  keys, knobs/faders and caller-valued meters. Procedural brushed/spun materials,
  analytical lighting, depth-tested rendering and bounded offscreen readback.
- `device-gpui`: GPUI 0.2.2 image lifecycle, one-active/one-latest-pending worker,
  and reusable `DeviceView`. The view emits typed Actions; application hosts own
  selection previews, shortcut recording, persistence and every external effect.
  The native fixture example is only a synthetic host of that same view.
  Transparent viewports leave the host canvas clear. `is_chassis_at` exposes
  current non-control geometry for host-owned dragging; controls (including
  disabled ones), empty canvas and stale frames never become drag handles.

The libraries never open fixture/font files, microphones, networks or app config.
Only explicitly invoked proof executables read their supplied paths. No consumer
is wired to this workspace; future consumers need pinned vendoring or publication,
not a shipped sibling-checkout dependency.

## Run

From this directory, with Rust and `just`:

```sh
just check
just test
just lint
just gpu-test
just build-proof
mkdir -p proofs
just render ../../spec/fixtures/device-scene-v1/ears-input.json proofs/ears-input.png
just proof ../../spec/fixtures/device-scene-v1/ears-shortcuts.json
```

`device-render <fixture.json> <output.png>` writes an explicit geometry-only PNG
and adjacent projection/measurement JSON. Labels remain in the returned Frame for
host typography; the PNG is not a complete settings screenshot.

`specimen <fixture.json> [printed-font-file] [ui-font-file]` opens a native window.
The host may explicitly supply Michroma and Inter font bytes. Missing Michroma is
reported and uses native sans fallback. Font paths are never scene fields.
Tab/Shift-Tab focus controls; Enter/Space and pointer activation emit JSON Actions.
The example host routes Dictation to Input, preserves Input/Shortcuts as subpages
and cycles synthetic select options. Other destinations remain visible and report
that this bounded fixture has no corresponding page. Wake phrases and their
selection/action bindings remain app-owned Settings coverage outside this first
fixture; no wake action execution or policy change is authorized. It does not open
live eaRS.

## Interface and invariants

`Snapshot::parse(&str)` returns `ValidatedSnapshot`; typed callers can use
`Snapshot::validate()` or `ValidatedSnapshot::new(snapshot)`.
Errors identify a path and invariant. IDs/references, dimensions, pages, value
types/ranges/options, active focus and grapheme-safe UTF-16 selections are checked.
Range step is an edit increment, not state quantization.

`Projection::new(&snapshot)` supplies identical view-projection/inverse matrices
to the GPU and hit testing. Z is the front surface; depth extends backwards.
Near/far derive from scene bounds. Behind-camera/intersecting bounds and
nonrepresentable projections fail, rather than silently fitting a different camera.
Viewport extent is uniformly reduced to its pixel cap; per-axis effective scales
record integer rounding.

`OffscreenRenderer::render(&snapshot, generation)` runs on a worker and returns
one `RenderFrame`: revisions, viewport, pixels, projection, exact hit triangles
and world-space label anchors. Hit testing includes noninteractive occluders.
Host code must also reject disabled, stale or nonactionable controls.

`RenderWorker::start()` returns a worker and an async notification receiver.
`submit` validates monotonic revisions; same-revision content changes fail, while
viewport-only changes get a new request generation. Superseded results are dropped.
`take_completed` returns pixels and their matching snapshot together.
`ImageSlot::expect/install` installs only the expected generation and explicitly
retires the old GPUI image. Errors retain the old complete image/projection pair.
Call `clear` on view release. Do not stretch an old image into a new viewport and
reuse its old hit regions; the specimen disables pointer input during a mismatch.

No idle render timer exists. The worker sleeps on a condition variable and the UI
awaits notifications. Resource limits are 4,194,304 output pixels, 600,000 vertices,
8,192 aperture fragments per panel, one active/latest pending/completed request,
one persistent color/MSAA/depth/staging target set. A resize releases the old set
before allocating its replacement. Readback waits at most ten seconds on the worker.
At the pixel cap, 4x color/depth, resolve and staging use roughly 160 MiB, plus mesh,
CPU frame and GPUI sprite copies; this is bounded, not a low-memory mobile claim.

Readback removes 256-byte row padding and explicitly converts RGBA to **BGRA8,
top-left, premultiplied in encoded sRGB**. MSAA coverage is unpremultiplied in linear
space then premultiplied after sRGB encoding. Fully transparent RGB is zero.
PNG export reverses the final encoded premultiplication because PNG uses straight
RGBA. `image::Frame`'s RGBA container does not change GPUI's BGRA byte contract.

## Evidence and limits

Verified locally on Apple M2 Ultra, wgpu 30.0.1/Metal, GPUI 0.2.2:

- Workspace check, tests, native example build and Clippy.
- Independent scene validation suite: 19 tests.
- Worker burst/idle/revision tests, padded-channel and alpha conversion tests.
- Explicit actual-GPU test: all three fixtures, 1x/2x, front/angled, nonaligned
  readback rows, transparent edges, red-channel sentinel, hidden-page filtering
  and occluder-aware hit testing.
- First cold Input render/readback at 2560x1440: approximately 259 ms, 1,130
  triangles. This includes first-use GPU costs; it is not an interaction benchmark.

Not yet established:

- GPUI composite screenshot/color-edge verification and image-retirement soak.
  Compilation is not visual proof. Linux/Windows were not built or exercised.
- Full receiver-reference fidelity: no shadow map/environment map, tactile
  reflection matching or motion curve. Feet are bevelled boxes in this proof.
- Angled typography: labels are sharp native text at projected centres/extents,
  but GPUI does not perspective-warp them. A host glyph-atlas pass is required.
  Labels are not yet depth-clipped glyph by glyph. Brand Geo is not exercised.
- Matrix text now uses shared case-preserving 5x7 phosphor geometry on the 3D
  display plane, not native-font overlays. Pitch/cells/padding come only from the
  scene; values never resize the dots. Long values word-wrap, then show a matrix
  ellipsis when capacity is exhausted; hover reveals the unchanged full value.
  Unsupported Unicode uses whole-string native shaping with a disclosed tooltip,
  never replacement glyphs, uppercasing or transliteration.
  `render-wgpu/src/matrix-glyphs.json` preserves the original r10 glyphs from
  Studio revision `a710fbc`, extending them with ASCII/lowercase and common German
  letters. Explicit GPU tests cover lit/unlit contrast, perspective and hit mapping
  at 1x/2x. Native print/Unicode fallback still has the angled-text limitations above.
- Native text/shortcut editing, IME, caret/selection projection and accessibility
  trees. Shortcut activation emits Preview with the current value to request
  app-owned capture; it never pretends that a chord has been recorded. Continuous knob/
  fader input, interaction cancellation protocol and app conflict policy are not
  implemented by the specimen; geometry can render caller-supplied values.
- Whole-app settings coverage. No existing settings/navigation/configuration has
  been replaced or removed.

The shader and mesh are live native geometry, not a screenshot or WebView. The
remaining work must preserve this shared scene/renderer/adapter seam rather than
flatten the device into a form or invent a separate app-local renderer.
