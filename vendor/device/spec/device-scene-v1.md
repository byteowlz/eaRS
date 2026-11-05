# Device scene v1 (draft implementation contract)

Schema: `device-scene-v1.schema.json` (Draft 7, root is Snapshot; `$defs.action` is Action).
First native fixtures: `fixtures/device-scene-v1/ears-input.json` and `ears-shortcuts.json`.
Browser reference: Design Studio `artifacts/spqx/voice-studio-r6/` (work in progress).
No visual selection or native-parity claim is implied by this interface agreement.

## Ownership and home

`impls/device-rs/` is a standalone Rust workspace: `scene`, `render-wgpu`, `gpui`.
Scene validation/projection is independent of GPUI and wgpu; renderer owns geometry/materials/frame creation;
GPUI adapter owns native input, focus, accessibility, IME and image lifetime. Apps own config, action dispatch,
validation and persistence. No renderer filesystem, microphone, network or application lifecycle effects.
Initial native versions: GPUI 0.2.2, wgpu 30.0.1, subject to compiling the actual proof. Do not ship sibling
`../` path dependencies in consumers. Native fixture proof uses the canonical workspace; distribution must
use a pinned revision/vendor or package path that works in a clean consumer checkout.

## Vocabulary

A **Scene** is immutable descriptor data: device chassis, panels, controls, material values, camera and light.
A **Snapshot** pairs a Scene, app-owned State and Viewport for one render request. A **Control** has a stable
ID, geometry, semantics and an action name, not a closure or a file path. A **Panel** groups controls; it is not
a second coordinate system. A **Frame** pairs pixels with the exact projection and revisions that produced them.
An **Interaction** is a preview sequence ending in one commit or cancel. Materials here are concrete Look
values, not a replacement for the repository's Scheme/Role mechanism.

## Coordinates and projection

- Right-handed device units: +X right, +Y up, +Z toward the viewer. Chassis front is Z=0; body extends toward
  negative Z. Rect X/Y are centres; Z is the front surface, so its volume is **[z-depth, z]**, including
  faceplates. Width/height/depth are dimensions. A faceplate at z=0 of depth .65 occupies [-.65, 0].
- Panel and control rects are in their device's coordinates, **not** relative to their containing panel.
  Device origin adds a translation; v1 has no arbitrary per-device rotation or custom shader code.
- Chassis dimensions include cover but not feet. Faceplate is a panel, not an implicit duplicate mesh.
  The native renderer creates holes/recesses for displays and keys, not a faceplate covering the controls.
- Camera angles orbit its target: direction is `(sin(yaw)*cos(pitch), sin(pitch), cos(yaw)*cos(pitch))`;
  position = target + direction * distance. Y up, perspective FOV vertical, degrees. No auto-fit that silently
  overrides the supplied distance. Hosts may calculate a new camera, then submit a new Snapshot.
- Renderer selects finite near/far planes from validated scene bounds (including feet and proud controls),
  with a small positive near floor. Reject unrepresentable/degenerate camera poses rather than clipping the
  device silently. Publish chosen near/far in Frame projection metadata; all input uses that same matrix.
- Frame carries one view-projection matrix and its inverse, viewport and revisions. Rendered pixels, native
  hit regions, selection/caret bounds and IME anchor all use this transform. Pointer hit-testing casts through
  the same camera and uses frontmost *visible* geometry; never click through a hidden key or an occluding panel.
- Viewport uses DIP dimensions and device scale. Physical extent = ceil(DIP * scale), clamped with uniform
  downscale to pixel_cap (preserve aspect). Projection metadata records the effective scale, not just requested
  DPR. Resize/DPR change invalidates the prior projection. Zero/invalid extents are rejected, not submitted.

## State, pages, validation

- Scene revision changes when descriptors/geometry change; state revision changes when values/interaction
  state changes. Revisions are monotonically increasing safe integers within a scene lifetime, not timestamps.
  Replace the scene ID or host generation when starting a new lifetime. An identical revision pair must not
  describe different data. Viewport changes still require a distinct render-request generation.
- IDs are globally unique across devices, panels and controls within a Scene. Material IDs form a separate map.
  Every module/material reference resolves. Every control has exactly one state entry; unknown state IDs fail.
- Validate finite positive dimensions, non-overflowing extents, numeric ranges with min < max and step > 0,
  control values matching kind/range/option IDs, one focused enabled visible control at most, and selection
  bounds `0 <= start <= end <= UTF16_length(value)`. Do not split grapheme clusters visually.
- Controls lie inside their parent panel in X/Y; their Z can stand proud (keys/knobs). No world-coordinate
  duplication in state. Page overlap is intentional, not permission for simultaneously overlapping hit regions.
- Page selectors: `*` always, exact name, or a terminal `.*` prefix matching descendants. A panel and its control
  must both match `state.active_page`, and state.visible must be true. `dictation.*` matches descendants only,
  not the exact `dictation` string. Navigation destination `dictation` is resolved by the app to its remembered
  subpage or default `dictation.input`; the fixture runner must do that routing rather than display a blank
  exact-root page. No regex/glob engine otherwise.
- Nonactive pages retain app draft values but receive no input or keyboard focus. tab_index >= 0 orders semantic
  controls on the active page; -1 is nonfocusable output. Disabled controls stay visible but emit no Actions.
- `value`: button can use null (momentary) or a destination string (navigation); toggle boolean; knob/fader/meter
  finite number within descriptor range; select option ID; text/shortcut/readout string; indicator boolean.
  Range step is an **editing increment**, not a requirement that authoritative values lie on a step lattice.
  Preserve in-range off-grid state and continuous meter samples. Do not quantize existing state on load.
  `error` is user-facing app feedback. No fake meter values: absent capture is labelled inactive, not live.
- Generic schema validates shape; the scene crate must enforce these cross-reference and semantic invariants.
  Schema loading alone does not establish a safe/usable scene.

## Input and actions

- Action includes scene ID + scene revision + base state revision + stable control ID + interaction ID +
  monotonic sequence + phase + value + source. `action` in descriptor is an app binding name, never executable.
- `preview` changes ephemeral presentation only; one `commit` asks the app to accept the proposed value;
  `cancel` discards that interaction's preview, with value=null. Button/keyboard/API activation may emit just
  one commit. Cancel and commit are terminal, duplicates are rejected or idempotently ignored by the host.
- `state_revision` is the base captured at interaction start, unchanged across its previews. App replies with
  authoritative state revisions; it decides conflict/rebase policy when other sources edited the same control.
  The renderer does not silently overwrite newer app state. Scene changes invalidate an active interaction.
- On pointer cancellation, lost capture, blur/removal or disabled state, cancel continuous interactions and
  release pressed visuals. Keyboard equivalents and explicit reset/cancel operations must exist.
- Native text/shortcut controls own editing and IME. Matrix is a presentation, never a second text buffer;
  native selection uses UTF-16 offsets for cross-renderer interchange. Preserve full Unicode; unsupported
  matrix glyphs use a disclosed normal-glyph fallback, not replacement or uppercasing of user data.
- Shortcut capture is typed app input, not key-name concatenation in the renderer. Escape cancels capture;
  global shortcuts must not intercept composing/editing input. Clear only optional shortcuts; the first
  specimen deliberately offers reset/cancel but no unsupported clear operation.

## Host typography registry

Fonts are host-provided resources, never remote URLs or filesystem paths in a Scene. For the receiver Look,
map printed labels to bundled **Michroma Regular**, brand wordmark `byteowlz` to **Geo Regular**, compact digits
and semantic native controls to the host's bundled legible UI face; Unicode native text fallback uses the
platform shaping stack. The browser reference bundles Michroma/Geo/Inter/JetBrains with licence notices.
Matrix glyphs are a shared glyph table, not browser text converted to uppercase. Missing host fonts/glyphs
must be reported as a fidelity limitation. This role mapping is normative for the proof; a future explicit
font-role field must be versioned instead of smuggling font paths into label strings. Front-facing print
needs device-pixel alignment/density testing; a raw texture size is not a sharpness guarantee.

## RenderFrame and scheduling (native adapter agreement)

A successful result is `{request_generation, scene_id, scene_revision, state_revision, viewport, projection,
width_px, height_px, stride_bytes, pixels}`. Pixels are **BGRA8, top-left origin, premultiplied alpha in encoded
sRGB** for the GPUI RenderImage seam; stride_bytes = width_px * 4. Swap RGBA channels and remove wgpu padded
rows explicitly. GPU target choice and intermediate alpha convention stay renderer-private. Verify the actual
GPUI composition with red/blue/transparent-edge fixtures before calling this proven; image::Frame's RGBA
wrapper is not a reason to skip the agreed BGRA conversion. Fully transparent RGB is zero. For transparent
viewports the chassis surroundings are alpha zero; no page-colour rectangle is baked into the frame.

Allow one active render + one replaceable pending snapshot. Superseded results are discarded by generation,
including same scene/state revisions at different viewports. Pixels and projection are delivered atomically.
UI-thread code never blocks on device polling/readback. Explicitly retire prior RenderImages using supported
GPUI lifecycle mechanisms and bound retained GPU buffers/textures. Idle means **zero render requests**, not a
continuous loop drawing unchanged frames. Animation and audio meters request bounded updates only while live;
reduced motion removes key-travel/cursor blinking where applicable. A failed frame retains previous complete
state with a truthful unavailable/error indicator, not new hit regions over old pixels.

## First eaRS specimen: full device, bounded functional coverage

Keep the incumbent four destinations **Dictation / Overlay / Dictionaries / Profiles**, with stable navigation
IDs. Input, Shortcuts and **Wake phrases** are Dictation subpages. Wake phrases must retain selection and
binding-to-action capability in the application's Settings coverage. Existing CLI operations are
`ears wake catalog/install/list/bind/unbind/remove`, with built-in `start_dictation`, `stop_dictation` and
`ignore` bindings and restart-to-apply behavior. Future external bindings use stable provider/action IDs,
not arbitrary shell macros, and require the app's catalog/prepare/confirmation seam. No policy change or
automatic external execution is authorized by this scene contract. Wake provider/model semantics remain
outside the shared renderer, and Wake phrases is not added to the bounded first renderer fixture. Existing settings, drafts and scroll state remain reachable in
real eaRS; this fixture is not permission to delete unshown fields or collapse them into an invented nav model.

`ears-input.json`: entire 64 x 16 x 24 chassis, vents/feet, header/navigation, Dictation subnavigation,
microphone selection (synthetic options), inactive scalar meter and truthful status. **No microphone test key
or gain control**: opening Input grants no capture authority and those actions are not established.

`ears-shortcuts.json`: same scene, Shortcuts selected, native shortcut capture, reset defaults and cancel
capture. Synthetic shortcut value only. App validates supported/required bindings and persists only on commit.

Fixture checks: `DEVICE_SCENE_AJV=/path/to/ajv/dist/ajv.js bun spec/fixtures/device-scene-v1/validate.mjs`.
Uses installed Ajv 8 without changing this repository's dependency tree. Covers positive shapes, Action phases,
and negative references/bounds/types; it is not the complete native semantic validator.

Both are native renderer/input fixtures, not selected visual designs. Required native proof: both pages at
1x/2x, front/angled projection, click/keyboard parity, disable/cancel, native text capture, resize during
in-flight render, no stale hit regions, alpha edges, BGRA sentinel and image retirement, zero idle renders.
