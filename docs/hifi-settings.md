# Native receiver Settings

Settings has **one navigation level**: Dictation, Overlay, Dictionaries, Profiles.
Dictation is a single receiver page; there are no sub-pages. Overlay, Dictionaries
and Profiles still use the earlier native panes until they are converted.

## Layout

- **Top display** (centered dot matrix): help for the control under the pointer,
  confirmations and errors, recording prompts, and Test transcripts. When idle it
  shows readiness and the start shortcut.
- **Test** streams your words into the top display. Nothing is typed, copied,
  kept in audio history or written to transcript logs; Test again or Esc stops it.
  Auto-pause ends a test like a dictation. Refused while dictation is running.
  The engine command is `SetTestCapture`; output targets stay a sink until the
  capture ends, even if a context profile changes the insertion mode meanwhile.
- **Tabs** line up with the two columns; the selected tab's lamp is lit.
- **Microphone**: one latching key per input, lamp on the active one. Selecting
  switches immediately (opened muted, swapped on success, saved after the engine
  confirms). Listing inputs on open is metadata only. More than four inputs page.
- **Shortcuts**: Start / stop, Hold to talk, Next profile, Key mode, Esc cancels,
  Restore defaults. Click a display, press the keys; it saves and applies at once.
  Esc cancels recording. × removes an optional shortcut. Physical left/right
  modifiers are recorded through the running engine's listener.
- **Speech**: Language, Auto-pause, Speech threshold. **Wake word**: Sensitivity.
  **Text**: Insert as, Context profiles. **Server**: choose with ◀ ▶, then Use
  (restarts dictation); Status shows the connection.

Changes apply immediately; there are no Save, Apply or Cancel rows. Unavailable
keys keep their place, dimmed (for example Auto-pause when `idle_ms = 0`).
Custom server URLs and enabling auto-pause or the wake word remain config.toml
edits, as before.

## Grid

`settings/device_layout.rs` defines it: both columns are stacks of 2.4-unit slots
(headings take a slot), so rows line up across the device and both columns end
together. Row labels align to the column edge, keys to the control column, and
every display to the inner value column. Spacing uses 0.4 / 0.8 / 1.2 units;
every display uses the same 0.16-unit dot pitch. Geometry never depends on values;
legend size follows the legend's role, not its length. Tests check edges, the
shared grid, pitch and non-overlap.

## Window

No OS title bar, frame or opaque stage. Drag unused chassis metal to move it.
× (top right) or Command-W closes; Esc cancels a recording or a test.

## Headless use

```sh
ears-ui --shortcuts
ears-ui --shortcuts-schema
printf '%s\n' '{"expected":{"toggle":"ctrl+shift+v","push_to_talk":null,"profile_cycle":null},"bindings":{"toggle":"ctrl+shift+v","push_to_talk":"right_option","profile_cycle":null}}' | ears-ui --set-shortcuts
```

Use the current saved bindings from `--shortcuts` as `expected`, not the example
blindly. All three keys must be explicit; null disables an optional binding.
The CLI uses the same narrow validation/save path but does **not** control an
already-running process: its receipt explicitly says `live_applied: false` and
`applies: next_engine_start`. None of these commands starts audio or a listener.

## Implementation and evidence

The pinned `vendor/device` dependency contains the shared scene, wgpu renderer
and GPUI adapter, imported unchanged from design-system. App-owned layout/state
and effect dispatch live in `apps/ears-ui/src/settings/device*.rs`. It is neither a
WebView nor an app-local duplicate renderer, and needs no sibling checkout.
Printed legends use bundled Michroma and Geo. Orange displays use the shared
case-preserving 5x7 glyph table, rendered as round phosphors on the actual 3D
surfaces with the same camera and depth buffer. Pitch is fixed at 0.16 device
units for every eaRS display, independent of string length. Long values wrap at
word boundaries and show an ellipsis if still too long; hovering shows the full,
unchanged value. Unsupported Unicode uses a disclosed native-shaped fallback.
Overlay rendering is unchanged.

Synthetic, non-persisting preview (Test streams a labelled synthetic sentence;
no microphone is opened and no config is written):

```sh
just --command cargo run --locked -p ears-ui -- --preview-settings-device
```

Verification includes the engine test-capture regression, core recorder and
generation regressions, layout/grid tests, scene/action-phase validation,
compare-and-save and CLI tests, plus actual native VM rendering and pointer and
keyboard interaction: hover help, Test streaming, F6 recording, microphone
selection, threshold steps, Overlay and back. The shared renderer additionally has
actual Metal readback/hit tests at 1x/2x. This is not physical global-hotkey,
high-refresh, accessibility-tree, Linux/Windows or IME acceptance. Shared material
fidelity and perspective-warped printed labels remain incomplete; other settings panes are not
yet rendered as hardware. Wake phrase controls and the ctx invocation seam remain
separate work; no external action execution was enabled.

The full parallel core harness reproduces two wake subprocess failures on the
unchanged base; serial core tests pass (`trx-4kyy.19.2`). Strict Clippy also fails
on the unchanged base with the current compiler. These limits are recorded, not
represented as a clean full-repository lint/parallel-test gate.
