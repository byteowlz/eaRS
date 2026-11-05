# Native receiver Settings

Dictation now opens the native hi-fi **Input / Shortcuts** slice. The four
primary destinations remain Dictation, Overlay, Dictionaries and Profiles.
**General** opens the existing dictation controls; the other destinations retain
their existing panes and drafts. Select Dictation again to return to the device.
This is an incremental implementation, not a completed whole-app redesign.

## Window

The device has no OS title bar, frame or opaque stage. Drag unused chassis metal
to move it; controls and empty transparent space are not drag handles. The top
right × closes Settings without applying drafts; Command-W on macOS (Control-W
elsewhere) also closes, except while actively recording that chord. Escape cancels
a draft. Existing non-device panes retain a Close button and header-padding drag
area. Opening Settings again uses the normal tray entry.

GPUI's titlebar setting alone does not remove macOS chrome: the Settings-only
AppKit setup removes the titled/full-content mask and native shadow before the
focused view is constructed. The shared view honors the transparent viewport;
microphone capture, overlays and permissions are unchanged.

## Input

Refresh enumerates device metadata without opening a microphone. Activate the
microphone display to cycle available choices, then Apply to confirm. A running
engine opens the new device muted, swaps only on success, and confirms before the
preference is saved. A stopped engine saves the next-start preference. Duplicate
names are not selectable; a missing saved device stays visible rather than being
silently replaced. Opening Settings never starts capture or performs a mic test.

## Shortcuts

Select Dictation, Hold to talk or Next profile, press a chord, then Save. With the
engine running, recording uses its existing physical-key listener and distinguishes
left/right modifiers and Fn. Without that listener, focused window chords work;
the Modifier key selector supplies explicit side-only presets without starting an
engine or asking for input permission. Function-key prefixes are not arbitrary
modifier combinations: unsupported bindings are refused rather than guessed.

- Defaults stages the default bindings; it does not save immediately.
- Clear disables an optional binding, never the required dictation binding.
- Escape, Cancel, leaving the page, losing window activation or closing the
  window discards the shortcut draft and releases its suspension guard.
- eaRS shortcut actions are suspended during capture/editing. Physical key-up
  drains before rearming. OS shortcuts remain OS shortcuts; no keyboard grab or
  security-policy change is attempted.
- Validation rejects unsupported keys, duplicate bindings and Escape when the
  running engine reserves it. Conflicts with other applications are not detectable.
- A narrow atomic save preserves comments, symlinks, permissions and unrelated
  configuration. Concurrent shortcut edits refuse the save. Persistence succeeds
  before live reconfiguration; if the listener stops, the UI says saved but not
  active instead of claiming success. Mode and runtime Escape overrides are not
  overwritten by binding edits.

Recorder cancellation is generation-fenced through the engine queue, so a delayed
hotkey-owned discard cannot stop or erase a newer wake/externally started capture.
Existing permissive startup syntax remains supported; editor validation is strict.

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

`settings/device_layout.rs` defines the common row tracks, column edges, gutters
and padding. Shortcut labels and fields share baselines; action keys have equal
widths and gutters; Input/Shortcuts share the same outer edges and footer. Geometry
tests check all visible pairs for overlap and every orange display for matrix
presentation. A wider string never changes item geometry or dot size.

Synthetic, non-persisting previews:

```sh
just --command cargo run --locked -p ears-ui -- --preview-settings-device
just --command cargo run --locked -p ears-ui -- --preview-settings-shortcuts
```

Verification includes core recorder/generation regressions, scene/action-phase
validation, compare-and-save and CLI tests, plus actual native VM rendering and
pointer/keyboard interaction for selection, conflicting shortcuts, cancellation,
capture, preview Save and return to General. The shared renderer additionally has
actual Metal readback/hit tests at 1x/2x. This is not physical global-hotkey,
high-refresh, accessibility-tree, Linux/Windows or IME acceptance. Shared material
fidelity and perspective-warped printed labels remain incomplete; other settings panes are not
yet rendered as hardware. Wake phrase controls and the ctx invocation seam remain
separate work; no external action execution was enabled.

The full parallel core harness reproduces two wake subprocess failures on the
unchanged base; serial core tests pass (`trx-4kyy.19.2`). Strict Clippy also fails
on the unchanged base with the current compiler. These limits are recorded, not
represented as a clean full-repository lint/parallel-test gate.
