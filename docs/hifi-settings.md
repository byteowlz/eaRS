# Native receiver Settings

Dictation now opens the native hi-fi **Input / Shortcuts** slice. The four
primary destinations remain Dictation, Overlay, Dictionaries and Profiles.
**General** opens the existing dictation controls; the other destinations retain
their existing panes and drafts. Select Dictation again to return to the device.
This is an incremental implementation, not a completed whole-app redesign.

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
Front presentation avoids perspective-warping native text. Michroma and Geo are
bundled; native text supplies Unicode fallback. Overlay rendering is unchanged.

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
and fixed-pitch glyph fidelity are still incomplete; other settings panes are not
yet rendered as hardware. Wake phrase controls and the ctx invocation seam remain
separate work; no external action execution was enabled.

The full parallel core harness reproduces two wake subprocess failures on the
unchanged base; serial core tests pass (`trx-4kyy.19.2`). Strict Clippy also fails
on the unchanged base with the current compiler. These limits are recorded, not
represented as a clean full-repository lint/parallel-test gate.
