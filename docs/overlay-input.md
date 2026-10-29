# Notch pointer routing

The macOS notch should capture input only within its painted black silhouette.
The 640 × 110 popup's transparent margins, lower corners and empty area below
it must pass through to the underlying application. Click-to-Settings, initiated
inside the body, and dragging remain available; explicit full click-through is
still a separate setting.

GPUI uses a faint nonzero native background for shadow stability and presents
through a Metal layer. Clearing the background and avoiding an explicit
`setIgnoresMouseEvents(false)` alone did **not** fix native routing. A public
Core Animation silhouette mask on the GPUI layer did. The mask shares the
rendered width, height, lower-corner radius and cubic outward throats. It updates
with the rendered morph, without an independent implicit animation, and uses the
renderer’s backing scale. Native window shadows remain disabled. No private
WindowServer API, global input monitor, polling or forwarded/reposted clicks.

AppKit has three effective input states: untouched alpha-based routing, explicit
all-input and explicit no-input. Never force false on a new overlay; switching
back from full click-through recreates the popup instead. The native mask applies
to the notched mode; the existing floating-pill rendering is not redesigned.

## Reproduce

```bash
cargo build -p ears-ui
swift scripts/probe_overlay_hit_test.swift "$PWD/target/debug/ears-ui"
swift scripts/probe_overlay_hit_test.swift "$PWD/target/debug/ears-ui" --click-through
swift scripts/probe_overlay_hit_test.swift "$PWD/target/debug/ears-ui" --expanded
```

Run on an unlocked macOS desktop with **existing** event-posting permission.
The bounded fixture starts only a synthetic UI and its own blue underlay, isolates
XDG state, posts clicks without pointer dwell, verifies exact underlay coordinates
and checks that a black-body click opens Settings. It restores the pointer and
terminates only its own child. No microphone, hotkeys, ASR session, real transcript
or desktop screenshot is collected. Lack of permission is a blocker, not a reason
to change system permissions or use another capture path.

The real GPUI fixture reproduced rectangular interception (zero of three outside
clicks reached the underlay), then passed all three outside clicks plus inside
Settings activation with the mask; explicit click-through passed all four clicks.
Geometry tests cover collapsed/active/expanded dimensions, lower corners and
throats. Native physical-notch fit, live dragging, changing input mode in Settings,
multiple display scales, full animation transitions and user acceptance remain
separate qualification. Nothing in these tests installs or restarts the live UI.

## Visual scope

Studio line `ears-brand-island`, desktop GPUI, keeps the user-directed
`ears-speech-motion-r9` contract and incumbent UI. This is the owner's requested
input correction, not selection of a new icon, animation or control-center mock.
The Settings slider reuses existing section/row/theme components; it is not a
Settings redesign. The Studio currently has no separate selected Settings target.
Native routing evidence is functional, not a screenshot-based visual acceptance.

Studio: http://100.64.0.12:4782/apps/ears-ui/?view=desktop
