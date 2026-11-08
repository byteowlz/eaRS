# Pinned native device renderer

Source: byteowlz/design-system, commit `276b06956cccae09360104cdd019e0221784884b`.
Imported unchanged via `git archive`: `impls/device-rs`, the device-scene v1
specification/schema and its three fixtures. MIT-licensed crate manifests.

This is a vendored dependency, not an application-local renderer. Changes belong
in the canonical design-system checkout, followed by a deliberate repin. No
sibling checkout, network dependency or globally installed shader is needed to
build eaRS. Application-owned scenes and actions live in ears-ui Settings.

Source validation: workspace check, 32 unit/semantic tests, strict Clippy, and
three explicit Metal actual-GPU geometry, matrix and fader tests at 1x/2x.
Faders include horizontal/vertical mesh geometry and preview/commit/cancel
pointer gestures with bounded keyboard editing; external effects remain host-owned. This does not claim complete visual,
accessibility, IME or cross-platform parity. See the upstream README for limits.
