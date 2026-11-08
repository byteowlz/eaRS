# Pinned native device renderer

Source: byteowlz/design-system, commit `265c0a19b7fd28d3c83383b9aa703d36dd966a75`.
Imported unchanged via `git archive`: `impls/device-rs`, the device-scene v1
specification/schema and its three fixtures. MIT-licensed crate manifests.

This is a vendored dependency, not an application-local renderer. Changes belong
in the canonical design-system checkout, followed by a deliberate repin. No
sibling checkout, network dependency or globally installed shader is needed to
build eaRS. Application-owned scenes and actions live in ears-ui Settings.

Source validation: workspace tests (including the continuous-input frame
presentation regression), strict Clippy, and three explicit Metal actual-GPU
geometry, matrix and fader tests at 1x/2x. Faders render the retro hi-fi r10
channel, value fill, scale ticks and raised cap inside the declared volume.
Faders include horizontal/vertical mesh geometry and preview/commit/cancel
pointer gestures with bounded keyboard editing; external effects remain host-owned. This does not claim complete visual,
accessibility, IME or cross-platform parity. See the upstream README for limits.
