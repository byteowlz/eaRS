# Pinned native device renderer

Source: byteowlz/design-system, commit `9afac5745a90fb23f15de95acee2fa44f94daafd`.
Imported unchanged via `git archive`: `impls/device-rs`, the device-scene v1
specification/schema and its three fixtures. MIT-licensed crate manifests.

This is a vendored dependency, not an application-local renderer. Changes belong
in the canonical design-system checkout, followed by a deliberate repin. No
sibling checkout, network dependency or globally installed shader is needed to
build eaRS. Application-owned scenes and actions live in ears-ui Settings.

Source validation: workspace check, 26 unit/semantic tests, Clippy, and the
explicit Metal actual-GPU geometry and matrix tests at 1x/2x. This does not claim complete visual,
accessibility, IME or cross-platform parity. See the upstream README for limits.
