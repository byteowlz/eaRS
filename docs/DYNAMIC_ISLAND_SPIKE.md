# Dynamic-island feasibility in GPUI — spike verdict (trx-4kyy.1)

Time-boxed probe of the "macOS dynamic island" presentation for the dictation
companion UI, executed against **gpui 0.2.2** (crates.io, Zed's published
package). The outcome is not a throwaway prototype: the island shipped as
`apps/ears-ui/src/overlay.rs` + `apps/ears-ui/src/platform/mac.rs`, and this
document records the per-capability verdicts the spike spec asked for.

Verdict up front: **shippable with one modest hack** — GPUI's `WindowKind::PopUp`
already creates a real `NSPanel` with `NSWindowStyleMaskNonactivatingPanel`,
`NSPopUpWindowLevel`, `canJoinAllSpaces` and `fullScreenAuxiliary`, which is
exactly the island's window contract. The two things GPUI does not expose
(click-through toggle, arbitrary repositioning) are reachable through the
`raw_window_handle` escape hatch in ~30 lines of objc2 code each.

## Per-capability results

Format: works / works-with-hack / blocked.

1. **Transparent, borderless, rounded-pill window** — *works.*
   `WindowOptions { titlebar: None, window_background: WindowBackgroundAppearance::Transparent, kind: PopUp }`
   gives a fully transparent window; the pill is drawn by GPUI (rounded rect +
   border + shadow). No OS chrome. Verified on macOS 26 (aarch64).

2. **Non-activating panel** — *works, no hack needed.*
   gpui 0.2.2's mac platform allocates `PANEL_CLASS` (an `NSPanel`) for
   `WindowKind::PopUp` and ORs in `NSWindowStyleMaskNonactivatingPanel`
   (`src/platform/mac/window.rs`, open()). Opening, updating and closing the
   window never takes key focus; `focus: false` additionally keeps GPUI from
   calling `makeKeyAndOrderFront`. Verified: dictating into another app while
   the island appears never steals keystrokes.

3. **Window level + spaces** — *works.*
   `PopUp` sets `NSPopUpWindowLevel` (101, above normal windows) plus
   `NSWindowCollectionBehaviorCanJoinAllSpaces | FullScreenAuxiliary`, so the
   pill is visible on every Space and over fullscreen apps where the OS
   permits. Not configurable per-window from the GPUI API, but PopUp's fixed
   behaviour is precisely the spec'd "floating, all spaces" level.

4. **Morph animation** — *works.*
   Pure GPUI: `window.request_animation_frame()` + an eased lerp on the pill's
   width/height each frame (~200 ms expand/collapse, spring-ish easing via
   `gpui::bounce(ease_in_out)` on the spinner only; the morph uses a quintic
   ease-out). Reduce-motion is honoured by consulting
   `NSWorkspace.sharedWorkspace.accessibilityDisplayShouldReduceMotion` and
   disabling the animation (exposed as `platform::reduce_motion()`).

5. **Per-display anchoring** — *works-with-hack (small).*
   `WindowOptions.display_id` opens the window on a chosen display and
   `cx.displays()` lists them; the island computes its origin
   display-relative (top-center or bottom-center + persisted drag offset).
   Hack: GPUI has no `set_position` after open, so re-anchoring (cursor moved
   displays, user dragged, settings changed) closes and re-opens the window
   — a one-frame operation that is invisible in practice because the island
   is re-created only while idle or at drag end.

6. **Click-through toggle** — *works-with-hack.*
   `NSWindow.setIgnoresMouseEvents` is not exposed by GPUI. The raw
   `NSView` pointer from `HasWindowHandle` reaches the window, and ~15 lines
   of objc2 flip the flag at runtime (`platform::set_click_through`).

   Repositioning uses the same escape hatch (`platform::set_window_origin`
   via `setFrameOrigin`) for drag-follow; everything else stays in GPUI.

## Fallback plan (not needed)

Had PopUp's panel behaviour been missing, the fallback was a `Normal`
transparent window with the objc2 escape hatch doing the panel conversion.
That escape hatch proved unnecessary for the window class but is kept for
click-through/movement.

## Cross-platform notes (from upstream sources, not runtime-tested here)

- `WindowKind::PopUp` exists on all platforms; the non-activating panel
  behaviour is macOS-specific. Linux (Wayland layer-shell / X11
  override-redirect) and Windows top-level transparent windows get the same
  island *shape language* (transparent window + drawn pill) but the
  focus-steal-free and all-spaces guarantees depend on the platform:
  - Wayland: gpui's wayland backend has no layer-shell surface yet — the
    pill is a normal window; expect compositor-dependent behaviour (no
    guaranteed always-on-top). X11: setting `_NET_WM_STATE_ABOVE` is
    reachable through the same raw-handle hack.
  - Windows: a `WS_POPUP` transparent topmost window covers the use case;
    gpui already renders transparent backgrounds.
- The engine-side (dictation, dictionaries, profiles) is fully
  cross-platform; degradation is visual only.

## Effort estimate for the real implementation

The spike's estimate of ~2-3 days for the island was accurate: window
plumbing + placement + morph + platform shims + tray + settings landed as
~3.5k lines. Remaining known work is polish (multi-display drag edge cases,
Windows/Linux smoke tests) rather than feasibility risk.
