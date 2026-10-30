# Optional transcript presentation

The incumbent pill is retained and remains the default. Settings → Overlay →
Transcript presentation adds independent, opt-in choices. No migration selects
an experiment or changes the indicator, capture, wake, insertion or recording.

## Configuration and agent interface

Edit `[overlay]` in `$XDG_CONFIG_HOME/ears/ui.toml` (otherwise
`~/.config/ears/ui.toml`). GUI choices persist through the same `UiConfig`.
File edits take effect at UI startup; editing the file does not restart the UI.
Read the overlay without starting dictation with `ears-ui --overlay-config`.
`ears-ui --overlay-schema` prints `examples/ui.schema.json`, which validates
these additional keys and permits the existing UI fields.

```toml
[overlay]
appearance = "classic"    # classic | black; independent of content
transcript = "word"       # current (Phrase) | word | meter | hidden; default current
preview = "inline"        # inline | subtitle | cursor; default inline
font = "departure_mono"   # system | departure_mono | jetbrains_mono; default system
word_width = 300.0        # 220–440 logical px; floating word pill only
word_size = 18.0          # 12–24 maximum logical px; camera-safe cap applies
word_fade = true
```

- **Classic / Black appearance:** an explicit skin choice, independent of text.
  Missing appearance defaults to Classic without rewriting existing files. Compact
  floating pills retain Classic; real hardware notches always remain black.
- **Phrase (stored as `current`):** existing content-sized phrase and indicator/motion.
- **Word:** stable-width floating capsule with centered text and balanced logo space;
  either appearance works. The 220 px size is available alongside existing widths.
  Latest received word, no growing
  text entrance. Camera mode retains the existing centered continuous silhouette,
  cubic throats and hardware gap between logo/content. It does not draw three slabs.
- **Meter:** retain the selected indicator without top text.
- **Hidden:** hide the top indicator; an independently enabled passive preview may
  remain visible. The tray still provides Settings. Disabling Overlay disables both.
- **Subtitle:** opaque, lifted black provisional-text bar.
- **Cursor:** pointer-adjacent bubble that follows the mouse at the existing bounded
  UI tick (33 ms while visible), clamped/flipped at display edges. This supersedes
  the earlier anchored prototype. It is **not an AX/caret implementation**.

Departure Mono and JetBrains Mono regular faces are bundled under OFL; there is
no runtime font download. System remains the default. Exceptionally long words
shrink to a 10 px floor, then ellipsize rather than widening the camera wings or
fixed pill. This is display-only: no recognized/insertion text is altered.

## Quiet fade and ownership

Only live Word transcript text fades. Error, armed, completion and dormant status
labels remain fully visible. Opacity uses the existing local Silero scored-sample
quiet/idle ratio; ASR word timing, server Speech and wall time do not advance it.
Physical holds and disabled animation/reduced motion keep text visible (unless the
existing explicit reduced-motion override is selected).

The core emits at most approximately 10 progress events/second, after actual
scored input only. The event contains generation, quiet/idle durations, physical
hold and a producer `Instant`. No audio, transcripts or identity are included.
Disabled/no scorer/unscored input does not fabricate zeroes or silence. Stale
(older than 500 ms), future or superseded progress is discarded; the model tick
notifies once at expiry so even a stationary Dot repaints at full opacity.
This is **not** the planned per-frame VAD diagnostic panel or pause-duration editor.

Passive previews use a separate, nonactivating, full-click-through popup. Text is
bounded to the newest 384 Unicode scalar values, with at most three visible native
wrapped lines and automatic tail scrolling. Nothing is copied, typed, dispatched,
retained beyond the capture or sent by the preview. Empty intermediate ASR segment
boundaries retain only this capture's last bounded displayed segment, avoiding
blank flashes. It clears at local capture stop/cancel, focus/security invalidation
or a new generation. Frontmost PID changes, unknown identity and macOS Secure Event
Input fence old ownership until a fresh capture. Pointer movement across displays
changes placement, not capture/text ownership; subtitles stay on their bound display.
Settings shows an explicit active/waiting/paused/focus/security/unavailable reason. The guard is best-effort application
identity polling, **not** same-app window/field authority or universal secure-field
recognition. Existing insertion/cancellation safeguards remain independent.

Cursor placement flips above the pointer near the bottom edge, keeping the visible
body edge 28 px away when unconstrained. Coordinates are display-local logical
pixels; AppKit conversion includes actual screen origin. Passive windows currently
require macOS. Elsewhere the option is reported unavailable; inline text remains.
No accessibility or event-posting permission is requested by normal rendering.

## Verification and synthetic fixtures

```bash
just ui-test
just ui-presentation-check
# Preview only; never starts engine/mic/hotkeys/tray or writes Settings:
ears-ui --preview-word --preview-floating --preview-departure
# Both skins, plus classic compact state (synthetic only):
ears-ui --preview-word --preview-floating --preview-black
ears-ui --preview-word --preview-floating --preview-black --preview-compact
ears-ui --preview-word --preview-subtitle
ears-ui --preview-word --preview-cursor --preview-jetbrains
```

Use isolated XDG config/data/state directories for synthetic Settings review;
existing preview construction may read profile/dictionary configuration. New
preview modifiers and unknown options never fall through to engine startup.

`ears-ui --overlay-font-check` uses GPUI's **headless** text system and terminates
without a native event loop or window. The native mask tests check full capsule
ends/top gap and unchanged notch lobes/corners. With existing event-posting access,
`just ui-input-test` and the extended Swift probe can additionally verify routing
(`--word --floating`, `--word`, `--expanded`, `--click-through`). They use synthetic
underlays, temporary XDG paths, an owned bounded child and pointer restoration.

`--preview-layout-report` emits deduplicated numeric layout evidence only in
synthetic fixtures: no text, identity or production state. The extended
`scripts/probe_overlay_feedback.swift` checks actual native centered glyph bounds,
resolved appearance, compact state, pointer displacement and Settings tab/action
geometry plus keyboard navigation. It does not capture pixels or request screen
recording permission. `--preview-settings-minimum` and `--preview-settings-light`
provide bounded synthetic size/theme coverage.

Source tests/font registration are not native pixel, hardware safe-area, live
focus/scale, accessibility or microphone acceptance. This change is not an install;
production replacement, live dictation checks and physical acceptance are separate.

Visual reference: Design Studio `artifacts/ears/transcription-previews-r3`,
source `ae959c0`, standalone SHA-256
`0f89036cbf7b8bc3c0626b276a04ab4b2b5efb13cead81463411ee2ecf65afe6`.
Owner approved additive native implementation, not default replacement or deployment.
Native compact words remain single-line with shrink/ellipsis (browser stress fixture
wraps); actual platform camera dimensions and the existing animation renderer are
used rather than the browser simulation. Native preview font sizes/three-line cap
are implementation bounds, not evidence of pixel-identical browser reproduction.
