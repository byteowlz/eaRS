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
recording_notch = false   # macOS software camera silhouette; opt-in
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
- **Waveform indicator** (`IndicatorStyle::Waveform`, Settings → Overlay →
  Indicator → *Waveform*): a dot-matrix level display in the island, as its own
  choice; the Dot, Standing wave and Animated bars indicators are unchanged.
  The core sends a scalar envelope (peak and RMS as 0-255 bytes, at most ~50
  times a second, no samples or text) only while capturing; the island keeps the
  newest 32 values and draws 12 columns of round dots mirrored on the centre
  line, newest at the right. One dot pitch is used for the whole display, dots
  are never shrunk to fit, and the level is the loudest of peak and RMS with no
  gain or gating. Silence keeps a single baseline dot. Reduced motion stops the
  per-frame animation. Inspired by the voxtype OSD's scrolling peak waveform,
  rendered in the greyrot dot-matrix manner rather than as smooth bars.
- **Meter:** retain the selected indicator without top text.
- **Hidden:** hide the top indicator; independently enabled passive previews and
  Recording notch may remain visible. The tray still provides Settings. Disabling
  Overlay disables all of these surfaces.
- **Subtitle:** opaque, lifted black provisional-text bar.
- **Cursor:** pointer-adjacent bubble that follows the mouse on the display clock,
  clamped/flipped at display edges. This supersedes the earlier anchored prototype
  and its 30 Hz service-timer motion. It is **not an AX/caret implementation**.

Departure Mono and JetBrains Mono regular faces are bundled under OFL; there is
no runtime font download. System remains the default. Exceptionally long words
shrink to a 10 px floor, then ellipsize rather than widening the camera wings or
fixed pill. This is display-only: no recognized/insertion text is altered.

## Recording notch (macOS)

Settings → Overlay → **Recording notch** paints a software camera-notch replica
on **every attached display**, including built-in and external displays. Default
off. The actual camera cutout is hardware, not painted image content; these black
software pixels can therefore appear in display recordings. App-window-only
capture may omit these separate panels; inclusion depends on the recorder/filter.

- A matching real notch uses its existing logical hardware dimensions. Displays
  without a cutout use the accepted synthetic fixture geometry (185 × 32 logical
  px, the same lower corner radius and cubic top throats). Displays narrower than
  the existing 640 px island viewport are not eligible.
- Collapsed replicas persist without an engine, are nonactivating and entirely
  click-through. They have no text, camera icon or animation clock. Display changes
  reconcile through the existing UI tick; unchanged replicas are not invalidated.
- While enabled, the active island temporarily hugs the top software/hardware
  notch. Saved anchor, island-mode, margin and drag offsets are untouched and
  return when switched off. Expanded click-to-Settings and the saved click-through
  preference remain; dragging is disabled while pinned. A lingering collapse becomes
  passive, and quick resume recreates the native popup before restoring interaction.
- Hidden still hides speech/indicator content, but not this explicitly requested
  replica. The global Overlay **Enabled** switch hides everything immediately.
- This starts **no recorder, microphone, model, wake listener or dictation engine**,
  requests no capture/input permission and changes no insertion authority. Existing
  active dictation remains independently controlled.

The same persisted/file interface is `[overlay] recording_notch = true` in
`ui.toml`; `ears-ui --overlay-config` reads it without starting a GUI/engine. As
with other overlay file edits, restart the UI to apply an external edit; the
Settings control is live. Other operating systems preserve but ignore the field.
Synthetic, no-engine/no-save native fixtures:

```sh
ears-ui --preview-recording-notch          # collapsed replicas only
ears-ui --preview-recording-notch-active   # accepted expanded island + replicas
```

VM display pixels and a real click delivered through the collapsed replica can
verify software rendering/passivity. These are not physical MacBook, multi-monitor,
recorder-product compatibility or live-dictation acceptance.

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

## Cursor motion scheduling

On macOS 14+, a public screen-bound `CADisplayLink` drives visible cursor previews at
that screen's advertised refresh rate (up to 240 Hz). It samples the latest absolute
pointer position, without tweening/prediction lag. Movement does not pump ASR events,
tray work or theme refresh. Every frame retains capture/generation/PID/Secure Event
Input eligibility checks; it adds no input monitors, permission requests or actions.

Native callbacks coalesce to one queued GPUI main-thread update. Closing/hiding or
invalidating the preview drops its frame clock; crossing displays recreates it for
the new screen without changing capture ownership. A missing/stalled clock falls
back to the existing service tick after 50 ms, rather than freezing the preview.
Text/font/size/above-below changes invalidate cached height; origin-only movement
keeps that native text-height result and does not explicitly notify unchanged text.
GPUI may still redraw in response to its native window-bounds observer.

A synthetic 60 Hz VM replay measured native movement gaps of 33.5 ms median/50.2 ms p95
before, versus 17.1 ms/22.5 ms after. These are public window-bounds observations,
**not physical compositor scanout or 120 Hz hardware acceptance**. The observer in
`scripts/probe_preview_motion.swift` reads only an owned fixture's numeric geometry;
it never moves the cursor, captures pixels/text or grants permissions. Run motion
input only inside an isolated, owned desktop; retain raw evidence privately.
`just ui-cursor-motion-check before.json after.json 60` verifies the old replay
fails and the new replay meets fixed median/p95 budgets of 1.25/1.5 display periods.
Its 120 Hz unit examples test arithmetic, not 120 Hz hardware. In a separate debug-only
replay without the observer, UI-process CPU was 0.79 s before/0.89 s after; these are
not release/idle/real-ASR CPU budgets.

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
`scripts/probe_overlay_feedback.swift` checks actual native centered text-layout bounds,
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
