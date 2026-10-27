# ears-ui — eaRS companion UI

A native, GPU-rendered companion app for eaRS dictation, built with
[GPUI](https://crates.io/crates/gpui). The dictation engine stays headless in
the core (`ears::dictation`); this binary is a thin client that renders its
state and forwards commands.

## What it does

- **Floating dictation island** — a borderless, transparent, non-activating
  panel that appears on the active display while you dictate. It morphs
  between phases (dormant dot → listening with live partials → spinner →
  check), anchors top- or bottom-center, is draggable (offset persisted),
  and can hug the camera notch in **island mode** (macOS). On a notched
  display the compact shape widens around the camera without increasing its
  height. The dot or ears indicator, smooth or spring shape motion, and
  grow, fade or combined content motion are configurable in Overlay settings.
  The ears mark can use the original speech-active standing wave or the
  parametric, state-dependent sliding bars, or the local Oqtopus vector player.
  Click-through and macOS reduced-motion
  are supported; an explicit Overlay option can animate despite system Reduce Motion.
  See `docs/DYNAMIC_ISLAND_SPIKE.md` for the
  window-feasibility findings.
- **Tray / menu-bar presence** — state-coloured mic icon, menu with
  pause/resume, insertion mode, settings, launch-at-login and quit. Works
  with the overlay disabled.
- **Settings window** — four tabs:
  - *Dictation*: backend identity + latency, server switcher (alias or
    `ws://` URL — enables dictate-locally/transcribe-remotely setups),
    insertion mode, hotkey mode, escape-cancels, session language.
  - *Overlay*: enable, anchor, dormant dot, island mode, click-through,
    animation, indicator, shape/content motion, expanded details, scheme,
    radius, opacity, margin and linger.
  - *Dictionaries*: list/add/remove replacement entries with a **live test
    box**.
  - *Profiles*: editor for context profiles (bundle id / app name / window
    title regex → dictionary set, language, insertion mode; priority
    ordered) backed by `~/.config/ears/profiles.toml`.
- **Design system** — colours are a Rust port of the byteowlz design-system
  mechanism (base24 slots → 16 closed roles → components), with the built-in
  `oqto-dark` / `oqto-light` schemes, auto appearance following, and the
  proportional radius dial (0 = sharp).

## Usage

```bash
cargo build -p ears-ui
target/debug/ears-ui              # island + tray; dictation starts paused/unpaused per config
target/debug/ears-ui --settings   # also open the settings window
target/debug/ears-ui --no-overlay # tray only
target/debug/ears-ui --preview-synthetic # synthetic notch, no dictation engine; Ctrl-C to exit
target/debug/ears-ui --preview-synthetic-expanded # synthetic expanded details
target/debug/ears-ui --preview-parametric # animated, synthetic; no dictation engine
target/debug/ears-ui --preview-parametric-reduced # static reduced-motion fixture
target/debug/ears-ui --preview-parametric-override # explicit reduced-motion override
target/debug/ears-ui --oqtopus-check # validate local Oqtopus pack
target/debug/ears-ui --preview-oqtopus # play local pack without dictation
target/debug/ears-ui --preview-oqtopus-reduced # static Oqtopus rest
target/debug/ears-ui --preview-placement # real top anchor, synthetic content
target/debug/ears-ui --preview-floating  # bottom floating fixture
target/debug/ears-ui --preview-settings  # synthetic settings fixture
```

Dictation control stays with the global hotkeys from `config.toml`
(`hotkeys.toggle`, default `ctrl+shift+v`); the UI mirrors and can switch
hotkey mode at runtime. `ears-ui` replaces a running `ears-dictation`
instance so text is never typed twice.

## Config files

| File | Owner |
|------|-------|
| `~/.config/ears/config.toml` | engine (servers, hotkeys, dictionaries) |
| `~/.config/ears/ui.toml`     | UI (overlay, theme, tray, per-UI dictation prefs) |
| `~/.config/ears/profiles.toml` | context profiles (shared with the CLI) |

The Overlay settings write the same `ui.toml` fields available to scripts:

```toml
[overlay]
island_mode = true
indicator = "parametric_ears" # dot | ears (standing wave) | parametric_ears | oqtopus
animate_with_reduced_motion = false # opt in to motion despite the system setting
shape_motion = "spring"  # smooth | spring
content_motion = "grow_fade" # grow | fade | grow_fade
show_details = false     # optional second row on a detected notch
```

The existing dot remains the default. Island mode requires a detected macOS
notch; without one the regular pill remains in use. For the parametric indicator,
`theme.scheme = "auto"` follows macOS system colours or the active Omarchy
`colors.toml`: `$OMARCHY_THEME_FILE`, then `$XDG_STATE_HOME/omarchy/current/theme/colors.toml`,
then the legacy `$XDG_CONFIG_HOME/omarchy/current/theme/colors.toml` (with standard
XDG defaults). An explicit eaRS scheme takes precedence. Missing or invalid
colours fall back to the active eaRS theme, and low-contrast colours are replaced
with readable black or white. Changes are checked once per second.

Oqtopus reads a local pack from `$XDG_DATA_HOME/ears/oqtopus.oqp` (default
`~/.local/share/ears/oqtopus.oqp`), or `EARS_OQTOPUS_PACK` when set. Run
`ears-ui --oqtopus-check` to validate it without starting dictation. Missing or
invalid packs fall back to the ears indicator. Listening and the time-based
completion mark use neutral idle motion; finalizing and error request thinking
and surprise clips if present. The mapping is provisional, and no transcript
or audio is sent to the player. Animation honors Reduce Motion unless the per-app override is set.

## Current limits

The Oqtopus vector indicator requires a separately supplied local pack; its
state mapping and physical-notch appearance need live user verification.
LostIsland integration is separate and not connected; eaRS keeps
its own pill. Notch corner radii are approximate because AppKit exposes the
camera housing's bounding rectangle, not its physical curve. Native rendering on
a physical notched Mac still needs verification. On Linux the existing platform
shim does not yet detect the OS reduced-motion setting; turn Overlay animation
off to disable motion.

## Permissions

- **macOS**: grant *Accessibility* (and *Input Monitoring*) to the `ears-ui`
  binary so it can type into other apps; grant *Microphone* for capture.
- **Linux/Wayland**: uinput setup as documented in the main README
  (`docs/WAYLAND_VIRTUAL_KEYBOARD.md`).
