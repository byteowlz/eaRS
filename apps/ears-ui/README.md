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
  and can hug the camera notch in **island mode** (macOS). Click-through and
  reduced-motion are supported. See `docs/DYNAMIC_ISLAND_SPIKE.md` for the
  window-feasibility findings.
- **Tray / menu-bar presence** — state-coloured mic icon, menu with
  pause/resume, insertion mode, settings, launch-at-login and quit. Works
  with the overlay disabled.
- **Settings window** — four tabs:
  - *Dictation*: backend identity + latency, server switcher (alias or
    `ws://` URL — enables dictate-locally/transcribe-remotely setups),
    insertion mode, hotkey mode, escape-cancels, session language.
  - *Overlay*: enable, anchor, dormant dot, island mode, click-through,
    animation, scheme, radius dial, opacity, margin, linger.
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

## Permissions

- **macOS**: grant *Accessibility* (and *Input Monitoring*) to the `ears-ui`
  binary so it can type into other apps; grant *Microphone* for capture.
- **Linux/Wayland**: uinput setup as documented in the main README
  (`docs/WAYLAND_VIRTUAL_KEYBOARD.md`).
