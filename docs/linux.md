# eaRS on Linux

eaRS runs on Linux (X11 and Wayland) with the same engine, hotkeys and Settings.
This page covers the platform differences; installation is in
[desktop-installation.md](desktop-installation.md).

## Tray icon

The tray is a status-notifier item (SNI, libayatana-appindicator). Bars that
implement an SNI host show it: waybar's `tray` module and omarchy's Quickshell
bar (`Quickshell.Services.SystemTray`) both do.

- GTK is initialized on the main thread before the tray is created; without
  that the app used to abort with `GTK has not been initialized`.
- Where GTK cannot initialize (a tty or SSH session with no display) the tray is
  skipped and `ears-ui` prints `tray disabled: …`. Everything else still works.
- The startup log says `tray installed` or `tray unavailable`.

## Window rules on a tiling compositor (Hyprland)

GPUI's Wayland backend creates ordinary toplevel windows, so a tiling
compositor tiles the floating island unless it is told otherwise. Print the
rules and append them to `~/.config/hypr/hyprland.conf`:

```sh
ears-ui --linux-window-rules
hyprctl reload
```

The island uses its own app id, `dev.byteowlz.ears-ui.island`, so rules can
float, unframe and pin it without touching the Settings window
(`dev.byteowlz.ears-ui`). Without the rules the island appears as a tiled,
framed window.

## Microphone

"System default" is the ALSA `default` PCM. On a PipeWire system that PCM can be
unusable while the hardware device works: observed on Arch with PipeWire 1.6 as
`snd_pcm_hw_params` failing for `default` and `pipewire`, while `hw:PCH`,
`plughw:PCH` and `sysdefault:CARD=PCH` capture normally.

eaRS therefore falls back on Linux only when the selection is **System
default**: it tries the other enumerated inputs in order and uses the first one
that opens, logging `input: system default unusable (…); using HDA Intel PCH`
and showing that device as **In use** in Settings. A microphone chosen by name
never falls back; if it disappears, eaRS waits for that exact device and
reconnects when it returns.

You can also pick the working device directly in Settings → Dictation →
Microphone (`ears-ui --input-devices` lists them as JSON).

## Typing into other applications

Text insertion uses `enigo`, which is built with both Linux backends:

- **Wayland**: the wlroots virtual-keyboard protocol
  (`zwp_virtual_keyboard_manager_v1`), available on Hyprland and sway. No extra
  permission is needed.
- **X11**: the x11rb backend.
- **uinput**: used where the compositor offers neither (it needs access to
  `/dev/uinput`).

Earlier builds only enabled the x11rb backend, so on Hyprland enigo fell back to
uinput, which failed without `/dev/uinput` access and left typing broken. If a
compositor has no virtual-keyboard protocol, allow `/dev/uinput` instead:

```sh
sudo usermod -aG input "$USER"      # then log out and back in
sudo modprobe uinput                # load the module now
```

A udev rule can set the group/mode instead, e.g.
`KERNEL=="uinput", GROUP="input", MODE="0660"`. eaRS does not change system
permissions itself.
