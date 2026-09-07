//! Global hotkey listener (rdev).
//!
//! Runs on its own thread for the lifetime of the process (`rdev::listen`
//! cannot be stopped). State changes are applied synchronously through
//! [`Shared`]; loop-affecting actions (cancel, profile cycling) are queued as
//! [`DictationCommand`]s.

use super::DictationCommand;
use super::Shared;
use super::vlog;
use crate::config::DictationHotkeyMode;
use crate::config::HotkeyConfig;
use rdev::EventType;
use rdev::listen;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Tap vs. hold threshold for [`DictationHotkeyMode::Hybrid`].
const TAP_THRESHOLD_MS: u128 = 220;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Combo {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub key: rdev::Key,
}

impl Combo {
    fn matches(&self, ctrl: bool, shift: bool, alt: bool, key: rdev::Key) -> bool {
        self.ctrl == ctrl && self.shift == shift && self.alt == alt && self.key == key
    }
}

/// Parse a combo such as `ctrl+shift+v`.
pub fn parse_combo(s: &str) -> Combo {
    let mut combo = Combo {
        ctrl: false,
        shift: false,
        alt: false,
        key: rdev::Key::Unknown(0),
    };

    for part in s.to_lowercase().split('+') {
        match part.trim() {
            "ctrl" | "control" => combo.ctrl = true,
            "shift" => combo.shift = true,
            "alt" | "option" => combo.alt = true,
            "space" => combo.key = rdev::Key::Space,
            "escape" | "esc" => combo.key = rdev::Key::Escape,
            "tab" => combo.key = rdev::Key::Tab,
            k if k.len() == 1 => {
                if let Some(ch) = k.chars().next() {
                    combo.key = key_for_char(ch);
                }
            }
            _ => {}
        }
    }
    combo
}

fn key_for_char(ch: char) -> rdev::Key {
    match ch {
        'a' => rdev::Key::KeyA,
        'b' => rdev::Key::KeyB,
        'c' => rdev::Key::KeyC,
        'd' => rdev::Key::KeyD,
        'e' => rdev::Key::KeyE,
        'f' => rdev::Key::KeyF,
        'g' => rdev::Key::KeyG,
        'h' => rdev::Key::KeyH,
        'i' => rdev::Key::KeyI,
        'j' => rdev::Key::KeyJ,
        'k' => rdev::Key::KeyK,
        'l' => rdev::Key::KeyL,
        'm' => rdev::Key::KeyM,
        'n' => rdev::Key::KeyN,
        'o' => rdev::Key::KeyO,
        'p' => rdev::Key::KeyP,
        'q' => rdev::Key::KeyQ,
        'r' => rdev::Key::KeyR,
        's' => rdev::Key::KeyS,
        't' => rdev::Key::KeyT,
        'u' => rdev::Key::KeyU,
        'v' => rdev::Key::KeyV,
        'w' => rdev::Key::KeyW,
        'x' => rdev::Key::KeyX,
        'y' => rdev::Key::KeyY,
        'z' => rdev::Key::KeyZ,
        '0' => rdev::Key::Num0,
        '1' => rdev::Key::Num1,
        '2' => rdev::Key::Num2,
        '3' => rdev::Key::Num3,
        '4' => rdev::Key::Num4,
        '5' => rdev::Key::Num5,
        '6' => rdev::Key::Num6,
        '7' => rdev::Key::Num7,
        '8' => rdev::Key::Num8,
        '9' => rdev::Key::Num9,
        _ => rdev::Key::Unknown(0),
    }
}

/// Human readable description of the hotkey mode.
pub fn mode_description(mode: DictationHotkeyMode) -> &'static str {
    match mode {
        DictationHotkeyMode::Toggle => "toggle pause/resume",
        DictationHotkeyMode::PushToTalk => "hold to talk",
        DictationHotkeyMode::Hybrid => "tap to toggle, hold to talk while paused",
    }
}

/// Start the listener thread.
pub(crate) fn spawn_listener(config: HotkeyConfig, escape_cancels: bool, shared: Arc<Shared>) {
    let _ = escape_cancels; // kept for signature stability; flag lives on Shared
    vlog!("Initializing hotkey listener for: {}", config.toggle);
    thread::Builder::new()
        .name("ears-hotkeys".to_string())
        .spawn(move || {
            let toggle = parse_combo(&config.toggle);
            let profile_cycle = config
                .profile_cycle
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .map(parse_combo);
            vlog!(
                "Parsed combo - ctrl:{} shift:{} alt:{} key:{:?} mode:{:?}",
                toggle.ctrl,
                toggle.shift,
                toggle.alt,
                toggle.key,
                shared.hotkey_mode()
            );

            let mut ctrl = false;
            let mut shift = false;
            let mut alt = false;
            let mut held_from_paused = false;
            let mut down_at: Option<Instant> = None;

            let result = listen(move |event| match event.event_type {
                EventType::KeyPress(rdev::Key::ControlLeft)
                | EventType::KeyPress(rdev::Key::ControlRight) => ctrl = true,
                EventType::KeyRelease(rdev::Key::ControlLeft)
                | EventType::KeyRelease(rdev::Key::ControlRight) => ctrl = false,
                EventType::KeyPress(rdev::Key::ShiftLeft)
                | EventType::KeyPress(rdev::Key::ShiftRight) => shift = true,
                EventType::KeyRelease(rdev::Key::ShiftLeft)
                | EventType::KeyRelease(rdev::Key::ShiftRight) => shift = false,
                EventType::KeyPress(rdev::Key::Alt) | EventType::KeyPress(rdev::Key::AltGr) => {
                    alt = true
                }
                EventType::KeyRelease(rdev::Key::Alt) | EventType::KeyRelease(rdev::Key::AltGr) => {
                    alt = false
                }
                EventType::KeyPress(key) => {
                    if !shared.is_running() {
                        return;
                    }
                    if shared.escape_cancels()
                        && key == rdev::Key::Escape
                        && !ctrl
                        && !shift
                        && !alt
                        && shared.is_capturing()
                    {
                        shared.command(DictationCommand::Cancel);
                        return;
                    }
                    if toggle.matches(ctrl, shift, alt, key) {
                        down_at = Some(Instant::now());
                        if matches!(
                            shared.hotkey_mode(),
                            DictationHotkeyMode::PushToTalk | DictationHotkeyMode::Hybrid
                        ) {
                            if shared.set_capturing(true) {
                                held_from_paused = true;
                                vlog!("Audio capture started (hold)");
                            } else {
                                held_from_paused = false;
                            }
                        }
                    }
                }
                EventType::KeyRelease(key) => {
                    if !shared.is_running() {
                        return;
                    }
                    if let Some(cycle) = profile_cycle {
                        if cycle.matches(ctrl, shift, alt, key) {
                            shared.command(DictationCommand::CycleProfile);
                            return;
                        }
                    }
                    if !toggle.matches(ctrl, shift, alt, key) {
                        return;
                    }
                    let press_duration_ms = down_at.map(|ts| ts.elapsed().as_millis()).unwrap_or(0);
                    down_at = None;

                    match shared.hotkey_mode() {
                        DictationHotkeyMode::Toggle => {
                            shared.toggle();
                        }
                        DictationHotkeyMode::PushToTalk => {
                            if held_from_paused {
                                shared.set_capturing(false);
                            }
                            held_from_paused = false;
                        }
                        DictationHotkeyMode::Hybrid => {
                            if held_from_paused {
                                if press_duration_ms <= TAP_THRESHOLD_MS {
                                    vlog!("Audio capture latched on (tap)");
                                } else if shared.set_capturing(false) {
                                    vlog!("Audio capture stopped (hold release)");
                                }
                            } else {
                                shared.set_capturing(false);
                                vlog!("Audio capture stopped (toggle off)");
                            }
                            held_from_paused = false;
                        }
                    }
                }
                _ => {}
            });
            if let Err(e) = result {
                eprintln!("Hotkey listener error: {:?}", e);
            }
        })
        .expect("failed to spawn hotkey listener thread");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modifier_combos() {
        let combo = parse_combo("Ctrl+Shift+V");
        assert!(combo.ctrl && combo.shift && !combo.alt);
        assert_eq!(combo.key, rdev::Key::KeyV);
        let combo = parse_combo("alt+space");
        assert!(combo.alt && !combo.ctrl);
        assert_eq!(combo.key, rdev::Key::Space);
    }
}
