//! Global hotkey listener.
//!
//! Runs on its own thread for the lifetime of the process (the native event
//! loop cannot currently be stopped). State changes are applied synchronously through
//! [`Shared`]; loop-affecting actions (cancel, profile cycling) are queued as
//! [`DictationCommand`]s.

use super::DictationCommand;
use super::Shared;
use super::vlog;
use crate::config::DictationHotkeyMode;
use crate::config::HotkeyConfig;
use rdev::EventType;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Tap vs. hold threshold for [`DictationHotkeyMode::Hybrid`].
const TAP_THRESHOLD_MS: u128 = 220;
/// A press shorter than this is a stray tap, not dictation (as in Hex): the
/// take is discarded instead of pasted.
const PTT_HOLD_MS: u128 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Combo {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    /// Command on macOS, the Windows/Super key elsewhere.
    pub meta: bool,
    pub key: rdev::Key,
}

impl Combo {
    fn matches(&self, ctrl: bool, shift: bool, alt: bool, meta: bool, key: rdev::Key) -> bool {
        self.ctrl == ctrl
            && self.shift == shift
            && self.alt == alt
            && self.meta == meta
            && self.key == key
    }

    /// A binding that is only modifiers (e.g. right Option), as Hex allows:
    /// holding it must not require any other key.
    pub fn modifier_only(&self) -> bool {
        is_modifier(self.key)
    }
}

fn is_modifier(key: rdev::Key) -> bool {
    matches!(
        key,
        rdev::Key::Alt
            | rdev::Key::AltGr
            | rdev::Key::ControlLeft
            | rdev::Key::ControlRight
            | rdev::Key::ShiftLeft
            | rdev::Key::ShiftRight
            | rdev::Key::MetaLeft
            | rdev::Key::MetaRight
            | rdev::Key::Function
    )
}

/// Parse a combo such as `ctrl+shift+v`, `cmd+space`, or `right_option`.
pub fn parse_combo(s: &str) -> Combo {
    let mut combo = Combo {
        ctrl: false,
        shift: false,
        alt: false,
        meta: false,
        key: rdev::Key::Unknown(0),
    };
    for part in s.to_lowercase().split('+') {
        let part = part.trim();
        if apply_modifier(part, &mut combo) {
            continue;
        }
        if let Some(key) = named_key(part) {
            combo.key = key;
        }
    }
    combo
}

/// Aggregate modifier words. Right-hand bindings also set their flag, so a
/// `right_option` hold matches with Option held.
fn apply_modifier(part: &str, combo: &mut Combo) -> bool {
    match part {
        "ctrl" | "control" | "left_ctrl" => combo.ctrl = true,
        "right_ctrl" => {
            combo.ctrl = true;
            combo.key = rdev::Key::ControlRight;
        }
        "shift" | "left_shift" => combo.shift = true,
        "right_shift" => {
            combo.shift = true;
            combo.key = rdev::Key::ShiftRight;
        }
        "alt" | "option" | "left_alt" | "left_option" => combo.alt = true,
        "right_alt" | "right_option" | "roption" => {
            combo.alt = true;
            combo.key = rdev::Key::AltGr;
        }
        "cmd" | "command" | "meta" | "super" | "left_cmd" => combo.meta = true,
        "right_cmd" | "right_meta" => {
            combo.meta = true;
            combo.key = rdev::Key::MetaRight;
        }
        _ => return false,
    }
    true
}

/// The non-modifier part of a binding.
fn named_key(part: &str) -> Option<rdev::Key> {
    Some(match part {
        "fn" | "function" | "globe" => rdev::Key::Function,
        "space" => rdev::Key::Space,
        "escape" | "esc" => rdev::Key::Escape,
        "tab" => rdev::Key::Tab,
        "return" | "enter" => rdev::Key::Return,
        f if (2..=3).contains(&f.len()) && f.starts_with('f') => match f[1..].parse::<u8>() {
            Ok(1) => rdev::Key::F1,
            Ok(2) => rdev::Key::F2,
            Ok(3) => rdev::Key::F3,
            Ok(4) => rdev::Key::F4,
            Ok(5) => rdev::Key::F5,
            Ok(6) => rdev::Key::F6,
            Ok(7) => rdev::Key::F7,
            Ok(8) => rdev::Key::F8,
            Ok(9) => rdev::Key::F9,
            Ok(10) => rdev::Key::F10,
            Ok(11) => rdev::Key::F11,
            Ok(12) => rdev::Key::F12,
            _ => rdev::Key::Unknown(0),
        },
        k if k.len() == 1 => key_for_char(k.chars().next()?),
        _ => return None,
    })
}

/// Why a push-to-talk take ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PttRelease {
    /// Held long enough: finish and insert.
    Finish,
    /// Stray tap, or another key joined the hold: discard, insert nothing.
    Discard,
    /// The capture was already running before the press: leave it alone.
    Leave,
}

/// Hold-to-talk state, as the Hex app does it: the press starts capture, a
/// release before [`PTT_HOLD_MS`] discards the take, another key pressed during
/// a modifier-only hold discards it, and a real hold finishes on release.
#[derive(Debug, Default)]
pub struct PushToTalk {
    started: Option<Instant>,
    started_capture: bool,
    chorded: bool,
}

impl PushToTalk {
    /// Register the binding's press. Returns whether this press started a new
    /// capture (the caller starts one when dictation was paused).
    pub fn pressed(&mut self, now: Instant, was_capturing: bool) -> bool {
        self.started = Some(now);
        self.started_capture = !was_capturing;
        self.chorded = false;
        self.started_capture
    }

    /// Another key joined a modifier-only hold: no longer a deliberate take.
    pub fn other_key(&mut self) {
        if self.started.is_some() {
            self.chorded = true;
        }
    }

    pub fn held(&self) -> bool {
        self.started.is_some()
    }

    /// The binding's key (or modifier) was released.
    pub fn released(&mut self, now: Instant) -> PttRelease {
        let Some(started) = self.started.take() else {
            return PttRelease::Leave;
        };
        let held_ms = now.saturating_duration_since(started).as_millis();
        let started_capture = std::mem::take(&mut self.started_capture);
        let chorded = std::mem::take(&mut self.chorded);
        if !started_capture {
            return PttRelease::Leave;
        }
        if chorded || held_ms < PTT_HOLD_MS {
            PttRelease::Discard
        } else {
            PttRelease::Finish
        }
    }
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

#[cfg(target_os = "macos")]
fn listen_events(callback: impl FnMut(EventType) + 'static) -> Result<(), String> {
    super::mac_hotkeys::listen(callback)
}

#[cfg(not(target_os = "macos"))]
fn listen_events(mut callback: impl FnMut(EventType) + 'static) -> Result<(), String> {
    rdev::listen(move |event| callback(event.event_type)).map_err(|e| format!("{e:?}"))
}

/// Bindings and their live state; each OS key event is one call.
pub(crate) struct Listener {
    toggle: Combo,
    profile_cycle: Option<Combo>,
    push_to_talk: Option<Combo>,
    ctrl: bool,
    shift: bool,
    alt: bool,
    meta: bool,
    toggle_down: Option<Instant>,
    held_from_paused: bool,
    ptt: PushToTalk,
}

impl Listener {
    pub(crate) fn new(config: &HotkeyConfig) -> Self {
        let combo = |s: &Option<String>| {
            s.as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(parse_combo)
        };
        Self {
            toggle: parse_combo(&config.toggle),
            profile_cycle: combo(&config.profile_cycle),
            push_to_talk: combo(&config.push_to_talk),
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
            toggle_down: None,
            held_from_paused: false,
            ptt: PushToTalk::default(),
        }
    }

    fn track_modifier(&mut self, key: rdev::Key, down: bool) -> bool {
        let slot = match key {
            rdev::Key::ControlLeft | rdev::Key::ControlRight => &mut self.ctrl,
            rdev::Key::ShiftLeft | rdev::Key::ShiftRight => &mut self.shift,
            rdev::Key::Alt | rdev::Key::AltGr => &mut self.alt,
            rdev::Key::MetaLeft | rdev::Key::MetaRight => &mut self.meta,
            _ => return false,
        };
        *slot = down;
        true
    }

    /// A key or modifier went down. `escape` is handled by the caller.
    pub(crate) fn press(&mut self, key: rdev::Key, shared: &Shared) {
        // The hold starts before anything else can be read as a chord.
        if let Some(combo) = self.push_to_talk
            && combo.matches(self.ctrl, self.shift, self.alt, self.meta, key)
        {
            let started = self.ptt.pressed(Instant::now(), shared.is_capturing());
            shared.set_ptt_held(true);
            if started {
                shared.set_capturing(true);
                vlog!("Audio capture started (push to talk)");
            }
            return;
        }
        // Any other key, modifiers included, joined a modifier-only hold: the
        // user is typing a chord, not dictating (Hex's rule).
        if self.ptt.held() && self.push_to_talk.is_some_and(|c| c.modifier_only()) {
            self.ptt.other_key();
        }
        if self.track_modifier(key, true) && !self.is_binding_modifier(key) {
            return;
        }
        if self
            .toggle
            .matches(self.ctrl, self.shift, self.alt, self.meta, key)
        {
            self.toggle_down = Some(Instant::now());
            if matches!(
                shared.hotkey_mode(),
                DictationHotkeyMode::PushToTalk | DictationHotkeyMode::Hybrid
            ) {
                shared.set_ptt_held(true);
                self.held_from_paused = shared.set_capturing(true);
                vlog!(
                    "Audio capture {} (hold)",
                    if self.held_from_paused {
                        "started"
                    } else {
                        "already on"
                    }
                );
            }
        }
    }

    fn is_binding_modifier(&self, key: rdev::Key) -> bool {
        // A modifier that is itself a binding must still be reported.
        self.push_to_talk.is_some_and(|c| c.key == key)
            || self.toggle.key == key
            || self.profile_cycle.is_some_and(|c| c.key == key)
    }

    pub(crate) fn release(&mut self, key: rdev::Key, shared: &Shared) {
        self.track_modifier(key, false);
        if self.ptt.held() && self.push_to_talk.is_some_and(|c| c.key == key) {
            shared.set_ptt_held(false);
            match self.ptt.released(Instant::now()) {
                PttRelease::Discard => {
                    shared.command(DictationCommand::Cancel);
                    vlog!("Audio capture discarded (push-to-talk tap or chord)");
                }
                PttRelease::Finish => {
                    if shared.set_capturing(false) {
                        vlog!("Audio capture stopped (push-to-talk release)");
                    }
                }
                PttRelease::Leave => {}
            }
            return;
        }
        // Release the hold guard even if modifiers were released first.
        if key == self.toggle.key {
            shared.set_ptt_held(false);
        }
        if let Some(cycle) = self.profile_cycle
            && cycle.matches(self.ctrl, self.shift, self.alt, self.meta, key)
        {
            shared.command(DictationCommand::CycleProfile);
            return;
        }
        if !self
            .toggle
            .matches(self.ctrl, self.shift, self.alt, self.meta, key)
        {
            return;
        }
        let press_duration_ms = self
            .toggle_down
            .take()
            .map(|ts| ts.elapsed().as_millis())
            .unwrap_or(0);
        match shared.hotkey_mode() {
            DictationHotkeyMode::Toggle => {
                shared.toggle();
            }
            DictationHotkeyMode::PushToTalk => {
                if self.held_from_paused {
                    shared.set_capturing(false);
                }
                self.held_from_paused = false;
            }
            DictationHotkeyMode::Hybrid => {
                if self.held_from_paused {
                    if press_duration_ms <= TAP_THRESHOLD_MS {
                        vlog!("Audio capture latched on (tap)");
                    } else if shared.set_capturing(false) {
                        vlog!("Audio capture stopped (hold release)");
                    }
                } else {
                    shared.set_capturing(false);
                    vlog!("Audio capture stopped (toggle off)");
                }
                self.held_from_paused = false;
            }
        }
    }

    /// Escape while capturing discards the take when the user enabled it.
    pub(crate) fn escape(&mut self, shared: &Shared) -> bool {
        if !shared.escape_cancels()
            || self.ctrl
            || self.shift
            || self.alt
            || self.meta
            || !shared.is_capturing()
        {
            return false;
        }
        shared.command(DictationCommand::Cancel);
        true
    }
}

/// Start the listener thread.
pub(crate) fn spawn_listener(config: HotkeyConfig, escape_cancels: bool, shared: Arc<Shared>) {
    let _ = escape_cancels; // kept for signature stability; flag lives on Shared
    vlog!("Initializing hotkey listener for: {}", config.toggle);
    thread::Builder::new()
        .name("ears-hotkeys".to_string())
        .spawn(move || {
            let mut listener = Listener::new(&config);
            vlog!(
                "hotkey listener ready: toggle={:?} mode={:?} push_to_talk={:?}",
                config.toggle,
                shared.hotkey_mode(),
                config.push_to_talk
            );
            let result = listen_events(move |event| match event {
                EventType::KeyPress(key) | EventType::KeyRelease(key) => {
                    if !shared.is_running() {
                        return;
                    }
                    if matches!(event, EventType::KeyPress(_)) {
                        if key == rdev::Key::Escape && listener.escape(&shared) {
                            return;
                        }
                        listener.press(key, &shared);
                    } else {
                        listener.release(key, &shared);
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
    fn parses_modifier_only_command_and_function_bindings() {
        let right_option = parse_combo("right_option");
        assert!(right_option.alt && right_option.modifier_only());
        assert_eq!(right_option.key, rdev::Key::AltGr);
        let command = parse_combo("cmd+space");
        assert!(command.meta && !command.alt);
        assert_eq!(command.key, rdev::Key::Space);
        assert_eq!(parse_combo("fn").key, rdev::Key::Function);
        assert_eq!(parse_combo("f5").key, rdev::Key::F5);
        assert_eq!(parse_combo("right_cmd").key, rdev::Key::MetaRight);
        assert!(parse_combo("right_cmd").modifier_only());
        assert!(!parse_combo("cmd+space").modifier_only());
    }

    #[test]
    fn push_to_talk_discards_taps_and_chords_and_finishes_real_holds() {
        let start = Instant::now();
        let after = |ms| start + std::time::Duration::from_millis(ms);
        // A hold long enough to speak finishes.
        let mut ptt = PushToTalk::default();
        assert!(ptt.pressed(start, false));
        assert_eq!(ptt.released(after(900)), PttRelease::Finish);
        // A quick tap (like Hex's under-300 ms rule) inserts nothing.
        let mut ptt = PushToTalk::default();
        assert!(ptt.pressed(start, false));
        assert_eq!(ptt.released(after(150)), PttRelease::Discard);
        assert_eq!(
            ptt.released(after(400)),
            PttRelease::Leave,
            "already released"
        );
        // Another key during a modifier-only hold discards it, however long.
        let mut ptt = PushToTalk::default();
        assert!(ptt.pressed(start, false));
        ptt.other_key();
        assert_eq!(ptt.released(after(2000)), PttRelease::Discard);
        // A capture that was already running is not the hold's to stop.
        let mut ptt = PushToTalk::default();
        assert!(!ptt.pressed(start, true));
        ptt.other_key();
        assert_eq!(ptt.released(after(50)), PttRelease::Leave);
        // Keys pressed before any hold are not a chord for the next one.
        let mut ptt = PushToTalk::default();
        ptt.other_key();
        assert!(ptt.pressed(start, false));
        assert_eq!(ptt.released(after(500)), PttRelease::Finish);
    }

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
