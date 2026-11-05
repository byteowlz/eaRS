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

#[path = "hotkey_edit.rs"]
mod edit;
pub use edit::{
    HotkeyController, HotkeyRecording, HotkeySuspension, parse_combo_checked, validate_hotkeys,
};
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

/// Legacy permissive config parser. Preserve token order, ignored unknown words
/// and aggregate left-side prefixes for existing configs; editor input must use
/// [`parse_combo_checked`] instead. An absent trigger remains `Unknown(0)`.
pub fn parse_combo(s: &str) -> Combo {
    let mut combo = Combo {
        ctrl: false,
        shift: false,
        alt: false,
        meta: false,
        key: rdev::Key::Unknown(0),
    };
    for part in s.to_lowercase().split('+') {
        // Historically these aliases only set a flag, even after a trigger.
        let part = match part.trim() {
            "left_ctrl" => "ctrl",
            "left_shift" => "shift",
            "left_alt" | "left_option" => "alt",
            "left_cmd" | "left_meta" => "cmd",
            part => part,
        };
        if apply_modifier(part, &mut combo) {
            continue;
        }
        if let Some(key) = named_key(part) {
            combo.key = key;
        }
    }
    combo
}

/// Prefer the editor's explicit-side bindings, falling back to legacy syntax
/// at startup instead of silently disabling a previously accepted shortcut.
fn runtime_combo(input: &str) -> Combo {
    parse_combo_checked(input).unwrap_or_else(|_| parse_combo(input))
}

/// Aggregate modifier words. Right-hand bindings also set their flag, so a
/// `right_option` hold matches with Option held.
fn apply_modifier(part: &str, combo: &mut Combo) -> bool {
    match part {
        "ctrl" | "control" => combo.ctrl = true,
        "left_ctrl" => {
            combo.ctrl = true;
            combo.key = rdev::Key::ControlLeft;
        }
        "right_ctrl" => {
            combo.ctrl = true;
            combo.key = rdev::Key::ControlRight;
        }
        "shift" => combo.shift = true,
        "left_shift" => {
            combo.shift = true;
            combo.key = rdev::Key::ShiftLeft;
        }
        "right_shift" => {
            combo.shift = true;
            combo.key = rdev::Key::ShiftRight;
        }
        "alt" | "option" => combo.alt = true,
        "left_alt" | "left_option" => {
            combo.alt = true;
            combo.key = rdev::Key::Alt;
        }
        "right_alt" | "right_option" | "roption" => {
            combo.alt = true;
            combo.key = rdev::Key::AltGr;
        }
        "cmd" | "command" | "meta" | "super" => combo.meta = true,
        "left_cmd" | "left_meta" => {
            combo.meta = true;
            combo.key = rdev::Key::MetaLeft;
        }
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
    profile_down: bool,
    /// Physical sides are tracked separately; releasing left must not clear right.
    down: Vec<rdev::Key>,
    held_from_paused: bool,
    /// A hold owns only its captured generation, never a later wake/restart.
    toggle_generation: Option<u64>,
    ptt_generation: Option<u64>,
    ptt: PushToTalk,
}

impl Listener {
    pub(crate) fn new(config: &HotkeyConfig) -> Self {
        let combo = |s: &Option<String>| {
            s.as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(runtime_combo)
        };
        Self {
            toggle: runtime_combo(&config.toggle),
            profile_cycle: combo(&config.profile_cycle),
            push_to_talk: combo(&config.push_to_talk),
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
            toggle_down: None,
            profile_down: false,
            down: Vec::new(),
            held_from_paused: false,
            toggle_generation: None,
            ptt_generation: None,
            ptt: PushToTalk::default(),
        }
    }

    /// Returns false for repeats/unpaired releases. Also used while suspended.
    fn track_key(&mut self, key: rdev::Key, down: bool) -> bool {
        let was_down = self.down.contains(&key);
        if was_down == down {
            return false;
        }
        if down {
            self.down.push(key);
        } else {
            self.down.retain(|held| *held != key);
        }
        self.ctrl = self.down.contains(&rdev::Key::ControlLeft)
            || self.down.contains(&rdev::Key::ControlRight);
        self.shift =
            self.down.contains(&rdev::Key::ShiftLeft) || self.down.contains(&rdev::Key::ShiftRight);
        self.alt = self.down.contains(&rdev::Key::Alt) || self.down.contains(&rdev::Key::AltGr);
        self.meta =
            self.down.contains(&rdev::Key::MetaLeft) || self.down.contains(&rdev::Key::MetaRight);
        true
    }

    /// Clear armed actions without forgetting keys still physically held.
    fn disarm(&mut self, shared: &Shared) {
        if self.ptt.started_capture
            && let Some(generation) = self.ptt_generation
        {
            shared.end_hotkey_capture(generation, true);
        }
        if self.held_from_paused
            && let Some(generation) = self.toggle_generation
        {
            shared.end_hotkey_capture(generation, true);
        }
        shared.set_ptt_held(false);
        self.toggle_down = None;
        self.profile_down = false;
        self.held_from_paused = false;
        self.toggle_generation = None;
        self.ptt_generation = None;
        self.ptt = PushToTalk::default();
    }

    /// Modifier-only shortcuts must not fire after they become a typing chord.
    fn disarm_chorded_shortcuts(&mut self, key: rdev::Key, shared: &Shared) {
        // Modifier-only shortcuts are not ordinary typing chords. Once another
        // key joins, don't toggle/cycle on the trailing modifier release.
        if self.toggle.modifier_only() && key != self.toggle.key && self.toggle_down.is_some() {
            if self.held_from_paused
                && let Some(generation) = self.toggle_generation
            {
                shared.end_hotkey_capture(generation, true);
                shared.set_ptt_held(false);
            }
            self.toggle_down = None;
            self.held_from_paused = false;
            self.toggle_generation = None;
        }
        if self
            .profile_cycle
            .is_some_and(|combo| combo.modifier_only() && combo.key != key)
        {
            self.profile_down = false;
        }
    }

    /// A key or modifier went down. `escape` and physical state are handled by
    /// the caller, before matching (modifier-only PTT includes its own flag).
    pub(crate) fn press(&mut self, key: rdev::Key, shared: &Shared) {
        self.disarm_chorded_shortcuts(key, shared);
        // The hold starts before anything else can be read as a chord.
        if let Some(combo) = self.push_to_talk
            && combo.matches(self.ctrl, self.shift, self.alt, self.meta, key)
        {
            let (generation, started) = shared.begin_hotkey_hold();
            self.ptt_generation = Some(generation);
            self.ptt.pressed(Instant::now(), !started);
            if started {
                vlog!("Audio capture started (push to talk)");
            }
            return;
        }
        // Any other key, modifiers included, joined a modifier-only hold: the
        // user is typing a chord, not dictating (Hex's rule).
        if self.ptt.held() && self.push_to_talk.is_some_and(|c| c.modifier_only()) {
            self.ptt.other_key();
        }
        if is_modifier(key) && !self.is_binding_modifier(key) {
            return;
        }
        if self
            .profile_cycle
            .is_some_and(|combo| combo.matches(self.ctrl, self.shift, self.alt, self.meta, key))
        {
            self.profile_down = true;
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
                let (generation, started) = shared.begin_hotkey_hold();
                self.toggle_generation = Some(generation);
                self.held_from_paused = started;
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
        if self.ptt.held() && self.push_to_talk.is_some_and(|c| c.key == key) {
            shared.set_ptt_held(false);
            let release = self.ptt.released(Instant::now());
            let generation = self.ptt_generation.take();
            if let Some(generation) = generation
                && release != PttRelease::Leave
            {
                let applied = shared.end_hotkey_capture(generation, release == PttRelease::Discard);
                vlog!("Push-to-talk release: {release:?}, applied={applied}");
            }
            return;
        }
        // Release the hold guard even if modifiers were released first.
        if key == self.toggle.key {
            shared.set_ptt_held(false);
        }
        if self.profile_down && self.profile_cycle.is_some_and(|cycle| cycle.key == key) {
            self.profile_down = false;
            shared.command(DictationCommand::CycleProfile);
            return;
        }
        if key != self.toggle.key {
            return;
        }
        let Some(pressed_at) = self.toggle_down.take() else {
            return;
        };
        let press_duration_ms = pressed_at.elapsed().as_millis();
        let generation = self.toggle_generation.take();
        match shared.hotkey_mode() {
            DictationHotkeyMode::Toggle => {
                shared.toggle();
            }
            DictationHotkeyMode::PushToTalk => {
                if self.held_from_paused
                    && let Some(generation) = generation
                {
                    shared.end_hotkey_capture(generation, false);
                }
                self.held_from_paused = false;
            }
            DictationHotkeyMode::Hybrid => {
                if self.held_from_paused {
                    if press_duration_ms <= TAP_THRESHOLD_MS {
                        vlog!("Audio capture latched on (tap)");
                    } else if let Some(generation) = generation
                        && shared.end_hotkey_capture(generation, false)
                    {
                        vlog!("Audio capture stopped (hold release)");
                    }
                } else if let Some(generation) = generation {
                    shared.end_hotkey_capture(generation, false);
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
pub(crate) fn spawn_listener(
    mut config: HotkeyConfig,
    escape_cancels: bool,
    shared: Arc<Shared>,
) -> HotkeyController {
    config.escape_cancels = escape_cancels;
    vlog!("Initializing hotkey listener for: {}", config.toggle);
    let controller = HotkeyController::new(config.clone(), &shared);
    let events = controller.clone();
    let lifecycle = controller.clone();
    thread::Builder::new()
        .name("ears-hotkeys".to_string())
        .spawn(move || {
            vlog!(
                "hotkey listener ready: toggle={:?} mode={:?} push_to_talk={:?}",
                config.toggle,
                shared.hotkey_mode(),
                config.push_to_talk
            );
            let callback_shared = shared.clone();
            let result = listen_events(move |event| match event {
                EventType::KeyPress(key) | EventType::KeyRelease(key) => {
                    if !callback_shared.is_running() {
                        return;
                    }
                    events.event(
                        key,
                        matches!(event, EventType::KeyPress(_)),
                        &callback_shared,
                    );
                }
                _ => {}
            });
            if let Err(e) = result {
                eprintln!("Hotkey listener error: {:?}", e);
                lifecycle.listener_failed(e, &shared);
            } else {
                lifecycle.listener_failed("event loop stopped".into(), &shared);
            }
        })
        .expect("failed to spawn hotkey listener thread");
    controller
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_parser_keeps_prefixes_order_and_permissive_defaults() {
        for (legacy, ordinary) in [
            ("left_ctrl+v", "ctrl+v"),
            ("left_option+e", "option+e"),
            ("left_shift+v", "shift+v"),
            ("left_cmd+space", "cmd+space"),
            ("v+ctrl", "ctrl+v"),
            ("v+left_ctrl", "ctrl+v"),
            ("ctrl+ctrl+v", "ctrl+v"),
            ("ctrl+ignored_name+v", "ctrl+v"),
            ("ctrl++v", "ctrl+v"),
        ] {
            assert_eq!(parse_combo(legacy), parse_combo(ordinary), "{legacy}");
        }
        assert_eq!(parse_combo("").key, rdev::Key::Unknown(0));
        assert_eq!(parse_combo("ctrl").key, rdev::Key::Unknown(0));
        assert!(parse_combo("ctrl").ctrl);
        assert_eq!(parse_combo("left_ctrl").key, rdev::Key::Unknown(0));
        assert_eq!(parse_combo("v+b").key, rdev::Key::KeyB);
        assert_eq!(parse_combo("ctrl+v+f13").key, rdev::Key::Unknown(0));
        assert!(parse_combo("ctrl+v+f13").ctrl);
    }

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
