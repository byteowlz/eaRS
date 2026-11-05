//! Validated, live hotkey edits and scoped shortcut-recorder suspension.
//!
//! The event callback and editors share one lock: when `suspend` returns no
//! hotkey action can still be in flight. Physical keys continue to be tracked
//! while suspended, and actions resume only after all recorder keys are up.
//! Persist a validated candidate before calling `reconfigure`; neither method
//! writes config files. Keep the suspension guard until save/cancel/close.

use super::{Combo, Listener, Shared, apply_modifier, named_key};
use crate::config::HotkeyConfig;
use anyhow::{Result, bail};
use std::sync::{Arc, Mutex, Weak};

/// Strict parser for editable shortcuts. Generic prefixes match either side;
/// an explicit side may only be the final (trigger) key, never a misleading
/// side-specific prefix. Modifier-only shortcuts must name a side (or `fn`).
pub fn parse_combo_checked(input: &str) -> Result<Combo> {
    let input = input.trim().to_ascii_lowercase();
    let mut combo = Combo {
        ctrl: false,
        shift: false,
        alt: false,
        meta: false,
        key: rdev::Key::Unknown(0),
    };
    let parts: Vec<_> = input.split('+').map(str::trim).collect();
    for (index, part) in parts.iter().enumerate() {
        if part.is_empty() {
            bail!("Shortcut must not be empty or contain an empty '+' component");
        }
        let previous = combo;
        if apply_modifier(part, &mut combo) {
            if (previous.ctrl, previous.shift, previous.alt, previous.meta)
                == (combo.ctrl, combo.shift, combo.alt, combo.meta)
            {
                bail!("Repeated modifier '{part}'");
            }
        } else {
            combo.key = named_key(part)
                .filter(|key| !matches!(key, rdev::Key::Unknown(_)))
                .ok_or_else(|| anyhow::anyhow!("Unsupported shortcut key '{part}'"))?;
        }
        if index + 1 < parts.len() && combo.key != rdev::Key::Unknown(0) {
            bail!(
                "The trigger key must be last; use generic modifier prefixes such as 'ctrl' or 'option'"
            );
        }
    }
    if combo.key == rdev::Key::Unknown(0) {
        bail!("Choose a trigger key or an explicit modifier side (for example 'right_option')");
    }
    Ok(combo)
}

/// Validate active dictation bindings and alias-equivalent conflicts. Empty
/// optional bindings disable that action. `language_cycle` is legacy config:
/// the dictation listener does not implement it, so it is not an active conflict.
/// This checks eaRS conflicts only, not OS/global shortcuts registered elsewhere.
pub fn validate_hotkeys(config: &HotkeyConfig) -> Result<()> {
    let mut parsed = Vec::new();
    for (name, text) in [
        ("toggle", Some(config.toggle.as_str())),
        ("push_to_talk", config.push_to_talk.as_deref()),
        ("profile_cycle", config.profile_cycle.as_deref()),
    ] {
        let Some(text) = text else { continue };
        if name != "toggle" && text.trim().is_empty() {
            continue;
        }
        let combo =
            parse_combo_checked(text).map_err(|error| anyhow::anyhow!("{name}: {error}"))?;
        if combo.key == rdev::Key::Escape
            && !combo.ctrl
            && !combo.shift
            && !combo.alt
            && !combo.meta
            && config.escape_cancels
        {
            bail!("{name}: Escape is reserved for cancelling dictation");
        }
        if let Some((other, _)) = parsed.iter().find(|(_, binding)| *binding == combo) {
            bail!("{name} conflicts with {other}");
        }
        parsed.push((name, combo));
    }
    Ok(())
}

struct State {
    config: HotkeyConfig,
    listener: Listener,
    suspensions: usize,
    wait_for_release: bool,
    listener_error: Option<String>,
    recording: Option<Recording>,
}

#[derive(Default)]
struct Recording {
    armed: bool,
    candidate: Option<String>,
}

/// A recorder reads physical keys from the existing listener, including modifier
/// sides. It never starts another event tap or grants input permission.
pub struct HotkeyRecording {
    guard: HotkeySuspension,
}
impl HotkeyRecording {
    pub fn candidate(&self) -> Option<String> {
        self.guard
            .controller
            .state
            .lock()
            .unwrap()
            .recording
            .as_ref()
            .and_then(|r| r.candidate.clone())
    }
}
impl Drop for HotkeyRecording {
    fn drop(&mut self) {
        self.guard.controller.state.lock().unwrap().recording = None;
        // The guard then drains held keys before actions resume.
    }
}

fn recorded_combo(key: rdev::Key, held: &[rdev::Key]) -> Option<String> {
    use rdev::Key;
    let trigger = match key {
        Key::Alt => "left_option".into(),
        Key::AltGr => "right_option".into(),
        Key::ControlLeft => "left_ctrl".into(),
        Key::ControlRight => "right_ctrl".into(),
        Key::ShiftLeft => "left_shift".into(),
        Key::ShiftRight => "right_shift".into(),
        Key::MetaLeft => "left_cmd".into(),
        Key::MetaRight => "right_cmd".into(),
        Key::Function => "fn".into(),
        _ => (b'a'..=b'z')
            .chain(b'0'..=b'9')
            .map(|c| (c as char).to_string())
            .chain(["space", "tab", "escape", "return"].map(str::to_owned))
            .chain((1..=12).map(|n| format!("f{n}")))
            .find(|name| named_key(name) == Some(key))?,
    };
    let mut parts = Vec::new();
    for (name, family) in [
        ("ctrl", [Key::ControlLeft, Key::ControlRight]),
        ("shift", [Key::ShiftLeft, Key::ShiftRight]),
        ("option", [Key::Alt, Key::AltGr]),
        ("cmd", [Key::MetaLeft, Key::MetaRight]),
    ] {
        if !family.contains(&key) && held.iter().any(|k| family.contains(k)) {
            parts.push(name.to_owned());
        }
    }
    // Fn as a prefix cannot be represented by the engine's binding contract.
    if key != Key::Function
        && held.contains(&Key::Function)
        && !matches!(
            key,
            Key::F1
                | Key::F2
                | Key::F3
                | Key::F4
                | Key::F5
                | Key::F6
                | Key::F7
                | Key::F8
                | Key::F9
                | Key::F10
                | Key::F11
                | Key::F12
        )
    {
        return None;
    }
    // macOS also sets its aggregate Fn flag for ordinary function-key events;
    // that flag must not make an otherwise supported F-key unrecordable.
    parts.push(trigger);
    Some(parts.join("+"))
}

/// Handle to one running listener; cloning it does not start another OS tap.
/// The engine may retain this handle without a reference cycle.
#[derive(Clone)]
pub struct HotkeyController {
    state: Arc<Mutex<State>>,
    shared: Weak<Shared>,
}

impl HotkeyController {
    pub(super) fn new(config: HotkeyConfig, shared: &Arc<Shared>) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                listener: Listener::new(&config),
                config,
                suspensions: 0,
                wait_for_release: false,
                listener_error: None,
                recording: None,
            })),
            shared: Arc::downgrade(shared),
        }
    }

    /// Snapshot of listener bindings and effective runtime flags. Do not use
    /// this as a disk config: runtime/CLI overrides may differ from persisted settings.
    pub fn config(&self) -> HotkeyConfig {
        let mut config = self.state.lock().unwrap().config.clone();
        if let Some(shared) = self.shared.upgrade() {
            config.mode = shared.hotkey_mode();
            config.escape_cancels = shared.escape_cancels();
        }
        config
    }

    /// Atomically replace the three dictation bindings only. Mode, Escape policy,
    /// language_cycle and enable_internal are deliberately not changed: shortcut
    /// editing must not overwrite CLI overrides or engine-owned safety policy.
    /// Use the existing handle commands for mode/Escape changes. Invalid input
    /// leaves both config and armed actions untouched. No disk persistence.
    /// Already-held keys cannot trigger either the old or the new binding.
    pub fn reconfigure(&self, config: &HotkeyConfig) -> Result<()> {
        validate_hotkeys(config)?;
        let shared = self
            .shared
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("Hotkey listener is unavailable"))?;
        if !shared.is_running() {
            bail!("Hotkey listener has stopped");
        }
        let mut state = self.state.lock().unwrap();
        if let Some(error) = &state.listener_error {
            bail!("Hotkey listener is unavailable: {error}");
        }
        let mut candidate = state.config.clone();
        candidate.toggle = config.toggle.clone();
        candidate.push_to_talk = config.push_to_talk.clone();
        candidate.profile_cycle = config.profile_cycle.clone();
        candidate.escape_cancels = shared.escape_cancels();
        validate_hotkeys(&candidate)?;
        state.listener.disarm(&shared);
        let mut listener = Listener::new(&candidate);
        for key in &state.listener.down {
            listener.track_key(*key, true);
        }
        state.wait_for_release = !listener.down.is_empty();
        state.listener = listener;
        state.config = candidate;
        Ok(())
    }

    /// Suspend all listener actions (including Escape/profile cycling) before
    /// focusing the shortcut recorder. Guards nest and resume safely on Drop.
    /// An in-progress hotkey-owned hold is discarded, not pasted. A capture
    /// started elsewhere is not paused; this suspends shortcuts, not the engine.
    #[must_use = "keep the guard alive for the entire shortcut recording/edit"]
    pub fn suspend(&self) -> HotkeySuspension {
        let mut state = self.state.lock().unwrap();
        if let Some(shared) = self.shared.upgrade() {
            state.listener.disarm(&shared);
        }
        state.suspensions += 1;
        HotkeySuspension {
            controller: self.clone(),
        }
    }

    /// Begin one physical recorder after synchronously suspending all actions.
    /// Keys already held at activation must be released before a new chord counts.
    pub fn record(&self) -> Result<HotkeyRecording> {
        let shared = self
            .shared
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("Hotkey listener is unavailable"))?;
        let mut state = self.state.lock().unwrap();
        if !shared.is_running() || state.listener_error.is_some() {
            bail!("Hotkey listener is unavailable; check input permission and restart dictation");
        }
        if state.recording.is_some() {
            bail!("Another shortcut recorder is active");
        }
        state.listener.disarm(&shared);
        state.recording = Some(Recording {
            armed: state.listener.down.is_empty(),
            candidate: None,
        });
        state.suspensions += 1;
        Ok(HotkeyRecording {
            guard: HotkeySuspension {
                controller: self.clone(),
            },
        })
    }

    pub(super) fn listener_failed(&self, error: String, shared: &Shared) {
        let mut state = self.state.lock().unwrap();
        state.listener.disarm(shared);
        state.listener_error = Some(error);
    }

    pub(super) fn event(&self, key: rdev::Key, down: bool, shared: &Shared) {
        let mut state = self.state.lock().unwrap();
        let held_before = state.listener.down.clone();
        if !state.listener.track_key(key, down) {
            return;
        }
        let held = state.listener.down.clone();
        if let Some(recording) = &mut state.recording {
            if recording.armed {
                if down && !super::is_modifier(key) {
                    recording.candidate = recorded_combo(key, &held);
                } else if !down && super::is_modifier(key) && recording.candidate.is_none() {
                    recording.candidate = recorded_combo(key, &held_before);
                }
            } else if held.is_empty() {
                recording.armed = true;
            }
        }
        if state.suspensions != 0
            || state.wait_for_release
            || !state.config.enable_internal
            || shared
                .hotkey_cancel_pending
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            state.wait_for_release = !state.listener.down.is_empty();
            return;
        }
        if down {
            if key == rdev::Key::Escape && state.listener.escape(shared) {
                state.listener.disarm(shared);
                return;
            }
            state.listener.press(key, shared);
        } else {
            state.listener.release(key, shared);
        }
    }
}

/// Scoped recorder suspension. Dropping the final guard waits for key-up before
/// rearming; cancellation, window closure and save must all drop their guard.
pub struct HotkeySuspension {
    controller: HotkeyController,
}

impl Drop for HotkeySuspension {
    fn drop(&mut self) {
        let mut state = self.controller.state.lock().unwrap();
        state.suspensions -= 1;
        state.wait_for_release = !state.listener.down.is_empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::AppConfig, dictation::DictationCommand};
    use rdev::Key;

    fn fixture(
        config: HotkeyConfig,
    ) -> (
        HotkeyController,
        Arc<Shared>,
        crossbeam_channel::Receiver<DictationCommand>,
    ) {
        let mut app = AppConfig::default();
        app.hotkeys = config.clone();
        app.dictation.notifications.enabled = false;
        let (events, _) = crossbeam_channel::unbounded();
        let (commands, rx) = crossbeam_channel::unbounded();
        let shared = Arc::new(Shared::new(
            &app,
            config.escape_cancels,
            events,
            commands,
            false,
        ));
        (HotkeyController::new(config, &shared), shared, rx)
    }

    fn tap(controller: &HotkeyController, shared: &Shared, key: Key) {
        controller.event(key, true, shared);
        controller.event(key, false, shared);
    }

    #[test]
    fn recorder_uses_physical_sides_suspends_actions_and_drains_activation_keys() {
        let (control, shared, _) = fixture(HotkeyConfig::default());
        control.event(Key::Return, true, &shared);
        let recording = control.record().unwrap();
        assert!(control.record().is_err());
        control.event(Key::Return, false, &shared);
        assert_eq!(recording.candidate(), None);
        tap(&control, &shared, Key::AltGr);
        assert_eq!(recording.candidate().as_deref(), Some("right_option"));
        control.event(Key::ControlLeft, true, &shared);
        control.event(Key::ShiftRight, true, &shared);
        tap(&control, &shared, Key::KeyV);
        assert_eq!(recording.candidate().as_deref(), Some("ctrl+shift+v"));
        assert!(!shared.is_capturing());
        drop(recording);
        tap(&control, &shared, Key::KeyV);
        assert!(
            !shared.is_capturing(),
            "held modifiers still drain after recorder closes"
        );
        control.event(Key::ControlLeft, false, &shared);
        control.event(Key::ShiftRight, false, &shared);
        assert!(control.record().is_ok());
    }

    #[test]
    fn recorder_representations_roundtrip_without_losing_sides() {
        for key in [
            Key::ControlLeft,
            Key::ControlRight,
            Key::Alt,
            Key::AltGr,
            Key::ShiftLeft,
            Key::ShiftRight,
            Key::MetaLeft,
            Key::MetaRight,
            Key::Function,
            Key::Return,
            Key::F12,
            Key::KeyZ,
        ] {
            let text = recorded_combo(key, &[key]).unwrap();
            assert_eq!(parse_combo_checked(&text).unwrap().key, key, "{text}");
        }
        assert!(recorded_combo(Key::KeyA, &[Key::Function, Key::KeyA]).is_none());
        assert_eq!(
            recorded_combo(Key::F6, &[Key::Function, Key::F6]).as_deref(),
            Some("f6")
        );
    }

    #[test]
    fn legacy_startup_chords_remain_live_without_editor_admission() {
        for (binding, modifier, key) in [
            ("left_ctrl+v", Key::ControlRight, Key::KeyV),
            ("left_option+e", Key::AltGr, Key::KeyE),
            ("v+ctrl", Key::ControlLeft, Key::KeyV),
            ("v+left_ctrl", Key::ControlRight, Key::KeyV),
        ] {
            assert!(parse_combo_checked(binding).is_err());
            let config = HotkeyConfig {
                toggle: binding.into(),
                ..Default::default()
            };
            let (control, shared, _) = fixture(config);
            control.event(modifier, true, &shared);
            tap(&control, &shared, key);
            assert!(
                shared.is_capturing(),
                "legacy binding {binding} must still work at startup"
            );
        }
    }

    #[test]
    fn rejects_unknown_partial_ambiguous_and_repeated_input() {
        for text in [
            "",
            "ctrl",
            "option",
            "ctrl++v",
            "ctrl+typo",
            "f13",
            "v+b",
            "right_option+v",
            "ctrl+control+v",
            "ctrl+v+shift",
        ] {
            assert!(parse_combo_checked(text).is_err(), "{text}");
        }
        for text in [
            "ctrl+shift+v",
            "cmd+space",
            "right_option",
            "left_option",
            "right_ctrl",
            "left_ctrl",
            "right_cmd",
            "left_cmd",
            "left_shift",
            "right_shift",
            "fn",
            "f12",
            "ctrl+right_option",
        ] {
            assert!(parse_combo_checked(text).is_ok(), "{text}");
        }
    }

    #[test]
    fn validates_alias_conflicts_optional_empty_and_escape() {
        let mut config = HotkeyConfig::default();
        config.push_to_talk = Some("control+shift+V".into());
        assert!(
            validate_hotkeys(&config)
                .unwrap_err()
                .to_string()
                .contains("conflicts")
        );
        config.push_to_talk = Some(" ".into());
        assert!(validate_hotkeys(&config).is_ok());
        config.toggle = "right_option".into();
        config.profile_cycle = Some("roption".into());
        assert!(validate_hotkeys(&config).is_err());
        config.profile_cycle = Some("left_option".into());
        assert!(validate_hotkeys(&config).is_ok());
        config.toggle = "escape".into();
        config.escape_cancels = true;
        assert!(validate_hotkeys(&config).is_err());
    }

    #[test]
    fn unavailable_listener_does_not_claim_a_live_update() {
        let config = HotkeyConfig::default();
        let (control, shared, _) = fixture(config.clone());
        control.listener_failed("input permission denied".into(), &shared);
        let candidate = HotkeyConfig {
            toggle: "f6".into(),
            ..config
        };
        assert!(
            control
                .reconfigure(&candidate)
                .unwrap_err()
                .to_string()
                .contains("input permission denied")
        );
        assert_eq!(control.config().toggle, "ctrl+shift+v");
    }

    #[test]
    fn invalid_reconfiguration_preserves_binding_and_armed_press() {
        let config = HotkeyConfig {
            toggle: "f5".into(),
            ..Default::default()
        };
        let (control, shared, _) = fixture(config.clone());
        control.event(Key::F5, true, &shared);
        let mut invalid = config;
        invalid.toggle = "ctrl+unknown".into();
        assert!(control.reconfigure(&invalid).is_err());
        assert_eq!(control.config().toggle, "f5");
        control.event(Key::F5, false, &shared);
        assert!(shared.is_capturing());
    }

    #[test]
    fn live_reconfiguration_waits_for_neutral_and_uses_new_binding() {
        let mut config = HotkeyConfig {
            toggle: "f5".into(),
            ..Default::default()
        };
        let (control, shared, _) = fixture(config.clone());
        control.event(Key::F5, true, &shared);
        config.toggle = "f6".into();
        control.reconfigure(&config).unwrap();
        tap(&control, &shared, Key::F6);
        control.event(Key::F5, false, &shared);
        assert!(!shared.is_capturing());
        tap(&control, &shared, Key::F5);
        assert!(!shared.is_capturing());
        tap(&control, &shared, Key::F6);
        assert!(shared.is_capturing());
    }

    #[test]
    fn nested_recorders_suppress_all_actions_and_trailing_release() {
        let config = HotkeyConfig {
            toggle: "f5".into(),
            profile_cycle: Some("f6".into()),
            escape_cancels: true,
            ..Default::default()
        };
        let (control, shared, commands) = fixture(config);
        shared.set_capturing(true);
        let first = control.suspend();
        let second = control.suspend();
        tap(&control, &shared, Key::Escape);
        tap(&control, &shared, Key::F6);
        drop(first);
        tap(&control, &shared, Key::F5);
        assert!(shared.is_capturing());
        assert!(commands.try_recv().is_err());
        shared.set_capturing(false);
        control.event(Key::F5, true, &shared);
        drop(second);
        control.event(Key::F5, false, &shared);
        assert!(!shared.is_capturing());
        tap(&control, &shared, Key::F5);
        assert!(shared.is_capturing());
    }

    #[test]
    fn modifier_only_ptt_starts_on_correct_side_and_repeat_does_not_rearm() {
        let config = HotkeyConfig {
            push_to_talk: Some("right_option".into()),
            ..Default::default()
        };
        let (control, shared, commands) = fixture(config);
        tap(&control, &shared, Key::Alt);
        assert!(!shared.is_capturing());
        control.event(Key::AltGr, true, &shared);
        assert!(shared.is_capturing());
        assert!(shared.ptt_held());
        let started = control.state.lock().unwrap().listener.ptt.started;
        control.event(Key::AltGr, true, &shared);
        assert_eq!(control.state.lock().unwrap().listener.ptt.started, started);
        control.event(Key::AltGr, false, &shared);
        assert!(!shared.ptt_held());
        assert!(matches!(
            commands.try_recv(),
            Ok(DictationCommand::CancelHotkey { .. })
        ));
    }

    #[test]
    fn recorder_does_not_cancel_a_newer_capture_after_an_owned_hold_ended() {
        for mode in [
            None,
            Some(crate::config::DictationHotkeyMode::PushToTalk),
            Some(crate::config::DictationHotkeyMode::Hybrid),
        ] {
            let mut config = HotkeyConfig {
                toggle: "f5".into(),
                ..Default::default()
            };
            let key = if let Some(mode) = mode {
                config.mode = mode;
                Key::F5
            } else {
                config.push_to_talk = Some("right_option".into());
                Key::AltGr
            };
            let (control, shared, commands) = fixture(config);
            control.event(key, true, &shared);
            assert!(shared.is_capturing());
            shared.set_capturing(false);
            shared.set_capturing(true); // A wake/other source owns a new capture.
            let newer_generation = shared.capture_generation();
            let _guard = control.suspend();
            assert!(
                shared.is_capturing(),
                "suspending {mode:?} must leave the newer capture alone"
            );
            assert_eq!(shared.capture_generation(), newer_generation);
            assert!(
                commands.try_recv().is_err(),
                "no stale Cancel may be queued"
            );
            assert!(
                !shared
                    .hotkey_cancel_pending
                    .load(std::sync::atomic::Ordering::SeqCst)
            );
        }
    }

    #[test]
    fn stale_hold_release_or_reconfiguration_leaves_newer_capture_alone() {
        for mode in [
            None,
            Some(crate::config::DictationHotkeyMode::PushToTalk),
            Some(crate::config::DictationHotkeyMode::Hybrid),
        ] {
            for action in ["tap_release", "hold_release", "reconfigure"] {
                let mut config = HotkeyConfig {
                    toggle: "f5".into(),
                    ..Default::default()
                };
                let key = if let Some(mode) = mode {
                    config.mode = mode;
                    Key::F5
                } else {
                    config.push_to_talk = Some("right_option".into());
                    Key::AltGr
                };
                let (control, shared, commands) = fixture(config.clone());
                control.event(key, true, &shared);
                if action == "hold_release" {
                    let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
                    let mut state = control.state.lock().unwrap();
                    state.listener.ptt.started = state.listener.ptt.started.map(|_| past);
                    state.listener.toggle_down = state.listener.toggle_down.map(|_| past);
                }
                shared.set_capturing(false);
                shared.set_capturing(true);
                let generation = shared.capture_generation();
                if action == "reconfigure" {
                    config.toggle = "f6".into();
                    control.reconfigure(&config).unwrap();
                } else {
                    control.event(key, false, &shared);
                }
                assert!(shared.is_capturing(), "{mode:?}: {action}");
                assert_eq!(shared.capture_generation(), generation);
                assert!(
                    commands.try_recv().is_err(),
                    "{mode:?}: {action} must not queue stale Cancel"
                );
            }
        }
    }

    #[test]
    fn recorder_discards_owned_hold_synchronously_before_engine_ack() {
        let config = HotkeyConfig {
            push_to_talk: Some("right_option".into()),
            ..Default::default()
        };
        let (control, shared, commands) = fixture(config);
        control.event(Key::AltGr, true, &shared);
        assert!(shared.is_capturing());
        let guard = control.suspend();
        assert!(
            !shared.is_capturing(),
            "capture must stop before queued Cancel is handled"
        );
        assert!(!shared.ptt_held());
        assert!(
            shared
                .hotkey_cancel_pending
                .load(std::sync::atomic::Ordering::SeqCst)
        );
        assert!(matches!(
            commands.try_recv(),
            Ok(DictationCommand::CancelHotkey { .. })
        ));
        drop(guard);
        control.event(Key::AltGr, false, &shared);
        assert!(
            commands.try_recv().is_err(),
            "release must not finalize or cancel twice"
        );
        assert!(!shared.is_capturing());
        tap(&control, &shared, Key::AltGr);
        assert!(
            !shared.is_capturing(),
            "do not rearm before delayed Cancel acknowledgement"
        );
        shared
            .hotkey_cancel_pending
            .store(false, std::sync::atomic::Ordering::SeqCst);
        control.event(Key::AltGr, true, &shared);
        assert!(
            shared.is_capturing(),
            "fresh press works after acknowledgement"
        );
    }

    #[test]
    fn reconfiguration_discards_owned_hold_without_finalizing_it() {
        let mut config = HotkeyConfig {
            toggle: "f5".into(),
            mode: crate::config::DictationHotkeyMode::PushToTalk,
            ..Default::default()
        };
        let (control, shared, commands) = fixture(config.clone());
        control.event(Key::F5, true, &shared);
        assert!(shared.is_capturing());
        config.toggle = "f6".into();
        control.reconfigure(&config).unwrap();
        assert!(!shared.is_capturing());
        assert!(!shared.ptt_held());
        assert!(matches!(
            commands.try_recv(),
            Ok(DictationCommand::CancelHotkey { .. })
        ));
        control.event(Key::F5, false, &shared);
        assert!(commands.try_recv().is_err());
    }

    #[test]
    fn binding_edits_preserve_runtime_overrides_and_escape_policy() {
        let mut config = HotkeyConfig {
            toggle: "f5".into(),
            ..Default::default()
        };
        let (control, shared, _) = fixture(config.clone());
        shared.set_escape_cancels(true);
        config.enable_internal = false;
        config.mode = crate::config::DictationHotkeyMode::PushToTalk;
        config.toggle = "f6".into();
        control.reconfigure(&config).unwrap();
        assert!(control.config().enable_internal);
        assert!(shared.escape_cancels());
        assert_eq!(
            shared.hotkey_mode(),
            crate::config::DictationHotkeyMode::Toggle
        );
        tap(&control, &shared, Key::F6);
        assert!(shared.is_capturing());
        config.toggle = "escape".into();
        assert!(
            control.reconfigure(&config).is_err(),
            "effective Escape policy wins over saved config"
        );
        assert_eq!(control.config().toggle, "f6");
    }

    #[test]
    fn chorded_modifier_toggle_is_not_a_shortcut() {
        let config = HotkeyConfig {
            toggle: "right_option".into(),
            ..Default::default()
        };
        let (control, shared, _) = fixture(config);
        control.event(Key::AltGr, true, &shared);
        tap(&control, &shared, Key::KeyV);
        control.event(Key::AltGr, false, &shared);
        assert!(!shared.is_capturing());
    }

    #[test]
    fn releasing_one_side_keeps_other_side_modifier_active() {
        let config = HotkeyConfig {
            toggle: "ctrl+v".into(),
            ..Default::default()
        };
        let (control, shared, _) = fixture(config);
        control.event(Key::ControlLeft, true, &shared);
        control.event(Key::ControlRight, true, &shared);
        control.event(Key::ControlLeft, false, &shared);
        tap(&control, &shared, Key::KeyV);
        assert!(shared.is_capturing());
    }

    #[test]
    fn modifier_toggle_and_profile_fire_on_release_without_stuck_flags() {
        let config = HotkeyConfig {
            toggle: "left_option".into(),
            profile_cycle: Some("right_cmd".into()),
            ..Default::default()
        };
        let (control, shared, commands) = fixture(config);
        tap(&control, &shared, Key::AltGr);
        assert!(!shared.is_capturing());
        tap(&control, &shared, Key::Alt);
        assert!(shared.is_capturing());
        tap(&control, &shared, Key::MetaRight);
        assert!(matches!(
            commands.try_recv(),
            Ok(DictationCommand::CycleProfile)
        ));
        control.event(Key::Alt, false, &shared);
        assert!(shared.is_capturing(), "unpaired key-up cannot toggle");
    }
}
