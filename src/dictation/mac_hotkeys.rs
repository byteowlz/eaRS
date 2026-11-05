//! Listen to macOS physical hotkeys without translating typed characters.
//!
//! rdev 0.5 translates every key through HIToolbox before invoking its callback;
//! on some macOS input sources that traps on its background listener thread.
//! eaRS only needs key identity and modifiers, never the characters typed.
use core_foundation::runloop::CFRunLoop;
use core_graphics::event::{
    CGEvent, CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement,
    CGEventType, CallbackResult, EventField, KeyCode,
};
use rdev::{EventType, Key};
use std::cell::RefCell;

fn key(code: u16) -> Option<Key> {
    Some(match code {
        KeyCode::ANSI_A => Key::KeyA,
        KeyCode::ANSI_B => Key::KeyB,
        KeyCode::ANSI_C => Key::KeyC,
        KeyCode::ANSI_D => Key::KeyD,
        KeyCode::ANSI_E => Key::KeyE,
        KeyCode::ANSI_F => Key::KeyF,
        KeyCode::ANSI_G => Key::KeyG,
        KeyCode::ANSI_H => Key::KeyH,
        KeyCode::ANSI_I => Key::KeyI,
        KeyCode::ANSI_J => Key::KeyJ,
        KeyCode::ANSI_K => Key::KeyK,
        KeyCode::ANSI_L => Key::KeyL,
        KeyCode::ANSI_M => Key::KeyM,
        KeyCode::ANSI_N => Key::KeyN,
        KeyCode::ANSI_O => Key::KeyO,
        KeyCode::ANSI_P => Key::KeyP,
        KeyCode::ANSI_Q => Key::KeyQ,
        KeyCode::ANSI_R => Key::KeyR,
        KeyCode::ANSI_S => Key::KeyS,
        KeyCode::ANSI_T => Key::KeyT,
        KeyCode::ANSI_U => Key::KeyU,
        KeyCode::ANSI_V => Key::KeyV,
        KeyCode::ANSI_W => Key::KeyW,
        KeyCode::ANSI_X => Key::KeyX,
        KeyCode::ANSI_Y => Key::KeyY,
        KeyCode::ANSI_Z => Key::KeyZ,
        KeyCode::ANSI_0 => Key::Num0,
        KeyCode::ANSI_1 => Key::Num1,
        KeyCode::ANSI_2 => Key::Num2,
        KeyCode::ANSI_3 => Key::Num3,
        KeyCode::ANSI_4 => Key::Num4,
        KeyCode::ANSI_5 => Key::Num5,
        KeyCode::ANSI_6 => Key::Num6,
        KeyCode::ANSI_7 => Key::Num7,
        KeyCode::ANSI_8 => Key::Num8,
        KeyCode::ANSI_9 => Key::Num9,
        KeyCode::SPACE => Key::Space,
        KeyCode::TAB => Key::Tab,
        KeyCode::ESCAPE => Key::Escape,
        36 => Key::Return,
        122 => Key::F1,
        120 => Key::F2,
        99 => Key::F3,
        118 => Key::F4,
        96 => Key::F5,
        97 => Key::F6,
        98 => Key::F7,
        100 => Key::F8,
        101 => Key::F9,
        109 => Key::F10,
        103 => Key::F11,
        111 => Key::F12,
        _ => return None,
    })
}

/// Modifier keys are identified by the keycode carried by `FlagsChanged`
/// events, which is how left/right (and Fn) stay distinguishable. The
/// aggregate flags reconcile a missed press or release.
#[derive(Default)]
struct Modifiers {
    down: Vec<(Key, CGEventFlags)>,
}

/// Physical keycode of a modifier key, with the flag that says it is held.
fn modifier_key(code: u16) -> Option<(Key, CGEventFlags)> {
    Some(match code {
        58 => (Key::Alt, CGEventFlags::CGEventFlagAlternate),
        61 => (Key::AltGr, CGEventFlags::CGEventFlagAlternate),
        55 => (Key::MetaLeft, CGEventFlags::CGEventFlagCommand),
        54 => (Key::MetaRight, CGEventFlags::CGEventFlagCommand),
        59 => (Key::ControlLeft, CGEventFlags::CGEventFlagControl),
        62 => (Key::ControlRight, CGEventFlags::CGEventFlagControl),
        56 => (Key::ShiftLeft, CGEventFlags::CGEventFlagShift),
        60 => (Key::ShiftRight, CGEventFlags::CGEventFlagShift),
        63 => (Key::Function, CGEventFlags::CGEventFlagSecondaryFn),
        _ => return None,
    })
}

impl Modifiers {
    fn emit(&mut self, key: Key, down: bool, callback: &mut impl FnMut(EventType)) {
        if down {
            if !self.down.iter().any(|(k, _)| *k == key) {
                self.down.push((key, KeyFlags::of(key)));
                callback(EventType::KeyPress(key));
            }
        } else if let Some(index) = self.down.iter().position(|(k, _)| *k == key) {
            self.down.remove(index);
            callback(EventType::KeyRelease(key));
        }
    }

    /// A `FlagsChanged` event names the exact modifier key.
    fn changed(&mut self, code: u16, flags: CGEventFlags, callback: &mut impl FnMut(EventType)) {
        if let Some((key, bit)) = modifier_key(code) {
            // Flags are aggregate across both sides: releasing right Option
            // while left is held still has Alternate set. FlagsChanged names
            // the side that transitioned, so toggle that side while its family
            // remains on; a cleared family always means release.
            let was_down = self.down.iter().any(|(held, _)| *held == key);
            let down = flags.contains(bit) && !was_down;
            self.emit(key, down, callback);
        }
        self.reconcile(flags, callback);
    }

    /// Release anything the flags say is no longer held and press a missing
    /// left-hand modifier, so a lost event cannot leave a modifier stuck.
    fn reconcile(&mut self, flags: CGEventFlags, callback: &mut impl FnMut(EventType)) {
        for (key, bit) in self.down.clone() {
            if !flags.contains(bit) {
                self.emit(key, false, callback);
            }
        }
        for (key, bit) in [
            (Key::ControlLeft, CGEventFlags::CGEventFlagControl),
            (Key::ShiftLeft, CGEventFlags::CGEventFlagShift),
            (Key::Alt, CGEventFlags::CGEventFlagAlternate),
            (Key::MetaLeft, CGEventFlags::CGEventFlagCommand),
            (Key::Function, CGEventFlags::CGEventFlagSecondaryFn),
        ] {
            // Only when no key of that modifier is tracked: a right-hand press
            // must not also produce the left-hand fallback.
            if flags.contains(bit) && !self.down.iter().any(|(k, _)| KeyFlags::of(*k) == bit) {
                self.emit(key, true, callback);
            }
        }
    }
}

struct KeyFlags;
impl KeyFlags {
    fn of(key: Key) -> CGEventFlags {
        match key {
            Key::Alt | Key::AltGr => CGEventFlags::CGEventFlagAlternate,
            Key::MetaLeft | Key::MetaRight => CGEventFlags::CGEventFlagCommand,
            Key::ControlLeft | Key::ControlRight => CGEventFlags::CGEventFlagControl,
            Key::ShiftLeft | Key::ShiftRight => CGEventFlags::CGEventFlagShift,
            Key::Function => CGEventFlags::CGEventFlagSecondaryFn,
            _ => CGEventFlags::CGEventFlagNull,
        }
    }
}

fn dispatch(
    kind: CGEventType,
    code: u16,
    flags: CGEventFlags,
    modifiers: &mut Modifiers,
    callback: &mut impl FnMut(EventType),
) {
    if matches!(kind, CGEventType::FlagsChanged) {
        modifiers.changed(code, flags, callback);
    } else {
        modifiers.reconcile(flags, callback);
    }
    if let Some(key) = key(code) {
        match kind {
            CGEventType::KeyDown => callback(EventType::KeyPress(key)),
            CGEventType::KeyUp => callback(EventType::KeyRelease(key)),
            _ => {}
        }
    }
}

fn dispatch_event(
    kind: CGEventType,
    event: &CGEvent,
    modifiers: &mut Modifiers,
    callback: &mut impl FnMut(EventType),
) {
    if !matches!(
        kind,
        CGEventType::KeyDown | CGEventType::KeyUp | CGEventType::FlagsChanged
    ) {
        return;
    }
    if let Ok(code) =
        u16::try_from(event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE))
    {
        dispatch(kind, code, event.get_flags(), modifiers, callback);
    }
}

pub(super) fn listen(callback: impl FnMut(EventType) + 'static) -> Result<(), String> {
    let state = RefCell::new((Modifiers::default(), callback));
    CGEventTap::with_enabled(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![
            CGEventType::KeyDown,
            CGEventType::KeyUp,
            CGEventType::FlagsChanged,
        ],
        move |_, kind, event| {
            if matches!(
                kind,
                CGEventType::KeyDown | CGEventType::KeyUp | CGEventType::FlagsChanged
            ) && let Ok(mut state) = state.try_borrow_mut()
            {
                let (modifiers, callback) = &mut *state;
                dispatch_event(kind, event, modifiers, callback);
            }
            CallbackResult::Keep
        },
        CFRunLoop::run_current,
    )
    .map_err(|_| {
        "cannot create macOS hotkey event tap; check Input Monitoring permissions".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_supported_hotkey_keycode_maps_correctly() {
        let pairs = [
            (KeyCode::ANSI_A, Key::KeyA),
            (KeyCode::ANSI_B, Key::KeyB),
            (KeyCode::ANSI_C, Key::KeyC),
            (KeyCode::ANSI_D, Key::KeyD),
            (KeyCode::ANSI_E, Key::KeyE),
            (KeyCode::ANSI_F, Key::KeyF),
            (KeyCode::ANSI_G, Key::KeyG),
            (KeyCode::ANSI_H, Key::KeyH),
            (KeyCode::ANSI_I, Key::KeyI),
            (KeyCode::ANSI_J, Key::KeyJ),
            (KeyCode::ANSI_K, Key::KeyK),
            (KeyCode::ANSI_L, Key::KeyL),
            (KeyCode::ANSI_M, Key::KeyM),
            (KeyCode::ANSI_N, Key::KeyN),
            (KeyCode::ANSI_O, Key::KeyO),
            (KeyCode::ANSI_P, Key::KeyP),
            (KeyCode::ANSI_Q, Key::KeyQ),
            (KeyCode::ANSI_R, Key::KeyR),
            (KeyCode::ANSI_S, Key::KeyS),
            (KeyCode::ANSI_T, Key::KeyT),
            (KeyCode::ANSI_U, Key::KeyU),
            (KeyCode::ANSI_V, Key::KeyV),
            (KeyCode::ANSI_W, Key::KeyW),
            (KeyCode::ANSI_X, Key::KeyX),
            (KeyCode::ANSI_Y, Key::KeyY),
            (KeyCode::ANSI_Z, Key::KeyZ),
            (KeyCode::ANSI_0, Key::Num0),
            (KeyCode::ANSI_1, Key::Num1),
            (KeyCode::ANSI_2, Key::Num2),
            (KeyCode::ANSI_3, Key::Num3),
            (KeyCode::ANSI_4, Key::Num4),
            (KeyCode::ANSI_5, Key::Num5),
            (KeyCode::ANSI_6, Key::Num6),
            (KeyCode::ANSI_7, Key::Num7),
            (KeyCode::ANSI_8, Key::Num8),
            (KeyCode::ANSI_9, Key::Num9),
            (KeyCode::SPACE, Key::Space),
            (KeyCode::TAB, Key::Tab),
            (KeyCode::ESCAPE, Key::Escape),
            (36, Key::Return),
            (122, Key::F1),
            (120, Key::F2),
            (99, Key::F3),
            (118, Key::F4),
            (96, Key::F5),
            (97, Key::F6),
            (98, Key::F7),
            (100, Key::F8),
            (101, Key::F9),
            (109, Key::F10),
            (103, Key::F11),
            (111, Key::F12),
        ];
        for (code, expected) in pairs {
            assert_eq!(key(code), Some(expected));
        }
    }

    #[test]
    fn unposted_core_graphics_event_uses_the_real_decode_path() {
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
        let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState).unwrap();
        let event = CGEvent::new_keyboard_event(source, KeyCode::ANSI_V, true).unwrap();
        event.set_flags(CGEventFlags::CGEventFlagControl | CGEventFlags::CGEventFlagShift);
        let mut modifiers = Modifiers::default();
        let mut events = Vec::new();
        dispatch_event(CGEventType::KeyDown, &event, &mut modifiers, &mut |e| {
            events.push(e)
        });
        assert_eq!(
            events,
            [
                EventType::KeyPress(Key::ControlLeft),
                EventType::KeyPress(Key::ShiftLeft),
                EventType::KeyPress(Key::KeyV)
            ]
        );
        // The synthetic event is never posted to the OS or delivered to other apps.
    }

    #[test]
    fn physical_hotkey_events_do_not_require_text_translation() {
        let mut modifiers = Modifiers::default();
        let mut events = Vec::new();
        let both = CGEventFlags::CGEventFlagControl | CGEventFlags::CGEventFlagShift;
        dispatch(
            CGEventType::FlagsChanged,
            KeyCode::CONTROL,
            CGEventFlags::CGEventFlagControl,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        dispatch(
            CGEventType::FlagsChanged,
            KeyCode::SHIFT,
            both,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        dispatch(
            CGEventType::KeyDown,
            KeyCode::ANSI_V,
            both,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        dispatch(
            CGEventType::KeyUp,
            KeyCode::ANSI_V,
            both,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        dispatch(
            CGEventType::FlagsChanged,
            KeyCode::SHIFT,
            CGEventFlags::CGEventFlagControl,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        dispatch(
            CGEventType::FlagsChanged,
            KeyCode::CONTROL,
            CGEventFlags::CGEventFlagNull,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        assert_eq!(
            events,
            [
                EventType::KeyPress(Key::ControlLeft),
                EventType::KeyPress(Key::ShiftLeft),
                EventType::KeyPress(Key::KeyV),
                EventType::KeyRelease(Key::KeyV),
                EventType::KeyRelease(Key::ShiftLeft),
                EventType::KeyRelease(Key::ControlLeft)
            ]
        );
    }
    const FN_KEY: u16 = 63;
    const RIGHT_OPTION: u16 = 61;
    const LEFT_OPTION: u16 = 58;
    const RIGHT_COMMAND: u16 = 54;
    const LEFT_SHIFT: u16 = 56;

    #[test]
    fn modifier_sides_command_and_fn_arrive_from_their_keycodes() {
        let mut modifiers = Modifiers::default();
        let mut events = Vec::new();
        let mut cb = |e| events.push(e);
        // Right Option alone is what a modifier-only binding names.
        modifiers.changed(RIGHT_OPTION, CGEventFlags::CGEventFlagAlternate, &mut cb);
        modifiers.changed(RIGHT_OPTION, CGEventFlags::CGEventFlagNull, &mut cb);
        modifiers.changed(RIGHT_COMMAND, CGEventFlags::CGEventFlagCommand, &mut cb);
        modifiers.changed(RIGHT_COMMAND, CGEventFlags::CGEventFlagNull, &mut cb);
        modifiers.changed(FN_KEY, CGEventFlags::CGEventFlagSecondaryFn, &mut cb);
        modifiers.changed(FN_KEY, CGEventFlags::CGEventFlagNull, &mut cb);
        assert_eq!(
            events,
            [
                EventType::KeyPress(Key::AltGr),
                EventType::KeyRelease(Key::AltGr),
                EventType::KeyPress(Key::MetaRight),
                EventType::KeyRelease(Key::MetaRight),
                EventType::KeyPress(Key::Function),
                EventType::KeyRelease(Key::Function),
            ]
        );
    }

    #[test]
    fn releasing_one_modifier_side_does_not_release_or_repress_the_other() {
        let mut modifiers = Modifiers::default();
        let mut events = Vec::new();
        let alt = CGEventFlags::CGEventFlagAlternate;
        for (code, flags) in [
            (LEFT_OPTION, alt),
            (RIGHT_OPTION, alt),
            (RIGHT_OPTION, alt),
            (LEFT_OPTION, CGEventFlags::CGEventFlagNull),
        ] {
            modifiers.changed(code, flags, &mut |event| events.push(event));
        }
        assert_eq!(
            events,
            [
                EventType::KeyPress(Key::Alt),
                EventType::KeyPress(Key::AltGr),
                EventType::KeyRelease(Key::AltGr),
                EventType::KeyRelease(Key::Alt),
            ]
        );
        assert!(modifiers.down.is_empty());
    }

    #[test]
    fn left_modifiers_keep_their_existing_keys_and_lost_events_reconcile() {
        let mut modifiers = Modifiers::default();
        let mut events = Vec::new();
        let mut cb = |e| events.push(e);
        modifiers.changed(LEFT_OPTION, CGEventFlags::CGEventFlagAlternate, &mut cb);
        modifiers.changed(LEFT_SHIFT, CGEventFlags::CGEventFlagShift, &mut cb);
        // The release event is lost; the next key event's flags reconcile it.
        modifiers.reconcile(CGEventFlags::CGEventFlagNull, &mut cb);
        assert_eq!(
            events,
            [
                EventType::KeyPress(Key::Alt),
                EventType::KeyPress(Key::ShiftLeft),
                EventType::KeyRelease(Key::Alt),
                EventType::KeyRelease(Key::ShiftLeft),
            ]
        );
        // A tap that reports a held modifier without a FlagsChanged still
        // yields the left-hand key.
        let mut modifiers = Modifiers::default();
        let mut events = Vec::new();
        let mut cb = |e| events.push(e);
        modifiers.reconcile(CGEventFlags::CGEventFlagAlternate, &mut cb);
        assert_eq!(events, [EventType::KeyPress(Key::Alt)]);
    }

    #[test]
    fn flags_are_reconciled_on_key_events_and_unknown_keys_are_ignored() {
        let mut modifiers = Modifiers::default();
        let mut events = Vec::new();
        dispatch(
            CGEventType::KeyDown,
            KeyCode::SPACE,
            CGEventFlags::CGEventFlagAlternate,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        dispatch(
            CGEventType::KeyUp,
            KeyCode::SPACE,
            CGEventFlags::CGEventFlagAlternate,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        dispatch(
            CGEventType::KeyDown,
            65535,
            CGEventFlags::CGEventFlagNull,
            &mut modifiers,
            &mut |e| events.push(e),
        );
        assert_eq!(
            events,
            [
                EventType::KeyPress(Key::Alt),
                EventType::KeyPress(Key::Space),
                EventType::KeyRelease(Key::Space),
                EventType::KeyRelease(Key::Alt)
            ]
        );
    }
}
