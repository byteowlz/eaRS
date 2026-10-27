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
        _ => return None,
    })
}

#[derive(Default)]
struct Modifiers {
    ctrl: bool,
    shift: bool,
    alt: bool,
}
impl Modifiers {
    fn update(&mut self, flags: CGEventFlags, callback: &mut impl FnMut(EventType)) {
        for (current, next, key) in [
            (
                &mut self.ctrl,
                flags.contains(CGEventFlags::CGEventFlagControl),
                Key::ControlLeft,
            ),
            (
                &mut self.shift,
                flags.contains(CGEventFlags::CGEventFlagShift),
                Key::ShiftLeft,
            ),
            (
                &mut self.alt,
                flags.contains(CGEventFlags::CGEventFlagAlternate),
                Key::Alt,
            ),
        ] {
            if *current != next {
                *current = next;
                callback(if next {
                    EventType::KeyPress(key)
                } else {
                    EventType::KeyRelease(key)
                });
            }
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
    modifiers.update(flags, callback);
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
