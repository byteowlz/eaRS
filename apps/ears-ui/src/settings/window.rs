//! Frameless host interaction; microphone/shortcut effects stay in their adapters.
use crate::platform;
use gpui::{KeyDownEvent, MouseMoveEvent, Window, prelude::*};

#[derive(Default)]
pub(super) struct Drag {
    anchor: Option<((f32, f32), (f32, f32))>,
}
impl Drag {
    pub fn begin(&mut self, window: &Window) {
        self.anchor = None;
        if !platform::start_native_drag(window) {
            self.anchor = platform::mouse_screen_position().zip(platform::window_origin(window));
        }
    }
    pub fn update(&mut self, event: &MouseMoveEvent, window: &Window) {
        if !event.dragging() {
            self.anchor = None;
            return;
        }
        if let (Some((start, origin)), Some(mouse)) =
            (self.anchor, platform::mouse_screen_position())
        {
            platform::set_window_origin(
                window,
                origin.0 + mouse.0 - start.0,
                origin.1 + mouse.1 - start.1,
            );
        }
    }
    pub fn end(&mut self) {
        self.anchor = None;
    }
}

pub(super) fn close_key(event: &KeyDownEvent) -> bool {
    let mods = event.keystroke.modifiers;
    let primary = if cfg!(target_os = "macos") {
        mods.platform
    } else {
        mods.control
    };
    event.keystroke.key == "w" && primary && !mods.alt && !mods.shift && !event.is_held
}

impl super::SettingsView {
    pub(super) fn window_shell(&self, cx: &gpui::Context<Self>) -> gpui::Div {
        gpui::div()
            .on_key_down(|event, window, cx| {
                if close_key(event) {
                    window.remove_window();
                    cx.stop_propagation();
                } else {
                    super::cycle_settings_focus(event, window, cx);
                }
            })
            .on_mouse_move(
                cx.listener(|this, event, window, _| this.window_drag.update(event, window)),
            )
            .on_mouse_up(
                gpui::MouseButton::Left,
                cx.listener(|this, _, _, _| this.window_drag.end()),
            )
            .on_mouse_up_out(
                gpui::MouseButton::Left,
                cx.listener(|this, _, _, _| this.window_drag.end()),
            )
    }

    pub(super) fn window_header(
        &self,
        tabs: gpui::AnyElement,
        theme: &crate::theme::Theme,
        cx: &gpui::Context<Self>,
    ) -> gpui::Div {
        gpui::div()
            .flex_shrink_0()
            .px(theme.space(5.0))
            .py(theme.space(3.0))
            .border_b_1()
            .border_color(theme.roles.border)
            .flex()
            .items_center()
            .gap(theme.space(2.0))
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(|this, _, window, _| this.window_drag.begin(window)),
            )
            .child(tabs)
            .child(
                crate::ui::button(
                    "settings-close",
                    "Close",
                    crate::ui::ButtonKind::Ghost,
                    theme,
                )
                .flex_shrink_0()
                .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .on_click(|_, window, _| window.remove_window()),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn close_chord_is_not_plain_w_or_a_shifted_recorder_chord() {
        let mut event = KeyDownEvent {
            keystroke: gpui::Keystroke::parse("w").unwrap(),
            is_held: false,
        };
        assert!(!close_key(&event));
        if cfg!(target_os = "macos") {
            event.keystroke.modifiers.platform = true;
        } else {
            event.keystroke.modifiers.control = true;
        }
        assert!(close_key(&event));
        event.keystroke.modifiers.shift = true;
        assert!(!close_key(&event));
    }
}
