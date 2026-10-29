//! Probability slider: pointer preview, one commit per gesture, keyboard steps.
use crate::theme::{Theme, theme};
use gpui::{
    App, Bounds, Context, EventEmitter, FocusHandle, Focusable, InteractiveElement, IntoElement,
    KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement, Pixels,
    Render, Styled, Window, canvas, div, px,
};

pub struct ProbabilitySlider {
    value: f32,
    bounds: Bounds<Pixels>,
    dragging: bool,
    focus: FocusHandle,
}

pub struct SliderCommit(pub f32);
impl EventEmitter<SliderCommit> for ProbabilitySlider {}
impl Focusable for ProbabilitySlider {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

fn pointer_value(x: f32, left: f32, width: f32) -> f32 {
    if width <= 0.0 {
        return 0.5;
    }
    (((x - left) / width * 100.0).round() / 100.0).clamp(0.01, 0.99)
}

fn key_value(value: f32, key: &str) -> Option<f32> {
    let next = match key {
        "left" | "down" => value - 0.01,
        "right" | "up" => value + 0.01,
        "home" => 0.01,
        "end" => 0.99,
        _ => return None,
    };
    Some((next * 100.0).round().clamp(1.0, 99.0) / 100.0)
}

impl ProbabilitySlider {
    pub fn new(value: f32, cx: &mut Context<Self>) -> Self {
        Self {
            value,
            bounds: Bounds::default(),
            dragging: false,
            focus: cx.focus_handle(),
        }
    }
    pub fn set_value(&mut self, value: f32, cx: &mut Context<Self>) {
        self.value = value;
        cx.notify();
    }
    fn preview(&mut self, x: Pixels, cx: &mut Context<Self>) {
        self.value = pointer_value(
            x.into(),
            self.bounds.origin.x.into(),
            self.bounds.size.width.into(),
        );
        cx.notify();
    }
    fn release(&mut self, event: &MouseUpEvent, cx: &mut Context<Self>) {
        if !self.dragging {
            return;
        }
        self.preview(event.position.x, cx);
        self.dragging = false;
        cx.emit(SliderCommit(self.value));
    }
}

impl Render for ProbabilitySlider {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme: Theme = theme(cx).clone();
        let r = &theme.roles;
        let focus_color = r.primary;
        let entity = cx.entity().downgrade();
        div()
            .id("probability-slider")
            .track_focus(&self.focus)
            .tab_index(0)
            .flex()
            .items_center()
            .gap(theme.space(3.0))
            .w(px(300.0))
            .border_1()
            .border_color(Theme::alpha(r.primary, 0.0))
            .rounded(theme.radius.sm())
            .focus(move |style| style.border_color(focus_color))
            .child(
                div()
                    .relative()
                    .flex_1()
                    .h(theme.space(7.0))
                    .cursor_pointer()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseDownEvent, window, cx| {
                            window.focus(&this.focus);
                            this.dragging = true;
                            this.preview(event.position.x, cx);
                            cx.stop_propagation();
                        }),
                    )
                    .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                        if this.dragging {
                            this.preview(event.position.x, cx);
                        }
                    }))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, event, _, cx| this.release(event, cx)),
                    )
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, event, _, cx| this.release(event, cx)),
                    )
                    .child(
                        canvas(
                            move |bounds, _, cx| {
                                let _ = entity.update(cx, |this, _| this.bounds = bounds);
                            },
                            |_, _, _, _| {},
                        )
                        .absolute()
                        .size_full(),
                    )
                    .child(
                        div()
                            .absolute()
                            .left_0()
                            .right_0()
                            .top(theme.space(3.0))
                            .h(theme.space(1.0))
                            .rounded_full()
                            .bg(r.border),
                    )
                    .child(
                        div()
                            .absolute()
                            .left(gpui::relative(self.value))
                            .ml(-theme.space(1.5))
                            .top(theme.space(2.0))
                            .size(theme.space(3.0))
                            .rounded_full()
                            .bg(r.primary),
                    ),
            )
            .child(
                div()
                    .w(px(42.0))
                    .text_size(px(super::FONT_MD))
                    .text_color(r.foreground)
                    .child(format!("{:.2}", self.value)),
            )
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                if let Some(value) = key_value(this.value, &event.keystroke.key) {
                    this.value = value;
                    cx.emit(SliderCommit(value));
                    cx.notify();
                    cx.stop_propagation();
                }
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pointer_mapping_and_keyboard_have_the_same_bounds() {
        assert_eq!(pointer_value(-20.0, 10.0, 100.0), 0.01);
        assert_eq!(pointer_value(60.0, 10.0, 100.0), 0.5);
        assert_eq!(pointer_value(120.0, 10.0, 100.0), 0.99);
        assert_eq!(key_value(0.5, "right"), Some(0.51));
        assert_eq!(key_value(0.99, "up"), Some(0.99));
        assert_eq!(key_value(0.01, "left"), Some(0.01));
        assert_eq!(key_value(0.5, "home"), Some(0.01));
        assert_eq!(key_value(0.5, "end"), Some(0.99));
        assert_eq!(key_value(0.5, "escape"), None);
    }
}
