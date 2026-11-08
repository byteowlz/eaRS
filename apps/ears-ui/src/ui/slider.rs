//! Range slider: pointer preview, one commit per gesture, keyboard steps.
//! A saved value outside the visible range is shown as-is and only changes
//! when the user actually moves the slider.
use crate::theme::{Theme, theme};
use gpui::{
    App, Bounds, Context, EventEmitter, FocusHandle, Focusable, InteractiveElement, IntoElement,
    KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement, Pixels,
    Render, Styled, Window, canvas, div, px,
};

/// Range, step and display of one slider.
#[derive(Clone, Copy)]
pub struct SliderSpec {
    pub id: &'static str,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub format: fn(f32) -> String,
}

impl SliderSpec {
    pub const PROBABILITY: Self = Self {
        id: "probability-slider",
        min: 0.01,
        max: 0.99,
        step: 0.01,
        format: |v| format!("{v:.2}"),
    };

    /// Snap to the step grid inside the range, free of float noise.
    pub(crate) fn snap(&self, value: f32) -> f32 {
        let steps = ((value - self.min) / self.step).round();
        let snapped = (self.min + steps * self.step).clamp(self.min, self.max);
        (snapped * 10_000.0).round() / 10_000.0
    }

    /// Thumb position 0..=1; out-of-range values pin to an end.
    fn fraction(&self, value: f32) -> f32 {
        if self.max <= self.min {
            return 0.0;
        }
        ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }

    fn pointer_value(&self, x: f32, left: f32, width: f32) -> f32 {
        if width <= 0.0 {
            return self.snap((self.min + self.max) / 2.0);
        }
        let fraction = ((x - left) / width).clamp(0.0, 1.0);
        self.snap(self.min + fraction * (self.max - self.min))
    }

    fn key_value(&self, value: f32, key: &str) -> Option<f32> {
        let current = self.snap(value);
        let next = match key {
            "left" | "down" => current - self.step,
            "right" | "up" => current + self.step,
            "pagedown" => current - 10.0 * self.step,
            "pageup" => current + 10.0 * self.step,
            "home" => self.min,
            "end" => self.max,
            _ => return None,
        };
        Some(self.snap(next))
    }
}

pub struct RangeSlider {
    spec: SliderSpec,
    value: f32,
    bounds: Bounds<Pixels>,
    dragging: bool,
    focus: FocusHandle,
}

pub struct SliderCommit(pub f32);
impl EventEmitter<SliderCommit> for RangeSlider {}
impl Focusable for RangeSlider {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl RangeSlider {
    pub fn new(spec: SliderSpec, value: f32, cx: &mut Context<Self>) -> Self {
        Self {
            spec,
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
        self.value = self.spec.pointer_value(
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

impl Render for RangeSlider {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme: Theme = theme(cx).clone();
        let r = &theme.roles;
        let focus_color = r.primary;
        let entity = cx.entity().downgrade();
        div()
            .id(self.spec.id)
            .track_focus(&self.focus)
            .tab_index(0)
            .flex()
            .items_center()
            .gap(theme.space(3.0))
            .w(px(super::FORM_CONTROL_WIDTH))
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
                            .left(gpui::relative(self.spec.fraction(self.value)))
                            .ml(-theme.space(1.5))
                            .top(theme.space(2.0))
                            .size(theme.space(3.0))
                            .rounded_full()
                            .bg(r.primary),
                    ),
            )
            .child(
                div()
                    .w(px(64.0))
                    .flex_shrink_0()
                    .text_size(px(super::FONT_MD))
                    .text_color(r.foreground)
                    .child((self.spec.format)(self.value)),
            )
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                if let Some(value) = this.spec.key_value(this.value, &event.keystroke.key) {
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
    const P: SliderSpec = SliderSpec::PROBABILITY;
    const LINGER: SliderSpec = SliderSpec {
        id: "linger",
        min: 0.0,
        max: 5000.0,
        step: 50.0,
        format: |v| format!("{v:.0} ms"),
    };
    const OPACITY: SliderSpec = SliderSpec {
        id: "opacity",
        min: 0.2,
        max: 1.0,
        step: 0.01,
        format: |v| format!("{v:.2}"),
    };

    #[test]
    fn probability_behavior_is_unchanged() {
        assert_eq!(P.pointer_value(-20.0, 10.0, 100.0), 0.01);
        assert_eq!(P.pointer_value(60.0, 10.0, 100.0), 0.5);
        assert_eq!(P.pointer_value(120.0, 10.0, 100.0), 0.99);
        assert_eq!(P.key_value(0.5, "right"), Some(0.51));
        assert_eq!(P.key_value(0.99, "up"), Some(0.99));
        assert_eq!(P.key_value(0.01, "left"), Some(0.01));
        assert_eq!(P.key_value(0.5, "home"), Some(0.01));
        assert_eq!(P.key_value(0.5, "end"), Some(0.99));
        assert_eq!(P.key_value(0.5, "escape"), None);
    }

    #[test]
    fn ranges_snap_to_their_step_without_float_noise() {
        assert_eq!(LINGER.pointer_value(10.0, 10.0, 100.0), 0.0);
        assert_eq!(LINGER.pointer_value(28.0, 10.0, 100.0), 900.0);
        assert_eq!(LINGER.key_value(900.0, "right"), Some(950.0));
        assert_eq!(LINGER.key_value(900.0, "pageup"), Some(1400.0));
        assert_eq!(LINGER.key_value(50.0, "pagedown"), Some(0.0));
        assert_eq!(OPACITY.key_value(0.96, "right"), Some(0.97));
        assert_eq!(OPACITY.key_value(0.2, "left"), Some(0.2));
        assert_eq!(OPACITY.pointer_value(110.0, 10.0, 100.0), 1.0);
        assert_eq!((LINGER.format)(900.0), "900 ms");
    }

    #[test]
    fn saved_values_outside_the_range_pin_the_thumb_until_moved() {
        // A hand-edited 12 s linger is displayed, never silently rewritten.
        assert_eq!(LINGER.fraction(12_000.0), 1.0);
        assert_eq!((LINGER.format)(12_000.0), "12000 ms");
        assert_eq!(LINGER.fraction(-5.0), 0.0);
        // Moving it brings it into range by one step from the visible end.
        assert_eq!(LINGER.key_value(12_000.0, "left"), Some(4950.0));
    }
}
