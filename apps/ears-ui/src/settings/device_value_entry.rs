//! Typing a value into a slider's dot-matrix display. Enter applies it through the
//! same snap/clamp and save/apply seam as the slider; Escape or leaving cancels.
//! The field covers the display on its own glass; opening or closing it never edits
//! the scene, so a click that ends an entry still reaches the control it lands on.
use super::NativeSettings;
use super::ranges::Numeric;
use crate::ui::text_input::{TextInput, TextInputEvent};
use gpui::{
    AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement, Styled, Window,
    div,
};

pub(in crate::settings) struct ValueEntry {
    pub setting: Numeric,
    field: Entity<TextInput>,
    invalid: bool,
}

/// The number in a typed entry, in the unit the display shows ("0.7", "3,5 s", "400ms").
pub(in crate::settings) fn parse_entry(text: &str) -> Option<f32> {
    let text = text.trim().replace(',', ".");
    let number: String = text
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+'))
        .collect();
    let rest = text[number.len()..].trim();
    // Only a unit may follow the number, never more digits or another value.
    if rest.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    number.parse::<f32>().ok().filter(|v| v.is_finite())
}

/// The editable form of a value: the display text without its unit.
fn initial_text(display: &str) -> String {
    display
        .split_whitespace()
        .next()
        .filter(|n| parse_entry(n).is_some())
        .unwrap_or("")
        .to_owned()
}

impl NativeSettings {
    /// Returns true when the action was a click/Enter on a slider's display.
    pub(super) fn value_entry_action(
        &mut self,
        id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(setting) = Numeric::find(id) else {
            return false;
        };
        if !self.range_enabled(setting, cx) {
            return true;
        }
        let shown = self
            .device
            .read(cx)
            .snapshot()
            .state
            .controls
            .get(id)
            .and_then(|s| s.value.string().map(initial_text))
            .unwrap_or_default();
        let field = cx.new(|cx| TextInput::new(cx, "").bare().with_text(shown));
        cx.subscribe_in(
            &field,
            window,
            |this, _, event: &TextInputEvent, window, cx| match event {
                TextInputEvent::Submit => this.submit_value_entry(window, cx),
                TextInputEvent::Change => {
                    if let Some(entry) = this.value_entry.as_mut() {
                        entry.invalid = false;
                    }
                    cx.notify();
                }
            },
        )
        .detach();
        let handle = gpui::Focusable::focus_handle(field.read(cx), cx);
        cx.on_blur(&handle, window, |this, _, cx| {
            this.cancel_value_entry(cx);
        })
        .detach();
        field.update(cx, |f, cx| f.select_everything(cx));
        field.read(cx).focus(window);
        self.value_entry = Some(ValueEntry {
            setting,
            field,
            invalid: false,
        });
        cx.notify();
        true
    }
    fn submit_value_entry(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.value_entry.as_mut() else {
            return;
        };
        let Some(value) = parse_entry(entry.field.read(cx).text()) else {
            entry.invalid = true;
            cx.notify();
            return;
        };
        let setting = entry.setting;
        self.value_entry = None;
        self.commit_range(setting, value, cx);
        self.sync(cx);
        self.focus(window, cx);
    }
    pub(super) fn cancel_value_entry(&mut self, cx: &mut Context<Self>) -> bool {
        if self.value_entry.take().is_none() {
            return false;
        }
        cx.notify();
        true
    }
    pub(super) fn value_entry_overlay(&self, cx: &gpui::App) -> Option<gpui::AnyElement> {
        let entry = self.value_entry.as_ref()?;
        let device = self.device.read(cx);
        let (p, s) = (device.projection()?, device.snapshot());
        let (d, c, _) = s.control(entry.setting.id())?;
        let (o, r) = (d.origin, &c.rect);
        let pad = c.matrix.as_ref().map_or(0.4, |m| m.padding_x);
        let corner =
            |x: f64, y: f64| p.project(glam::DVec3::new(x + o[0], y + o[1], r.z + o[2] + 0.03));
        let a = corner(r.x - r.width / 2.0 + pad, r.y + r.height / 2.0)?;
        let b = corner(r.x + r.width / 2.0 - pad, r.y - r.height / 2.0)?;
        let height = (b[1] - a[1]) as f32;
        Some(
            div()
                .absolute()
                .left(gpui::px(a[0] as f32))
                .top(gpui::px(a[1] as f32))
                .w(gpui::px((b[0] - a[0]) as f32))
                .h(gpui::px(height))
                .flex()
                .items_center()
                .justify_center()
                .bg(gpui::rgb(0x0a0908))
                .font_family("Departure Mono")
                .text_size(gpui::px((height * 0.56).clamp(9.0, 22.0)))
                .line_height(gpui::px(height))
                .text_color(gpui::rgb(if entry.invalid { 0xff5a4a } else { 0xff9e52 }))
                .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .child(entry.field.clone())
                .into_any_element(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_values_accept_the_display_unit_and_reject_everything_else() {
        assert_eq!(parse_entry("0.7"), Some(0.7));
        assert_eq!(parse_entry(" 3,5 s "), Some(3.5));
        assert_eq!(parse_entry("400ms"), Some(400.0));
        assert_eq!(parse_entry("-2"), Some(-2.0));
        for bad in ["", "abc", "1.2.3", "3 4", "NaN", "inf", "--1"] {
            assert_eq!(parse_entry(bad), None, "{bad:?}");
        }
    }
    #[test]
    fn the_field_starts_with_the_number_without_its_unit() {
        assert_eq!(initial_text("0.50"), "0.50");
        assert_eq!(initial_text("3.00 s"), "3.00");
        assert_eq!(initial_text("400 ms"), "400");
        assert_eq!(initial_text("Off"), "");
    }
}
