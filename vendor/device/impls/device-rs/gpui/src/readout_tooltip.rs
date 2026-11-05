//! Exact semantic value for clipped matrix text and disclosed Unicode fallback.
use gpui::{Context, SharedString, Window, div, prelude::*, px, rgb};
pub(crate) struct ReadoutTooltip(pub String, pub SharedString);
impl Render for ReadoutTooltip {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .max_w(px(480.0))
            .p(px(10.0))
            .bg(rgb(0x171a1e))
            .text_color(rgb(0xf0f0f0))
            .text_size(px(13.0))
            .font_family(self.1.clone())
            .child(self.0.clone())
    }
}
