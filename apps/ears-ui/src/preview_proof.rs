//! Opt-in synthetic layout evidence. No pixels, text, identity or production state.
use gpui::{App, BorrowAppContext, Bounds, Global, Pixels, Window};
use std::collections::HashMap;
#[derive(Default)]
struct Reports(HashMap<&'static str, serde_json::Value>);
impl Global for Reports {}

pub fn bounds(
    fixture: bool,
    tag: &'static str,
    state: u32,
) -> impl Fn(Vec<Bounds<Pixels>>, &mut Window, &mut App) + 'static {
    move |bounds, window, cx| {
        if !fixture || !std::env::args().any(|arg| arg == "--preview-layout-report") {
            return;
        }
        let rectangles: Vec<_> = bounds
            .iter()
            .map(|b| {
                [
                    f32::from(b.origin.x),
                    f32::from(b.origin.y),
                    f32::from(b.size.width),
                    f32::from(b.size.height),
                ]
            })
            .collect();
        let viewport = window.viewport_size();
        let value = serde_json::json!({"tag":tag,"state":state,"viewport":[f32::from(viewport.width),f32::from(viewport.height)],"bounds":rectangles});
        if !cx.has_global::<Reports>() {
            cx.set_global(Reports::default());
        }
        cx.update_global::<Reports, _>(|reports, _| {
            if reports.0.get(tag) != Some(&value) {
                eprintln!("ears-preview-layout: {value}");
                reports.0.insert(tag, value);
            }
        });
    }
}
