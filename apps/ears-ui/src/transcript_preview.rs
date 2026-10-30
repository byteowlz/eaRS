//! Passive display of provisional text, never a text-entry/action surface.
//! Cursor placement snapshots the pointer once per capture; no AX/caret access.
use crate::{
    app::AppModel,
    config::PreviewPlacement,
    platform,
    presentation::{self, PreviewRect},
    runtime,
};
use gpui::{
    App, AppContext, BorrowAppContext, Bounds, Context, Entity, Global, InteractiveElement,
    IntoElement, ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Window,
    WindowBounds, WindowHandle, WindowKind, WindowOptions, div, point, px, rgb, size,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Owner {
    generation: u64,
    display: u32,
    pid: u32,
}

#[derive(Default)]
struct Ownership {
    owner: Option<Owner>,
    blocked: Option<u64>,
}
impl Ownership {
    fn observe(&mut self, candidate: Option<Owner>) -> bool {
        let Some(candidate) = candidate else {
            self.owner = None;
            return false;
        };
        if self.blocked == Some(candidate.generation) {
            return false;
        }
        if let Some(owner) = &self.owner {
            if owner.generation == candidate.generation && *owner != candidate {
                self.blocked = Some(candidate.generation);
                self.owner = None;
                return false;
            }
        }
        self.blocked = None;
        self.owner = Some(candidate);
        true
    }
    fn invalidate(&mut self, generation: Option<u64>) {
        self.blocked = generation;
        self.owner = None;
    }
}

#[derive(Default)]
struct PreviewManager {
    window: Option<WindowHandle<PreviewView>>,
    ownership: Ownership,
    placement: Option<PreviewPlacement>,
    rect: Option<PreviewRect>,
}
impl Global for PreviewManager {}

pub fn sync(cx: &mut App) {
    if !cx.has_global::<PreviewManager>() {
        cx.set_global(PreviewManager::default());
    }
    cx.update_global::<PreviewManager, _>(|manager, cx| manager.sync(cx));
}

impl PreviewManager {
    fn hide(&mut self, cx: &mut App) {
        if let Some(window) = self.window.take() {
            let _ = window.update(cx, |_, window, _| window.remove_window());
        }
    }
    fn close(&mut self, cx: &mut App) {
        self.hide(cx);
        self.rect = None;
    }
    fn sync(&mut self, cx: &mut App) {
        let model = runtime::model(cx);
        let (overlay, active, generation, synthetic, text) = model.read_with(cx, |m, _| {
            (
                m.ui.overlay.clone(),
                m.is_capturing()
                    && m.engine
                        .as_ref()
                        .map_or(m.configuration_is_preview, |engine| engine.is_capturing()),
                m.capture_generation(),
                m.configuration_is_preview,
                presentation::preview_text(&m.partial),
            )
        });
        if !overlay.enabled
            || overlay.preview == PreviewPlacement::Inline
            || !active
            || !platform::passive_previews_supported()
        {
            self.close(cx);
            self.ownership = Ownership::default();
            return;
        }
        let Some(generation) = generation else {
            self.close(cx);
            return;
        };
        let Some(display) = crate::overlay::target_display(cx) else {
            self.close(cx);
            self.ownership.invalidate(Some(generation));
            return;
        };
        let display_id = u32::from(display.id());
        let pid = if synthetic {
            Some(0)
        } else if ears::frontmost::secure_input_active() {
            None
        } else {
            ears::frontmost::frontmost_app_identity()
                .and_then(|app| app.pid)
                .filter(|pid| *pid > 0)
        };
        let Some(pid) = pid else {
            self.close(cx);
            self.ownership.invalidate(Some(generation));
            return;
        };
        let owner = Owner {
            generation,
            display: display_id,
            pid,
        };
        let is_new = self.ownership.owner.as_ref() != Some(&owner);
        if !self.ownership.observe(Some(owner)) {
            self.close(cx);
            return;
        }
        if is_new || self.placement != Some(overlay.preview) {
            self.close(cx);
            self.placement = Some(overlay.preview);
        }
        if self.rect.is_none() {
            let bounds = display.bounds();
            self.rect = presentation::preview_rect(
                (f32::from(bounds.size.width), f32::from(bounds.size.height)),
                overlay.preview,
                platform::cursor_on_display(display_id),
            );
        }
        let Some(rect) = self.rect else {
            self.close(cx);
            return;
        };
        if text.is_empty() {
            self.hide(cx);
            return;
        }
        if let Some(window) = self.window {
            if window.update(cx, |_, _, cx| cx.notify()).is_ok() {
                return;
            }
            self.window = None;
        }
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                point(px(rect.x), px(rect.y)),
                size(px(rect.width), px(rect.height)),
            ))),
            display_id: Some(display.id()),
            titlebar: None,
            focus: false,
            show: true,
            kind: WindowKind::PopUp,
            is_movable: false,
            is_resizable: false,
            is_minimizable: false,
            window_background: gpui::WindowBackgroundAppearance::Transparent,
            app_id: Some("dev.byteowlz.ears-ui.preview".into()),
            ..Default::default()
        };
        if let Ok(handle) = cx.open_window(options, |window, cx| {
            platform::prepare_overlay_panel(window);
            platform::set_click_through(window, true);
            cx.new(|_| PreviewView {
                model: model.clone(),
                rect,
                scroll: gpui::ScrollHandle::new(),
            })
        }) {
            let positioned = handle
                .update(cx, |_, window, _| {
                    platform::position_preview(window, display_id, rect.x, rect.y)
                })
                .unwrap_or(false);
            self.window = Some(handle);
            if !positioned {
                self.close(cx);
                self.ownership.invalidate(Some(generation));
            }
        }
    }
}

struct PreviewView {
    model: Entity<AppModel>,
    rect: PreviewRect,
    scroll: gpui::ScrollHandle,
}
impl Render for PreviewView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (text, font, placement) = self.model.read_with(cx, |m, _| {
            (
                presentation::preview_text(&m.partial),
                m.ui.overlay.font,
                m.ui.overlay.preview,
            )
        });
        let subtitle = placement == PreviewPlacement::Subtitle;
        let size = if subtitle { 22.0 } else { 18.0 };
        let height =
            presentation::preview_height(window.text_system(), &text, font, size, self.rect.width)
                .min(self.rect.height);
        self.scroll.scroll_to_bottom();
        div().size_full().relative().overflow_hidden().child(
            div()
                .absolute()
                .top(px(self.rect.body_top(height) - self.rect.y))
                .left(px(0.0))
                .w_full()
                .h(px(height))
                .px(px(20.0))
                .py(px(14.0))
                .rounded(px(if subtitle { 16.0 } else { 14.0 }))
                .bg(rgb(0x000000))
                .text_color(rgb(0xf4f4f4))
                .font_family(font.family())
                .text_size(px(size))
                .line_height(px(size * 1.4))
                .text_align(if subtitle {
                    gpui::TextAlign::Center
                } else {
                    gpui::TextAlign::Left
                })
                .child(
                    div()
                        .id("preview-text")
                        .h(px(height - 28.0))
                        .overflow_y_scroll()
                        .track_scroll(&self.scroll)
                        .child(SharedString::from(text)),
                ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn owner(generation: u64, pid: u32, display: u32) -> Option<Owner> {
        Some(Owner {
            generation,
            pid,
            display,
        })
    }
    #[test]
    fn changed_application_blocks_same_capture_even_when_returning() {
        let mut guard = Ownership::default();
        assert!(guard.observe(owner(1, 11, 1)));
        assert!(!guard.observe(owner(1, 12, 1)));
        assert!(!guard.observe(owner(1, 11, 1)));
        assert!(guard.observe(owner(2, 11, 1)));
    }
    #[test]
    fn changed_display_and_unknown_context_do_not_reanchor_old_text() {
        let mut guard = Ownership::default();
        assert!(guard.observe(owner(1, 11, 1)));
        assert!(!guard.observe(owner(1, 11, 2)));
        assert!(!guard.observe(owner(1, 11, 1)));
        guard.invalidate(Some(1));
        assert!(!guard.observe(owner(1, 11, 1)));
        assert!(guard.observe(owner(3, 11, 1)));
    }
}
