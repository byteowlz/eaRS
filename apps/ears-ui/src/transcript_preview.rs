//! Passive display of provisional text, never a text-entry/action surface.
//! Pointer following is presentation-only; no AX/caret or insertion-target access.
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
            if owner.generation == candidate.generation && owner.pid != candidate.pid {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreviewStatus {
    #[default]
    Inactive,
    Active,
    Waiting,
    Paused,
    FocusChanged,
    SecureInput,
    UnknownOwner,
    DisplayUnavailable,
    Unavailable,
}
impl PreviewStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Inactive => "No separate preview selected.",
            Self::Active => "Showing provisional text; no text-entry actions.",
            Self::Waiting => "Waiting for this capture's first words.",
            Self::Paused => "Hidden while capture is paused.",
            Self::FocusChanged => "Hidden after focus changed. Start a fresh dictation.",
            Self::SecureInput => "Hidden for Secure Event Input. Start a fresh dictation.",
            Self::UnknownOwner => {
                "Hidden: foreground identity unavailable. Start a fresh dictation."
            }
            Self::DisplayUnavailable => "Hidden: display unavailable. Start a fresh dictation.",
            Self::Unavailable => "Passive previews are unavailable on this platform.",
        }
    }
}
#[derive(Default)]
struct DisplayText(String);
impl DisplayText {
    fn observe(&mut self, text: &str) {
        if !text.trim().is_empty() {
            self.0 = presentation::preview_text(text);
        }
    }
    fn clear(&mut self) {
        self.0.clear();
    }
}

#[derive(Clone, Copy)]
struct Eligibility {
    enabled: bool,
    placement: PreviewPlacement,
    active: bool,
    generation: Option<u64>,
    synthetic: bool,
}
fn eligibility(model: &AppModel) -> Eligibility {
    Eligibility {
        enabled: model.ui.overlay.enabled,
        placement: model.ui.overlay.preview,
        active: model.is_capturing()
            && model
                .engine
                .as_ref()
                .map_or(model.configuration_is_preview, |engine| {
                    engine.is_capturing()
                }),
        generation: model.capture_generation(),
        synthetic: model.configuration_is_preview,
    }
}

fn motion_clock_live(last: Option<std::time::Instant>, now: std::time::Instant) -> bool {
    last.and_then(|last| now.checked_duration_since(last))
        .is_some_and(|age| age <= std::time::Duration::from_millis(50))
}

fn needs_paint(previous: PreviewRect, next: PreviewRect, content_changed: bool) -> bool {
    content_changed
        || previous.width != next.width
        || previous.height != next.height
        || previous.align_bottom != next.align_bottom
}

// Queue at most one main-thread update; never re-enter GPUI from an AppKit callback.
fn motion_frames(display: u32, cx: &App) -> Option<platform::PreviewFrames> {
    let app = cx.to_async();
    let executor = cx.foreground_executor().clone();
    let pending = std::rc::Rc::new(std::cell::Cell::new(false));
    platform::preview_frames(display, move || {
        if pending.replace(true) {
            return;
        }
        let pending = pending.clone();
        let app = app.clone();
        executor
            .spawn(async move {
                pending.set(false);
                let _ = app.update(|cx| update_manager(cx, PreviewManager::follow_frame));
            })
            .detach();
    })
}
fn update_manager(cx: &mut App, update: fn(&mut PreviewManager, &mut App) -> PreviewStatus) {
    if !cx.has_global::<PreviewManager>() {
        cx.set_global(PreviewManager::default());
    }
    let status = cx.update_global::<PreviewManager, _>(|manager, cx| {
        manager.status = update(manager, cx);
        manager.status
    });
    runtime::model(cx).update(cx, |model, cx| {
        if model.preview_status != status {
            model.preview_status = status;
            cx.notify();
        }
    });
}

#[derive(Default)]
struct PreviewManager {
    window: Option<WindowHandle<PreviewView>>,
    frames: Option<platform::PreviewFrames>,
    last_frame: Option<std::time::Instant>,
    ownership: Ownership,
    placement: Option<PreviewPlacement>,
    rect: Option<PreviewRect>,
    text: DisplayText,
    status: PreviewStatus,
}
impl Global for PreviewManager {}

pub fn sync(cx: &mut App) {
    update_manager(cx, PreviewManager::sync);
}

pub fn is_active(cx: &App) -> bool {
    cx.try_global::<PreviewManager>()
        .is_some_and(|manager| manager.window.is_some())
}

impl PreviewManager {
    fn hide(&mut self, cx: &mut App) {
        self.frames = None;
        self.last_frame = None;
        if let Some(window) = self.window.take() {
            let _ = window.update(cx, |_, window, _| window.remove_window());
        }
    }
    fn close(&mut self, cx: &mut App) {
        self.hide(cx);
        self.rect = None;
    }
    fn clear(&mut self, cx: &mut App) {
        self.close(cx);
        self.text.clear();
    }
    fn block(&mut self, generation: u64, cx: &mut App, status: PreviewStatus) -> PreviewStatus {
        self.clear(cx);
        self.ownership.invalidate(Some(generation));
        status
    }
    fn sync(&mut self, cx: &mut App) -> PreviewStatus {
        let model = runtime::model(cx);
        let (eligible, overlay, text) = model.read_with(cx, |m, _| {
            (
                eligibility(m),
                m.ui.overlay.clone(),
                presentation::preview_text(&m.partial),
            )
        });
        let (generation, display) = match self.prepare(eligible, cx) {
            Ok(ready) => ready,
            Err(status) => return status,
        };
        let display_id = u32::from(display.id());
        self.text.observe(&text);
        let bounds = display.bounds();
        let screen = (f32::from(bounds.size.width), f32::from(bounds.size.height));
        let rect = if (overlay.preview == PreviewPlacement::Cursor
            && !motion_clock_live(self.last_frame, std::time::Instant::now()))
            || self.rect.is_none()
        {
            presentation::preview_rect(
                screen,
                overlay.preview,
                platform::cursor_on_display(display_id),
            )
        } else {
            self.rect
        };
        let Some(rect) = rect else {
            return self.block(generation, cx, PreviewStatus::DisplayUnavailable);
        };
        self.rect = Some(rect);
        if self.text.0.is_empty() {
            self.hide(cx);
            return PreviewStatus::Waiting;
        }
        let positioned = self
            .update_window(rect, display_id, cx)
            .unwrap_or_else(|| self.open(model, display, rect, cx));
        if positioned {
            PreviewStatus::Active
        } else {
            self.block(generation, cx, PreviewStatus::DisplayUnavailable)
        }
    }

    fn follow_frame(&mut self, cx: &mut App) -> PreviewStatus {
        if self.window.is_none() || self.placement != Some(PreviewPlacement::Cursor) {
            return self.status;
        }
        self.last_frame = Some(std::time::Instant::now());
        let eligible = runtime::model(cx).read_with(cx, |m, _| eligibility(m));
        let (generation, display) = match self.prepare(eligible, cx) {
            Ok(ready) => ready,
            Err(status) => return status,
        };
        // Screen/placement changes retain text but recreate the passive window and
        // its screen-bound frame clock. Capture/insertion ownership never changes.
        let Some(handle) = self.window else {
            return self.sync(cx);
        };
        let display_id = u32::from(display.id());
        let bounds = display.bounds();
        let rect = presentation::preview_rect(
            (f32::from(bounds.size.width), f32::from(bounds.size.height)),
            PreviewPlacement::Cursor,
            platform::cursor_on_display(display_id),
        );
        let Some(rect) = rect else {
            return self.block(generation, cx, PreviewStatus::DisplayUnavailable);
        };
        let moved = handle
            .update(cx, |view, window, cx| {
                view.move_to(rect, display_id, window, cx)
            })
            .unwrap_or(false);
        if !moved {
            return self.block(generation, cx, PreviewStatus::DisplayUnavailable);
        }
        self.rect = Some(rect);
        PreviewStatus::Active
    }

    fn update_window(&mut self, rect: PreviewRect, display: u32, cx: &mut App) -> Option<bool> {
        let handle = self.window?;
        match handle.update(cx, |view, window, cx| {
            if !view.move_to(rect, display, window, cx) {
                return false;
            }
            view.set_content(&self.text.0, cx);
            true
        }) {
            Ok(positioned) => Some(positioned),
            Err(_) => {
                self.hide(cx);
                None
            }
        }
    }

    /// Capture/security eligibility is separate from moving or creating windows.
    fn prepare(
        &mut self,
        eligible: Eligibility,
        cx: &mut App,
    ) -> Result<(u64, std::rc::Rc<dyn gpui::PlatformDisplay>), PreviewStatus> {
        let Eligibility {
            enabled,
            placement,
            active,
            generation,
            synthetic,
        } = eligible;
        if !enabled || placement == PreviewPlacement::Inline || !active {
            self.clear(cx);
            self.ownership = Ownership::default();
            return Err(if enabled && placement != PreviewPlacement::Inline {
                PreviewStatus::Paused
            } else {
                PreviewStatus::Inactive
            });
        }
        if !platform::passive_previews_supported() {
            self.clear(cx);
            return Err(PreviewStatus::Unavailable);
        }
        let Some(generation) = generation else {
            self.clear(cx);
            return Err(PreviewStatus::UnknownOwner);
        };
        if self.ownership.blocked == Some(generation) {
            return Err(self.status);
        }
        let display = if placement == PreviewPlacement::Subtitle {
            self.ownership
                .owner
                .as_ref()
                .filter(|o| o.generation == generation)
                .and_then(|o| {
                    cx.displays()
                        .into_iter()
                        .find(|d| u32::from(d.id()) == o.display)
                })
                .or_else(|| crate::overlay::target_display(cx))
        } else {
            crate::overlay::target_display(cx)
        };
        let Some(display) = display else {
            return Err(self.block(generation, cx, PreviewStatus::DisplayUnavailable));
        };
        let display_id = u32::from(display.id());
        let pid = if synthetic {
            Some(0)
        } else {
            ears::frontmost::frontmost_app_identity()
                .and_then(|app| app.pid)
                .filter(|pid| *pid > 0)
        };
        // A background app holding Secure Input must not hide the preview everywhere.
        if !synthetic && ears::frontmost::secure_input_protects(pid) {
            return Err(self.block(generation, cx, PreviewStatus::SecureInput));
        }
        let Some(pid) = pid else {
            return Err(self.block(generation, cx, PreviewStatus::UnknownOwner));
        };
        let owner = Owner {
            generation,
            display: display_id,
            pid,
        };
        let fresh = self
            .ownership
            .owner
            .as_ref()
            .is_none_or(|old| old.generation != generation);
        let display_changed = self
            .ownership
            .owner
            .as_ref()
            .is_some_and(|old| old.display != display_id);
        if !self.ownership.observe(Some(owner)) {
            self.clear(cx);
            return Err(PreviewStatus::FocusChanged);
        }
        if fresh {
            self.clear(cx);
        } else if display_changed || self.placement != Some(placement) {
            self.close(cx);
        }
        self.placement = Some(placement);
        Ok((generation, display))
    }

    fn open(
        &mut self,
        model: Entity<AppModel>,
        display: std::rc::Rc<dyn gpui::PlatformDisplay>,
        rect: PreviewRect,
        cx: &mut App,
    ) -> bool {
        let display_id = u32::from(display.id());
        let font = model.read(cx).ui.overlay.font;
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

        let Ok(handle) = cx.open_window(options, |window, cx| {
            platform::prepare_overlay_panel(window);
            platform::set_click_through(window, true);
            cx.new(|_| PreviewView {
                model,
                rect,
                text: self.text.0.clone(),
                font,
                cached_height: None,
                scroll: gpui::ScrollHandle::new(),
            })
        }) else {
            return false;
        };
        let positioned = handle
            .update(cx, |_, window, _| {
                platform::position_preview(window, display_id, rect.x, rect.y)
            })
            .unwrap_or(false);
        self.window = Some(handle);
        if positioned && self.placement == Some(PreviewPlacement::Cursor) {
            self.frames = motion_frames(display_id, cx);
        }
        positioned
    }
}

struct PreviewView {
    model: Entity<AppModel>,
    rect: PreviewRect,
    scroll: gpui::ScrollHandle,
    text: String,
    font: crate::config::OverlayFont,
    cached_height: Option<f32>,
}
impl PreviewView {
    fn move_to(
        &mut self,
        rect: PreviewRect,
        display: u32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.rect == rect {
            return true;
        }
        if self.rect.width != rect.width || self.rect.height != rect.height {
            window.resize(size(px(rect.width), px(rect.height)));
        }
        if !platform::position_preview(window, display, rect.x, rect.y) {
            return false;
        }
        if needs_paint(self.rect, rect, false) {
            self.cached_height = None;
            cx.notify();
        }
        self.rect = rect;
        true
    }
    fn set_content(&mut self, text: &str, cx: &mut Context<Self>) {
        let font = self.model.read(cx).ui.overlay.font;
        if self.text != text || self.font != font {
            self.text.clear();
            self.text.push_str(text);
            self.font = font;
            self.cached_height = None;
            cx.notify();
        }
    }
}
impl Render for PreviewView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (font, placement) = self
            .model
            .read_with(cx, |m, _| (m.ui.overlay.font, m.ui.overlay.preview));
        let text = self.text.clone();
        let subtitle = placement == PreviewPlacement::Subtitle;
        let size = if subtitle { 22.0 } else { 18.0 };
        let height = *self.cached_height.get_or_insert_with(|| {
            presentation::preview_height(window.text_system(), &text, font, size, self.rect.width)
                .min(self.rect.height)
        });
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
#[path = "cursor_follow_tests.rs"]
mod cursor_follow_tests;

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
    fn empty_asr_boundaries_keep_only_this_capture_text() {
        let mut text = DisplayText::default();
        text.observe("one segment");
        text.observe("");
        assert_eq!(text.0, "one segment");
        text.observe("next words");
        assert_eq!(text.0, "next words");
        text.clear();
        text.observe("");
        assert!(text.0.is_empty());
        text.observe(&"ä".repeat(500));
        assert_eq!(text.0.chars().count(), 384);
    }

    #[test]
    fn ownership_accepts_pointer_displays_but_fences_focus_and_unknown_context() {
        for (pid, display, accepted) in [(11, 2, true), (12, 1, false)] {
            let mut guard = Ownership::default();
            assert!(guard.observe(owner(1, 11, 1)));
            assert_eq!(guard.observe(owner(1, pid, display)), accepted);
            if accepted {
                assert!(guard.observe(owner(1, 11, 1)));
                guard.invalidate(Some(1));
            }
            assert!(!guard.observe(owner(1, 11, 1)));
            assert!(guard.observe(owner(3, 11, 1)));
        }
    }
}
