//! The floating dictation island.
//!
//! One borderless, transparent, non-activating `PopUp` window per active
//! display, positioned display-relative at the anchor. The pill inside is
//! drawn by GPUI (rounded, shadowed) and morphs between phases:
//!
//! dormant (dot) → armed (dot + hint) → listening (dot + live partial)
//! → finalizing (spinner → check) → back to dormant after the linger.
//!
//! On macOS with `island_mode` the pill turns pure black and hugs the camera
//! notch (see docs/DYNAMIC_ISLAND_SPIKE.md).

use crate::app::AppModel;
use crate::app::FINALIZE_MS;
use crate::app::Phase;
use crate::config::ContentMotion;
use crate::config::IndicatorStyle;
use crate::config::OverlayAnchor;
use crate::config::OverlayConfig;
use crate::config::ShapeMotion;
use crate::ears_mark;
use crate::indicator_theme::IndicatorPalette;
use crate::parametric_mark::{self, State as MarkState};
use crate::platform;
use crate::platform::Notch;
use crate::runtime;
use crate::theme::Theme;
use crate::theme::theme;
use gpui::Animation;
use gpui::AnimationExt;
use gpui::App;
use gpui::AppContext;
use gpui::Bounds;
use gpui::BoxShadow;
use gpui::Context;
use gpui::Div;
use gpui::ElementId;
use gpui::Entity;
use gpui::Global;
use gpui::Hsla;
use gpui::InteractiveElement;
use gpui::IntoElement;
use gpui::MouseButton;
use gpui::MouseDownEvent;
use gpui::MouseMoveEvent;
use gpui::MouseUpEvent;
use gpui::ParentElement;
use gpui::PathBuilder;
use gpui::Pixels;
use gpui::Render;
use gpui::SharedString;
use gpui::StatefulInteractiveElement;
use gpui::Styled;
use gpui::TextRun;
use gpui::Window;
use gpui::WindowBackgroundAppearance;
use gpui::WindowBounds;
use gpui::WindowHandle;
use gpui::WindowKind;
use gpui::WindowOptions;
use gpui::canvas;
use gpui::div;
use gpui::point;
use gpui::prelude::FluentBuilder;
use gpui::px;
use gpui::size;
use std::time::Duration;
use std::time::Instant;

/// Window size. The pill is centred inside; the margin leaves room for the
/// shadow and for the island to grow.
const WIN_W: f32 = 640.0;
const WIN_H: f32 = 110.0;
/// Regular pill.
const PILL_H: f32 = 36.0;
const PILL_MAX_W: f32 = 560.0;
const PILL_MIN_W: f32 = 44.0;
const DOT_W: f32 = 10.0;
const PAD_X: f32 = 14.0;
const GAP: f32 = 9.0;
const FONT_SIZE: f32 = 13.0;
/// Morph timings (spec: 150-250 ms).
const MORPH_MS: u64 = 200;
const ISLAND_MORPH_MS: u64 = 260;
const CONTENT_MORPH_MS: u64 = 180;
const DRAG_THRESHOLD: f32 = 4.0;
/// Provisional outward throat, shared in shape with LostIsland's host.
const THROAT: f32 = 12.0;

pub struct OverlayManager {
    window: Option<(WindowHandle<IslandView>, u32)>,
    hide_at: Option<Instant>,
    applied_click_through: Option<bool>,
    /// Placement config the open window was built with; a change re-opens
    /// the window on the next tick (drag offsets included).
    applied_placement: Option<String>,
}

impl Global for OverlayManager {}

impl OverlayManager {
    fn get(cx: &mut App) -> &mut OverlayManager {
        if !cx.has_global::<OverlayManager>() {
            cx.set_global(OverlayManager {
                window: None,
                hide_at: None,
                applied_click_through: None,
                applied_placement: None,
            });
        }
        cx.global_mut::<OverlayManager>()
    }
}

pub fn is_active(cx: &App) -> bool {
    cx.try_global::<OverlayManager>()
        .map(|m| m.window.is_some())
        .unwrap_or(false)
}

/// Reconcile the island window with the model. Called every tick.
pub fn sync(cx: &mut App) {
    let model = runtime::model(cx);
    let (overlay, phase, engine_running) = model.read_with(cx, |m, _| {
        (m.ui.overlay.clone(), m.phase(), m.engine_running())
    });

    if !overlay.enabled {
        close(cx);
        return;
    }

    let want_visible = phase != Phase::Dormant || (overlay.dormant_dot && engine_running);
    let has_window = OverlayManager::get(cx).window.is_some();

    if want_visible {
        OverlayManager::get(cx).hide_at = None;
        if !has_window {
            open(cx, &overlay, model.clone());
        } else if phase != Phase::Listening {
            // Follow the cursor between displays while nothing is being
            // dictated; never yank the pill away mid-sentence.
            let current = OverlayManager::get(cx).window.map(|(_, d)| d);
            let target = platform::cursor_display();
            if let (Some(current), Some(target)) = (current, target) {
                if current != target {
                    close(cx);
                    open(cx, &overlay, model.clone());
                }
            }
        }
    } else if has_window {
        let linger = Duration::from_millis(overlay.linger_ms);
        let manager = OverlayManager::get(cx);
        match manager.hide_at {
            None => manager.hide_at = Some(Instant::now() + linger),
            Some(at) if Instant::now() >= at => close(cx),
            _ => {}
        }
    }

    // Displays can disappear (dock/undock): drop windows whose display is
    // gone so they are re-created on a live one next tick.
    let window = OverlayManager::get(cx).window;
    if let Some((handle, display)) = window {
        let alive = cx.displays().iter().any(|d| u32::from(d.id()) == display);
        if !alive {
            let _ = handle.update(cx, |_, window, _| window.remove_window());
            OverlayManager::get(cx).window = None;
            return;
        }
        let (applied, placement) = {
            let manager = OverlayManager::get(cx);
            (
                manager.applied_click_through,
                Some(placement_signature(&overlay)),
            )
        };
        if applied != Some(overlay.click_through) {
            let _ = handle.update(cx, |_, window, _| {
                platform::set_click_through(window, overlay.click_through);
            });
            OverlayManager::get(cx).applied_click_through = Some(overlay.click_through);
        }
        if OverlayManager::get(cx).applied_placement.as_deref() != placement.as_deref() {
            close(cx);
            open(cx, &overlay, model.clone());
        }
    }
}

/// Everything that determines where the island window sits.
fn placement_signature(overlay: &OverlayConfig) -> String {
    format!(
        "{:?}|{:.1}|{}|{:.1}|{:.1}",
        overlay.anchor, overlay.margin, overlay.island_mode, overlay.offset_x, overlay.offset_y
    )
}

fn close(cx: &mut App) {
    let manager = OverlayManager::get(cx);
    let window = manager.window.take();
    manager.hide_at = None;
    manager.applied_click_through = None;
    manager.applied_placement = None;
    if let Some((handle, _)) = window {
        let _ = handle.update(cx, |_, window, _| window.remove_window());
    }
}

/// Compute the window origin (display-relative) for the anchor + offset.
fn window_origin(
    display: &Bounds<Pixels>,
    display_id: u32,
    overlay: &OverlayConfig,
    island: bool,
) -> (f32, f32) {
    let width: f32 = display.size.width.into();
    let height: f32 = display.size.height.into();
    let x = (width - WIN_W) / 2.0 + overlay.offset_x;
    let menu_height = if overlay.anchor == OverlayAnchor::Top && !island {
        platform::menu_bar_height(display_id)
    } else {
        0.0
    };
    let y = window_origin_y(height, menu_height, overlay, island);
    (x.max(0.0), y.max(0.0))
}

/// GPUI's final popup frame uses a downward display-local top coordinate on
/// macOS as well. Its initial AppKit content rect is repositioned afterwards.
fn window_origin_y(height: f32, menu_height: f32, overlay: &OverlayConfig, island: bool) -> f32 {
    let y = match overlay.anchor {
        OverlayAnchor::Top if island => 0.0,
        OverlayAnchor::Top => menu_height + overlay.margin,
        OverlayAnchor::Bottom => height - WIN_H - overlay.margin,
    } + overlay.offset_y;
    y.max(0.0)
}

#[cfg(test)]
mod placement_tests {
    use super::*;

    #[test]
    fn top_anchors_account_for_platform_window_coordinates() {
        let overlay = OverlayConfig::default();
        let expected_top = 0.0;
        assert_eq!(window_origin_y(1080.0, 0.0, &overlay, true), expected_top);
        assert_eq!(
            window_origin_y(1080.0, 24.0, &overlay, false),
            expected_top + 24.0 + overlay.margin
        );
    }

    #[test]
    fn attached_top_is_screen_edge_not_window_height_below_it() {
        let overlay = OverlayConfig::default();
        assert_eq!(window_origin_y(1080.0, 0.0, &overlay, true), 0.0);
    }

    #[test]
    fn bottom_anchor_and_drag_offset_are_unchanged() {
        let overlay = OverlayConfig {
            anchor: OverlayAnchor::Bottom,
            offset_y: 7.0,
            ..OverlayConfig::default()
        };
        assert_eq!(
            window_origin_y(1080.0, 24.0, &overlay, false),
            1080.0 - WIN_H - overlay.margin + 7.0
        );
    }
}

pub(crate) fn open(cx: &mut App, overlay: &OverlayConfig, model: Entity<AppModel>) {
    let displays = cx.displays();
    let cursor = platform::cursor_display();
    let display = displays
        .iter()
        .find(|d| Some(u32::from(d.id())) == cursor)
        .cloned()
        .or_else(|| cx.primary_display())
        .or_else(|| displays.first().cloned());
    let Some(display) = display else {
        return;
    };
    let display_id = u32::from(display.id());
    let notch = if overlay.island_mode && overlay.anchor == OverlayAnchor::Top {
        platform::notch(display_id)
    } else {
        None
    };
    let (x, y) = window_origin(&display.bounds(), display_id, overlay, notch.is_some());
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(Bounds {
            origin: point(px(x), px(y)),
            size: size(px(WIN_W), px(WIN_H)),
        })),
        titlebar: None,
        focus: false,
        show: true,
        kind: WindowKind::PopUp,
        is_movable: false,
        is_resizable: false,
        is_minimizable: false,
        display_id: Some(display.id()),
        window_background: WindowBackgroundAppearance::Transparent,
        app_id: Some("dev.byteowlz.ears-ui".into()),
        ..Default::default()
    };
    let click_through = overlay.click_through;
    let result = cx.open_window(options, |window, cx| {
        platform::prepare_overlay_panel(window);
        if notch.is_some() {
            platform::pin_to_screen_top(window, display_id, y);
        }
        platform::set_click_through(window, click_through);
        let handle = window.window_handle();
        cx.new(|cx| IslandView::new(model, display_id, notch, handle, cx))
    });
    match result {
        Ok(handle) => {
            let manager = OverlayManager::get(cx);
            manager.window = Some((handle, display_id));
            manager.applied_click_through = Some(click_through);
            manager.applied_placement = Some(placement_signature(overlay));
        }
        Err(err) => eprintln!("ears-ui: cannot open overlay: {err:#}"),
    }
}

/// Open a synthetic notched fixture without touching the dictation engine.
pub fn open_preview(cx: &mut App, model: Entity<AppModel>) {
    let display = cx
        .primary_display()
        .or_else(|| cx.displays().first().cloned());
    let Some(display) = display else {
        eprintln!("ears-ui: no display for synthetic preview");
        return;
    };
    let display_id = u32::from(display.id());
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
            None,
            size(px(WIN_W), px(WIN_H)),
            cx,
        ))),
        titlebar: None,
        focus: false,
        show: true,
        kind: WindowKind::PopUp,
        is_movable: false,
        is_resizable: false,
        is_minimizable: false,
        display_id: Some(display.id()),
        window_background: WindowBackgroundAppearance::Transparent,
        app_id: Some("dev.byteowlz.ears-ui.synthetic-preview".into()),
        ..Default::default()
    };
    if let Err(error) = cx.open_window(options, |window, cx| {
        platform::prepare_overlay_panel(window);
        platform::set_click_through(window, true);
        cx.new(|cx| {
            IslandView::new(
                model,
                display_id,
                Some(Notch {
                    width: 185.0,
                    height: 32.0,
                }),
                window.window_handle(),
                cx,
            )
        })
    }) {
        eprintln!("ears-ui: cannot open synthetic preview: {error:#}");
    }
}

// ---------------------------------------------------------------------------

/// A compact island widens around the camera without changing its height.
fn notch_target(notch_width: f32, notch_height: f32, dormant: bool) -> (f32, f32) {
    // Keep the active wings stable while partial transcripts change or clear.
    let width = if dormant {
        notch_width
    } else {
        PILL_MAX_W.max(notch_width)
    };
    (width, notch_height)
}

fn throat_extent(width: f32, height: f32) -> f32 {
    THROAT
        .min(((WIN_W - width) / 2.0).max(0.0))
        .min(height / 2.0)
        .max(0.0)
}

/// Draw only the outward top throats; the existing pill paints the body.
fn notch_throats(width: f32, height: f32) -> impl IntoElement {
    canvas(
        |_, _, _| (),
        move |bounds, (), window, _| {
            let t = throat_extent(width, height);
            if t <= 0.0 {
                return;
            }
            let l = (WIN_W - width) / 2.0;
            let r = l + width;
            let p = |x, y| point(bounds.origin.x + px(x), bounds.origin.y + px(y));
            let mut path = PathBuilder::fill();
            path.move_to(p(l - t, 0.0));
            path.line_to(p(l, 0.0));
            path.line_to(p(l, t));
            path.cubic_bezier_to(p(l - t, 0.0), p(l, t * 0.3), p(l, 0.0));
            path.close();
            path.move_to(p(r, 0.0));
            path.line_to(p(r + t, 0.0));
            path.cubic_bezier_to(p(r, t), p(r, 0.0), p(r, t * 0.3));
            path.close();
            if let Ok(path) = path.build() {
                window.paint_path(path, gpui::black());
            }
        },
    )
    .absolute()
    .size_full()
}

fn detail_height(notch_height: f32, dormant: bool, show_details: bool) -> f32 {
    notch_height
        + if !dormant && show_details {
            PILL_H
        } else {
            0.0
        }
}

#[cfg(test)]
mod notch_geometry_tests {
    use super::*;

    #[test]
    fn compact_notch_widens_without_growing_downward() {
        assert_eq!(notch_target(185.0, 32.0, true), (185.0, 32.0));
        let (width, height) = notch_target(185.0, 32.0, false);
        assert_eq!(height, 32.0);
        assert!(width > 185.0);
        assert!((width - 185.0) / 2.0 >= 50.0 + 2.0 * PAD_X);
    }

    #[test]
    fn active_notch_width_is_stable_as_partial_text_changes() {
        // Text width is deliberately absent from the active geometry contract.
        assert_eq!(notch_target(185.0, 32.0, false), (PILL_MAX_W, 32.0));
    }

    #[test]
    fn throat_is_symmetric_and_bounded_by_window_and_height() {
        assert_eq!(throat_extent(560.0, 38.0), 12.0);
        assert_eq!(throat_extent(636.0, 38.0), 2.0);
        assert_eq!(throat_extent(640.0, 38.0), 0.0);
        assert_eq!(throat_extent(560.0, 8.0), 4.0);
    }

    #[test]
    fn details_expand_only_when_active_and_enabled() {
        assert_eq!(detail_height(32.0, false, true), 32.0 + PILL_H);
        assert_eq!(detail_height(32.0, false, false), 32.0);
        assert_eq!(detail_height(32.0, true, true), 32.0);
    }

    #[test]
    fn compact_notch_width_is_bounded_without_clipping_the_camera() {
        assert_eq!(notch_target(185.0, 32.0, false), (PILL_MAX_W, 32.0));
        assert_eq!(notch_target(600.0, 32.0, false), (600.0, 32.0));
    }
}

/// Width/height morph state.
#[derive(Debug, Clone, Copy)]
struct Morph {
    from: (f32, f32),
    to: (f32, f32),
    started: Instant,
    duration: Duration,
    generation: usize,
}

impl Morph {
    fn new(size: (f32, f32)) -> Self {
        Self {
            from: size,
            to: size,
            started: Instant::now(),
            duration: Duration::from_millis(MORPH_MS),
            generation: 0,
        }
    }

    fn progress(&self) -> f32 {
        if self.duration.is_zero() {
            return 1.0;
        }
        (self.started.elapsed().as_secs_f32() / self.duration.as_secs_f32()).clamp(0.0, 1.0)
    }

    fn current(&self, easing: impl Fn(f32) -> f32) -> (f32, f32) {
        let t = easing(self.progress());
        (
            self.from.0 + (self.to.0 - self.from.0) * t,
            self.from.1 + (self.to.1 - self.from.1) * t,
        )
    }

    fn retarget(
        &mut self,
        to: (f32, f32),
        duration: Duration,
        delay: Duration,
        easing: impl Fn(f32) -> f32,
    ) {
        if (to.0 - self.to.0).abs() < 0.5 && (to.1 - self.to.1).abs() < 0.5 {
            return;
        }
        self.from = self.current(easing);
        self.to = to;
        self.started = Instant::now() + delay;
        self.duration = duration;
        self.generation += 1;
    }

    fn done(&self) -> bool {
        self.progress() >= 1.0
    }
}

/// Content stays centered in a fixed wing while its scale and opacity change.
fn content_style(style: ContentMotion, reveal: f32) -> (f32, f32) {
    match style {
        ContentMotion::Grow => (reveal.max(0.01), if reveal > 0.0 { 1.0 } else { 0.0 }),
        ContentMotion::Fade => (1.0, reveal),
        ContentMotion::GrowFade => (reveal.max(0.01), reveal),
    }
}

/// A content transition can start after the shape has finished expanding.
struct ContentReveal {
    from: f32,
    to: f32,
    started: Instant,
    duration: Duration,
}

impl ContentReveal {
    fn new() -> Self {
        Self {
            from: 0.0,
            to: 0.0,
            started: Instant::now(),
            duration: Duration::ZERO,
        }
    }
    fn current(&self) -> f32 {
        if self.duration.is_zero() {
            return self.to;
        }
        let t =
            (self.started.elapsed().as_secs_f32() / self.duration.as_secs_f32()).clamp(0.0, 1.0);
        self.from + (self.to - self.from) * t
    }
    fn retarget(&mut self, visible: bool, duration: Duration, delay: Duration) {
        let value = self.current();
        self.from = value;
        self.to = if visible { 1.0 } else { 0.0 };
        self.started = Instant::now() + delay;
        self.duration = duration;
    }
    fn done(&self) -> bool {
        self.duration.is_zero() || Instant::now() >= self.started + self.duration
    }
}

struct DetailsTransition {
    reveal: ContentReveal,
    active: bool,
}

impl DetailsTransition {
    fn new() -> Self {
        Self {
            reveal: ContentReveal::new(),
            active: false,
        }
    }

    /// Returns whether visibility changed, so callers can defer shape collapse.
    fn update(
        &mut self,
        visible: bool,
        content_duration: Duration,
        shape_duration: Duration,
    ) -> bool {
        if self.active == visible {
            return false;
        }
        self.active = visible;
        self.reveal.retarget(
            visible,
            content_duration,
            if visible {
                shape_duration
            } else {
                Duration::ZERO
            },
        );
        true
    }
}

#[cfg(test)]
mod transition_tests {
    use super::*;

    #[test]
    fn content_waits_for_shape_and_reversal_preserves_position() {
        let mut reveal = ContentReveal::new();
        let duration = Duration::from_millis(180);
        reveal.retarget(true, duration, Duration::from_millis(260));
        assert_eq!(reveal.current(), 0.0);
        assert!(!reveal.done());
        reveal.started = Instant::now() - Duration::from_millis(90);
        let before = reveal.current();
        assert!((0.45..0.55).contains(&before));
        reveal.retarget(false, duration, Duration::ZERO);
        assert!((reveal.current() - before).abs() < 0.02);
        reveal.started = Instant::now() - duration;
        assert!(reveal.done());
        assert_eq!(reveal.current(), 0.0);
    }

    #[test]
    fn content_styles_remain_distinct_and_hidden_at_zero() {
        assert_eq!(content_style(ContentMotion::Grow, 0.5), (0.5, 1.0));
        assert_eq!(content_style(ContentMotion::Fade, 0.5), (1.0, 0.5));
        assert_eq!(content_style(ContentMotion::GrowFade, 0.5), (0.5, 0.5));
        for style in [
            ContentMotion::Grow,
            ContentMotion::Fade,
            ContentMotion::GrowFade,
        ] {
            assert_eq!(content_style(style, 0.0).1, 0.0);
            assert_eq!(content_style(style, 1.0), (1.0, 1.0));
        }
    }

    #[test]
    fn details_wait_for_expansion_and_reverse_without_jumping() {
        let mut details = DetailsTransition::new();
        let content = Duration::from_millis(180);
        let shape = Duration::from_millis(260);
        assert!(details.update(true, content, shape));
        assert_eq!(details.reveal.current(), 0.0);
        assert!(!details.update(true, content, shape));
        details.reveal.started = Instant::now() - content;
        assert_eq!(details.reveal.current(), 1.0);
        assert!(details.update(false, content, shape));
        details.reveal.started = Instant::now() - Duration::from_millis(90);
        let before = details.reveal.current();
        assert!(details.update(true, content, shape));
        assert!((details.reveal.current() - before).abs() < 0.02);
    }

    #[test]
    fn details_and_shape_stage_in_both_directions() {
        let content_duration = Duration::from_millis(CONTENT_MORPH_MS);
        let shape_duration = Duration::from_millis(ISLAND_MORPH_MS);
        let mut details = DetailsTransition::new();
        let mut shape = Morph::new((300.0, 32.0));
        assert!(details.update(true, content_duration, shape_duration));
        shape.retarget((300.0, 68.0), shape_duration, Duration::ZERO, |x| x);
        assert_eq!(details.reveal.current(), 0.0);
        details.reveal.started = Instant::now() - content_duration;
        shape.started = Instant::now() - shape_duration;
        assert_eq!(details.reveal.current(), 1.0);
        assert_eq!(shape.current(|x| x).1, 68.0);
        assert!(details.update(false, content_duration, shape_duration));
        shape.retarget((300.0, 32.0), shape_duration, content_duration, |x| x);
        assert_eq!(shape.current(|x| x).1, 68.0);
        assert!(details.reveal.current() > 0.99);
    }

    #[test]
    fn reduced_motion_snaps_and_delayed_shape_waits() {
        let mut reveal = ContentReveal::new();
        reveal.retarget(true, Duration::ZERO, Duration::ZERO);
        assert_eq!(reveal.current(), 1.0);
        reveal.retarget(false, Duration::ZERO, Duration::ZERO);
        assert_eq!(reveal.current(), 0.0);
        let mut shape = Morph::new((300.0, 32.0));
        shape.retarget(
            (185.0, 32.0),
            Duration::from_millis(260),
            Duration::from_millis(180),
            |x| x,
        );
        assert_eq!(shape.current(|x| x), (300.0, 32.0));
        assert!(!shape.done());
    }
}

struct DragState {
    start_mouse: (f32, f32),
    start_origin: (f32, f32),
    moved: bool,
    native: bool,
}

pub struct IslandView {
    model: Entity<AppModel>,
    display: u32,
    notch: Option<Notch>,
    morph: Morph,
    reveal: ContentReveal,
    details: DetailsTransition,
    notch_active: bool,
    last_notch_content: Option<Content>,
    drag: Option<DragState>,
    hovered: bool,
    appeared: Instant,
    parametric_state: Option<MarkState>,
    parametric_started: Instant,
    oqtopus: Option<crate::oqtopus::Worker>,
    oqtopus_attempted: bool,
}

/// Small easing with overshoot for the island's springy feel.
fn ease_out_back(t: f32) -> f32 {
    let c1 = 1.70158;
    let c3 = c1 + 1.0;
    1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
}

fn ease_out_quint(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(5)
}

fn parametric_state(phase: Phase, speech: bool, check: bool) -> MarkState {
    match phase {
        Phase::Dormant => MarkState::Ready,
        Phase::Armed => MarkState::Listening,
        Phase::Listening if speech => MarkState::Speech,
        Phase::Listening => MarkState::Listening,
        Phase::Finalizing if check => MarkState::Done,
        Phase::Finalizing => MarkState::Processing,
        Phase::Error => MarkState::Attention,
    }
}

#[cfg(test)]
mod parametric_state_tests {
    use super::*;
    #[test]
    fn maps_only_real_engine_phases() {
        assert_eq!(
            parametric_state(Phase::Dormant, false, false),
            MarkState::Ready
        );
        assert_eq!(
            parametric_state(Phase::Armed, false, false),
            MarkState::Listening
        );
        assert_eq!(
            parametric_state(Phase::Listening, true, false),
            MarkState::Speech
        );
        assert_eq!(
            parametric_state(Phase::Listening, false, false),
            MarkState::Listening
        );
        assert_eq!(
            parametric_state(Phase::Finalizing, false, false),
            MarkState::Processing
        );
        assert_eq!(
            parametric_state(Phase::Finalizing, false, true),
            MarkState::Done
        );
        assert_eq!(
            parametric_state(Phase::Error, false, false),
            MarkState::Attention
        );
    }
}

#[derive(Clone)]
struct Content {
    phase: Phase,
    text: String,
    dot: Hsla,
    text_color: Hsla,
    hint: bool,
    check: bool,
}

// GPUI's request_animation_frame can stall for a non-activating popup. Drive
// its transitions while the view exists, without retaining a closed window.
fn spawn_animation_driver(window: gpui::AnyWindowHandle, cx: &Context<IslandView>) {
    cx.spawn(async move |view, cx| {
        loop {
            let active = view.update(cx, |this, cx| {
                let model = this.model.read(cx);
                let animate = model.ui.overlay.motion_enabled(model.reduce_motion);
                let visible = !this.island()
                    || model.phase() != Phase::Dormant
                    || this.reveal.current() > 0.0;
                let oqtopus = visible && model.ui.overlay.indicator == IndicatorStyle::Oqtopus;
                let rapid = animate
                    && (!this.morph.done()
                        || !this.reveal.done()
                        || !this.details.reveal.done()
                        || (visible
                            && model.ui.overlay.indicator == IndicatorStyle::ParametricEars)
                        || (visible
                            && model.ui.overlay.indicator == IndicatorStyle::Ears
                            && model.speech_active)
                        || oqtopus);
                let refresh = rapid || oqtopus;
                if refresh {
                    cx.notify();
                }
                (refresh, rapid)
            });
            let Ok((active, rapid)) = active else { break };
            if active
                && window
                    .update(cx, |_, window, _| {
                        window.refresh();
                        platform::request_redraw(window);
                    })
                    .is_err()
            {
                break;
            }
            cx.background_executor()
                .timer(Duration::from_millis(if rapid { 33 } else { 250 }))
                .await;
        }
    })
    .detach();
}

impl IslandView {
    fn new(
        model: Entity<AppModel>,
        display: u32,
        notch: Option<Notch>,
        window: gpui::AnyWindowHandle,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.observe(&model, |_, _, cx| cx.notify()).detach();
        spawn_animation_driver(window, cx);
        let initial = match notch {
            Some(n) => (n.width, n.height),
            None => (PILL_MIN_W, PILL_H),
        };
        Self {
            model,
            display,
            notch,
            morph: Morph::new(initial),
            reveal: ContentReveal::new(),
            details: DetailsTransition::new(),
            notch_active: false,
            last_notch_content: None,
            drag: None,
            hovered: false,
            appeared: Instant::now(),
            parametric_state: None,
            parametric_started: Instant::now(),
            oqtopus: None,
            oqtopus_attempted: false,
        }
    }

    fn island(&self) -> bool {
        self.notch.is_some()
    }

    fn oqtopus_indicator(
        &mut self,
        state: MarkState,
        animate: bool,
        color: Hsla,
        bg: Hsla,
        size: f32,
    ) -> gpui::AnyElement {
        if !self.oqtopus_attempted {
            self.oqtopus_attempted = true;
            if let Some(path) = crate::oqtopus::pack_path() {
                self.oqtopus = crate::oqtopus::Worker::start(path).ok();
            }
        }
        if let Some(worker) = &self.oqtopus {
            worker.set(state, animate);
        }
        if let Some(snapshot) = self.oqtopus.as_ref().and_then(|worker| worker.snapshot()) {
            crate::oqtopus::mark(snapshot, color, bg, size).into_any_element()
        } else {
            // Missing or unreadable packs retain a working indicator.
            parametric_mark::mark(color, size * (24.0 / 36.0), state, 0.0, false).into_any_element()
        }
    }

    fn easing(&self, shape_motion: ShapeMotion) -> fn(f32) -> f32 {
        if self.island() && shape_motion == ShapeMotion::Spring {
            ease_out_back
        } else {
            ease_out_quint
        }
    }

    fn update_motion(
        &mut self,
        target: (f32, f32),
        content: &mut Content,
        overlay: &OverlayConfig,
        animate: bool,
    ) -> ((f32, f32), f32, f32) {
        let island = self.island();
        let dormant = content.phase == Phase::Dormant;
        let duration = if animate {
            Duration::from_millis(if island { ISLAND_MORPH_MS } else { MORPH_MS })
        } else {
            Duration::ZERO
        };
        let content_duration = if animate {
            Duration::from_millis(CONTENT_MORPH_MS)
        } else {
            Duration::ZERO
        };
        let changed = island && self.notch_active == dormant;
        if changed {
            self.notch_active = !dormant;
            self.reveal.retarget(
                !dormant,
                content_duration,
                if dormant { Duration::ZERO } else { duration },
            );
        }
        let show_details = island && !dormant && overlay.show_details;
        let details_changed = self
            .details
            .update(show_details, content_duration, duration);
        if island && !dormant {
            self.last_notch_content = Some(content.clone());
        }
        let delay = if (changed && dormant) || (details_changed && !show_details) {
            content_duration
        } else {
            Duration::ZERO
        };
        let easing = self.easing(overlay.shape_motion);
        self.morph.retarget(target, duration, delay, easing);
        if !animate {
            self.morph.duration = Duration::ZERO;
            self.reveal.duration = Duration::ZERO;
            self.details.reveal.duration = Duration::ZERO;
        }
        let size = self.morph.current(easing);
        let reveal = if island { self.reveal.current() } else { 1.0 };
        if island && dormant && reveal > 0.0 {
            if let Some(last) = &self.last_notch_content {
                *content = last.clone();
            }
        }
        (size, reveal, self.details.reveal.current())
    }

    fn measure(&self, window: &Window, text: &str, color: Hsla) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        let font = window.text_style().font();
        let run = TextRun {
            len: text.len(),
            font,
            color,
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        let line = window.text_system().shape_line(
            SharedString::from(text.to_string()),
            px(FONT_SIZE),
            &[run],
            None,
        );
        f32::from(line.width)
    }

    /// Keep the tail of the text so the newest words stay visible.
    fn tail(&self, window: &Window, text: &str, max_width: f32, color: Hsla) -> String {
        let text = text.trim();
        if self.measure(window, text, color) <= max_width {
            return text.to_string();
        }
        let chars: Vec<char> = text.chars().collect();
        let mut lo = 0usize;
        let mut hi = chars.len();
        // Binary search the longest suffix that fits with a leading ellipsis.
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            let candidate: String = std::iter::once('…')
                .chain(chars[chars.len() - mid..].iter().copied())
                .collect();
            if self.measure(window, &candidate, color) <= max_width {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        std::iter::once('…')
            .chain(chars[chars.len() - lo..].iter().copied())
            .collect()
    }
}

impl Render for IslandView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme(cx).clone();
        let r = &theme.roles;
        let island = self.island();

        let (
            phase,
            partial,
            voice_mode,
            error_text,
            last_utterance,
            finalizing_elapsed,
            speech,
            overlay,
            reduce_motion,
        ) = self.model.read_with(cx, |m, _| {
            (
                m.phase(),
                m.partial.clone(),
                m.voice_mode.clone(),
                m.error_text(),
                m.last_utterance.clone().unwrap_or_default(),
                m.finalizing_since.map(|t| t.elapsed()).unwrap_or_default(),
                m.speech_active,
                m.ui.overlay.clone(),
                m.reduce_motion,
            )
        });

        // Colours: the regular pill is a `surface` card; the island is the
        // notch's own black (a fixed identity effect, not a role).
        let (bg, fg, muted) = if island {
            (
                gpui::black(),
                Theme::alpha(gpui::white(), 0.92),
                Theme::alpha(gpui::white(), 0.55),
            )
        } else {
            (
                Theme::alpha(
                    if theme.is_dark() {
                        r.surface
                    } else {
                        r.background
                    },
                    overlay.opacity,
                ),
                r.foreground,
                r.muted_foreground,
            )
        };

        let mut content = match phase {
            Phase::Dormant => Content {
                phase,
                text: String::new(),
                dot: Theme::alpha(muted, 0.8),
                text_color: fg,
                hint: false,
                check: false,
            },
            Phase::Armed => Content {
                phase,
                text: voice_mode.clone(),
                dot: r.primary,
                text_color: muted,
                hint: true,
                check: false,
            },
            Phase::Listening => Content {
                phase,
                text: if partial.trim().is_empty() {
                    "…".to_string()
                } else {
                    partial.clone()
                },
                dot: if speech {
                    r.primary
                } else {
                    Theme::alpha(r.primary, 0.6)
                },
                text_color: fg,
                hint: false,
                check: false,
            },
            Phase::Finalizing => Content {
                phase,
                text: last_utterance.clone(),
                dot: r.success,
                text_color: muted,
                hint: false,
                check: finalizing_elapsed >= Duration::from_millis(FINALIZE_MS / 2),
            },
            Phase::Error => Content {
                phase,
                text: error_text.clone(),
                dot: r.danger,
                text_color: fg,
                hint: false,
                check: false,
            },
        };

        // Geometry ------------------------------------------------------------
        let dormant = matches!(content.phase, Phase::Dormant);
        let indicator_width = match overlay.indicator {
            IndicatorStyle::Oqtopus => 36.0,
            IndicatorStyle::Ears | IndicatorStyle::ParametricEars => 24.0,
            IndicatorStyle::Dot => DOT_W,
        };
        let max_text_w = match self.notch {
            Some(n) => ((PILL_MAX_W - n.width) / 2.0 - 2.0 * PAD_X).max(0.0),
            None => PILL_MAX_W - 2.0 * PAD_X - indicator_width - GAP,
        };
        let text = if dormant {
            String::new()
        } else {
            self.tail(window, &content.text, max_text_w, content.text_color)
        };
        let text_w = self.measure(window, &text, content.text_color);
        let target = match self.notch {
            Some(n) => {
                let (width, height) = notch_target(n.width, n.height, dormant);
                (width, detail_height(height, dormant, overlay.show_details))
            }
            None => {
                if dormant {
                    ((2.0 * PAD_X + indicator_width).max(PILL_MIN_W), PILL_H)
                } else {
                    (
                        (2.0 * PAD_X + indicator_width + GAP + text_w)
                            .clamp(PILL_MIN_W, PILL_MAX_W),
                        PILL_H,
                    )
                }
            }
        };
        let animate = overlay.motion_enabled(reduce_motion);
        let ((w, h), reveal, detail_reveal) =
            self.update_motion(target, &mut content, &overlay, animate);
        let display_text = if island && dormant && reveal > 0.0 {
            self.tail(window, &content.text, max_text_w, content.text_color)
        } else {
            text
        };
        if !self.morph.done() || (island && (!self.reveal.done() || !self.details.reveal.done())) {
            window.request_animation_frame();
        }
        let (scale, content_opacity) = if island {
            content_style(overlay.content_motion, reveal)
        } else {
            (1.0, 1.0)
        };

        // Pill ------------------------------------------------------------------
        let interactive = !overlay.click_through;
        let entity = cx.entity().downgrade();
        let hover_bg = if island {
            bg
        } else {
            Theme::alpha(r.accent, overlay.opacity)
        };

        let mut pill = div()
            .id("island")
            .flex()
            .items_center()
            .w(px(w))
            .h(px(h))
            .bg(if self.hovered && interactive {
                hover_bg
            } else {
                bg
            })
            .font_family(theme.font_sans.clone())
            .text_size(px(FONT_SIZE))
            .text_color(content.text_color)
            .overflow_hidden();

        if island {
            // AppKit supplies the notch's bounding rectangle, not its corner
            // curve; the radius remains an approximation until measured.
            let radius = px(self.notch.map_or(PILL_H, |n| n.height) * 0.35);
            pill = pill
                .rounded_bl(radius)
                .rounded_br(radius)
                .items_start()
                .relative();
        } else {
            pill = pill
                .rounded_full()
                .border_1()
                .border_color(Theme::alpha(
                    r.foreground,
                    if theme.is_dark() { 0.10 } else { 0.06 },
                ))
                .shadow(vec![
                    BoxShadow {
                        color: Theme::alpha(
                            gpui::black(),
                            if theme.is_dark() { 0.45 } else { 0.18 },
                        ),
                        offset: point(px(0.0), px(6.0)),
                        blur_radius: px(18.0),
                        spread_radius: px(0.0),
                    },
                    BoxShadow {
                        color: Theme::alpha(gpui::black(), 0.12),
                        offset: point(px(0.0), px(1.0)),
                        blur_radius: px(2.0),
                        spread_radius: px(0.0),
                    },
                ])
                .px(px(PAD_X));
        }

        // Dot / spinner / check ------------------------------------------------
        let wave_active = overlay.indicator == IndicatorStyle::Ears
            && content.phase == Phase::Listening
            && speech
            && animate;
        if wave_active {
            window.request_animation_frame();
        }
        let mark_state = parametric_state(content.phase, speech, content.check);
        if self.parametric_state != Some(mark_state) {
            self.parametric_state = Some(mark_state);
            self.parametric_started = Instant::now();
        }
        if overlay.indicator != IndicatorStyle::Oqtopus {
            self.oqtopus = None;
            self.oqtopus_attempted = false;
        }
        let indicator: gpui::AnyElement = if overlay.indicator == IndicatorStyle::Oqtopus {
            let background = if island {
                crate::theme::Color::BLACK
            } else if theme.is_dark() {
                theme.slots[0x01]
            } else {
                theme.slots[0x00]
            };
            let color = cx
                .global::<IndicatorPalette>()
                .color(mark_state, background);
            self.oqtopus_indicator(mark_state, animate, color, bg, indicator_width * scale)
        } else if overlay.indicator == IndicatorStyle::ParametricEars {
            if animate && (!island || !dormant || reveal > 0.0) {
                window.request_animation_frame();
            }
            let background = if island {
                crate::theme::Color::BLACK
            } else if theme.is_dark() {
                theme.slots[0x01]
            } else {
                theme.slots[0x00]
            };
            let color = cx
                .global::<IndicatorPalette>()
                .color(mark_state, background);
            parametric_mark::mark(
                color,
                24.0 * scale,
                mark_state,
                self.parametric_started.elapsed().as_secs_f32(),
                animate,
            )
            .into_any_element()
        } else if overlay.indicator == IndicatorStyle::Ears && content.phase != Phase::Finalizing {
            ears_mark::mark(
                content.dot,
                24.0 * scale,
                self.appeared.elapsed().as_secs_f32() / 0.095,
                wave_active,
            )
            .into_any_element()
        } else {
            match content.phase {
                Phase::Finalizing if !content.check => {
                    spinner(r.success, animate, scale).into_any_element()
                }
                Phase::Finalizing => div()
                    .w(px((DOT_W + 4.0) * scale))
                    .h(px((DOT_W + 4.0) * scale))
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_size(px(13.0 * scale))
                    .text_color(r.success)
                    .child("✓")
                    .into_any_element(),
                Phase::Armed if animate => dot(content.dot, scale)
                    .with_animation(
                        "armed-pulse",
                        Animation::new(Duration::from_millis(1400))
                            .repeat()
                            .with_easing(gpui::bounce(gpui::ease_in_out)),
                        |d, t| d.opacity(0.35 + 0.65 * t),
                    )
                    .into_any_element(),
                _ => dot(content.dot, scale).into_any_element(),
            }
        };

        let detail_text = format!("{} · {}", voice_mode, content.text);
        let label = div()
            .whitespace_nowrap()
            .overflow_hidden()
            .when(content.hint, |d| d.text_color(muted))
            .text_size(px(FONT_SIZE * scale))
            .child(SharedString::from(display_text));
        if let Some(n) = self.notch {
            // The physical camera covers the center. Place the icon near the
            // outer edge and reserve the full notch width for the camera.
            let wing = ((w - n.width) / 2.0).max(0.0);
            if !dormant || reveal > 0.0 {
                pill = pill
                    .child(
                        div()
                            .w(px(wing))
                            .h(px(n.height))
                            .flex()
                            .items_center()
                            .justify_start()
                            .pl(px(PAD_X.min((wing - indicator_width).max(0.0))))
                            .overflow_hidden()
                            .opacity(content_opacity)
                            .child(indicator),
                    )
                    .child(div().w(px(n.width)).flex_shrink_0())
                    .child(
                        div()
                            .w(px(wing))
                            .h(px(n.height))
                            .flex()
                            .items_center()
                            .justify_center()
                            .overflow_hidden()
                            .opacity(content_opacity)
                            .child(label),
                    );
            }
            if self.details.active || detail_reveal > 0.0 {
                let (detail_scale, detail_opacity) =
                    content_style(overlay.content_motion, detail_reveal);
                let detail_text = self.tail(
                    window,
                    &detail_text,
                    (w - 2.0 * PAD_X).max(0.0),
                    content.text_color,
                );
                pill = pill.child(
                    div()
                        .absolute()
                        .top(px(n.height))
                        .left_0()
                        .w(px(w))
                        .h(px(PILL_H))
                        .flex()
                        .items_center()
                        .justify_center()
                        .overflow_hidden()
                        .opacity(detail_opacity)
                        .text_size(px(FONT_SIZE * detail_scale))
                        .child(SharedString::from(detail_text)),
                );
            }
        } else {
            let mut inner = div().flex().items_center().gap(px(GAP)).child(indicator);
            if !dormant {
                inner = inner.child(label);
            }
            pill = pill.child(inner);
        }

        // Interaction ------------------------------------------------------------
        if interactive {
            let e1 = entity.clone();
            let e2 = entity.clone();
            let e3 = entity.clone();
            let e4 = entity.clone();
            let e5 = entity;
            pill = pill
                .cursor_pointer()
                .on_hover(move |hovered, _, cx| {
                    let _ = e1.update(cx, |this, cx| {
                        this.hovered = *hovered;
                        cx.notify();
                    });
                })
                .on_mouse_down(MouseButton::Left, move |_: &MouseDownEvent, window, cx| {
                    let native = platform::start_native_drag(window);
                    let start_mouse = platform::mouse_screen_position().unwrap_or((0.0, 0.0));
                    let start_origin = platform::window_origin(window).unwrap_or((0.0, 0.0));
                    let _ = e2.update(cx, |this, _| {
                        this.drag = Some(DragState {
                            start_mouse,
                            start_origin,
                            moved: false,
                            native,
                        });
                    });
                })
                .on_mouse_move(move |event: &MouseMoveEvent, window, cx| {
                    if !event.dragging() {
                        return;
                    }
                    let Some(mouse) = platform::mouse_screen_position() else {
                        return;
                    };
                    let _ = e3.update(cx, |this, _| {
                        if let Some(drag) = this.drag.as_mut() {
                            if drag.native {
                                drag.moved = true;
                                return;
                            }
                            let dx = mouse.0 - drag.start_mouse.0;
                            let dy = mouse.1 - drag.start_mouse.1;
                            if dx.abs() > DRAG_THRESHOLD || dy.abs() > DRAG_THRESHOLD {
                                drag.moved = true;
                            }
                            if drag.moved {
                                platform::set_window_origin(
                                    window,
                                    drag.start_origin.0 + dx,
                                    drag.start_origin.1 + dy,
                                );
                            }
                        }
                    });
                })
                .on_mouse_up(MouseButton::Left, move |_: &MouseUpEvent, window, cx| {
                    let mouse = platform::mouse_screen_position();
                    let outcome = e4
                        .update(cx, |this, _| {
                            let drag = this.drag.take()?;
                            if !drag.moved {
                                return Some(None);
                            }
                            let mouse = mouse?;
                            // AppKit y grows upwards; our offset_y grows downwards.
                            Some(Some((
                                mouse.0 - drag.start_mouse.0,
                                -(mouse.1 - drag.start_mouse.1),
                            )))
                        })
                        .ok()
                        .flatten();
                    match outcome {
                        Some(None) => runtime::open_settings(cx),
                        Some(Some((dx, dy))) => {
                            let model = runtime::model(cx);
                            model.update(cx, |m, cx| {
                                m.update_ui(cx, |ui| {
                                    ui.overlay.offset_x += dx;
                                    ui.overlay.offset_y += dy;
                                });
                            });
                            let _ = window;
                        }
                        None => {}
                    }
                })
                .on_mouse_up_out(MouseButton::Left, move |_: &MouseUpEvent, _, cx| {
                    let _ = e5.update(cx, |this, _| {
                        this.drag = None;
                    });
                });
        }

        // Appear animation: fade in on first frames.
        let appear = if animate {
            (self.appeared.elapsed().as_secs_f32() / 0.18).clamp(0.0, 1.0)
        } else {
            1.0
        };
        if appear < 1.0 {
            window.request_animation_frame();
        }

        let mut root = div()
            .size_full()
            .relative()
            .flex()
            .justify_center()
            .opacity(appear);
        if island {
            root = root.child(notch_throats(w, h));
        }
        root = if island {
            root.items_start()
        } else {
            match overlay.anchor {
                OverlayAnchor::Top => root.items_start().pt(px(12.0)),
                OverlayAnchor::Bottom => root.items_end().pb(px(12.0)),
            }
        };
        let _ = self.display;
        root.child(pill)
    }
}

fn dot(color: Hsla, scale: f32) -> Div {
    div()
        .w(px(DOT_W * scale))
        .h(px(DOT_W * scale))
        .flex_shrink_0()
        .rounded_full()
        .bg(color)
}

/// Three dots pulsing in sequence.
fn spinner(color: Hsla, animate: bool, scale: f32) -> Div {
    let mut row = div()
        .flex()
        .items_center()
        .gap(px(3.0 * scale))
        .flex_shrink_0();
    for i in 0..3 {
        let d = div()
            .w(px(4.0 * scale))
            .h(px(4.0 * scale))
            .rounded_full()
            .bg(color);
        if animate {
            row = row.child(
                d.with_animation(
                    ElementId::Name(format!("spin-{}", i).into()),
                    Animation::new(Duration::from_millis(600))
                        .repeat()
                        .with_easing(move |t| {
                            let phase = (t + i as f32 / 3.0) % 1.0;
                            gpui::bounce(gpui::ease_in_out)(phase)
                        }),
                    |d, t| d.opacity(0.25 + 0.75 * t),
                ),
            );
        } else {
            row = row.child(d);
        }
    }
    row
}
