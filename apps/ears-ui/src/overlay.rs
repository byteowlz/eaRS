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
use crate::config::OverlayAnchor;
use crate::config::OverlayConfig;
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
const DRAG_THRESHOLD: f32 = 4.0;

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
    let y = match overlay.anchor {
        OverlayAnchor::Top if island => 0.0,
        OverlayAnchor::Top => platform::menu_bar_height(display_id) + overlay.margin,
        OverlayAnchor::Bottom => height - WIN_H - overlay.margin,
    } + overlay.offset_y;
    (x.max(0.0), y.max(0.0))
}

fn open(cx: &mut App, overlay: &OverlayConfig, model: Entity<AppModel>) {
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
        platform::set_shadow(window, false);
        platform::set_click_through(window, click_through);
        cx.new(|cx| IslandView::new(model, display_id, notch, cx))
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

// ---------------------------------------------------------------------------

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

    fn retarget(&mut self, to: (f32, f32), duration: Duration, easing: impl Fn(f32) -> f32) {
        if (to.0 - self.to.0).abs() < 0.5 && (to.1 - self.to.1).abs() < 0.5 {
            return;
        }
        self.from = self.current(easing);
        self.to = to;
        self.started = Instant::now();
        self.duration = duration;
        self.generation += 1;
    }

    fn done(&self) -> bool {
        self.progress() >= 1.0
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
    drag: Option<DragState>,
    hovered: bool,
    appeared: Instant,
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

struct Content {
    phase: Phase,
    text: String,
    dot: Hsla,
    text_color: Hsla,
    hint: bool,
    check: bool,
}

impl IslandView {
    fn new(
        model: Entity<AppModel>,
        display: u32,
        notch: Option<Notch>,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.observe(&model, |_, _, cx| cx.notify()).detach();
        let initial = match notch {
            Some(n) => (n.width + 24.0, n.height),
            None => (PILL_MIN_W, PILL_H),
        };
        Self {
            model,
            display,
            notch,
            morph: Morph::new(initial),
            drag: None,
            hovered: false,
            appeared: Instant::now(),
        }
    }

    fn island(&self) -> bool {
        self.notch.is_some()
    }

    fn easing(&self) -> fn(f32) -> f32 {
        if self.island() {
            ease_out_back
        } else {
            ease_out_quint
        }
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

        let content = match phase {
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
                text: "Listening".to_string(),
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
        let max_text_w = PILL_MAX_W - 2.0 * PAD_X - DOT_W - GAP;
        let text = if dormant {
            String::new()
        } else {
            self.tail(window, &content.text, max_text_w, content.text_color)
        };
        let text_w = self.measure(window, &text, content.text_color);
        let target = match self.notch {
            Some(n) => {
                if dormant {
                    (n.width + 24.0, n.height)
                } else {
                    let w = (n.width + 2.0 * PAD_X + DOT_W + GAP + text_w + 24.0)
                        .clamp(n.width + 60.0, PILL_MAX_W.max(n.width + 60.0));
                    (w, n.height + PILL_H)
                }
            }
            None => {
                if dormant {
                    (PILL_MIN_W, PILL_H)
                } else {
                    (
                        (2.0 * PAD_X + DOT_W + GAP + text_w).clamp(PILL_MIN_W, PILL_MAX_W),
                        PILL_H,
                    )
                }
            }
        };
        let animate = overlay.animation && !reduce_motion;
        let duration = if animate {
            Duration::from_millis(if island { ISLAND_MORPH_MS } else { MORPH_MS })
        } else {
            Duration::ZERO
        };
        let easing = self.easing();
        self.morph.retarget(target, duration, easing);
        let (w, h) = self.morph.current(easing);
        if !self.morph.done() {
            window.request_animation_frame();
        }

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
            let radius = px(PILL_H * 0.6);
            pill = pill
                .rounded_bl(radius)
                .rounded_br(radius)
                .items_end()
                .pb(px((PILL_H - 20.0) / 2.0))
                .px(px(PAD_X + 6.0));
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
        let indicator: gpui::AnyElement = match content.phase {
            Phase::Finalizing if !content.check => spinner(r.success, animate).into_any_element(),
            Phase::Finalizing => div()
                .w(px(DOT_W + 4.0))
                .h(px(DOT_W + 4.0))
                .flex()
                .items_center()
                .justify_center()
                .text_size(px(13.0))
                .text_color(r.success)
                .child("✓")
                .into_any_element(),
            Phase::Armed if animate => dot(content.dot)
                .with_animation(
                    "armed-pulse",
                    Animation::new(Duration::from_millis(1400))
                        .repeat()
                        .with_easing(gpui::bounce(gpui::ease_in_out)),
                    |d, t| d.opacity(0.35 + 0.65 * t),
                )
                .into_any_element(),
            _ => dot(content.dot).into_any_element(),
        };

        let mut inner = div().flex().items_center().gap(px(GAP)).child(indicator);
        if !dormant {
            inner = inner.child(
                div()
                    .whitespace_nowrap()
                    .overflow_hidden()
                    .when(content.hint, |d| d.text_color(muted))
                    .child(SharedString::from(text)),
            );
        }
        if island && dormant {
            inner = div();
        }
        pill = pill.child(inner);

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

        let mut root = div().size_full().flex().justify_center().opacity(appear);
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

fn dot(color: Hsla) -> Div {
    div()
        .w(px(DOT_W))
        .h(px(DOT_W))
        .flex_shrink_0()
        .rounded_full()
        .bg(color)
}

/// Three dots pulsing in sequence.
fn spinner(color: Hsla, animate: bool) -> Div {
    let mut row = div().flex().items_center().gap(px(3.0)).flex_shrink_0();
    for i in 0..3 {
        let d = div().w(px(4.0)).h(px(4.0)).rounded_full().bg(color);
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
