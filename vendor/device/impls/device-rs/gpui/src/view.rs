//! Reusable native device view. Emits typed actions; the host owns every app effect.
#[path = "readout_tooltip.rs"]
mod readout_tooltip;
use crate::{ImageSlot, RenderWorker};
use device_scene::{Action, Kind, Phase, Snapshot, Source, ValidatedSnapshot, Value};
use gpui::{
    Context, FocusHandle, KeyDownEvent, MouseButton, SharedString, Window, div, img, prelude::*,
    px, rgb,
};
use readout_tooltip::ReadoutTooltip;

pub struct DeviceView {
    snapshot: Snapshot,
    worker: RenderWorker,
    slot: ImageSlot,
    focus: FocusHandle,
    pressed: Option<String>,
    interaction: u64,
    printed_font: SharedString,
    ui_font: SharedString,
    keyboard_enabled: bool,
    font_overrides: std::collections::HashMap<String, SharedString>,
    hovered: Option<String>,
}
/// The control under the pointer changed (host-owned help; no app effect).
#[derive(Clone, Debug, PartialEq)]
pub struct Hover(pub Option<String>);
impl DeviceView {
    pub fn new(
        snapshot: Snapshot,
        printed_font: SharedString,
        ui_font: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (worker, notifications) = RenderWorker::start();
        let focus = cx.focus_handle();
        window.focus(&focus);
        cx.spawn_in(window, async move |view, cx| {
            while notifications.recv().await.is_ok() {
                if view
                    .update_in(cx, |this, window, cx| {
                        if let Some(completed) = this.worker.take_completed() {
                            this.slot.install(completed, window);
                            if let Some(error) = &this.slot.error {
                                eprintln!("renderer unavailable: {error}");
                            }
                            cx.notify();
                        }
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
        cx.observe_window_bounds(window, |this, window, cx| this.resize(window, cx))
            .detach();
        cx.observe_window_activation(window, |this, window, cx| {
            if !window.is_window_active() {
                this.pressed = None;
                this.clear_pressed();
                this.request(cx);
            }
        })
        .detach();
        cx.on_release_in(window, |this, window, _| this.slot.clear(window))
            .detach();
        let mut this = Self {
            snapshot,
            worker,
            slot: ImageSlot::default(),
            focus,
            pressed: None,
            interaction: 0,
            printed_font,
            ui_font,
            keyboard_enabled: true,
            font_overrides: Default::default(),
            hovered: None,
        };
        this.resize(window, cx);
        this
    }
    fn resize(&mut self, window: &Window, cx: &mut Context<Self>) {
        let size = window.viewport_size();
        self.snapshot.viewport.width_dip = f32::from(size.width) as f64;
        self.snapshot.viewport.height_dip = f32::from(size.height) as f64;
        self.snapshot.viewport.scale_factor = window.scale_factor() as f64;
        self.pressed = None;
        self.clear_pressed();
        self.request(cx);
    }
    fn clear_pressed(&mut self) {
        let mut changed = false;
        for state in self.snapshot.state.controls.values_mut() {
            changed |= state.pressed;
            state.pressed = false;
        }
        if changed {
            self.snapshot.state.revision += 1;
        }
    }
    fn request(&mut self, cx: &mut Context<Self>) {
        match ValidatedSnapshot::new(self.snapshot.clone())
            .map_err(|e| e.to_string())
            .and_then(|s| self.worker.submit(s))
        {
            Ok(g) => self.slot.expect(g),
            Err(error) => {
                self.slot.error = Some(error.clone());
                eprintln!("request rejected: {error}");
            }
        }
        cx.notify();
    }
    fn hit(&self, x: f32, y: f32) -> Option<String> {
        let p = self.slot.current()?;
        // Do not accept input against a newer state or a resized viewport over old pixels.
        if p.snapshot.snapshot() != &self.snapshot {
            return None;
        }
        let id = p.frame.hit_test(x as f64, y as f64)?;
        let (_, c, state) = self.snapshot.control(id)?;
        (state.enabled && c.action != "none").then(|| id.to_owned())
    }
    /// Host-owned dragging may begin only on current, visible non-control geometry.
    pub fn is_chassis_at(&self, x: f32, y: f32) -> bool {
        self.slot.current().is_some_and(|p| {
            p.snapshot.snapshot() == &self.snapshot && p.frame.is_chassis_at(x as f64, y as f64)
        })
    }
    fn clear_focus(&mut self) {
        let mut changed = false;
        for state in self.snapshot.state.controls.values_mut() {
            changed |= state.focused;
            state.focused = false;
        }
        if changed {
            self.snapshot.state.revision += 1;
        }
    }
    fn focus_control(&mut self, id: &str) {
        for (key, state) in &mut self.snapshot.state.controls {
            state.focused = key == id;
        }
        self.snapshot.state.revision += 1;
    }
    pub fn focus(&self, window: &mut Window) {
        window.focus(&self.focus);
    }
    /// Camera mapping of the displayed frame, for host tools such as layout editors.
    pub fn projection(&self) -> Option<device_scene::Projection> {
        self.slot.current().map(|p| p.frame.projection.clone())
    }
    pub fn error(&self) -> Option<&str> {
        self.slot.error.as_deref()
    }
    /// Host font registry override, keyed by semantic control id; never a scene path.
    pub fn set_control_font(&mut self, id: &str, font: SharedString) {
        self.font_overrides.insert(id.into(), font);
    }
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }
    pub fn set_keyboard_enabled(&mut self, enabled: bool) {
        self.keyboard_enabled = enabled;
    }
    /// Apply app-owned state without I/O. Viewport/focus remain in the native adapter.
    pub fn edit_snapshot(&mut self, edit: impl FnOnce(&mut Snapshot), cx: &mut Context<Self>) {
        let before = self.snapshot.clone();
        edit(&mut self.snapshot);
        let active = self
            .snapshot
            .focus_order()
            .iter()
            .map(|c| c.id.clone())
            .collect::<Vec<_>>();
        for (id, state) in &mut self.snapshot.state.controls {
            if !active.contains(id) {
                state.focused = false;
                state.pressed = false;
            }
        }
        if self.snapshot != before {
            self.snapshot.scene.revision += 1;
            self.snapshot.state.revision += 1;
            self.pressed = None;
            self.request(cx);
        }
    }
    fn activate(&mut self, id: &str, source: Source, cx: &mut Context<Self>) {
        let Some((_, control, state)) = self.snapshot.control(id) else {
            return;
        };
        if !state.enabled {
            return;
        }
        let mut value = state.value.clone();
        let mut phase = Phase::Commit;
        if control.kind == Kind::Select
            && let Some(options) = &control.options
        {
            let current = options
                .iter()
                .position(|o| Some(o.id.as_str()) == value.string())
                .unwrap_or(0);
            value = Value(serde_json::Value::String(
                options[(current + 1) % options.len()].id.clone(),
            ));
            phase = Phase::Preview;
        }
        // Activation requests host capture; it is not a recorded chord.
        if control.kind == Kind::Shortcut {
            phase = Phase::Preview;
        }
        self.interaction += 1;
        let action = Action {
            schema_version: 1,
            scene_id: self.snapshot.scene.id.clone(),
            scene_revision: self.snapshot.scene.revision,
            state_revision: self.snapshot.state.revision,
            control_id: id.to_owned(),
            interaction_id: format!("native.i{}", self.interaction),
            sequence: 0,
            phase,
            value,
            source,
        };
        if action.validate(&self.snapshot).is_ok() {
            cx.emit(action);
        }
        self.request(cx);
    }
    fn key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        if !self.keyboard_enabled {
            return;
        }
        match event.keystroke.key.as_str() {
            "tab" => {
                let order = self.snapshot.focus_order();
                if order.is_empty() {
                    return;
                }
                let current = order
                    .iter()
                    .position(|c| self.snapshot.state.controls[&c.id].focused);
                let index = if event.keystroke.modifiers.shift {
                    current
                        .map(|i| (i + order.len() - 1) % order.len())
                        .unwrap_or(order.len() - 1)
                } else {
                    current.map(|i| (i + 1) % order.len()).unwrap_or(0)
                };
                let id = order[index].id.clone();
                self.focus_control(&id);
                self.request(cx);
                cx.stop_propagation();
            }
            "enter" | "space" => {
                if let Some(id) = self
                    .snapshot
                    .state
                    .controls
                    .iter()
                    .find(|(_, s)| s.focused)
                    .map(|(id, _)| id.clone())
                {
                    self.activate(&id, Source::Keyboard, cx);
                }
                cx.stop_propagation();
            }
            "escape" => {
                self.pressed = None;
                self.clear_pressed();
                self.request(cx);
                // Host editors own cancellation; do not swallow their Escape.
            }
            _ => {}
        }
    }
}
impl gpui::EventEmitter<Action> for DeviceView {}
impl gpui::EventEmitter<Hover> for DeviceView {}
impl DeviceView {
    fn hover(&mut self, x: f32, y: f32, cx: &mut Context<Self>) {
        let id = self.slot.current().and_then(|p| {
            (p.snapshot.snapshot() == &self.snapshot)
                .then(|| p.frame.hit_test(x as f64, y as f64))
                .flatten()
                .map(str::to_owned)
        });
        if id != self.hovered {
            self.hovered = id.clone();
            cx.emit(Hover(id));
        }
    }
}
impl DeviceView {
    // Native print/fallback and exact-value tooltips; matrix ink is already in GPU pixels.
    fn label_element(
        &self,
        label: &device_render_wgpu::Label,
        projection: &device_scene::Projection,
        label_index: usize,
    ) -> Option<gpui::Stateful<gpui::Div>> {
        let center = projection.project(label.center)?;
        let right = projection.project(label.center + glam::DVec3::X * label.width / 2.0)?;
        let top = projection.project(label.center + glam::DVec3::Y * label.height / 2.0)?;
        let width = ((right[0] - center[0]).abs() * 2.0) as f32;
        let height = ((top[1] - center[1]).abs() * 2.0) as f32;
        let color = ((label.color[0] as u32) << 16)
            | ((label.color[1] as u32) << 8)
            | label.color[2] as u32;
        // Native sharp text: projected centre/extent, but no perspective warp in this proof.
        // Exact-value inspection only where the device cannot show it all.
        let tooltip_text = match label.raster {
            device_render_wgpu::TextRaster::UnicodeFallback => {
                Some(format!("Native glyph fallback: {}", label.text))
            }
            device_render_wgpu::TextRaster::MatrixClipped => Some(label.text.trim().to_owned()),
            _ => None,
        };
        let tooltip_font = self.ui_font.clone();
        Some(
            div()
                .id(("device-readout", label_index))
                .when_some(tooltip_text, |element, text| {
                    element.tooltip(move |_, cx| {
                        cx.new(|_| ReadoutTooltip(text.clone(), tooltip_font.clone()))
                            .into()
                    })
                })
                .absolute()
                .left(px(center[0] as f32 - width / 2.0))
                .top(px(center[1] as f32 - height / 2.0))
                .w(px(width))
                .h(px(height))
                .flex()
                .items_center()
                .when(label.align == device_render_wgpu::TextAlign::Center, |e| {
                    e.justify_center()
                })
                .when(label.align == device_render_wgpu::TextAlign::Start, |e| {
                    e.justify_start()
                })
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .text_color(rgb(color))
                .font_family(
                    label
                        .control_id
                        .as_ref()
                        .and_then(|id| self.font_overrides.get(id))
                        .cloned()
                        .unwrap_or_else(|| {
                            if label.printed {
                                self.printed_font.clone()
                            } else {
                                self.ui_font.clone()
                            }
                        }),
                )
                // Size follows the legend's role (its height), never the string length.
                .text_size(px((height * 0.72).clamp(5.0, 28.0)))
                .when(!label.raster.is_matrix(), |element| {
                    element.child(etched(label, (height * 0.72).clamp(5.0, 28.0), color))
                }),
        )
    }
}

/// Printed legends read as engraved: a dark lip above (light from above) and a faint
/// lit edge below, with the ink fill slightly sunk. Native text stays sharp.
fn etched(label: &device_render_wgpu::Label, size: f32, color: u32) -> gpui::AnyElement {
    let text = label.text.clone();
    if !label.printed {
        return text.into_any_element();
    }
    let lip = (size * 0.07).clamp(0.6, 1.6);
    let sunk = |c: u32, k: f32| {
        let ch = |shift: u32| (((c >> shift) & 0xff) as f32 * k) as u32;
        (ch(16) << 16) | (ch(8) << 8) | ch(0)
    };
    div()
        .relative()
        .child(
            div()
                .absolute()
                .left_0()
                .top(px(lip))
                .text_color(gpui::rgba(0xffffff1f))
                .child(text.clone()),
        )
        .child(
            div()
                .absolute()
                .left_0()
                .top(px(-lip * 0.8))
                .text_color(gpui::rgba(0x000000c0))
                .child(text.clone()),
        )
        .child(
            div()
                .relative()
                .text_color(rgb(sunk(color, 0.9)))
                .child(text),
        )
        .into_any_element()
}

impl Render for DeviceView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut root = div()
            .id("native-device-specimen")
            .relative()
            .size_full()
            .overflow_hidden()
            .when(!self.snapshot.viewport.transparent, |root| {
                root.bg(rgb(0x101316))
            })
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|this, event, _, cx| this.key(event, cx)))
            .on_mouse_move(cx.listener(|this, event: &gpui::MouseMoveEvent, _, cx| {
                this.hover(event.position.x.into(), event.position.y.into(), cx)
            }))
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !hovered && this.hovered.take().is_some() {
                    cx.emit(Hover(None));
                }
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseDownEvent, window, cx| {
                    window.focus(&this.focus);
                    if let Some(id) = this.hit(event.position.x.into(), event.position.y.into()) {
                        // Pointer use shows no focus ring; the ring is for keyboard navigation.
                        this.clear_focus();
                        this.pressed = Some(id.clone());
                        this.snapshot
                            .state
                            .controls
                            .get_mut(&id)
                            .expect("state")
                            .pressed = true;
                        // Every state edit gets a new revision, or the frame is rejected.
                        this.snapshot.state.revision += 1;
                        this.request(cx);
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseUpEvent, _, cx| {
                    let pressed = this.pressed.take();
                    // Current frame may still be in flight for key travel; use the displayed frame's same geometry.
                    let hit = this
                        .slot
                        .current()
                        .and_then(|p| {
                            p.frame.hit_test(
                                f32::from(event.position.x) as f64,
                                f32::from(event.position.y) as f64,
                            )
                        })
                        .map(str::to_owned);
                    this.clear_pressed();
                    if let Some(id) = pressed.filter(|id| Some(id) == hit.as_ref()) {
                        this.activate(&id, Source::Pointer, cx);
                    } else {
                        this.request(cx);
                    }
                }),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.pressed = None;
                    this.clear_pressed();
                    this.request(cx);
                }),
            );
        if let Some(p) = self.slot.current() {
            let viewport = &p.frame.viewport;
            root = root.child(
                img(p.image.clone())
                    .absolute()
                    .left_0()
                    .top_0()
                    .w(px(viewport.width_dip as f32))
                    .h(px(viewport.height_dip as f32)),
            );
            for (index, label) in p.frame.labels.iter().enumerate() {
                if let Some(element) = self.label_element(label, &p.frame.projection, index) {
                    root = root.child(element);
                }
            }
        }
        if let Some(error) = &self.slot.error {
            root = root.child(
                div()
                    .absolute()
                    .bottom_0()
                    .left_0()
                    .text_color(rgb(0xffaa66))
                    .text_size(px(12.0))
                    .child(format!("Render unavailable: {error}")),
            );
        }
        root
    }
}
