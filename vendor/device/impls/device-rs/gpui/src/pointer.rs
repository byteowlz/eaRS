//! Pointer routing for momentary keys and range gestures. Geometry is shared
//! with the presented GPU frame; no host side effect lives here.
use super::DeviceView;
use device_scene::{Phase, Source};
use gpui::{Context, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Window};

impl DeviceView {
    pub(super) fn pointer_move(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        if self.range_drag.is_some() {
            self.move_fader(
                event.position.x.into(),
                event.position.y.into(),
                Phase::Preview,
                cx,
            );
            cx.stop_propagation();
        } else {
            self.hover(event.position.x.into(), event.position.y.into(), cx);
        }
    }
    pub(super) fn pointer_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus);
        self.cancel_fader(cx);
        let Some(id) = self.hit(event.position.x.into(), event.position.y.into()) else {
            return;
        };
        // Pointer use shows no focus ring; the ring is for keyboard navigation.
        self.clear_focus();
        self.range_keyboard = None;
        if self.start_fader(&id, event.position.x.into(), event.position.y.into(), cx) {
            cx.stop_propagation();
            return;
        }
        self.pressed = Some(id.clone());
        self.snapshot
            .state
            .controls
            .get_mut(&id)
            .expect("state")
            .pressed = true;
        self.snapshot.state.revision += 1;
        self.request(cx);
    }
    pub(super) fn pointer_up(
        &mut self,
        event: &MouseUpEvent,
        outside: bool,
        cx: &mut Context<Self>,
    ) {
        if self.range_drag.is_some() {
            self.move_fader(
                event.position.x.into(),
                event.position.y.into(),
                Phase::Commit,
                cx,
            );
            cx.stop_propagation();
            return;
        }
        let pressed = self.pressed.take();
        // A key-travel/text frame may still be in flight; hit the displayed frame.
        let hit = self
            .slot
            .current()
            .and_then(|p| {
                p.frame.hit_test(
                    f32::from(event.position.x) as f64,
                    f32::from(event.position.y) as f64,
                )
            })
            .map(str::to_owned);
        self.clear_pressed();
        if let Some(id) = pressed.filter(|id| !outside && Some(id) == hit.as_ref()) {
            self.activate(&id, Source::Pointer, cx);
        } else {
            self.request(cx);
        }
    }
}
