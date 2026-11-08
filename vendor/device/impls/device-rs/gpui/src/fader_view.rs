//! DeviceView's fader input seam. Preview/cancel never imply host persistence.
use super::{DeviceView, fader};
use device_scene::{Phase, Source, Value, validate_value};
use gpui::Context;

impl DeviceView {
    pub(super) fn start_fader(&mut self, id: &str, x: f32, y: f32, cx: &mut Context<Self>) -> bool {
        let next = self.interaction + 1;
        let Some(drag) = fader::Drag::start(&self.snapshot, id, next) else {
            return false;
        };
        self.interaction = next;
        self.range_drag = Some(drag);
        self.range_keyboard = Some(id.into());
        self.move_fader(x, y, Phase::Preview, cx);
        true
    }
    pub(super) fn move_fader(&mut self, x: f32, y: f32, phase: Phase, cx: &mut Context<Self>) {
        let Some(drag) = self.range_drag.take() else {
            return;
        };
        if !drag.valid(&self.snapshot) {
            self.range_drag = Some(drag);
            self.cancel_fader(cx);
            return;
        }
        let value = self
            .slot
            .current()
            .and_then(|p| drag.value_at(&p.frame.projection, x as f64, y as f64));
        let Some(value) = value else {
            self.range_drag = Some(drag);
            self.cancel_fader(cx);
            return;
        };
        self.emit_fader(drag, value, phase, Source::Pointer, cx);
    }
    fn emit_fader(
        &mut self,
        mut drag: fader::Drag,
        value: f64,
        phase: Phase,
        source: Source,
        cx: &mut Context<Self>,
    ) {
        let unchanged = self.snapshot.control(&drag.id).is_some_and(|(_, c, s)| {
            c.range
                .as_ref()
                .zip(s.value.number())
                .is_some_and(|(range, current)| !fader::changed(range, current, value))
        });
        if phase == Phase::Preview && unchanged {
            self.range_drag = Some(drag);
            return;
        }
        self.snapshot
            .state
            .controls
            .get_mut(&drag.id)
            .expect("fader state")
            .value = Value(serde_json::json!(value));
        self.snapshot.state.revision += 1;
        let action = drag.action(
            &self.snapshot,
            phase,
            Value(serde_json::json!(value)),
            source,
        );
        if phase == Phase::Preview {
            self.range_drag = Some(drag)
        }
        if action.validate(&self.snapshot).is_ok() {
            cx.emit(action)
        }
        self.request(cx);
    }
    pub(super) fn cancel_fader(&mut self, cx: &mut Context<Self>) {
        let Some(mut drag) = self.range_drag.take() else {
            return;
        };
        if self
            .snapshot
            .control(&drag.id)
            .is_some_and(|(_, c, _)| validate_value(c, &drag.original).is_ok())
        {
            self.snapshot
                .state
                .controls
                .get_mut(&drag.id)
                .expect("fader state")
                .value = drag.original.clone();
            self.snapshot.state.revision += 1;
        }
        let action = drag.action(
            &self.snapshot,
            Phase::Cancel,
            Value(serde_json::Value::Null),
            Source::Pointer,
        );
        if action.validate(&self.snapshot).is_ok() {
            cx.emit(action)
        }
        self.request(cx);
    }
    pub(super) fn key_fader(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        let edit = fader::key_edit(&self.snapshot, self.range_keyboard.as_deref(), key);
        let Some((id, value)) = edit else {
            return false;
        };
        self.cancel_fader(cx);
        let next = self.interaction + 1;
        let Some(drag) = fader::Drag::start(&self.snapshot, &id, next) else {
            return false;
        };
        self.interaction = next;
        self.focus_control(&id);
        self.emit_fader(drag, value, Phase::Commit, Source::Keyboard, cx);
        true
    }
}
