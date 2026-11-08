//! Pointer/key mapping and gesture fencing for native faders. No host effects.
use device_scene::{
    Action, Control, Kind, Phase, Projection, Range, Snapshot, Source, Value, fader_thumb,
};
use glam::DVec3;

/// Hosts may store values as f32. Do not turn conversion noise or GPUI's
/// replayed mouse position after a redraw into another preview/render cycle.
pub(super) fn changed(range: &Range, current: f64, next: f64) -> bool {
    (current - next).abs() > range.step * 1e-4
}

pub(super) fn snap(range: &Range, value: f64) -> f64 {
    if value <= range.min {
        return range.min;
    }
    if value >= range.max {
        return range.max;
    }
    (range.min + ((value - range.min) / range.step).round() * range.step)
        .clamp(range.min, range.max)
}

pub(super) fn key_value(control: &Control, current: f64, key: &str) -> Option<f64> {
    if control.kind != Kind::Fader {
        return None;
    }
    let r = control.range.as_ref()?;
    let value = match key {
        "left" | "down" => current - r.step,
        "right" | "up" => current + r.step,
        "pagedown" => current - 10.0 * r.step,
        "pageup" => current + 10.0 * r.step,
        "home" => r.min,
        "end" => r.max,
        _ => return None,
    };
    Some(snap(r, value))
}

/// Pointer use retains a keyboard target without drawing a focus ring. Tab's
/// explicit focus takes precedence, so arrows never edit a previously clicked
/// slider while another control is focused.
pub(super) fn key_edit(
    snapshot: &Snapshot,
    pointer_target: Option<&str>,
    key: &str,
) -> Option<(String, f64)> {
    let focused = snapshot
        .state
        .controls
        .iter()
        .find(|(_, s)| s.focused)
        .map(|(id, _)| id.as_str());
    let id = focused.or(pointer_target)?;
    let (d, c, s) = snapshot.control(id)?;
    if !s.enabled || !snapshot.control_visible(d, c) {
        return None;
    }
    Some((id.into(), key_value(c, s.value.number()?, key)?))
}

pub(super) struct Drag {
    pub id: String,
    pub original: Value,
    snapshot: Snapshot,
    interaction: u64,
    sequence: u64,
}
impl Drag {
    pub fn start(snapshot: &Snapshot, id: &str, interaction: u64) -> Option<Self> {
        let (d, c, s) = snapshot.control(id)?;
        if c.kind != Kind::Fader
            || !s.enabled
            || c.action == "none"
            || !snapshot.control_visible(d, c)
        {
            return None;
        }
        Some(Self {
            id: id.into(),
            original: s.value.clone(),
            snapshot: snapshot.clone(),
            interaction,
            sequence: 0,
        })
    }
    pub fn valid(&self, snapshot: &Snapshot) -> bool {
        super::same_geometry(&self.snapshot, snapshot)
            && self
                .snapshot
                .control(&self.id)
                .zip(snapshot.control(&self.id))
                .is_some_and(|((_, a, _), (_, b, _))| a.range == b.range && a.action == b.action)
    }
    pub fn value_at(&self, projection: &Projection, x: f64, y: f64) -> Option<f64> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let (d, c, _) = self.snapshot.control(&self.id)?;
        let ray = projection.ray(
            x.clamp(0.0, projection.viewport.width_dip),
            y.clamp(0.0, projection.viewport.height_dip),
        )?;
        if ray.direction.z.abs() < 1e-9 {
            return None;
        }
        let t = (c.rect.z + d.origin[2] - ray.origin.z) / ray.direction.z;
        if t < 0.0 {
            return None;
        }
        let point = ray.origin + ray.direction * t - DVec3::from_array(d.origin);
        let thumb = fader_thumb(&c.rect, 0.5);
        let f = if c.rect.width >= c.rect.height {
            (point.x - c.rect.x) / (c.rect.width - thumb.width) + 0.5
        } else {
            (point.y - c.rect.y) / (c.rect.height - thumb.height) + 0.5
        }
        .clamp(0.0, 1.0);
        let r = c.range.as_ref()?;
        Some(snap(r, r.min + f * (r.max - r.min)))
    }
    pub fn action(
        &mut self,
        snapshot: &Snapshot,
        phase: Phase,
        value: Value,
        source: Source,
    ) -> Action {
        let sequence = self.sequence;
        self.sequence += 1;
        Action {
            schema_version: 1,
            scene_id: snapshot.scene.id.clone(),
            scene_revision: snapshot.scene.revision,
            state_revision: snapshot.state.revision,
            control_id: self.id.clone(),
            interaction_id: format!("native.i{}", self.interaction),
            sequence,
            phase,
            value,
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Snapshot, String) {
        let mut s = Snapshot::parse(include_str!(
            "../../../../spec/fixtures/device-scene-v1/spqx-receiver.json"
        ))
        .unwrap()
        .into_inner();
        let c = s.scene.devices[0]
            .controls
            .iter_mut()
            .find(|c| c.kind == Kind::Knob)
            .unwrap();
        c.kind = Kind::Fader;
        c.rect.width = 4.0;
        c.rect.height = 1.2;
        let id = c.id.clone();
        (s, id)
    }
    #[test]
    fn replayed_positions_and_f32_rounding_do_not_restart_preview_rendering() {
        let range = Range {
            min: 0.01,
            max: 0.99,
            step: 0.01,
        };
        assert!(!changed(&range, 0.43, 0.43));
        assert!(!changed(&range, 0.43_f32 as f64, 0.43));
        assert!(changed(&range, 0.43, 0.44));
        assert!(changed(&range, 0.433, 0.43));
    }
    #[test]
    fn pointer_endpoints_steps_and_both_orientations_match_the_mesh() {
        for horizontal in [false, true] {
            let (mut s, id) = fixture();
            let c = s.scene.devices[0]
                .controls
                .iter_mut()
                .find(|c| c.id == id)
                .unwrap();
            if !horizontal {
                std::mem::swap(&mut c.rect.width, &mut c.rect.height)
            }
            let range = c.range.clone().unwrap();
            let p =
                Projection::new(&device_scene::ValidatedSnapshot::new(s.clone()).unwrap()).unwrap();
            let drag = Drag::start(&s, &id, 1).unwrap();
            let (d, c, _) = s.control(&id).unwrap();
            for (fraction, value) in [(0.0, range.min), (1.0, range.max)] {
                let cap = fader_thumb(&c.rect, fraction);
                let pixel = p
                    .project(DVec3::new(
                        cap.x + d.origin[0],
                        cap.y + d.origin[1],
                        cap.z + d.origin[2],
                    ))
                    .unwrap();
                assert!((drag.value_at(&p, pixel[0], pixel[1]).unwrap() - value).abs() < 1e-8);
            }
            assert_eq!(key_value(c, range.min, "left"), Some(range.min));
            assert_eq!(key_value(c, range.max, "right"), Some(range.max));
            assert_eq!(key_value(c, range.min, "end"), Some(range.max));
        }
    }
    #[test]
    fn pointer_target_supports_keys_but_never_overrides_tab_or_disabled_state() {
        let (mut s, id) = fixture();
        for state in s.state.controls.values_mut() {
            state.focused = false;
        }
        assert!(key_edit(&s, None, "right").is_none());
        assert_eq!(key_edit(&s, Some(&id), "end").unwrap().0, id);
        let other = s.state.controls.keys().find(|k| **k != id).unwrap().clone();
        s.state.controls.get_mut(&other).unwrap().focused = true;
        assert!(key_edit(&s, Some(&id), "right").is_none());
        s.state.controls.get_mut(&other).unwrap().focused = false;
        s.state.controls.get_mut(&id).unwrap().enabled = false;
        assert!(key_edit(&s, Some(&id), "right").is_none());
    }
    #[test]
    fn value_updates_do_not_cancel_but_disable_page_geometry_and_ranges_do() {
        let (s, id) = fixture();
        let mut drag = Drag::start(&s, &id, 3).unwrap();
        let mut edited = s.clone();
        edited.state.revision += 1;
        edited.state.controls.get_mut(&id).unwrap().value = Value(serde_json::json!(
            edited.control(&id).unwrap().1.range.as_ref().unwrap().min
        ));
        assert!(drag.valid(&edited));
        let preview = drag.action(
            &edited,
            Phase::Preview,
            edited.state.controls[&id].value.clone(),
            Source::Pointer,
        );
        let commit = drag.action(
            &edited,
            Phase::Commit,
            preview.value.clone(),
            Source::Pointer,
        );
        let cancel = drag.action(
            &edited,
            Phase::Cancel,
            Value(serde_json::Value::Null),
            Source::Pointer,
        );
        assert_eq!(preview.interaction_id, commit.interaction_id);
        assert_eq!(
            (preview.sequence, commit.sequence, cancel.sequence),
            (0, 1, 2)
        );
        preview.validate(&edited).unwrap();
        commit.validate(&edited).unwrap();
        cancel.validate(&edited).unwrap();
        edited.state.controls.get_mut(&id).unwrap().enabled = false;
        assert!(!drag.valid(&edited));
        assert!(Drag::start(&edited, &id, 4).is_none());
        let mut resized = s.clone();
        resized.viewport.width_dip += 1.0;
        assert!(!drag.valid(&resized));
        let mut moved = s.clone();
        moved.scene.devices[0]
            .controls
            .iter_mut()
            .find(|c| c.id == id)
            .unwrap()
            .rect
            .x += 1.0;
        assert!(!drag.valid(&moved));
    }
}
