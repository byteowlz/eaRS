//! App-owned scene state and typed action validation. Geometry lives in device_layout.
pub(super) use super::device_layout::initial;
use super::shortcuts::Binding;
use device_scene::{Snapshot, Value};
use ears::config::HotkeyConfig;
#[cfg(test)]
use serde_json::Value as Json;
use serde_json::json;
pub(super) const INPUT: &str = "dictation.input";
pub(super) const SHORTCUTS: &str = "dictation.shortcuts";

pub(super) fn text(snapshot: &mut Snapshot, id: &str, value: impl Into<String>) {
    snapshot
        .state
        .controls
        .get_mut(id)
        .expect("app control")
        .value = Value(json!(value.into()));
}
pub(super) fn bindings(
    snapshot: &mut Snapshot,
    config: &HotkeyConfig,
    selected: Option<Binding>,
    has_draft: bool,
) {
    for binding in Binding::ALL {
        text(
            snapshot,
            &format!("shortcut.{}", binding.id()),
            binding
                .get(config)
                .filter(|s| !s.is_empty())
                .unwrap_or("Not assigned"),
        );
    }
    snapshot
        .state
        .controls
        .get_mut("shortcut.modifier")
        .unwrap()
        .enabled = has_draft;
    for id in ["shortcut.save", "shortcut.cancel"] {
        snapshot.state.controls.get_mut(id).unwrap().enabled = has_draft;
    }
    snapshot
        .state
        .controls
        .get_mut("shortcut.clear")
        .unwrap()
        .enabled = selected.is_some_and(|b| b != Binding::Toggle);
}
/// Phase semantics belong to the app adapter, not just schema validation.
pub(super) fn accepts(action: &device_scene::Action, snapshot: &Snapshot) -> bool {
    if action.validate(snapshot).is_err() {
        return false;
    }
    match action.phase {
        device_scene::Phase::Commit | device_scene::Phase::Cancel => true,
        device_scene::Phase::Preview => {
            snapshot
                .control(&action.control_id)
                .is_some_and(|(_, control, _)| {
                    matches!(
                        control.kind,
                        device_scene::Kind::Select | device_scene::Kind::Shortcut
                    )
                })
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_or_preview_buttons_cannot_commit_host_effects() {
        let snapshot = initial();
        let mut action = device_scene::Action {
            schema_version: 1,
            scene_id: snapshot.scene.id.clone(),
            scene_revision: snapshot.scene.revision,
            state_revision: snapshot.state.revision,
            control_id: "input.refresh".into(),
            interaction_id: "test.i1".into(),
            sequence: 0,
            phase: device_scene::Phase::Preview,
            value: Value(Json::Null),
            source: device_scene::Source::Api,
        };
        assert!(!accepts(&action, &snapshot));
        action.phase = device_scene::Phase::Commit;
        assert!(accepts(&action, &snapshot));
        action.state_revision += 1;
        assert!(!accepts(&action, &snapshot));
    }
    #[test]
    fn app_scene_preserves_navigation_and_all_editable_bindings() {
        let mut snapshot = initial();
        assert_eq!(snapshot.scene.devices[0].chassis.depth, 24.0);
        for page in [INPUT, SHORTCUTS] {
            snapshot.state.active_page = page.into();
            bindings(&mut snapshot, &HotkeyConfig::default(), None, false);
            snapshot.validate().unwrap();
            for id in [
                "nav.dictation",
                "nav.overlay",
                "nav.dictionaries",
                "nav.profiles",
            ] {
                assert!(snapshot.focus_order().iter().any(|c| c.id == id));
            }
        }
        for binding in Binding::ALL {
            assert!(
                snapshot
                    .control(&format!("shortcut.{}", binding.id()))
                    .is_some()
            );
        }
    }
}
