//! App-owned layout/state for the approved receiver direction. No rendering or effects.
use super::shortcuts::Binding;
use device_scene::{Snapshot, Value};
use ears::config::HotkeyConfig;
use serde_json::{Value as Json, json};

const BASE: &str =
    include_str!("../../../../vendor/device/spec/fixtures/device-scene-v1/ears-input.json");
pub(super) const INPUT: &str = "dictation.input";
pub(super) const SHORTCUTS: &str = "dictation.shortcuts";

fn control(id: &str, page: &str, kind: &str, label: &str, x: f64, y: f64, width: f64) -> Json {
    json!({"id":id,"module_id":"ears.front","page":page,"kind":kind,
        "rect":{"x":x,"y":y,"z":0.8,"width":width,"height":1.65,"depth":0.15},
        "material":if kind == "button" {"key"} else {"glass"},"label":label,
        "accessible_name":label,"description":label,"tab_index":0,
        "action":id,"presentation":if kind == "button" {"printed"} else {"native-text"}})
}
fn append(controls: &mut Vec<Json>, states: &mut Json, mut descriptor: Json, value: Json) {
    let id = descriptor["id"].as_str().unwrap().to_owned();
    let enabled = descriptor["kind"] != "readout";
    descriptor["tab_index"] = json!(if enabled { controls.len() as i32 } else { -1 });
    if !enabled {
        descriptor["action"] = json!("none");
    }
    states[&id] = json!({"value":value,"enabled":enabled,"visible":true,"pressed":false,"focused":false,"busy":false,"error":""});
    controls.push(descriptor);
}
pub(super) fn initial() -> Snapshot {
    let mut base: Json = serde_json::from_str(BASE).expect("pinned base scene");
    base["scene"]["id"] = json!("ears.settings.live");
    // Front presentation keeps native typography and hit geometry aligned. Chassis
    // depth is still real geometry, not a picture behind an unrelated form.
    base["scene"]["camera"]["yaw_deg"] = json!(0);
    base["scene"]["camera"]["pitch_deg"] = json!(0);
    base["scene"]["camera"]["target"] = json!([0, 0, -2]);
    base["scene"]["camera"]["distance"] = json!(115);
    base["scene"]["devices"][0]["chassis"]["height"] = json!(22);
    let panels = base["scene"]["devices"][0]["panels"]
        .as_array_mut()
        .unwrap();
    panels[0]["rect"]["height"] = json!(22);
    panels[0]["label"] = json!("");
    panels[1]["rect"]["height"] = json!(13);
    panels[1]["label"] = json!("");
    panels[2]["rect"]["height"] = json!(13);
    panels[2]["label"] = json!("");
    let mut controls = Vec::new();
    let mut states = json!({});
    for (i, (id, label)) in [
        ("dictation", "Dictation"),
        ("overlay", "Overlay"),
        ("dictionaries", "Dictionaries"),
        ("profiles", "Profiles"),
    ]
    .iter()
    .enumerate()
    {
        append(
            &mut controls,
            &mut states,
            control(
                &format!("nav.{id}"),
                "*",
                "button",
                label,
                -23.0 + 15.0 * i as f64,
                8.5,
                13.0,
            ),
            json!(id),
        );
    }
    for (i, (id, label)) in [
        (INPUT, "Input"),
        (SHORTCUTS, "Shortcuts"),
        ("dictation.general", "General"),
    ]
    .iter()
    .enumerate()
    {
        append(
            &mut controls,
            &mut states,
            control(
                id,
                "dictation.*",
                "button",
                label,
                -25.0,
                3.5 - 3.0 * i as f64,
                8.8,
            ),
            json!(id),
        );
    }
    let mut modifier = control(
        "shortcut.modifier",
        SHORTCUTS,
        "select",
        "Modifier key preset",
        -25.0,
        -5.4,
        8.8,
    );
    modifier["options"] = json!([
        {"id":"choose","label":"Modifier key"}, {"id":"right_option","label":"Right Option"},
        {"id":"right_cmd","label":"Right Cmd"}, {"id":"fn","label":"Fn"},
        {"id":"left_option","label":"Left Option"}, {"id":"left_cmd","label":"Left Cmd"},
        {"id":"right_ctrl","label":"Right Ctrl"}, {"id":"left_ctrl","label":"Left Ctrl"},
        {"id":"right_shift","label":"Right Shift"}, {"id":"left_shift","label":"Left Shift"}
    ]);
    append(&mut controls, &mut states, modifier, json!("choose"));
    let mut input = control(
        "input.device",
        INPUT,
        "select",
        "Microphone",
        3.0,
        3.5,
        32.0,
    );
    input["options"] = json!([{"id":"default","label":"System default"}]);
    append(&mut controls, &mut states, input, json!("default"));
    append(
        &mut controls,
        &mut states,
        control("input.refresh", INPUT, "button", "Refresh", 24.0, 3.5, 8.0),
        Json::Null,
    );
    append(
        &mut controls,
        &mut states,
        control(
            "input.apply",
            INPUT,
            "button",
            "Apply input",
            -3.0,
            0.2,
            17.0,
        ),
        Json::Null,
    );
    append(
        &mut controls,
        &mut states,
        control("input.cancel", INPUT, "button", "Cancel", 17.0, 0.2, 17.0),
        Json::Null,
    );
    append(
        &mut controls,
        &mut states,
        control(
            "input.saved",
            INPUT,
            "readout",
            "Saved microphone",
            6.0,
            -3.0,
            40.0,
        ),
        json!("System default"),
    );
    for (i, binding) in Binding::ALL.iter().enumerate() {
        let y = 3.8 - i as f64 * 2.8;
        let mut title = control(
            &format!("title.{}", binding.id()),
            SHORTCUTS,
            "readout",
            binding.label(),
            -7.0,
            y,
            11.0,
        );
        title["presentation"] = json!("printed");
        title["material"] = json!("face");
        append(&mut controls, &mut states, title, json!(binding.label()));
        append(
            &mut controls,
            &mut states,
            control(
                &format!("shortcut.{}", binding.id()),
                SHORTCUTS,
                "shortcut",
                binding.label(),
                14.0,
                y,
                27.0,
            ),
            json!(""),
        );
    }
    for (i, (id, label)) in [
        ("defaults", "Defaults"),
        ("clear", "Clear"),
        ("save", "Save"),
        ("cancel", "Cancel"),
    ]
    .iter()
    .enumerate()
    {
        append(
            &mut controls,
            &mut states,
            control(
                &format!("shortcut.{id}"),
                SHORTCUTS,
                "button",
                label,
                -8.5 + i as f64 * 11.5,
                -5.2,
                10.0,
            ),
            Json::Null,
        );
    }
    append(
        &mut controls,
        &mut states,
        control(
            "status",
            "*",
            "readout",
            "Settings status",
            -4.0,
            -8.8,
            50.0,
        ),
        json!("Choose a setting"),
    );
    let mut brand = control("brand", "*", "readout", "eaRS", 26.0, -8.8, 7.0);
    brand["presentation"] = json!("printed");
    brand["material"] = json!("face");
    append(&mut controls, &mut states, brand, json!("eaRS"));
    base["scene"]["devices"][0]["controls"] = json!(controls);
    base["state"]["controls"] = states;
    base["state"]["active_page"] = json!(INPUT);
    Snapshot::parse(&serde_json::to_string(&base).unwrap())
        .expect("app scene validates")
        .into_inner()
}
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
