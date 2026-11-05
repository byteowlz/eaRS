//! Receiver grid: 0.2-unit spacing, common row tracks, fixed 0.16-unit phosphor pitch.
//! Geometry is independent of current values. This module owns no app effects.
use super::{
    device_scene::{INPUT, SHORTCUTS},
    shortcuts::Binding,
};
use device_scene::Snapshot;
use serde_json::{Value as Json, json};

const BASE: &str =
    include_str!("../../../../vendor/device/spec/fixtures/device-scene-v1/ears-input.json");
const PITCH: f64 = 0.16;
const GAP: f64 = 0.8;
const LEFT: f64 = -30.4;
const RIGHT: f64 = 30.4;
const CONTENT_LEFT: f64 = -17.2;
const CONTENT_RIGHT: f64 = 29.2;
const ROWS: [f64; 4] = [4.4, 1.2, -2.0, -5.2];
const KEY_HEIGHT: f64 = 1.6;

#[derive(Clone, Copy)]
struct Slot {
    left: f64,
    right: f64,
    y: f64,
    height: f64,
}
fn slot(left: f64, right: f64, y: f64) -> Slot {
    Slot {
        left,
        right,
        y,
        height: KEY_HEIGHT,
    }
}
fn track(left: f64, right: f64, count: usize, index: usize, y: f64) -> Slot {
    let width = (right - left - GAP * (count - 1) as f64) / count as f64;
    let start = left + index as f64 * (width + GAP);
    slot(start, start + width, y)
}
fn descriptor(id: &str, page: &str, kind: &str, label: &str, slot: Slot) -> Json {
    let width = slot.right - slot.left;
    let mut value = json!({"id":id,"module_id":"ears.front","page":page,"kind":kind,
        "rect":{"x":(slot.left+slot.right)/2.0,"y":slot.y,"z":0.8,"width":width,"height":slot.height,"depth":0.15},
        "material":if kind=="button" {"key"} else {"glass"},"label":label,
        "accessible_name":label,"description":label,"tab_index":0,"action":id,
        "presentation":if kind=="button" {"printed"} else {"matrix"}});
    if kind != "button" {
        let cols = ((width - 0.8 + PITCH) / (6.0 * PITCH)).floor() as u32;
        let rows = if slot.height >= 2.8 { 2 } else { 1 };
        let grid_width = (cols * 6 - 1) as f64 * PITCH;
        let grid_height = ((rows - 1) * 8 + 7) as f64 * PITCH;
        value["matrix"] = json!({"cols":cols,"rows":rows,"glyph_width":5,"glyph_height":7,
            "cell_width":6,"cell_height":8,"pitch":PITCH,
            "padding_x":(width-grid_width)/2.0,"padding_y":(slot.height-grid_height)/2.0});
    }
    value
}
fn printed(mut value: Json) -> Json {
    value["presentation"] = json!("printed");
    value["material"] = json!("face");
    value.as_object_mut().unwrap().remove("matrix");
    value
}
#[derive(Default)]
struct Controls {
    descriptors: Vec<Json>,
    states: serde_json::Map<String, Json>,
}
impl Controls {
    fn add(&mut self, mut descriptor: Json, value: Json) {
        let id = descriptor["id"].as_str().unwrap().to_owned();
        let enabled = descriptor["kind"] != "readout";
        descriptor["tab_index"] = json!(if enabled {
            self.descriptors.len() as i32
        } else {
            -1
        });
        if !enabled {
            descriptor["action"] = json!("none");
        }
        self.states.insert(
            id,
            json!({"value":value,"enabled":enabled,"visible":true,
            "pressed":false,"focused":false,"busy":false,"error":""}),
        );
        self.descriptors.push(descriptor);
    }
    fn navigation(&mut self) {
        for (i, (id, label)) in [
            ("dictation", "Dictation"),
            ("overlay", "Overlay"),
            ("dictionaries", "Dictionaries"),
            ("profiles", "Profiles"),
        ]
        .iter()
        .enumerate()
        {
            self.add(
                descriptor(
                    &format!("nav.{id}"),
                    "*",
                    "button",
                    label,
                    track(LEFT, RIGHT - 3.2, 4, i, 9.4),
                ),
                json!(id),
            );
        }
        let mut close = descriptor(
            "window.close",
            "*",
            "button",
            "×",
            slot(RIGHT - 2.4, RIGHT, 9.4),
        );
        close["accessible_name"] = json!("Close settings");
        close["description"] = json!("Close settings without applying drafts");
        self.add(close, Json::Null);
        for (i, (id, label)) in [
            (INPUT, "Input"),
            (SHORTCUTS, "Shortcuts"),
            ("dictation.general", "General"),
        ]
        .iter()
        .enumerate()
        {
            self.add(
                descriptor(
                    id,
                    "dictation.*",
                    "button",
                    label,
                    slot(-29.6, -20.8, ROWS[i]),
                ),
                json!(id),
            );
        }
    }
    fn input(&mut self) {
        let mut input = descriptor(
            "input.device",
            INPUT,
            "select",
            "Microphone",
            slot(CONTENT_LEFT, CONTENT_RIGHT, ROWS[0]),
        );
        input["options"] = json!([{"id":"default","label":"System default"}]);
        self.add(input, json!("default"));
        self.add(
            descriptor(
                "input.saved",
                INPUT,
                "readout",
                "Saved microphone",
                slot(CONTENT_LEFT, CONTENT_RIGHT, ROWS[1]),
            ),
            json!("System default"),
        );
        let first = track(CONTENT_LEFT, CONTENT_RIGHT, 4, 0, ROWS[3]);
        let second = track(CONTENT_LEFT, CONTENT_RIGHT, 4, 1, ROWS[3]);
        self.add(
            descriptor(
                "input.refresh",
                INPUT,
                "button",
                "Refresh inputs",
                slot(first.left, second.right, ROWS[3]),
            ),
            Json::Null,
        );
        for (index, id, label) in [
            (2, "input.apply", "Apply input"),
            (3, "input.cancel", "Cancel"),
        ] {
            self.add(
                descriptor(
                    id,
                    INPUT,
                    "button",
                    label,
                    track(CONTENT_LEFT, CONTENT_RIGHT, 4, index, ROWS[3]),
                ),
                Json::Null,
            );
        }
    }
    fn shortcuts(&mut self) {
        for (i, binding) in Binding::ALL.iter().enumerate() {
            self.add(
                printed(descriptor(
                    &format!("title.{}", binding.id()),
                    SHORTCUTS,
                    "readout",
                    binding.label(),
                    slot(CONTENT_LEFT, -5.2, ROWS[i]),
                )),
                json!(binding.label()),
            );
            self.add(
                descriptor(
                    &format!("shortcut.{}", binding.id()),
                    SHORTCUTS,
                    "shortcut",
                    binding.label(),
                    slot(-4.4, CONTENT_RIGHT, ROWS[i]),
                ),
                json!(""),
            );
        }
        let mut modifier = descriptor(
            "shortcut.modifier",
            SHORTCUTS,
            "select",
            "Modifier key preset",
            Slot {
                height: 2.8,
                ..slot(-29.6, -20.8, ROWS[3])
            },
        );
        modifier["options"] = json!([
            {"id":"choose","label":"Modifier key"},{"id":"right_option","label":"Right Option"},
            {"id":"right_cmd","label":"Right Cmd"},{"id":"fn","label":"Fn"},
            {"id":"left_option","label":"Left Option"},{"id":"left_cmd","label":"Left Cmd"},
            {"id":"right_ctrl","label":"Right Ctrl"},{"id":"left_ctrl","label":"Left Ctrl"},
            {"id":"right_shift","label":"Right Shift"},{"id":"left_shift","label":"Left Shift"}]);
        self.add(modifier, json!("choose"));
        for (i, (id, label)) in [
            ("defaults", "Defaults"),
            ("clear", "Clear"),
            ("save", "Save"),
            ("cancel", "Cancel"),
        ]
        .iter()
        .enumerate()
        {
            self.add(
                descriptor(
                    &format!("shortcut.{id}"),
                    SHORTCUTS,
                    "button",
                    label,
                    track(CONTENT_LEFT, CONTENT_RIGHT, 4, i, ROWS[3]),
                ),
                Json::Null,
            );
        }
    }
    fn footer(&mut self) {
        self.add(
            descriptor(
                "status",
                "*",
                "readout",
                "Settings status",
                Slot {
                    height: 2.8,
                    ..slot(LEFT, 20.8, -8.8)
                },
            ),
            json!("Choose a setting"),
        );
        self.add(
            printed(descriptor(
                "brand",
                "*",
                "readout",
                "eaRS",
                Slot {
                    height: 2.8,
                    ..slot(22.4, RIGHT, -8.8)
                },
            )),
            json!("eaRS"),
        );
    }
}
pub(super) fn initial() -> Snapshot {
    let mut base: Json = serde_json::from_str(BASE).expect("pinned base scene");
    base["scene"]["id"] = json!("ears.settings.live");
    base["scene"]["camera"]["yaw_deg"] = json!(0);
    base["scene"]["camera"]["pitch_deg"] = json!(0);
    base["scene"]["camera"]["target"] = json!([0, 0, -2]);
    base["scene"]["camera"]["distance"] = json!(115);
    base["scene"]["devices"][0]["chassis"]["height"] = json!(22);
    let panels = base["scene"]["devices"][0]["panels"]
        .as_array_mut()
        .unwrap();
    panels[0]["rect"]["height"] = json!(22);
    for panel in panels.iter_mut() {
        panel["label"] = json!("");
    }
    for (panel, left, right) in [(1, -30.4, -20.0), (2, -18.4, 30.4)] {
        panels[panel]["rect"]["x"] = json!((left + right) / 2.0);
        panels[panel]["rect"]["y"] = json!(-0.2);
        panels[panel]["rect"]["width"] = json!(right - left);
        panels[panel]["rect"]["height"] = json!(13.2);
    }
    let mut controls = Controls::default();
    controls.navigation();
    controls.input();
    controls.shortcuts();
    controls.footer();
    base["scene"]["devices"][0]["controls"] = json!(controls.descriptors);
    base["state"]["controls"] = json!(controls.states);
    base["state"]["active_page"] = json!(INPUT);
    Snapshot::parse(&serde_json::to_string(&base).unwrap())
        .expect("receiver grid validates")
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn edges(snapshot: &Snapshot, id: &str) -> (f64, f64, f64) {
        let r = &snapshot.control(id).unwrap().1.rect;
        (r.x - r.width / 2.0, r.x + r.width / 2.0, r.y)
    }
    fn equal(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-8, "{a} != {b}");
    }
    #[test]
    fn receiver_rows_edges_gutters_and_matrix_pitch_are_content_independent() {
        let s = initial();
        for (i, binding) in Binding::ALL.iter().enumerate() {
            let title = edges(&s, &format!("title.{}", binding.id()));
            let field = edges(&s, &format!("shortcut.{}", binding.id()));
            equal(title.0, CONTENT_LEFT);
            equal(field.1, CONTENT_RIGHT);
            equal(field.0 - title.1, GAP);
            equal(title.2, ROWS[i]);
            equal(field.2, ROWS[i]);
        }
        let mut last = None;
        for id in [
            "shortcut.defaults",
            "shortcut.clear",
            "shortcut.save",
            "shortcut.cancel",
        ] {
            let (l, r, y) = edges(&s, id);
            equal(y, ROWS[3]);
            equal(r - l, 11.0);
            if let Some(end) = last {
                equal(l - end, GAP);
            }
            last = Some(r);
        }
        for c in &s.scene.devices[0].controls {
            if let Some(m) = &c.matrix {
                equal(m.pitch, PITCH);
                assert_eq!(m.cell_width, 6);
            }
        }
        equal(edges(&s, "input.device").0, CONTENT_LEFT);
        equal(edges(&s, "input.saved").1, CONTENT_RIGHT);
        assert!(s.viewport.transparent);
        let scene_before = s.scene.clone();
        let mut long = s.clone();
        long.state.controls.get_mut("status").unwrap().value =
            device_scene::Value(json!("A long synthetic status ".repeat(16)));
        long.validate().unwrap();
        assert_eq!(
            long.scene, scene_before,
            "values cannot resize dots or controls"
        );
    }
    #[test]
    fn visible_controls_never_overlap_and_every_orange_display_is_matrix() {
        let mut s = initial();
        for page in [INPUT, SHORTCUTS] {
            s.state.active_page = page.into();
            s.validate().unwrap();
            let d = &s.scene.devices[0];
            let visible: Vec<_> = d
                .controls
                .iter()
                .filter(|c| c.page == "*" || c.page == page || c.page == "dictation.*")
                .collect();
            for (i, a) in visible.iter().enumerate() {
                if a.material == "glass" {
                    assert_eq!(a.presentation, device_scene::Presentation::Matrix);
                }
                for b in &visible[i + 1..] {
                    let x = (a.rect.x - b.rect.x).abs() - (a.rect.width + b.rect.width) / 2.0;
                    let y = (a.rect.y - b.rect.y).abs() - (a.rect.height + b.rect.height) / 2.0;
                    assert!(
                        x >= 0.2 - 1e-8 || y >= 0.2 - 1e-8,
                        "{} overlaps {}",
                        a.id,
                        b.id
                    );
                }
            }
        }
    }
}
