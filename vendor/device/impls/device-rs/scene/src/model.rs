use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// Optional fields may be absent, but explicit null is not in the v1 schema.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

macro_rules! record {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[serde(deny_unknown_fields)]
        pub struct $name { $(pub $field: $ty),* }
    };
}
macro_rules! choices {
    ($name:ident { $($variant:ident),* $(,)? }) => {
        #[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
        #[serde(rename_all = "kebab-case")]
        pub enum $name { $($variant),* }
    };
}

choices!(Finish {
    Matte,
    Brushed,
    Spun,
    Glass
});
choices!(Surface {
    Faceplate,
    Module,
    Display
});
choices!(Kind {
    Button,
    Toggle,
    Knob,
    Fader,
    Select,
    Text,
    Shortcut,
    Meter,
    Indicator,
    Readout
});
choices!(Presentation {
    NativeText,
    Matrix,
    Printed,
    None
});
choices!(CameraKind { Perspective });
choices!(Phase {
    Preview,
    Commit,
    Cancel
});
choices!(Source {
    Pointer,
    Keyboard,
    Accessibility,
    Api
});

record!(Rect {
    x: f64,
    y: f64,
    z: f64,
    width: f64,
    height: f64,
    depth: f64
});
record!(Material {
    base: String,
    ink: String,
    emissive: String,
    roughness: f64,
    metalness: f64,
    finish: Finish
});
record!(Camera {
    projection: CameraKind,
    fov_y_deg: f64,
    yaw_deg: f64,
    pitch_deg: f64,
    target: [f64; 3],
    distance: f64
});
record!(Light {
    azimuth_deg: f64,
    elevation_deg: f64,
    intensity: f64,
    ambient: f64
});
record!(Matrix {
    cols: u32,
    rows: u32,
    glyph_width: u32,
    glyph_height: u32,
    cell_width: u32,
    cell_height: u32,
    pitch: f64,
    padding_x: f64,
    padding_y: f64
});
record!(Panel {
    id: String,
    rect: Rect,
    surface: Surface,
    material: String,
    label: String,
    page: String
});
record!(OptionItem {
    id: String,
    label: String
});
record!(Range {
    min: f64,
    max: f64,
    step: f64
});

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Control {
    pub id: String,
    pub module_id: String,
    pub page: String,
    pub kind: Kind,
    pub rect: Rect,
    pub material: String,
    pub label: String,
    pub accessible_name: String,
    pub description: String,
    pub tab_index: i32,
    pub action: String,
    pub presentation: Presentation,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub unit: Option<String>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub options: Option<Vec<OptionItem>>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub range: Option<Range>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub matrix: Option<Matrix>,
}
record!(Chassis {
    width: f64,
    height: f64,
    depth: f64,
    front_thickness: f64,
    feet_height: f64,
    material: String,
    vents: bool
});
record!(Device { id: String, origin: [f64; 3], chassis: Chassis, panels: Vec<Panel>, controls: Vec<Control> });
record!(Scene { id: String, revision: u64, materials: BTreeMap<String, Material>, camera: Camera, light: Light, devices: Vec<Device> });
record!(Selection {
    start: u64,
    end: u64,
    composing: bool
});

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlState {
    // Missing value is an error, including for momentary buttons.
    pub value: Value,
    pub enabled: bool,
    pub visible: bool,
    pub pressed: bool,
    pub focused: bool,
    pub busy: bool,
    pub error: String,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub selection: Option<Selection>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct Value(pub serde_json::Value);

impl Value {
    pub fn string(&self) -> Option<&str> {
        self.0.as_str()
    }
    pub fn number(&self) -> Option<f64> {
        self.0.as_f64()
    }
    pub fn boolean(&self) -> Option<bool> {
        self.0.as_bool()
    }
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}
record!(State { revision: u64, active_page: String, controls: BTreeMap<String, ControlState> });
record!(Viewport {
    width_dip: f64,
    height_dip: f64,
    scale_factor: f64,
    pixel_cap: u32,
    transparent: bool
});
record!(Snapshot {
    schema_version: u32,
    scene: Scene,
    state: State,
    viewport: Viewport
});
record!(Action {
    schema_version: u32,
    scene_id: String,
    scene_revision: u64,
    state_revision: u64,
    control_id: String,
    interaction_id: String,
    sequence: u64,
    phase: Phase,
    value: Value,
    source: Source
});

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_optional_fields_are_allowed_but_explicit_null_is_rejected() {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../spec/fixtures/device-scene-v1/ears-input.json"
        ))
        .unwrap();
        assert!(serde_json::from_value::<Snapshot>(value.clone()).is_ok());
        value["scene"]["devices"][0]["controls"][0]["range"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<Snapshot>(value).is_err());
    }
}
