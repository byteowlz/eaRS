//! Independently authored contract tests: fixture data, invalid mutations and projection agreement.
use device_scene::{Projection, Rect, Snapshot, ValidatedSnapshot};
use glam::DVec3;
use serde_json::{Value, json};

const INPUT: &str = include_str!("../../../../spec/fixtures/device-scene-v1/ears-input.json");
const SHORTCUTS: &str =
    include_str!("../../../../spec/fixtures/device-scene-v1/ears-shortcuts.json");
const SPQX: &str = include_str!("../../../../spec/fixtures/device-scene-v1/spqx-receiver.json");
fn mutate(source: &str, f: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(source).unwrap();
    f(&mut v);
    serde_json::to_string(&v).unwrap()
}
fn reject(source: &str, f: impl FnOnce(&mut Value)) {
    let raw = mutate(source, f);
    assert!(
        Snapshot::parse(&raw).is_err(),
        "accepted invalid snapshot: {raw}"
    );
}
#[test]
fn fixtures_roundtrip_and_project() {
    for source in [INPUT, SHORTCUTS, SPQX] {
        let validated = Snapshot::parse(source).unwrap();
        let again = serde_json::to_string(validated.snapshot()).unwrap();
        assert_eq!(
            validated.snapshot(),
            Snapshot::parse(&again).unwrap().snapshot()
        );
        let projection = Projection::new(&validated).unwrap();
        assert!(projection.near > 0.0 && projection.far > projection.near);
        assert!(projection.view_projection.is_finite());
        assert!(projection.inverse_view_projection.is_finite());
    }
}
#[test]
fn shape_is_closed_and_versioned() {
    reject(INPUT, |v| v["unexpected"] = json!(true));
    reject(INPUT, |v| v["schema_version"] = json!(2));
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][0]["run"] = json!("command")
    });
}
#[test]
fn missing_state_value_is_not_an_implicit_null() {
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.nav.overlay"]
            .as_object_mut()
            .unwrap()
            .remove("value");
    });
}
#[test]
fn control_references_and_ids_are_validated() {
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][1]["id"] = json!("ears.nav.dictation")
    });
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][0]["module_id"] = json!("unknown")
    });
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][0]["material"] = json!("unknown")
    });
    reject(INPUT, |v| {
        v["state"]["controls"]
            .as_object_mut()
            .unwrap()
            .remove("ears.input.device");
    });
    reject(INPUT, |v| {
        v["state"]["controls"]["unknown"] = v["state"]["controls"]["ears.input.device"].clone()
    });
}
#[test]
fn control_bounds_are_device_local() {
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][0]["rect"]["x"] = json!(100)
    });
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][0]["rect"]["depth"] = json!(-1)
    });
}
#[test]
fn state_matches_descriptor_kind_and_options() {
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.input.device"]["value"] = json!("nonexistent")
    });
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.input.level"]["value"] = json!("0.5")
    });
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.input.level"]["value"] = json!(1.1)
    });
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.input.status"]["value"] = json!({"unsafe":"object"})
    });
}
#[test]
fn duplicate_option_ids_fail() {
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][6]["options"][1]["id"] = json!("default")
    });
}
#[test]
fn off_grid_meter_samples_are_valid() {
    let raw = mutate(INPUT, |v| {
        v["state"]["controls"]["ears.input.level"]["value"] = json!(0.01347)
    });
    assert!(Snapshot::parse(&raw).is_ok());
}
#[test]
fn off_grid_knob_state_is_not_quantized() {
    let raw = mutate(SPQX, |v| {
        v["state"]["controls"]["spqx.volume"]["value"] = json!(83.2)
    });
    let s = Snapshot::parse(&raw).unwrap();
    assert_eq!(
        s.snapshot().state.controls["spqx.volume"].value.number(),
        Some(83.2)
    );
}
#[test]
fn utf16_bounds_preserve_full_unicode() {
    let raw = mutate(SPQX, |v| {
        v["state"]["controls"]["spqx.manuscript"]["value"] = json!("a\u{1F600}界");
        v["state"]["controls"]["spqx.manuscript"]["selection"] =
            json!({"start":1,"end":3,"composing":false});
    });
    assert!(Snapshot::parse(&raw).is_ok());
    reject(&raw, |v| {
        v["state"]["controls"]["spqx.manuscript"]["selection"]["end"] = json!(5)
    });
}
#[test]
fn utf16_selection_must_not_bisect_surrogate_pair() {
    reject(SPQX, |v| {
        v["state"]["controls"]["spqx.manuscript"]["value"] = json!("a\u{1F600}界");
        v["state"]["controls"]["spqx.manuscript"]["selection"] =
            json!({"start":2,"end":2,"composing":false});
    });
}
#[test]
fn focus_requires_active_enabled_control() {
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.shortcuts.toggle"]["focused"] = json!(true)
    });
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.input.level"]["focused"] = json!(true)
    });
    reject(INPUT, |v| {
        v["state"]["controls"]["ears.input.device"]["focused"] = json!(true);
        v["state"]["controls"]["ears.nav.overlay"]["focused"] = json!(true);
    });
}
#[test]
fn pages_preserve_four_nav_destinations_and_subpages() {
    for (source, included, excluded) in [
        (INPUT, "ears.input.device", "ears.shortcuts.toggle"),
        (SHORTCUTS, "ears.shortcuts.toggle", "ears.input.device"),
    ] {
        let v = Snapshot::parse(source).unwrap();
        let names: Vec<_> = v
            .snapshot()
            .focus_order()
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        for nav in ["dictation", "overlay", "dictionaries", "profiles"] {
            assert!(names.contains(&format!("ears.nav.{nav}").as_str()));
        }
        assert!(names.contains(&included));
        assert!(!names.contains(&excluded));
    }
    assert!(!device_scene::page_matches("dictation.*", "dictation"));
    assert!(device_scene::page_matches("dictation.*", "dictation.input"));
    assert!(!device_scene::page_matches(
        "dictation.*",
        "dictation-other.input"
    ));
}
#[test]
fn invalid_page_selectors_fail() {
    reject(INPUT, |v| {
        v["scene"]["devices"][0]["controls"][0]["page"] = json!("dic*tion")
    });
    reject(INPUT, |v| v["state"]["active_page"] = json!("dictation.*"));
}
#[test]
fn rect_z_is_front_and_depth_extends_back() {
    let rect = Rect {
        x: 2.0,
        y: 3.0,
        z: 4.0,
        width: 6.0,
        height: 8.0,
        depth: 10.0,
    };
    let corners = rect.corners([1.0, 2.0, 3.0]);
    for p in corners {
        assert!([-3.0, 7.0].contains(&p.z));
        assert!([0.0, 6.0].contains(&p.x));
        assert!([1.0, 9.0].contains(&p.y));
    }
}
#[test]
fn pixel_budget_is_bounded_even_for_extreme_aspects() {
    let mut s: Snapshot = serde_json::from_str(INPUT).unwrap();
    for (w, h, cap) in [
        (16384.0, 1.0, 1),
        (1.0, 16384.0, 2),
        (16384.0, 16384.0, 4096),
        (1280.0, 720.0, 4194304),
    ] {
        s.viewport.width_dip = w;
        s.viewport.height_dip = h;
        s.viewport.scale_factor = 8.0;
        s.viewport.pixel_cap = cap;
        let e = s.viewport.extent().unwrap();
        assert!(e.width_px > 0 && e.height_px > 0);
        assert!(u64::from(e.width_px) * u64::from(e.height_px) <= u64::from(cap));
    }
}
#[test]
fn nonfinite_typed_geometry_is_rejected() {
    let mut s: Snapshot = serde_json::from_str(INPUT).unwrap();
    s.scene.devices[0].controls[0].rect.width = f64::NAN;
    assert!(ValidatedSnapshot::new(s).is_err());
}
#[test]
fn projection_and_ray_agree_after_resize_and_camera_change() {
    let mut s: Snapshot = serde_json::from_str(INPUT).unwrap();
    for (yaw, pitch, scale, width, height) in [
        (0.0, 0.0, 1.0, 1280.0, 720.0),
        (-18.0, 16.0, 2.0, 1440.0, 900.0),
        (30.0, 25.0, 1.5, 800.0, 700.0),
    ] {
        s.scene.camera.yaw_deg = yaw;
        s.scene.camera.pitch_deg = pitch;
        s.viewport.scale_factor = scale;
        s.viewport.width_dip = width;
        s.viewport.height_dip = height;
        let p = Projection::new(&ValidatedSnapshot::new(s.clone()).unwrap()).unwrap();
        for point in [
            DVec3::ZERO,
            DVec3::new(-10.0, 2.0, 0.8),
            DVec3::new(20.0, -4.0, 1.0),
        ] {
            let screen = p.project(point).unwrap();
            let ray = p.ray(screen[0], screen[1]).unwrap();
            assert!((point - ray.origin).cross(ray.direction).length() < 1e-7);
        }
        assert!(p.ray(-1.0, 0.0).is_none());
        assert!(p.ray(f64::NAN, 0.0).is_none());
    }
}
#[test]
fn wholly_behind_camera_scene_is_not_a_successful_projection() {
    let mut s: Snapshot = serde_json::from_str(INPUT).unwrap();
    s.scene.camera.yaw_deg = 0.0;
    s.scene.camera.pitch_deg = 0.0;
    s.scene.devices[0].origin = [0.0, 0.0, 1000.0];
    let validated = ValidatedSnapshot::new(s).unwrap();
    assert!(
        Projection::new(&validated).is_err(),
        "silently clipped entire device behind camera"
    );
}
