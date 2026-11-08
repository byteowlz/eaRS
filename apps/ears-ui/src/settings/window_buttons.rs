//! Anchor native etched window icons to the *painted* receiver, not the transparent
//! window's top-left. The renderer's presented projection is the source of truth.
use device_scene::{Projection, Snapshot};
use glam::DVec3;

fn region(snapshot: &Snapshot, projection: &Projection) -> Option<[f64; 4]> {
    let (device, control, _) = snapshot.control("window.buttons")?;
    let r = &control.rect;
    let point = |x, y, z| {
        projection.project(DVec3::new(
            x + device.origin[0],
            y + device.origin[1],
            z + device.origin[2],
        ))
    };
    let a = point(r.x - r.width / 2.0, r.y + r.height / 2.0, r.z)?;
    let b = point(r.x + r.width / 2.0, r.y - r.height / 2.0, r.z)?;
    let front = &device.panels.iter().find(|p| p.id == "ears.front")?.rect;
    let plate = &device.panels.iter().find(|p| p.id == "ears.top")?.rect;
    let front_y = front.y + front.height / 2.0;
    let plate_y = plate.y + plate.height / 2.0;
    // The two border edges have different depths. Centre between their painted
    // positions, not their world-space midpoint; preserve editor nudge offsets.
    let border_center = (point(0.0, front_y, front.z)?[1] + point(0.0, plate_y, plate.z)?[1]) / 2.0;
    let nominal_center = point(0.0, (front_y + plate_y) / 2.0, front.z)?[1];
    let correction = border_center - nominal_center;
    Some([a[0], a[1] + correction, b[0], b[1] + correction])
}

pub(super) fn sync(
    device: &device_gpui::DeviceView,
    buttons: &crate::platform::SettingsButtons,
    window: &gpui::Window,
) {
    let viewport = window.viewport_size();
    let anchor = device.projection().and_then(|projection| {
        // Do not put native controls over an old frame during a viewport change.
        let current = projection.viewport.width_dip == f64::from(f32::from(viewport.width))
            && projection.viewport.height_dip == f64::from(f32::from(viewport.height));
        current
            .then(|| region(device.snapshot(), &projection))
            .flatten()
    });
    buttons.position(window, anchor);
}

#[cfg(test)]
mod tests {
    use super::*;
    use device_scene::ValidatedSnapshot;

    #[test]
    fn native_buttons_follow_the_chassis_at_each_window_size_and_scale() {
        for (width, height) in [(640.0, 420.0), (1240.0, 860.0), (1600.0, 1000.0)] {
            let mut first: Option<[f64; 4]> = None;
            for scale in [1.0, 2.0] {
                let mut snapshot = crate::settings::device_layout::generated();
                snapshot.viewport.width_dip = width;
                snapshot.viewport.height_dip = height;
                snapshot.viewport.scale_factor = scale;
                let valid = ValidatedSnapshot::new(snapshot).unwrap();
                let projection = Projection::new(&valid).unwrap();
                let [left, top, right, bottom] = region(valid.snapshot(), &projection).unwrap();
                assert!(
                    left > 0.0 && top > 0.0,
                    "inside the transparent window margin"
                );
                assert!(right < width && bottom < height);
                // Three 16pt icon targets with 20pt centre spacing. Never
                // shrink them with the device or overlap the brand or tab row.
                assert!(
                    right - left >= 56.0,
                    "native buttons fit at {width}x{height}"
                );
                assert!(bottom > top);
                let snapshot = valid.snapshot();
                let d = &snapshot.scene.devices[0];
                let front = &d.panels[0].rect;
                let plate = &d.panels.iter().find(|p| p.id == "ears.top").unwrap().rect;
                let project_top = |r: &device_scene::Rect| {
                    projection
                        .project(DVec3::new(
                            d.origin[0],
                            r.y + r.height / 2.0 + d.origin[1],
                            r.z + d.origin[2],
                        ))
                        .unwrap()[1]
                };
                let border_center = (project_top(front) + project_top(plate)) / 2.0;
                assert!(
                    ((top + bottom) / 2.0 - border_center).abs() < 1.0,
                    "native controls stay vertically centred in the actual border"
                );
                if let Some(previous) = first {
                    for (a, b) in [left, top, right, bottom].into_iter().zip(previous) {
                        assert!(
                            (a - b).abs() <= 1.0,
                            "positions use points (allow one pixel of extent rounding): {a} vs {b}"
                        );
                    }
                }
                first = Some([left, top, right, bottom]);
            }
        }
    }
}
