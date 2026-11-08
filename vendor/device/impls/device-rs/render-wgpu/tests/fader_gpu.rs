use device_render_wgpu::OffscreenRenderer;
use device_scene::{Kind, Presentation, Snapshot, ValidatedSnapshot, Value, fader_thumb};
use glam::DVec3;

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly for native proof"]
fn both_fader_axes_move_the_physical_cap_at_each_scale() {
    let mut renderer = pollster::block_on(OffscreenRenderer::new()).unwrap();
    for horizontal in [false, true] {
        for scale in [1.0, 2.0] {
            let mut s = Snapshot::parse(include_str!(
                "../../../../spec/fixtures/device-scene-v1/spqx-receiver.json"
            ))
            .unwrap()
            .into_inner();
            s.viewport.width_dip = 641.0;
            s.viewport.height_dip = 361.0;
            s.viewport.scale_factor = scale;
            let c = s.scene.devices[0]
                .controls
                .iter_mut()
                .find(|c| c.kind == Kind::Knob)
                .unwrap();
            c.kind = Kind::Fader;
            c.presentation = Presentation::None;
            (c.rect.width, c.rect.height) = if horizontal { (4.0, 1.2) } else { (1.2, 4.0) };
            let id = c.id.clone();
            let range = c.range.clone().unwrap();
            let mut previous: Option<Vec<u8>> = None;
            for (generation, value) in [(1, range.min), (2, range.max)] {
                s.state.controls.get_mut(&id).unwrap().value = Value(serde_json::json!(value));
                let frame = renderer
                    .render(&ValidatedSnapshot::new(s.clone()).unwrap(), generation)
                    .unwrap();
                let (d, c, _) = s.control(&id).unwrap();
                let cap = fader_thumb(&c.rect, if generation == 1 { 0.0 } else { 1.0 });
                let p = frame
                    .projection
                    .project(DVec3::new(
                        cap.x + d.origin[0],
                        cap.y + d.origin[1],
                        cap.z + d.origin[2],
                    ))
                    .unwrap();
                assert_eq!(frame.hit_test(p[0], p[1]), Some(id.as_str()));
                if let Some(previous) = previous {
                    let changed = frame
                        .pixels
                        .chunks_exact(4)
                        .zip(previous.chunks_exact(4))
                        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 8))
                        .count();
                    assert!(
                        changed > 12,
                        "native cap pixels must move, not just its semantic value"
                    );
                }
                previous = Some(frame.pixels);
            }
        }
    }
}
