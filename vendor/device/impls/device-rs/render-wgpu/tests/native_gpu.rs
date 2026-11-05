use device_render_wgpu::OffscreenRenderer;
use device_scene::{Snapshot, ValidatedSnapshot};
use glam::DVec3;
const INPUT: &str = include_str!("../../../../spec/fixtures/device-scene-v1/ears-input.json");
const SHORTCUTS: &str =
    include_str!("../../../../spec/fixtures/device-scene-v1/ears-shortcuts.json");
const SPQX: &str = include_str!("../../../../spec/fixtures/device-scene-v1/spqx-receiver.json");

/// Explicit GPU proof: invoke with --ignored; never silently passes without an adapter.
#[test]
#[ignore = "requires a native wgpu adapter; run explicitly for native proof"]
fn actual_gpu_frames_preserve_projection_hits_alpha_and_channel_order() {
    let mut renderer = pollster::block_on(OffscreenRenderer::new()).unwrap();
    for (fixture, page) in [(INPUT, "input"), (SHORTCUTS, "shortcuts"), (SPQX, "spqx")] {
        for scale in [1.0, 2.0] {
            for angled in [false, true] {
                let mut s = Snapshot::parse(fixture).unwrap().into_inner();
                s.viewport.scale_factor = scale;
                s.viewport.width_dip = 641.0;
                s.viewport.height_dip = 361.0; // non-256 aligned row
                if !angled {
                    s.scene.camera.yaw_deg = 0.0;
                    s.scene.camera.pitch_deg = 0.0;
                }
                let frame = renderer
                    .render(&ValidatedSnapshot::new(s.clone()).unwrap(), 7)
                    .unwrap();
                assert_eq!(
                    frame.pixels.len(),
                    frame.width_px as usize * frame.height_px as usize * 4
                );
                assert_eq!(frame.request_generation, 7);
                assert_eq!(&frame.pixels[..4], &[0, 0, 0, 0]);
                assert!(frame.pixels.chunks_exact(4).any(|p| p[3] == 255));
                assert!(
                    frame
                        .pixels
                        .chunks_exact(4)
                        .all(|p| p[0] <= p[3] && p[1] <= p[3] && p[2] <= p[3])
                );
                if page != "spqx" {
                    let (d, c, _) = s.control("ears.nav.dictation").unwrap();
                    let world = DVec3::new(
                        c.rect.x + d.origin[0],
                        c.rect.y + d.origin[1],
                        c.rect.z + d.origin[2] + 0.001,
                    );
                    let p = frame.projection.project(world).unwrap();
                    assert_eq!(frame.hit_test(p[0], p[1]), Some("ears.nav.dictation"));
                    let absent = if page == "input" {
                        "ears.shortcuts.toggle"
                    } else {
                        "ears.input.device"
                    };
                    assert!(
                        !frame
                            .hit_geometry
                            .iter()
                            .any(|t| t.control_id.as_deref() == Some(absent))
                    );
                }
            }
        }
    }
    let mut s = Snapshot::parse(INPUT).unwrap().into_inner();
    s.viewport.width_dip = 641.0;
    s.viewport.height_dip = 361.0;
    s.viewport.scale_factor = 1.0;
    s.scene.materials.get_mut("key").unwrap().base = "#ff0000".into();
    let frame = renderer
        .render(&ValidatedSnapshot::new(s.clone()).unwrap(), 8)
        .unwrap();
    let (_, c, _) = s.control("ears.nav.dictation").unwrap();
    let p = frame
        .projection
        .project(DVec3::new(c.rect.x, c.rect.y, c.rect.z))
        .unwrap();
    let x = (p[0] * frame.projection.extent.scale_x).floor() as usize;
    let y = (p[1] * frame.projection.extent.scale_y).floor() as usize;
    let pixel = &frame.pixels[(y * frame.width_px as usize + x) * 4..][..4];
    assert!(pixel[2] > pixel[0] + 50, "BGRA red sentinel: {pixel:?}");
    // Put the key behind the solid chassis. Its aperture does not allow clicking through the chassis.
    s.scene.devices[0]
        .controls
        .iter_mut()
        .find(|c| c.id == "ears.nav.dictation")
        .unwrap()
        .rect
        .z = -2.0;
    let frame = renderer
        .render(&ValidatedSnapshot::new(s).unwrap(), 9)
        .unwrap();
    assert_ne!(frame.hit_test(p[0], p[1]), Some("ears.nav.dictation"));
}
