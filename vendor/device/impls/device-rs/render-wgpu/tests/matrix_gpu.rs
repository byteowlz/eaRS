use device_render_wgpu::{OffscreenRenderer, TextRaster};
use device_scene::{Snapshot, ValidatedSnapshot};
use glam::DVec3;
const FIXTURE: &str = include_str!("../../../../spec/fixtures/device-scene-v1/spqx-receiver.json");

fn pixel(frame: &device_render_wgpu::RenderFrame, world: DVec3) -> &[u8] {
    let p = frame.projection.project(world).unwrap();
    let x = (p[0] * frame.projection.extent.scale_x).floor() as usize;
    let y = (p[1] * frame.projection.extent.scale_y).floor() as usize;
    &frame.pixels[(y * frame.width_px as usize + x) * 4..][..4]
}

#[test]
#[ignore = "actual native adapter required; run explicitly with --ignored"]
fn matrix_phosphors_are_gpu_pixels_at_fixed_pitch_in_front_and_angled_views() {
    let mut renderer = pollster::block_on(OffscreenRenderer::new()).unwrap();
    for scale in [1.0, 2.0] {
        for (yaw, pitch) in [(0.0, 0.0), (-18.0, 12.0)] {
            let mut s = Snapshot::parse(FIXTURE).unwrap().into_inner();
            s.viewport.width_dip = 1100.0;
            s.viewport.height_dip = 700.0;
            s.viewport.scale_factor = scale;
            s.scene.camera.yaw_deg = yaw;
            s.scene.camera.pitch_deg = pitch;
            let id = s.scene.devices[0]
                .controls
                .iter()
                .find(|c| c.matrix.is_some())
                .unwrap()
                .id
                .clone();
            s.state.controls.get_mut(&id).unwrap().value =
                device_scene::Value(serde_json::json!("A"));
            let (d, c, _) = s.control(&id).unwrap();
            let m = c.matrix.as_ref().unwrap();
            let origin = DVec3::new(
                d.origin[0] + c.rect.x - c.rect.width / 2.0 + m.padding_x,
                d.origin[1] + c.rect.y + c.rect.height / 2.0 - m.padding_y,
                d.origin[2] + c.rect.z + 0.025,
            );
            let off = origin + DVec3::new(m.pitch * 0.5, -m.pitch * 0.5, 0.0);
            let on = origin + DVec3::new(m.pitch * 1.5, -m.pitch * 0.5, 0.0);
            let frame = renderer
                .render(&ValidatedSnapshot::new(s.clone()).unwrap(), 1)
                .unwrap();
            assert_eq!(
                frame
                    .labels
                    .iter()
                    .find(|l| l.control_id.as_deref() == Some(&id))
                    .unwrap()
                    .raster,
                TextRaster::Matrix
            );
            let (lit, dark) = (pixel(&frame, on), pixel(&frame, off));
            assert!(
                lit[2] > dark[2] + 40,
                "lit={lit:?} off={dark:?} scale={scale} yaw={yaw}"
            );
            assert_eq!(lit[3], 255);
            assert_eq!(
                frame.hit_test(
                    frame.projection.project(on).unwrap()[0],
                    frame.projection.project(on).unwrap()[1]
                ),
                Some(id.as_str())
            );
            s.state.controls.get_mut(&id).unwrap().value =
                device_scene::Value(serde_json::json!("日本語"));
            let fallback = renderer
                .render(&ValidatedSnapshot::new(s).unwrap(), 2)
                .unwrap();
            let label = fallback
                .labels
                .iter()
                .find(|l| l.control_id.as_deref() == Some(&id))
                .unwrap();
            assert_eq!(label.raster, TextRaster::UnicodeFallback);
            assert_eq!(label.text, "日本語");
        }
    }
}
