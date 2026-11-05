//! Explicit fixture CLI. The library never opens files or writes proof artifacts.
use device_render_wgpu::OffscreenRenderer;
use device_scene::Snapshot;
use std::{error::Error, path::PathBuf, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: device-render <snapshot.json> <output.png>".into());
    }
    let input = std::fs::read_to_string(&args[0])?;
    let snapshot = Snapshot::parse(&input)?;
    let mut renderer = pollster::block_on(OffscreenRenderer::new())?;
    let start = Instant::now();
    let frame = renderer.render(&snapshot, 1)?;
    let elapsed = start.elapsed();
    // PNG stores straight RGBA; GPUI consumes the original premultiplied BGRA frame.
    let mut rgba = frame.pixels.clone();
    for p in rgba.chunks_exact_mut(4) {
        p.swap(0, 2);
        if p[3] > 0 {
            for i in 0..3 {
                p[i] = ((u32::from(p[i]) * 255) / u32::from(p[3])).min(255) as u8;
            }
        }
    }
    image::save_buffer(
        &args[1],
        &rgba,
        frame.width_px,
        frame.height_px,
        image::ColorType::Rgba8,
    )?;
    let metadata = serde_json::json!({
        "request_generation":frame.request_generation,"scene_id":frame.scene_id,
        "scene_revision":frame.scene_revision,"state_revision":frame.state_revision,
        "width_px":frame.width_px,"height_px":frame.height_px,"stride_bytes":frame.stride_bytes,
        "projection":frame.projection,"adapter":renderer.adapter_name,"render_readback_ms":elapsed.as_secs_f64()*1000.0,
        "triangles":frame.hit_geometry.len(),"labels_not_in_png":frame.labels.len(),
        "limitations":["geometry-only GPU specimen; labels are returned for native host rendering","no shadow map or environment map; analytical material approximation"]
    });
    let output = PathBuf::from(&args[1]).with_extension("json");
    std::fs::write(output, serde_json::to_vec_pretty(&metadata)?)?;
    println!(
        "{}x{} on {}: {:.1} ms; {} triangles; {} host labels",
        frame.width_px,
        frame.height_px,
        renderer.adapter_name,
        elapsed.as_secs_f64() * 1000.0,
        frame.hit_geometry.len(),
        frame.labels.len()
    );
    Ok(())
}
