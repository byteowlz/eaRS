//! Theme-aware GPUI adapter for shared motion vector output.
use gpui::{Hsla, IntoElement, Path, Rgba, Styled, canvas, point, px};
use motion_render::RenderFrame;
use std::sync::Arc;

pub fn mark(
    snapshot: Arc<RenderFrame>,
    body: Hsla,
    background: Hsla,
    size: f32,
) -> impl IntoElement {
    canvas(
        |_, _, _| (),
        move |bounds, (), window, _| {
            let body: Rgba = body.into();
            let bg: Rgba = background.into();
            let stripe = Rgba {
                r: body.r * 0.8 + bg.r * 0.2,
                g: body.g * 0.8 + bg.g * 0.2,
                b: body.b * 0.8 + bg.b * 0.2,
                a: 1.0,
            };
            let [width, height] = snapshot.view_box;
            let scale = (size / width).min(size / height);
            let ox = (size - width * scale) / 2.0;
            let oy = (size - height * scale) / 2.0;
            for region in &snapshot.regions {
                let color = Rgba {
                    r: body.r + (stripe.r - body.r) * region.stripe_mix,
                    g: body.g + (stripe.g - body.g) * region.stripe_mix,
                    b: body.b + (stripe.b - body.b) * region.stripe_mix,
                    a: region.opacity * body.a,
                };
                let to_point = |p: [f32; 2]| {
                    point(
                        bounds.origin.x + px(ox + p[0] * scale),
                        bounds.origin.y + px(oy + p[1] * scale),
                    )
                };
                let mut path = Path::new(bounds.origin);
                for t in region.triangles.chunks_exact(3) {
                    path.push_triangle(
                        (to_point(t[0]), to_point(t[1]), to_point(t[2])),
                        (point(0., 1.), point(0., 1.), point(0., 1.)),
                    );
                }
                window.paint_path(path, Hsla::from(color));
            }
        },
    )
    .size(px(size))
    .flex_shrink_0()
}
