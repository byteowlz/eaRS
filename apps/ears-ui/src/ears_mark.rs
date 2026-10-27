//! eaRS mark, built from the seven rectangles in `logo/ears_logo_white.svg`.
//! The standing wave offsets each bar locally; no source artwork is cropped.

use gpui::{Hsla, IntoElement, PathBuilder, Styled, canvas, point, px};
use std::f32::consts::{FRAC_1_SQRT_2, PI};

const BARS: [(f32, f32, f32, f32, f32); 7] = [
    (213.31, 107.45, 255.99, 235.45, -97.53),
    (268.16, 60.04, 460.51, 290.30, -120.25),
    (323.01, 78.38, 533.54, 345.15, -142.97),
    (377.86, 49.06, 701.88, 400.00, -165.69),
    (432.71, 188.08, 533.54, 454.85, -188.41),
    (487.56, 279.45, 460.51, 509.70, -211.13),
    (542.41, 436.56, 255.99, 564.55, -233.85),
];
const WIDTH: f32 = 44.28;
const VIEWBOX: f32 = 800.0;

fn bar_point(
    bar: (f32, f32, f32, f32, f32),
    u: f32,
    edge: f32,
    phase: f32,
    index: usize,
    animate: bool,
) -> (f32, f32) {
    let (x, y, height, tx, ty) = bar;
    let wave = if animate {
        22.0 * (u * 6.0 * PI).sin() * (phase + index as f32 * 0.25).sin()
    } else {
        0.0
    };
    let x = x + edge + wave;
    let y = y + u * height;
    ((x - y) * FRAC_1_SQRT_2 + tx, (x + y) * FRAC_1_SQRT_2 + ty)
}

pub fn mark(color: Hsla, size: f32, phase: f32, animate: bool) -> impl IntoElement {
    canvas(
        |_, _, _| (),
        move |bounds, _, window, _| {
            let width: f32 = bounds.size.width.into();
            let height: f32 = bounds.size.height.into();
            for (index, bar) in BARS.iter().copied().enumerate() {
                let mut points = Vec::with_capacity(50);
                for step in 0..=24 {
                    let (x, y) = bar_point(bar, step as f32 / 24.0, 0.0, phase, index, animate);
                    points.push(point(
                        bounds.origin.x + px(x * width / VIEWBOX),
                        bounds.origin.y + px(y * height / VIEWBOX),
                    ));
                }
                for step in (0..=24).rev() {
                    let (x, y) = bar_point(bar, step as f32 / 24.0, WIDTH, phase, index, animate);
                    points.push(point(
                        bounds.origin.x + px(x * width / VIEWBOX),
                        bounds.origin.y + px(y * height / VIEWBOX),
                    ));
                }
                let mut path = PathBuilder::fill();
                path.add_polygon(&points, true);
                if let Ok(path) = path.build() {
                    window.paint_path(path, color);
                }
            }
        },
    )
    .w(px(size))
    .h(px(size))
    .flex_shrink_0()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_wave_preserves_endpoints_and_fixed_nodes() {
        let bar = BARS[3];
        for u in [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] {
            let a = bar_point(bar, u, 0.0, 0.0, 3, true);
            let b = bar_point(bar, u, 0.0, 1.5, 3, true);
            assert!((a.0 - b.0).abs() < 0.001 && (a.1 - b.1).abs() < 0.001);
        }
        let a = bar_point(bar, 1.0 / 12.0, 0.0, 0.0, 3, true);
        let b = bar_point(bar, 1.0 / 12.0, 0.0, 1.5, 3, true);
        assert!((a.0 - b.0).abs() > 1.0);
    }
}
