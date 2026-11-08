//! Shared fader geometry: a wide rectangle is horizontal, a tall one vertical.
//! Rendering and pointer editing use the same cap and travel, at either scale.
use crate::Rect;

pub fn fader_thumb(rect: &Rect, fraction: f64) -> Rect {
    let mut thumb = rect.clone();
    if rect.width >= rect.height {
        // Compact horizontal cap; the full scene rectangle remains the hit area.
        thumb.width = (rect.width * 0.18).min(0.8);
        thumb.height = rect.height * 0.75;
        thumb.x += (fraction.clamp(0.0, 1.0) - 0.5) * (rect.width - thumb.width);
    } else {
        thumb.height = (rect.height * 0.18).min(1.1);
        thumb.y += (fraction.clamp(0.0, 1.0) - 0.5) * (rect.height - thumb.height);
    }
    thumb
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_axes_have_bounded_caps_and_exact_endpoints() {
        for (width, height) in [(10.0, 1.6), (1.6, 10.0)] {
            let rect = Rect {
                x: 2.0,
                y: 3.0,
                z: 0.8,
                width,
                height,
                depth: 0.15,
            };
            let low = fader_thumb(&rect, -1.0);
            let high = fader_thumb(&rect, 2.0);
            let center = fader_thumb(&rect, 0.5);
            assert_eq!((center.x, center.y), (rect.x, rect.y));
            if width >= height {
                assert!((low.x - low.width / 2.0 - (rect.x - width / 2.0)).abs() < 1e-9);
                assert!((high.x + high.width / 2.0 - (rect.x + width / 2.0)).abs() < 1e-9);
                assert_eq!(low.height, height * 0.75);
            } else {
                assert!((low.y - low.height / 2.0 - (rect.y - height / 2.0)).abs() < 1e-9);
                assert!((high.y + high.height / 2.0 - (rect.y + height / 2.0)).abs() < 1e-9);
                assert_eq!(low.width, width);
            }
        }
    }
}
