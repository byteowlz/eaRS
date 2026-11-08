//! Shared fader geometry: a wide rectangle is horizontal, a tall one vertical.
//! Rendering and pointer editing use the same cap and travel, at either scale.
use crate::Rect;

/// A physical cap in the retro hi-fi r10 proportions: narrow along the travel and
/// spanning most of the slot's cross axis. It fills the control's declared volume
/// [z - depth, z]: hosts give a fader real depth so the cap stands proud of the
/// plate it is mounted on (z - depth) and casts a shadow.
pub fn fader_thumb(rect: &Rect, fraction: f64) -> Rect {
    let mut thumb = rect.clone();
    let horizontal = rect.width >= rect.height;
    let (along, across) = if horizontal {
        (rect.width, rect.height)
    } else {
        (rect.height, rect.width)
    };
    let cross = across * 0.75;
    let travel = (across * 0.42).min(along * 0.18);
    let offset = (fraction.clamp(0.0, 1.0) - 0.5) * (along - travel);
    if horizontal {
        (thumb.width, thumb.height) = (travel, cross);
        thumb.x += offset;
    } else {
        (thumb.width, thumb.height) = (cross, travel);
        thumb.y += offset;
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
            // The cap fills the declared volume; it never rises past the control face.
            assert_eq!((center.z, center.depth), (rect.z, rect.depth));
            assert_eq!((center.x, center.y), (rect.x, rect.y));
            if width >= height {
                assert!((low.x - low.width / 2.0 - (rect.x - width / 2.0)).abs() < 1e-9);
                assert!((high.x + high.width / 2.0 - (rect.x + width / 2.0)).abs() < 1e-9);
                assert_eq!(low.height, height * 0.75);
                assert!(low.width < low.height, "narrow along the travel");
            } else {
                assert!((low.y - low.height / 2.0 - (rect.y - height / 2.0)).abs() < 1e-9);
                assert!((high.y + high.height / 2.0 - (rect.y + height / 2.0)).abs() < 1e-9);
                assert_eq!(low.width, width * 0.75);
                assert!(low.height < low.width, "narrow along the travel");
            }
        }
    }
}
