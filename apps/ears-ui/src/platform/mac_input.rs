//! Public Core Animation mask for this GPUI Metal popup's input silhouette.
//! No global event monitors, polling, forwarded clicks or private window APIs.
use super::super::OverlayInputShape;
use gpui::Window;
use objc2_app_kit::NSView;
use objc2_core_foundation::CFRetained;
use objc2_core_graphics::CGMutablePath;
use objc2_foundation::NSSize;
use objc2_quartz_core::{CAShapeLayer, CATransaction};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::ptr::null;

fn silhouette(shape: OverlayInputShape, bounds: NSSize) -> CFRetained<CGMutablePath> {
    let path = CGMutablePath::new();
    let l = (bounds.width - f64::from(shape.width)) / 2.0;
    let r = l + f64::from(shape.width);
    let h = f64::from(shape.height);
    let radius = f64::from(shape.radius).min(h / 2.0).min((r - l) / 2.0);
    let t = f64::from(shape.throat);
    // Core Animation mask paths use unflipped layer coordinates.
    let y = |top: f64| bounds.height - top;
    let k = radius * 0.5522847498307936;
    // SAFETY: the path is owned, all transforms are explicitly null (identity),
    // and the caller supplies finite painted geometry on the main thread.
    unsafe {
        CGMutablePath::move_to_point(Some(&path), null(), l, y(0.0));
        CGMutablePath::add_line_to_point(Some(&path), null(), r, y(0.0));
        CGMutablePath::add_line_to_point(Some(&path), null(), r, y(h - radius));
        CGMutablePath::add_curve_to_point(
            Some(&path),
            null(),
            r,
            y(h - radius + k),
            r - radius + k,
            y(h),
            r - radius,
            y(h),
        );
        CGMutablePath::add_line_to_point(Some(&path), null(), l + radius, y(h));
        CGMutablePath::add_curve_to_point(
            Some(&path),
            null(),
            l + radius - k,
            y(h),
            l,
            y(h - radius + k),
            l,
            y(h - radius),
        );
        CGMutablePath::close_subpath(Some(&path));
        // The same two cubic outward throats painted by notch_throats.
        CGMutablePath::move_to_point(Some(&path), null(), l - t, y(0.0));
        CGMutablePath::add_line_to_point(Some(&path), null(), l, y(0.0));
        CGMutablePath::add_line_to_point(Some(&path), null(), l, y(t));
        CGMutablePath::add_curve_to_point(
            Some(&path),
            null(),
            l,
            y(t * 0.3),
            l,
            y(0.0),
            l - t,
            y(0.0),
        );
        CGMutablePath::close_subpath(Some(&path));
        CGMutablePath::move_to_point(Some(&path), null(), r, y(0.0));
        CGMutablePath::add_line_to_point(Some(&path), null(), r + t, y(0.0));
        CGMutablePath::add_curve_to_point(Some(&path), null(), r, y(0.0), r, y(t * 0.3), r, y(t));
        CGMutablePath::close_subpath(Some(&path));
    }
    path
}

pub fn set_overlay_input_shape(window: &Window, shape: OverlayInputShape) -> bool {
    let Ok(handle) = HasWindowHandle::window_handle(window) else {
        return false;
    };
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return false;
    };
    // SAFETY: GPUI owns this live NSView throughout this main-thread render.
    let view = unsafe { &*(handle.ns_view.as_ptr() as *mut NSView) };
    let Some(layer) = view.layer() else {
        return false;
    };
    let path = silhouette(shape, view.bounds().size);
    let mask = CAShapeLayer::new();
    CATransaction::begin();
    CATransaction::setDisableActions(true); // Never animate input independently of the painted frame.
    mask.setFrame(layer.bounds());
    mask.setContentsScale(layer.contentsScale());
    mask.setPath(Some(&path));
    // SAFETY: the mask is newly allocated, has no superlayer and is retained by layer.
    unsafe { layer.setMask(Some(&mask)) };
    CATransaction::commit();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_graphics::CGPath;
    use objc2_foundation::NSPoint;
    #[test]
    fn native_mask_matches_body_lower_corners_throats_and_resize() {
        for (width, height) in [(185.0, 32.0), (300.0, 32.0), (560.0, 32.0), (560.0, 68.0)] {
            let shape = OverlayInputShape {
                width,
                height,
                radius: 11.2,
                throat: 12.0,
            };
            let path = silhouette(shape, NSSize::new(640.0, 110.0));
            let inside = |x, y| {
                // SAFETY: identity transform and a live, immutable owned path.
                unsafe {
                    CGPath::contains_point(Some(&path), null(), NSPoint::new(x, 110.0 - y), false)
                }
            };
            let left = (640.0 - f64::from(width)) / 2.0;
            assert!(inside(320.0, 16.0));
            assert!(!inside(10.0, 50.0));
            assert!(!inside(320.0, f64::from(height) + 1.0));
            assert!(!inside(left + 0.5, f64::from(height) - 0.5));
            assert!(inside(left + 11.2, f64::from(height) - 1.0));
            assert!(inside(left - 1.0, 0.5));
            assert!(!inside(left - 11.5, 11.5));
        }
    }
}
