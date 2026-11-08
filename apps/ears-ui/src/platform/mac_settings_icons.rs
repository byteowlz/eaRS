//! Resolution-independent etched-style window glyphs, using the same dark lip
//! and faint lower highlight as the receiver's printed legends. No circles or
//! button plates; the NSButton frame remains a generous native hit target.
use block2::RcBlock;
use objc2::{rc::Retained, runtime::Bool};
use objc2_app_kit::{
    NSAccessibilityCloseButtonSubrole, NSAccessibilityMinimizeButtonSubrole,
    NSAccessibilitySubrole, NSAccessibilityZoomButtonSubrole, NSBezierPath, NSColor, NSImage,
    NSLineCapStyle,
};
use objc2_foundation::{NSPoint, NSRect, NSSize};

type Segment = ([f64; 2], [f64; 2]);
#[derive(Clone, Copy)]
pub(super) enum Glyph {
    Close,
    Minimize,
    Zoom,
}
impl Glyph {
    pub const ALL: [Self; 3] = [Self::Close, Self::Minimize, Self::Zoom];
    pub fn label(self) -> &'static str {
        match self {
            Self::Close => "Close settings",
            Self::Minimize => "Minimise settings",
            Self::Zoom => "Zoom settings",
        }
    }
    pub fn subrole(self) -> &'static NSAccessibilitySubrole {
        // SAFETY: public immutable AppKit constants, loaded with the framework.
        unsafe {
            match self {
                Self::Close => NSAccessibilityCloseButtonSubrole,
                Self::Minimize => NSAccessibilityMinimizeButtonSubrole,
                Self::Zoom => NSAccessibilityZoomButtonSubrole,
            }
        }
    }
    fn segments(self) -> &'static [Segment] {
        match self {
            Self::Close => &[([5.0, 5.0], [11.0, 11.0]), ([11.0, 5.0], [5.0, 11.0])],
            Self::Minimize => &[([4.5, 8.0], [11.5, 8.0])],
            Self::Zoom => &[
                ([4.5, 4.5], [11.5, 4.5]),
                ([11.5, 4.5], [11.5, 11.5]),
                ([11.5, 11.5], [4.5, 11.5]),
                ([4.5, 11.5], [4.5, 4.5]),
            ],
        }
    }
    pub fn image(self) -> Retained<NSImage> {
        let drawing = RcBlock::new(move |_: NSRect| -> Bool {
            // Top-left coordinates: highlight below, dark lip above, then ink.
            for (offset, white, alpha) in [(0.8, 1.0, 0.12), (-0.64, 0.0, 0.75), (0.0, 0.67, 1.0)] {
                NSColor::colorWithSRGBRed_green_blue_alpha(white, white, white, alpha).setStroke();
                let path = NSBezierPath::bezierPath();
                path.setLineWidth(1.0);
                path.setLineCapStyle(NSLineCapStyle::Round);
                for (a, b) in self.segments() {
                    path.moveToPoint(NSPoint::new(a[0], a[1] + offset));
                    path.lineToPoint(NSPoint::new(b[0], b[1] + offset));
                }
                path.stroke();
            }
            Bool::YES
        });
        NSImage::imageWithSize_flipped_drawingHandler(NSSize::new(16.0, 16.0), true, &drawing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_vector_images_render_monochrome_ink_at_one_and_two_x() {
        use objc2::AnyThread;
        use objc2_app_kit::{
            NSBitmapImageRep, NSCompositingOperation, NSDeviceRGBColorSpace, NSGraphicsContext,
        };
        let mut coverage = [[0usize; 3]; 2];
        for scale in [1, 2] {
            let pixels = 16 * scale;
            // SAFETY: null planes request AppKit-owned storage; explicit RGBA
            // row/pixel sizes match the buffer cleared and inspected below.
            let bitmap = unsafe {
                NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel(
                    NSBitmapImageRep::alloc(), std::ptr::null_mut(), pixels, pixels, 8, 4, true, false,
                    NSDeviceRGBColorSpace, pixels * 4, 32,
                )
            }.unwrap();
            bitmap.setSize(NSSize::new(16.0, 16.0));
            let context = NSGraphicsContext::graphicsContextWithBitmapImageRep(&bitmap).unwrap();
            let length = (bitmap.bytesPerRow() * pixels) as usize;
            let data = bitmap.bitmapData();
            assert!(!data.is_null());
            for (index, glyph) in Glyph::ALL.into_iter().enumerate() {
                // SAFETY: bitmap owns this writable packed buffer for the loop.
                unsafe {
                    data.write_bytes(0, length);
                }
                NSGraphicsContext::saveGraphicsState_class();
                NSGraphicsContext::setCurrentContext(Some(&context));
                glyph.image().drawInRect_fromRect_operation_fraction(
                    NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(16.0, 16.0)),
                    NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(16.0, 16.0)),
                    NSCompositingOperation::SourceOver,
                    1.0,
                );
                NSGraphicsContext::restoreGraphicsState_class();
                // SAFETY: drawing is complete and this immutable view does not
                // escape the bitmap's lifetime or overlap another draw.
                let bytes = unsafe { std::slice::from_raw_parts(data, length) };
                coverage[(scale - 1) as usize][index] =
                    bytes.chunks_exact(4).filter(|p| p[3] > 0).count();
                assert!(
                    bytes.chunks_exact(4).any(|p| p[3] > 0),
                    "glyph has visible strokes at {scale}x"
                );
                assert!(
                    bytes.chunks_exact(4).all(|p| p[0] == p[1] && p[1] == p[2]),
                    "no coloured traffic-light pixels"
                );
            }
        }
        assert!(
            (0..3).all(|i| coverage[1][i] > 2 * coverage[0][i]),
            "2x glyphs must actually use the larger backing: {coverage:?}"
        );
    }
    #[test]
    fn glyphs_are_distinct_compact_strokes_inside_the_native_hit_area() {
        assert_eq!(Glyph::Close.segments().len(), 2);
        assert_eq!(Glyph::Minimize.segments().len(), 1);
        assert_eq!(Glyph::Zoom.segments().len(), 4);
        for glyph in Glyph::ALL {
            assert!(!glyph.label().is_empty());
            for (a, b) in glyph.segments() {
                for p in [a, b] {
                    assert!(p.iter().all(|v| (4.0..=12.0).contains(v)));
                }
            }
        }
    }
}
