//! Display-clock liveness and content-vs-position invalidation contracts.
use super::*;

#[test]
fn frame_clock_supports_refresh_rates_and_falls_back_after_stall() {
    let start = std::time::Instant::now();
    for hz in [60_u32, 120, 240] {
        let frame = std::time::Duration::from_secs_f64(1.0 / f64::from(hz));
        assert!(motion_clock_live(Some(start), start + frame));
        assert!(motion_clock_live(Some(start), start + frame * 2));
    }
    assert!(!motion_clock_live(None, start));
    assert!(!motion_clock_live(
        Some(start),
        start - std::time::Duration::from_millis(1)
    ));
    assert!(!motion_clock_live(
        Some(start),
        start + std::time::Duration::from_millis(51)
    ));
}

#[test]
fn cursor_motion_does_not_invalidate_text_layout_but_content_and_edges_do() {
    let previous = PreviewRect {
        x: 20.0,
        y: 40.0,
        width: 370.0,
        height: 156.0,
        align_bottom: false,
    };
    let moved = PreviewRect {
        x: 200.5,
        y: 300.25,
        ..previous
    };
    assert!(!needs_paint(previous, previous, false));
    assert!(!needs_paint(previous, moved, false));
    assert!(needs_paint(previous, moved, true)); // new text or selected font
    assert!(needs_paint(
        previous,
        PreviewRect {
            align_bottom: true,
            ..moved
        },
        false
    ));
    assert!(needs_paint(
        previous,
        PreviewRect {
            width: 300.0,
            ..moved
        },
        false
    ));
    assert!(needs_paint(
        previous,
        PreviewRect {
            height: 120.0,
            ..moved
        },
        false
    ));
}
