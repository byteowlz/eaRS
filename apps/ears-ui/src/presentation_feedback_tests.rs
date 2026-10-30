use super::*;
#[test]
fn word_slot_is_centered_in_capsule_not_beside_logo() {
    for width in [220.0, 260.0, 300.0, 420.0] {
        let (left, space) = word_band(width, 47.0);
        assert_eq!(left + space / 2.0, width / 2.0);
        assert!(left >= 47.0 && space > 0.0);
    }
    assert_eq!(word_band(220.0, 47.0), (47.0, 126.0));
    assert_eq!(word_band(20.0, 47.0), (10.0, 0.0));
}

#[test]
fn pointer_path_changes_position_and_keeps_the_whole_body_on_screen() {
    let mut previous = None;
    for point in [(100.0, 100.0), (400.0, 400.0), (1800.0, 1000.0)] {
        let rect = preview_rect((1920.0, 1080.0), PreviewPlacement::Cursor, Some(point)).unwrap();
        if let Some(old) = previous {
            assert_ne!(rect, old);
        }
        assert!(
            rect.x >= 16.0
                && rect.y >= 16.0
                && rect.x + rect.width <= 1904.0
                && rect.y + rect.height <= 1064.0
        );
        previous = Some(rect);
    }
}
