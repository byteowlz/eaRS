use super::*;

#[test]
fn choices_use_bounded_equal_tracks_for_long_and_compact_labels() {
    for (count, columns) in [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 3),
        (8, 3),
        (9, 3),
    ] {
        assert_eq!(segment_columns(count, false), columns);
    }
    assert_eq!(segment_columns(7, true), 7); // language codes stay on one row
    assert_eq!(segment_columns(5, true), 5); // short numeric values need no wrapping
    assert_eq!(FORM_CONTROL_WIDTH, 300.0);
}
