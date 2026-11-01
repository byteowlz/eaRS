use super::*;

#[test]
fn opt_in_and_global_disable_win_without_requiring_engine_state() {
    let mut config = OverlayConfig::default();
    assert!(!enabled(&config, true));
    config.recording_notch = true;
    assert!(enabled(&config, true));
    assert!(!enabled(&config, false));
    config.enabled = false;
    assert!(!enabled(&config, true));
}

#[test]
fn temporary_top_notch_does_not_mutate_saved_placement_or_appearance() {
    let config = OverlayConfig {
        recording_notch: true,
        anchor: OverlayAnchor::Bottom,
        island_mode: false,
        offset_x: 17.0,
        offset_y: 9.0,
        margin: 25.0,
        ..OverlayConfig::default()
    };
    let before = serde_json::to_value(&config).unwrap();
    let view = effective(&config, true);
    assert_eq!(view.anchor, OverlayAnchor::Top);
    assert!(view.island_mode);
    assert_eq!((view.offset_x, view.offset_y, view.margin), (0.0, 0.0, 0.0));
    assert_eq!(view.appearance, config.appearance);
    assert_eq!(serde_json::to_value(&config).unwrap(), before);
    let mut off = config.clone();
    off.recording_notch = false;
    let restored = effective(&off, true);
    assert_eq!(restored.anchor, OverlayAnchor::Bottom);
    assert!(!restored.island_mode);
    assert_eq!(
        (restored.offset_x, restored.offset_y, restored.margin),
        (17.0, 9.0, 25.0)
    );
    assert_eq!(
        serde_json::to_value(effective(&config, false).as_ref()).unwrap(),
        before
    );
}

#[test]
fn real_cutout_is_matched_and_external_display_uses_existing_fixture_geometry() {
    let hardware = Notch {
        width: 207.0,
        height: 36.0,
    };
    assert_eq!(geometry(Some(hardware), 1728.0), Some(hardware));
    assert_eq!(
        geometry(None, 1920.0),
        Some(Notch {
            width: 185.0,
            height: 32.0
        })
    );
    for width in [f32::NAN, 0.0, 639.0] {
        assert!(geometry(None, width).is_none());
    }
    assert_eq!(
        geometry(
            Some(Notch {
                width: f32::NAN,
                height: 32.0
            }),
            1920.0
        ),
        geometry(None, 1920.0)
    );
}

#[test]
fn all_attached_displays_are_planned_and_toggle_or_detach_removes_them() {
    let screen = |x| Bounds::new(point(px(x), px(0.0)), size(px(1920.0), px(1080.0)));
    let screens = vec![
        (
            1,
            screen(0.0),
            Some(Notch {
                width: 207.0,
                height: 36.0,
            }),
        ),
        (2, screen(-1920.0), None),
    ];
    let both = placements(true, screens.clone());
    assert_eq!(both.len(), 2);
    assert_eq!(both[&1].notch.width, 207.0);
    assert_eq!(both[&2].notch.width, 185.0);
    assert_eq!(placements(true, screens[..1].to_vec()).len(), 1);
    assert!(placements(false, screens).is_empty());
}
