//! Regression: a paused Recording notch must not recreate its window every tick.
use super::*;

fn ticks(
    applied: &mut WindowConfiguration,
    overlay: &OverlayConfig,
    recording: bool,
    phase: Phase,
    count: usize,
) -> Vec<InputStep> {
    (0..count)
        .map(|_| reconcile_input(applied, overlay, recording, phase))
        .collect()
}

#[test]
fn paused_recording_notch_settles_passive_without_recreating_every_tick() {
    // Owner's live configuration: interactive island preference.
    let overlay = OverlayConfig {
        recording_notch: true,
        click_through: false,
        ..OverlayConfig::default()
    };
    // Opened interactively while dictating, then dictation pauses.
    let mut applied = window_configuration(&overlay, true);
    let paused = ticks(&mut applied, &overlay, true, Phase::Dormant, 30);
    let recreated = paused.iter().filter(|s| **s == InputStep::Recreate).count();
    assert_eq!(
        recreated, 0,
        "collapsed island recreated per tick: {paused:?}"
    );
    assert_eq!(paused[0], InputStep::MakePassive);
    assert!(paused[1..].iter().all(|s| *s == InputStep::Keep));
    assert!(applied.click_through);
    // Quick resume: exactly one recreation restores interaction.
    let resumed = ticks(&mut applied, &overlay, true, Phase::Listening, 10);
    assert_eq!(resumed[0], InputStep::Recreate);
    assert!(resumed[1..].iter().all(|s| *s == InputStep::Keep));
    assert!(!applied.click_through);
}

#[test]
fn ordinary_islands_and_genuine_changes_keep_their_existing_behavior() {
    for click_through in [false, true] {
        for recording in [false, true] {
            for phase in [Phase::Dormant, Phase::Armed, Phase::Listening] {
                let overlay = OverlayConfig {
                    recording_notch: recording,
                    click_through,
                    ..OverlayConfig::default()
                };
                let mut applied = window_configuration(&overlay, recording);
                if recording && phase == Phase::Dormant && !click_through {
                    continue; // covered above
                }
                let steps = ticks(&mut applied, &overlay, recording, phase, 20);
                assert!(steps.iter().all(|s| *s == InputStep::Keep), "{steps:?}");
            }
        }
    }
    // A real placement change still recreates once, then settles.
    let mut overlay = OverlayConfig::default();
    let mut applied = window_configuration(&overlay, false);
    overlay.anchor = OverlayAnchor::Bottom;
    let steps = ticks(&mut applied, &overlay, false, Phase::Listening, 5);
    assert_eq!(steps[0], InputStep::Recreate);
    assert!(steps[1..].iter().all(|s| *s == InputStep::Keep));
    // Turning Recording notch on or off retires the old window.
    let mut applied = window_configuration(&overlay, false);
    assert_eq!(
        reconcile_input(&mut applied, &overlay, true, Phase::Dormant),
        InputStep::Close
    );
}
