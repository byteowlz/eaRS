//! Capture-clock progress snapshots; no models, microphone or wall-clock simulation.
use super::tests::{gate, silent};
use super::*;

#[test]
fn progress_uses_scored_clock_threshold_reset_and_physical_hold() {
    let mut gate = gate();
    gate.start(7);
    assert_eq!(gate.progress(7, false), None); // No implicit initial zero.
    gate.observe(0.0).unwrap();
    let before = Instant::now();
    let initial = gate.progress(7, false).unwrap();
    let after = Instant::now();
    assert_eq!(
        (
            initial.capture_generation,
            initial.quiet_ms,
            initial.idle_ms,
            initial.held
        ),
        (7, 32, 3000, false)
    );
    assert!((before..=after).contains(&initial.scored_at));
    silent(&mut gate, 3);
    gate.observe(0.49).unwrap();
    let progress = gate.progress(7, true).unwrap();
    assert_eq!(progress.quiet_ms, 160);
    assert!(progress.held);
    silent(&mut gate, 3);
    gate.observe(0.5).unwrap(); // The configured threshold counts as speech.
    assert_eq!(gate.progress(7, false).unwrap().quiet_ms, 0);
    assert_eq!(gate.progress(7, true), None); // Hold alone cannot invent data.
    assert!(!gate.due(7));
    silent(&mut gate, 94);
    assert!(gate.due(7));
    assert_eq!(gate.progress(7, true).unwrap().quiet_ms, 3008);
    assert!(gate.due(7)); // Observation never changes pause/hold policy.
}

#[test]
fn progress_rate_is_bounded_and_does_not_catch_up_without_new_scores() {
    let mut gate = gate();
    gate.start(1);
    let mut snapshots = Vec::new();
    for frame in 1..=320 {
        gate.observe(0.0).unwrap();
        if gate.progress(1, false).is_some() {
            snapshots.push(frame * 32);
        }
    }
    assert_eq!(snapshots.len(), 80);
    assert!(snapshots.windows(2).all(|pair| pair[1] - pair[0] >= 100));
    for _ in 0..100 {
        assert_eq!(gate.progress(1, false), None);
    }
}

#[test]
fn progress_fences_generations_and_resets_only_its_emission_bookkeeping() {
    let mut gate = gate();
    gate.start(1);
    silent(&mut gate, 94);
    assert_eq!(gate.progress(2, false), None);
    assert!(gate.progress(1, false).is_some());
    let id = gate.begin(1, Instant::now()).unwrap();
    silent(&mut gate, 4);
    assert!(gate.progress(1, false).is_some());
    assert!(gate.draining());
    gate.final_seen(1);
    assert_eq!(gate.acknowledge(id, 1).unwrap(), Completion::Paused);
    assert_eq!(gate.progress(1, false), None);
    gate.start(2);
    assert_eq!(gate.progress(2, false), None);
    gate.observe(0.0).unwrap();
    assert_eq!(gate.progress(1, false), None);
    assert_eq!(gate.progress(2, false).unwrap().quiet_ms, 32);
    gate.stop();
    assert_eq!(gate.progress(2, false), None);
}
