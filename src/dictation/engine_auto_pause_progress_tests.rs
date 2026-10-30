//! Scalar progress observers exercised without models, microphone or OS typing.
use super::auto_pause_tests::auto_pause_engine;
use super::*;

fn progress_engine() -> (Engine, Receiver<DictationEvent>) {
    let (mut engine, _) = auto_pause_engine();
    let (events, rx) = unbounded();
    engine.shared = Arc::new(Shared::new(
        &engine.config,
        true,
        events,
        engine.shared.commands.clone(),
        true,
    ));
    (engine, rx)
}

fn progress(rx: &Receiver<DictationEvent>) -> Vec<super::super::AutoPauseProgress> {
    rx.try_iter()
        .filter_map(|event| match event {
            DictationEvent::AutoPauseProgress(snapshot) => Some(snapshot),
            _ => None,
        })
        .collect()
}

#[test]
fn connected_and_disconnected_paths_emit_scored_progress_with_actual_hold() {
    for disconnected in [false, true] {
        let (mut engine, events) = progress_engine();
        let (tx, audio) = unbounded();
        engine.shared.set_ptt_held(true);
        for _ in 0..16 {
            tx.send(vec![0.0; 1920]).unwrap();
        }
        if disconnected {
            assert!(engine.service_disconnected_capture(&audio).unwrap());
        } else {
            assert_eq!(engine.capture_batch(&audio, &mut None).unwrap(), None);
        }
        let snapshots = progress(&events);
        assert!(!snapshots.is_empty());
        assert!(snapshots.iter().all(|p| p.held && p.idle_ms == 500));
        assert!(snapshots.iter().all(|p| p.quiet_ms % 32 == 0));
        assert!(
            snapshots
                .windows(2)
                .all(|p| p[1].quiet_ms - p[0].quiet_ms >= 100)
        );
        assert!(engine.shared.is_capturing()); // Reporting must not release the hold.
        engine.shared.set_ptt_held(false);
        tx.send(vec![0.0; 7680]).unwrap();
        if disconnected {
            engine.service_disconnected_capture(&audio).unwrap();
        } else {
            engine.capture_batch(&audio, &mut None).unwrap();
        }
        let snapshots = progress(&events);
        assert!(!snapshots.is_empty());
        assert!(snapshots.iter().all(|p| !p.held));
    }
}

#[test]
fn empty_or_unscored_chunks_and_server_speech_do_not_invent_progress() {
    for disconnected in [false, true] {
        let (mut engine, events) = progress_engine();
        let (tx, audio) = unbounded();
        tx.send(Vec::new()).unwrap();
        tx.send(vec![0.0; 32]).unwrap(); // Less than a scored frame.
        if disconnected {
            engine.service_disconnected_capture(&audio).unwrap();
        } else {
            engine.capture_batch(&audio, &mut None).unwrap();
        }
        engine
            .handle_message(&serde_json::json!({"type":"speech", "active":false}))
            .unwrap();
        engine.emit_auto_pause_progress(0);
        assert!(progress(&events).is_empty());
    }
}

#[test]
fn disabled_runtime_emits_nothing_in_either_capture_path() {
    for disconnected in [false, true] {
        let (mut engine, events) = progress_engine();
        engine.auto_pause.runtime = None;
        let (tx, audio) = unbounded();
        tx.send(vec![0.0; 7680]).unwrap();
        if disconnected {
            engine.service_disconnected_capture(&audio).unwrap();
        } else {
            engine.capture_batch(&audio, &mut None).unwrap();
        }
        assert!(progress(&events).is_empty());
    }
}

#[test]
fn handle_generation_and_event_fence_reject_stopped_or_superseded_capture() {
    let (mut engine, events) = progress_engine();
    let handle = super::super::DictationHandle {
        shared: engine.shared.clone(),
    };
    let old = handle.capture_generation();
    engine
        .auto_pause
        .runtime
        .as_mut()
        .unwrap()
        .control
        .observe(0.0)
        .unwrap();
    handle.set_capturing(false);
    assert_ne!(handle.capture_generation(), old);
    engine.emit_auto_pause_progress(old);
    assert!(progress(&events).is_empty());
    handle.set_capturing(true);
    let new = handle.capture_generation();
    engine.emit_auto_pause_progress(old);
    engine.emit_auto_pause_progress(new); // Old detector clock cannot be relabeled.
    assert!(progress(&events).is_empty());
    engine.on_capture_edge(true).unwrap();
    engine.emit_auto_pause_progress(new); // A start is not a scored zero.
    assert!(progress(&events).is_empty());
    engine
        .auto_pause
        .runtime
        .as_mut()
        .unwrap()
        .control
        .observe(0.0)
        .unwrap();
    engine.emit_auto_pause_progress(old);
    engine.emit_auto_pause_progress(new);
    let snapshots = progress(&events);
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].capture_generation, new);
    assert_eq!(snapshots[0].quiet_ms, 32);
}

#[test]
fn observer_stalls_or_scored_replay_never_reemit_an_old_value() {
    let (mut engine, events) = progress_engine();
    let (tx, audio) = unbounded();
    tx.send(vec![0.0; 7680]).unwrap();
    let before = Instant::now();
    engine.capture_batch(&audio, &mut None).unwrap();
    let produced = Instant::now();
    std::thread::sleep(Duration::from_millis(5)); // Delay delivery, not scored audio.
    let received = Instant::now();
    let snapshots = progress(&events);
    assert!(!snapshots.is_empty());
    assert!(
        snapshots
            .iter()
            .all(|p| { (before..=produced).contains(&p.scored_at) && p.scored_at < received })
    );
    let last = *snapshots.last().unwrap();
    for _ in 0..20 {
        engine.capture_batch(&audio, &mut None).unwrap();
        engine.service_disconnected_capture(&audio).unwrap();
    }
    engine.auto_pause.replay.push_back(vec![0.0; 7680]);
    engine.auto_pause.replay_observed = true;
    engine.capture_batch(&audio, &mut None).unwrap();
    assert!(progress(&events).is_empty());
    tx.send(vec![0.0; 7680]).unwrap();
    engine.capture_batch(&audio, &mut None).unwrap();
    let next = progress(&events);
    assert!(!next.is_empty());
    assert!(next.iter().all(|p| p.scored_at > last.scored_at));
    assert!(next.iter().all(|p| p.quiet_ms > last.quiet_ms));
}
