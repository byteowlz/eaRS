//! Synthetic production-Engine ownership tests. No microphone, models or OS typing.
use super::auto_pause_tests::{auto_pause_engine, make_idle_due};
use super::*;

fn observe(engine: &mut Engine) -> Receiver<DictationEvent> {
    let (tx, rx) = unbounded();
    engine.shared = Arc::new(Shared::new(
        &engine.config,
        true,
        tx,
        engine.shared.commands.clone(),
        true,
    ));
    rx
}

fn pending(engine: &mut Engine, supported: bool) -> mpsc::UnboundedReceiver<WriterCommand> {
    make_idle_due(engine);
    engine.auto_pause.supported = supported;
    let (tx, rx) = mpsc::unbounded_channel();
    engine.writer = Some(tx);
    let (_tx, audio) = unbounded();
    assert_eq!(engine.begin_idle_drain(&audio, &mut None).unwrap(), None);
    assert!(!engine.shared.is_capturing());
    assert!(engine.auto_pause.draining());
    rx
}

#[test]
fn local_pause_has_pending_then_owned_confirmation_without_a_second_capture_edge() {
    let (mut engine, typed) = auto_pause_engine();
    let events = observe(&mut engine);
    let mut writer = pending(&mut engine, true);
    let WriterCommand::Text(request) = writer.try_recv().unwrap() else {
        panic!("expected ordered drain");
    };
    let id = serde_json::from_str::<Value>(&request).unwrap()["request_id"]
        .as_u64()
        .unwrap();
    let paused_generation = engine.shared.capture_generation();
    for json in [
        serde_json::json!({"type":"word","word":"before"}),
        serde_json::json!({"type":"final","text":"before tail"}),
        serde_json::json!({"type":"final","text":"before tail"}),
    ] {
        assert_eq!(engine.handle_session_message(&json).unwrap(), None);
    }
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained","request_id":id+1}))
            .unwrap(),
        None
    );
    let initial: Vec<_> = events.try_iter().collect();
    assert!(initial.iter().any(|event| matches!(
        event,
        DictationEvent::Finalization {
            state: FinalizationState::Pending,
            ..
        }
    )));
    assert!(!initial.iter().any(|event| matches!(
        event,
        DictationEvent::Finalization {
            state: FinalizationState::Confirmed,
            ..
        }
    )));
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained","request_id":id}))
            .unwrap(),
        Some(Flow::DrainedPause)
    );
    assert_eq!(engine.shared.capture_generation(), paused_generation);
    assert_eq!(&*typed.lock().unwrap(), "before tail ");
    assert!(events.try_iter().any(|event| matches!(
        event,
        DictationEvent::Finalization {
            state: FinalizationState::Confirmed,
            ..
        }
    )));
}

#[test]
fn new_capture_command_buffers_unobserved_pcm_until_old_finalization_finishes() {
    let (mut engine, typed) = auto_pause_engine();
    let events = observe(&mut engine);
    let mut writer = pending(&mut engine, true);
    while writer.try_recv().is_ok() {}
    assert_eq!(
        engine
            .handle_command(DictationCommand::SetCapturing(true))
            .unwrap(),
        Flow::Continue
    );
    let new_generation = engine.shared.capture_generation();
    let (tx, rx) = unbounded();
    tx.send(vec![0.3; 1920]).unwrap();
    engine.capture_batch(&rx, &mut None).unwrap();
    assert!(
        writer.try_recv().is_err(),
        "new PCM cannot reach the old socket"
    );
    engine
        .handle_session_message(&serde_json::json!({"type":"final","text":"old tail"}))
        .unwrap();
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained","request_id":1}))
            .unwrap(),
        Some(Flow::DrainedResume)
    );
    assert!(engine.shared.is_capturing());
    assert_eq!(engine.shared.capture_generation(), new_generation);
    assert_eq!(engine.auto_pause.replay.front(), Some(&vec![0.3; 1920]));
    assert!(!engine.auto_pause.replay_observed);
    assert!(
        typed.lock().unwrap().is_empty(),
        "old text cannot enter a new capture generation"
    );
    assert!(events.try_iter().any(|event| matches!(event, DictationEvent::DeferredTranscript { text, confirmed: true, .. } if text == "old tail")));
}

#[test]
fn legacy_final_and_unsolicited_ack_cannot_claim_confirmation() {
    let (mut engine, typed) = auto_pause_engine();
    let events = observe(&mut engine);
    let mut writer = pending(&mut engine, false);
    assert!(
        writer.try_recv().is_err(),
        "legacy protocol must not receive an unsupported drain"
    );
    engine
        .handle_session_message(&serde_json::json!({"type":"final","text":"known words"}))
        .unwrap();
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained","request_id":1}))
            .unwrap(),
        None
    );
    assert!(engine.auto_pause.draining());
    assert_eq!(
        engine.finish_local_pause(false, Some("legacy completion unconfirmed")),
        Flow::DrainedPause
    );
    assert_eq!(&*typed.lock().unwrap(), "known words ");
    let states: Vec<_> = events
        .try_iter()
        .filter_map(|event| match event {
            DictationEvent::Finalization { state, .. } => Some(state),
            _ => None,
        })
        .collect();
    assert_eq!(
        states,
        vec![FinalizationState::Pending, FinalizationState::Unconfirmed]
    );
}

#[test]
fn unknown_or_changed_target_defers_text_without_fabricating_recognition_failure() {
    for unknown in [false, true] {
        let (mut engine, typed) = auto_pause_engine();
        let events = observe(&mut engine);
        let _writer = pending(&mut engine, true);
        engine.target_reader = if unknown {
            || None
        } else {
            || {
                Some(crate::frontmost::FrontmostApp {
                    pid: Some(999),
                    ..Default::default()
                })
            }
        };
        assert_eq!(
            engine
                .handle_session_message(&serde_json::json!({"type":"final","text":"must not type"}))
                .unwrap(),
            None
        );
        assert_eq!(
            engine
                .handle_session_message(&serde_json::json!({"type":"drained","request_id":1}))
                .unwrap(),
            Some(Flow::DrainedPause)
        );
        assert!(typed.lock().unwrap().is_empty());
        assert!(!engine.shared.is_capturing());
        let events: Vec<_> = events.try_iter().collect();
        assert!(events.iter().any(|event| matches!(
            event,
            DictationEvent::Finalization {
                state: FinalizationState::Confirmed,
                ..
            }
        )));
        assert!(events.iter().any(|event| matches!(event, DictationEvent::DeferredTranscript { text, confirmed: true, .. } if text == "must not type")));
    }
}

#[test]
fn pending_timeout_preserves_a_new_capture_but_cancel_discards_its_pcm() {
    let (mut engine, _) = auto_pause_engine();
    let _writer = pending(&mut engine, false);
    engine
        .handle_command(DictationCommand::SetCapturing(true))
        .unwrap();
    engine.auto_pause.freeze(vec![0.25; 960]).unwrap();
    assert_eq!(
        engine.finish_local_pause(false, Some("timeout")),
        Flow::DrainedResume
    );
    assert!(engine.shared.is_capturing());
    assert_eq!(engine.auto_pause.replay.front(), Some(&vec![0.25; 960]));

    let (mut engine, typed) = auto_pause_engine();
    let _writer = pending(&mut engine, true);
    engine
        .handle_command(DictationCommand::SetCapturing(true))
        .unwrap();
    engine.auto_pause.freeze(vec![0.25; 960]).unwrap();
    assert_eq!(
        engine.handle_command(DictationCommand::Cancel).unwrap(),
        Flow::Restart
    );
    assert!(!engine.shared.is_capturing());
    assert!(engine.auto_pause.frozen.is_empty() && engine.auto_pause.replay.is_empty());
    engine
        .handle_session_message(&serde_json::json!({"type":"final","text":"stale"}))
        .unwrap();
    assert!(typed.lock().unwrap().is_empty());
}

#[test]
fn old_stop_phrase_cannot_stop_the_new_wake_capture() {
    let (mut engine, typed) = auto_pause_engine();
    engine.router.set_stop_phrases(&["okto stopp".into()]);
    let _writer = pending(&mut engine, true);
    engine
        .handle_command(DictationCommand::SetCapturing(true))
        .unwrap();
    engine
        .handle_session_message(&serde_json::json!({"type":"final","text":"okto stopp"}))
        .unwrap();
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained","request_id":1}))
            .unwrap(),
        Some(Flow::DrainedResume)
    );
    assert!(engine.shared.is_capturing());
    assert!(typed.lock().unwrap().is_empty());
}

#[test]
fn oversized_new_capture_buffer_fails_closed_without_partial_replay() {
    let (mut engine, typed) = auto_pause_engine();
    let _writer = pending(&mut engine, true);
    engine
        .handle_command(DictationCommand::SetCapturing(true))
        .unwrap();
    let (tx, rx) = unbounded();
    tx.send(vec![0.3; 48_001]).unwrap();
    assert_eq!(
        engine.capture_batch(&rx, &mut None).unwrap(),
        Some(Flow::DrainedPause)
    );
    assert!(!engine.shared.is_capturing());
    assert!(engine.auto_pause.replay.is_empty() && engine.auto_pause.frozen.is_empty());
    assert!(typed.lock().unwrap().is_empty());
}

async fn wait_for_local_pause(shared: &Shared) {
    tokio::time::timeout(Duration::from_secs(1), async {
        while shared.is_capturing() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("local pause must not wait for ASR connection or retry");
    shared.command(DictationCommand::Shutdown);
}

#[tokio::test]
async fn local_pause_runs_during_a_stalled_websocket_handshake() {
    let (mut engine, typed) = auto_pause_engine();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (_tcp, _) = listener.accept().await.unwrap();
        accepted_tx.send(()).unwrap();
        let _ = stop_rx.await; // Intentionally never complete the WebSocket handshake.
    });
    let shared = engine.shared.clone();
    let (tx, rx) = unbounded();
    let producer = tokio::spawn(async move {
        accepted_rx.await.unwrap();
        for _ in 0..16 {
            tx.send(vec![0.0; 1920]).unwrap();
        }
        wait_for_local_pause(&shared).await;
    });
    tokio::time::timeout(
        Duration::from_secs(2),
        engine.main_loop(&format!("ws://{address}"), rx),
    )
    .await
    .unwrap()
    .unwrap();
    producer.await.unwrap();
    stop_tx.send(()).unwrap();
    server.await.unwrap();
    assert!(!engine.shared.is_capturing());
    assert!(typed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn local_pause_runs_while_disconnected_in_the_retry_delay() {
    let (mut engine, typed) = auto_pause_engine();
    let events = observe(&mut engine);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let shared = engine.shared.clone();
    let (tx, rx) = unbounded();
    let producer = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if events.try_iter().any(|event| {
                    matches!(
                        event,
                        DictationEvent::Connection {
                            state: ConnectionState::Disconnected,
                            ..
                        }
                    )
                }) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        for _ in 0..16 {
            tx.send(vec![0.0; 1920]).unwrap();
        }
        wait_for_local_pause(&shared).await;
    });
    tokio::time::timeout(
        Duration::from_secs(2),
        engine.main_loop(&format!("ws://{address}"), rx),
    )
    .await
    .unwrap()
    .unwrap();
    producer.await.unwrap();
    assert!(!engine.shared.is_capturing());
    assert!(typed.lock().unwrap().is_empty());
}

#[test]
fn queued_cancel_fences_tail_even_before_the_engine_dequeues_it() {
    let (mut engine, typed) = auto_pause_engine();
    let events = observe(&mut engine);
    let _writer = pending(&mut engine, true);
    engine.shared.command(DictationCommand::Cancel);
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"final","text":"canceled tail"}))
            .unwrap(),
        Some(Flow::DrainedPause)
    );
    assert!(typed.lock().unwrap().is_empty());
    assert!(
        !events
            .try_iter()
            .any(|event| matches!(event, DictationEvent::DeferredTranscript { .. }))
    );
}

#[test]
fn physical_hold_arriving_with_owned_ack_starts_a_new_capture_not_a_second_pause() {
    let (mut engine, _) = auto_pause_engine();
    let _writer = pending(&mut engine, true);
    engine
        .handle_session_message(&serde_json::json!({"type":"final","text":""}))
        .unwrap();
    engine.shared.set_ptt_held(true);
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained","request_id":1}))
            .unwrap(),
        Some(Flow::DrainedResume)
    );
    assert!(engine.shared.is_capturing());
}

#[test]
fn offline_quiet_waits_for_physical_release_without_inventing_silence() {
    let (mut engine, _) = auto_pause_engine();
    let (tx, rx) = unbounded();
    assert!(engine.service_disconnected_capture(&rx).unwrap());
    assert!(
        engine.shared.is_capturing(),
        "wall time without scored input is not silence"
    );
    engine.shared.set_ptt_held(true);
    for _ in 0..16 {
        tx.send(vec![0.0; 1920]).unwrap();
    }
    engine.service_disconnected_capture(&rx).unwrap();
    assert!(engine.shared.is_capturing());
    engine.shared.set_ptt_held(false);
    engine.service_disconnected_capture(&rx).unwrap();
    assert!(!engine.shared.is_capturing());
}

#[test]
fn deferred_words_stay_ordered_and_uninserted_even_if_focus_returns() {
    let (mut engine, typed) = auto_pause_engine();
    let events = observe(&mut engine);
    let target = engine.target_reader;
    let _writer = pending(&mut engine, true);
    engine.target_reader = || None;
    engine
        .handle_session_message(&serde_json::json!({"type":"word","word":"first"}))
        .unwrap();
    engine.target_reader = target;
    engine
        .handle_session_message(&serde_json::json!({"type":"word","word":"second"}))
        .unwrap();
    engine
        .handle_session_message(&serde_json::json!({"type":"final","text":"first second"}))
        .unwrap();
    engine
        .handle_session_message(&serde_json::json!({"type":"drained","request_id":1}))
        .unwrap();
    assert!(typed.lock().unwrap().is_empty());
    assert!(events.try_iter().any(|event| matches!(event, DictationEvent::DeferredTranscript { text, confirmed: true, .. } if text == "first second")));
}
