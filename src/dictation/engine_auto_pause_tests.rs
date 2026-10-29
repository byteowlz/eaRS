//! Model-free client protocol and capture-lifecycle regression tests.
use super::tests::retention_engine;
use super::*;

#[test]
fn opus_drain_flush_delivers_a_marker_in_a_partial_encoder_frame() {
    let mut encoder = kaudio::ogg_opus::Encoder::new(24_000).unwrap();
    let mut decoder = kaudio::ogg_opus::Decoder::new(24_000, 1).unwrap();
    assert!(decoder.decode(encoder.header_data()).unwrap().is_none());
    let mut pcm = vec![0.0; 960];
    pcm.extend(
        (0..137).map(|i| 0.4 * (i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 24_000.0).sin()),
    );
    let mut decoded = decoder
        .decode(&encoder.encode_page(&pcm).unwrap())
        .unwrap()
        .unwrap()
        .to_vec();
    assert_eq!(decoded.len(), 960); // The marker is still in the encoder.
    decoded.extend_from_slice(
        decoder
            .decode(&flush_opus_for_drain(&mut encoder).unwrap())
            .unwrap()
            .unwrap(),
    );
    assert_eq!(decoded.len(), 1920);
    let energy = decoded[960..]
        .iter()
        .map(|sample| sample * sample)
        .sum::<f32>()
        / 960.0;
    assert!(
        energy.sqrt() > 0.01,
        "the retained partial-frame tone must reach the decoder"
    );
}

pub(super) fn auto_pause_engine() -> (Engine, Arc<Mutex<String>>) {
    struct SyntheticScorer;
    impl super::super::client_vad::SpeechScorer for SyntheticScorer {
        fn reset(&mut self) {}
        fn score(&mut self, frame: &[f32]) -> Result<f32> {
            Ok(if frame.iter().any(|sample| sample.abs() > 0.1) {
                0.9
            } else {
                0.0
            })
        }
    }
    let mut config = AppConfig::default();
    config.replacement.enabled = false;
    config.transcripts.enabled = false;
    config.dictation.notifications.enabled = false;
    let (mut engine, typed) = retention_engine(config, None);
    let mut auto = super::super::auto_pause::AutoPause::fixture(
        Box::new(SyntheticScorer),
        super::super::auto_pause::AutoPauseConfig {
            idle_ms: 500,
            drain_timeout_ms: 1000,
            ..Default::default()
        },
    );
    auto.start(engine.shared.capture_generation());
    engine.base_insertion_mode = InsertionMode::InsertAtCursor;
    engine.inserter = Inserter::fixture(Box::new(super::super::typing::tests::RecordingKeyboard {
        typed: typed.clone(),
    }));
    engine.target_reader = || {
        Some(crate::frontmost::FrontmostApp {
            pid: Some(123),
            window_title: Some("synthetic target".into()),
            ..Default::default()
        })
    };
    engine.auto_pause.runtime = Some(auto);
    engine.auto_pause.capture_generation = Some(engine.shared.capture_generation());
    (engine, typed)
}

pub(super) fn make_idle_due(engine: &mut Engine) {
    for _ in 0..16 {
        engine
            .auto_pause
            .runtime
            .as_mut()
            .unwrap()
            .control
            .observe(0.0)
            .unwrap();
    }
}

#[tokio::test]
async fn local_pause_precedes_legacy_server_tail_without_drain_capability() {
    let (mut engine, typed) = auto_pause_engine();
    let shared = engine.shared.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        ws.send(Message::text(r#"{"type":"capabilities","features":[]}"#))
            .await
            .unwrap();
        let mut bytes = 0;
        while bytes < 16 * 1920 * 4 {
            match ws.next().await.unwrap().unwrap() {
                Message::Binary(data) => bytes += data.len(),
                Message::Text(text) => assert_ne!(
                    serde_json::from_str::<Value>(&text).unwrap()["type"],
                    "drain"
                ),
                _ => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !shared.is_capturing(),
            "local silence must pause before any server final/ACK"
        );
        assert_eq!(shared.state(), DictationState::Suspended);
        ws.send(Message::text(r#"{"type":"word","word":"trailing"}"#))
            .await
            .unwrap();
        ws.send(Message::text(r#"{"type":"final","text":"trailing word"}"#))
            .await
            .unwrap();
        ws.close(None).await.unwrap();
    });
    let (tx, rx) = unbounded();
    for _ in 0..16 {
        tx.send(vec![0.0; 1920]).unwrap();
    }
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    let flow = tokio::time::timeout(Duration::from_secs(3), engine.session(ws, rx))
        .await
        .unwrap()
        .unwrap();
    server.await.unwrap();
    assert_eq!(flow, Flow::DrainedPause);
    assert_eq!(&*typed.lock().unwrap(), "trailing word ");
}

#[tokio::test]
async fn auto_pause_real_ws_preserves_delayed_final_tail_then_pauses_once() {
    let (mut engine, typed) = auto_pause_engine();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        ws.send(Message::text(
            r#"{"type":"capabilities","features":["dictation-drain/1"]}"#,
        ))
        .await
        .unwrap();
        let mut audio_bytes = 0;
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Binary(data) = &message {
                audio_bytes += data.len();
            }
            if let Message::Text(text) = message {
                let json: Value = serde_json::from_str(&text).unwrap();
                if json["type"] != "drain" {
                    continue;
                }
                assert_eq!(audio_bytes, 16 * 1920 * 4); // Every pre-cut sample precedes drain.
                let id = json["request_id"].as_u64().unwrap();
                ws.send(Message::text(
                    r#"{"type":"word","word":"final","start_time":0.0,"end_time":null}"#,
                ))
                .await
                .unwrap();
                // The tail exists only in Final, not in a Word event.
                let final_result =
                    serde_json::json!({"type":"final","text":"final tail","words":[]}).to_string();
                ws.send(Message::text(final_result.clone())).await.unwrap();
                ws.send(Message::text(final_result)).await.unwrap();
                ws.send(Message::text(
                    serde_json::json!({"type":"drained","request_id":id+1}).to_string(),
                ))
                .await
                .unwrap();
                ws.send(Message::text(
                    serde_json::json!({"type":"drained","request_id":id}).to_string(),
                ))
                .await
                .unwrap();
                break;
            }
        }
    });
    let (audio_tx, audio_rx) = unbounded();
    // Tiny callback fragments must not starve the queue-empty drain boundary.
    for _ in 0..960 {
        audio_tx.send(vec![0.0; 32]).unwrap();
    }
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    let flow = tokio::time::timeout(Duration::from_secs(5), engine.session(ws, audio_rx))
        .await
        .unwrap()
        .unwrap();
    server.await.unwrap();
    assert_eq!(flow, Flow::DrainedPause);
    assert!(!engine.shared.is_capturing());
    assert_eq!(&*typed.lock().unwrap(), "final tail ");
    assert!(!engine.auto_pause.draining());
}

#[tokio::test]
async fn auto_pause_hybrid_hold_defers_until_physical_release() {
    let (mut engine, _) = auto_pause_engine();
    engine.shared.set_ptt_held(true);
    let shared = engine.shared.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        ws.send(Message::text(
            r#"{"type":"capabilities","features":["dictation-drain/1"]}"#,
        ))
        .await
        .unwrap();
        let mut bytes = 0;
        while bytes < 16 * 1920 * 4 {
            if let Some(Ok(Message::Binary(data))) = ws.next().await {
                bytes += data.len();
            }
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(100), ws.next())
                .await
                .is_err()
        );
        assert!(shared.is_capturing());
        shared.set_ptt_held(false); // Synthetic physical release, not a real event.
        while let Some(Ok(Message::Text(text))) = ws.next().await {
            let json: Value = serde_json::from_str(&text).unwrap();
            if json["type"] == "drain" {
                ws.send(Message::text(r#"{"type":"final","text":"","words":[]}"#))
                    .await
                    .unwrap();
                ws.send(Message::text(
                    serde_json::json!({"type":"drained","request_id":json["request_id"]})
                        .to_string(),
                ))
                .await
                .unwrap();
                break;
            }
        }
    });
    let (audio_tx, audio_rx) = unbounded();
    for _ in 0..16 {
        audio_tx.send(vec![0.0; 1920]).unwrap();
    }
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), engine.session(ws, audio_rx))
            .await
            .unwrap()
            .unwrap(),
        Flow::DrainedPause
    );
    server.await.unwrap();
}

#[tokio::test]
async fn auto_pause_unsupported_server_pauses_locally_and_reports_unconfirmed_tail() {
    let (mut engine, _) = auto_pause_engine();
    let (events_tx, events_rx) = unbounded();
    let (commands_tx, _) = unbounded();
    engine.shared = Arc::new(Shared::new(
        &engine.config,
        true,
        events_tx,
        commands_tx,
        true,
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        ws.send(Message::text(r#"{"type":"capabilities","features":[]}"#))
            .await
            .unwrap();
        let mut pending = false;
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                for event in events_rx.try_iter() {
                    if let DictationEvent::Finalization { state, .. } = event {
                        match state {
                            FinalizationState::Pending => pending = true,
                            FinalizationState::Confirmed => {
                                panic!("legacy server cannot confirm an owned drain")
                            }
                            FinalizationState::Unconfirmed => {
                                assert!(pending);
                                return;
                            }
                        }
                    }
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let _ = ws.close(None).await;
    });
    let (audio_tx, audio_rx) = unbounded();
    for _ in 0..16 {
        audio_tx.send(vec![0.0; 1920]).unwrap();
    }
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    let flow = tokio::time::timeout(Duration::from_secs(5), engine.session(ws, audio_rx))
        .await
        .unwrap()
        .unwrap();
    server.await.unwrap();
    assert_eq!(flow, Flow::DrainedPause);
    assert!(!engine.shared.is_capturing());
    assert!(!engine.auto_pause.draining());
}

#[tokio::test]
async fn auto_pause_replays_speech_that_resumes_during_drain_exactly_once() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct SignalledScorer(Arc<AtomicBool>);
    impl super::super::client_vad::SpeechScorer for SignalledScorer {
        fn reset(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
        fn score(&mut self, frame: &[f32]) -> Result<f32> {
            let speech = frame.iter().any(|sample| sample.abs() > 0.1);
            if speech {
                self.0.store(true, Ordering::SeqCst);
            }
            Ok(if speech { 0.9 } else { 0.0 })
        }
    }
    let (mut engine, typed) = auto_pause_engine();
    let observed = Arc::new(AtomicBool::new(false));
    let mut auto = super::super::auto_pause::AutoPause::fixture(
        Box::new(SignalledScorer(observed.clone())),
        super::super::auto_pause::AutoPauseConfig {
            idle_ms: 500,
            drain_timeout_ms: 1000,
            ..Default::default()
        },
    );
    auto.start(engine.shared.capture_generation());
    engine.auto_pause.runtime = Some(auto);
    let (audio_tx, audio_rx) = unbounded();
    for _ in 0..16 {
        audio_tx.send(vec![0.0; 1920]).unwrap();
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let shared = engine.shared.clone();
    let server = tokio::spawn(async move {
        let mut previous_id = 0;
        for phase in 0..2 {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            ws.send(Message::text(
                r#"{"type":"capabilities","features":["dictation-drain/1"]}"#,
            ))
            .await
            .unwrap();
            if phase == 1 {
                for _ in 0..16 {
                    audio_tx.send(vec![0.0; 1920]).unwrap();
                }
            }
            let mut bytes = Vec::new();
            while let Some(Ok(message)) = ws.next().await {
                match message {
                    Message::Binary(data) => bytes.extend_from_slice(&data),
                    Message::Text(text) => {
                        let json: Value = serde_json::from_str(&text).unwrap();
                        if json["type"] != "drain" {
                            continue;
                        }
                        let id = json["request_id"].as_u64().unwrap();
                        assert!(id > previous_id);
                        previous_id = id;
                        if phase == 0 {
                            assert!(!shared.is_capturing());
                            shared.mark_wake_started();
                            assert!(shared.set_capturing(true));
                            for _ in 0..3 {
                                audio_tx.send(vec![0.3; 1920]).unwrap();
                            }
                            tokio::time::sleep(READ_TIMEOUT * 3).await;
                            assert!(
                                !observed.load(Ordering::SeqCst),
                                "new PCM cannot be scored under the old capture owner"
                            );
                            ws.send(Message::text(
                                r#"{"type":"final","text":"old tail","words":[]}"#,
                            ))
                            .await
                            .unwrap();
                        } else {
                            assert_eq!(bytes.len(), (3 + 16) * 1920 * 4);
                            assert!(
                                bytes[..3 * 1920 * 4]
                                    .chunks_exact(4)
                                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 0.3)
                            );
                            ws.send(Message::text(r#"{"type":"word","word":"resumed"}"#))
                                .await
                                .unwrap();
                            ws.send(Message::text(
                                r#"{"type":"final","text":"resumed","words":[]}"#,
                            ))
                            .await
                            .unwrap();
                        }
                        ws.send(Message::text(
                            serde_json::json!({"type":"drained","request_id":id}).to_string(),
                        ))
                        .await
                        .unwrap();
                        break;
                    }
                    _ => {}
                }
            }
        }
    });
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), engine.session(ws, audio_rx.clone()))
            .await
            .unwrap()
            .unwrap(),
        Flow::DrainedResume
    );
    assert!(engine.shared.is_capturing());
    assert_eq!(
        engine.auto_pause.replay.iter().map(Vec::len).sum::<usize>(),
        3 * 1920
    );
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), engine.session(ws, audio_rx))
            .await
            .unwrap()
            .unwrap(),
        Flow::DrainedPause
    );
    server.await.unwrap();
    assert_eq!(&*typed.lock().unwrap(), "resumed "); // Old tail is deferred, never inserted into this new generation.
}

#[test]
fn auto_pause_stale_generation_and_inconsistent_final_have_no_new_effects() {
    let (mut engine, typed) = auto_pause_engine();
    let auto = engine.auto_pause.runtime.as_mut().unwrap();
    for _ in 0..16 {
        auto.control.observe(0.0).unwrap();
    }
    let id = auto
        .control
        .begin(engine.shared.capture_generation(), Instant::now())
        .unwrap();
    engine.shared.set_capturing(false);
    engine.shared.set_capturing(true);
    engine
        .handle_session_message(&serde_json::json!({"type":"word","word":"stale"}))
        .unwrap();
    engine
        .handle_session_message(&serde_json::json!({"type":"final","text":"stale"}))
        .unwrap();
    engine
        .handle_session_message(&serde_json::json!({"type":"drained","request_id":id}))
        .unwrap();
    assert!(typed.lock().unwrap().is_empty());
    assert!(engine.shared.is_capturing()); // An old ACK cannot pause the new capture.

    let (mut engine, typed) = auto_pause_engine();
    engine
        .handle_session_message(&serde_json::json!({"type":"word","word":"known"}))
        .unwrap();
    let before = typed.lock().unwrap().clone();
    let auto = engine.auto_pause.runtime.as_mut().unwrap();
    for _ in 0..16 {
        auto.control.observe(0.0).unwrap();
    }
    auto.control
        .begin(engine.shared.capture_generation(), Instant::now())
        .unwrap();
    let flow = engine
        .handle_session_message(&serde_json::json!({"type":"final","text":"different revision"}))
        .unwrap();
    assert_eq!(flow, Some(Flow::Restart));
    assert_eq!(*typed.lock().unwrap(), before);
    assert!(!engine.shared.is_capturing());
}

#[tokio::test]
async fn auto_pause_queued_cancel_wins_over_a_late_word_waking_the_reader() {
    let (mut engine, typed) = auto_pause_engine();
    let shared = engine.shared.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        ws.send(Message::text(
            r#"{"type":"capabilities","features":["dictation-drain/1"]}"#,
        ))
        .await
        .unwrap();
        while let Some(Ok(message)) = ws.next().await {
            let Message::Text(text) = message else {
                continue;
            };
            let json: Value = serde_json::from_str(&text).unwrap();
            if json["type"] != "drain" {
                continue;
            }
            shared.command(DictationCommand::Cancel);
            let _ = ws
                .send(Message::text(r#"{"type":"word","word":"late"}"#))
                .await;
            let _ = ws
                .send(Message::text(
                    r#"{"type":"final","text":"late","words":[]}"#,
                ))
                .await;
            let _ = ws
                .send(Message::text(
                    serde_json::json!({"type":"drained","request_id":json["request_id"]})
                        .to_string(),
                ))
                .await;
            break;
        }
    });
    let (audio_tx, audio_rx) = unbounded();
    for _ in 0..16 {
        audio_tx.send(vec![0.0; 1920]).unwrap();
    }
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), engine.session(ws, audio_rx))
            .await
            .unwrap()
            .unwrap(),
        Flow::Restart
    );
    server.await.unwrap();
    assert!(typed.lock().unwrap().is_empty());
    assert!(!engine.shared.is_capturing());
}

#[tokio::test]
async fn auto_pause_missing_ack_times_out_without_dispatching_buffered_commands() {
    let (mut engine, typed) = auto_pause_engine();
    engine.router = Router::new(VoiceMode::CommandAware);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        ws.send(Message::text(
            r#"{"type":"capabilities","features":["dictation-drain/1"]}"#,
        ))
        .await
        .unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                let json: Value = serde_json::from_str(&text).unwrap();
                if json["type"] != "drain" {
                    continue;
                }
                ws.send(Message::text(r#"{"type":"word","word":"unconfirmed"}"#))
                    .await
                    .unwrap();
                // Neither a Final nor an ACK: client must visibly fail and close.
                while let Some(Ok(message)) = ws.next().await {
                    if matches!(message, Message::Close(_)) {
                        return;
                    }
                }
                return;
            }
        }
    });
    let (audio_tx, audio_rx) = unbounded();
    for _ in 0..16 {
        audio_tx.send(vec![0.0; 1920]).unwrap();
    }
    let (ws, _) = connect_async(format!("ws://{address}")).await.unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), engine.session(ws, audio_rx))
            .await
            .unwrap()
            .unwrap(),
        Flow::DrainedPause
    );
    server.await.unwrap();
    assert!(!engine.shared.is_capturing());
    assert!(typed.lock().unwrap().is_empty());
    assert!(engine.utterance.is_empty() && engine.auto_pause.frozen.is_empty());
}

#[test]
fn auto_pause_threshold_command_preserves_owned_drain_and_buffered_pcm() {
    let (mut engine, typed) = auto_pause_engine();
    let generation = engine.shared.capture_generation();
    let auto = engine.auto_pause.runtime.as_mut().unwrap();
    for _ in 0..16 {
        auto.control.observe(0.0).unwrap();
    }
    let id = auto.control.begin(generation, Instant::now()).unwrap();
    engine.auto_pause.freeze(vec![0.25; 960]).unwrap();
    for threshold in [0.75, f32::NAN] {
        assert_eq!(
            engine
                .handle_command(DictationCommand::SetAutoPauseSpeechThreshold(threshold))
                .unwrap(),
            Flow::Continue
        );
        assert!(engine.auto_pause.draining());
        assert_eq!(engine.shared.capture_generation(), generation);
        assert!(engine.shared.is_capturing());
        assert_eq!(engine.auto_pause.frozen.front().unwrap(), &vec![0.25; 960]);
    }
    // 0.6 no longer requests resumption; the invalid edit did not replace 0.75.
    engine
        .auto_pause
        .runtime
        .as_mut()
        .unwrap()
        .control
        .observe(0.6)
        .unwrap();
    engine
        .handle_session_message(&serde_json::json!({"type":"final", "text":""}))
        .unwrap();
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained", "request_id":id}))
            .unwrap(),
        Some(Flow::DrainedPause)
    );
    assert!(typed.lock().unwrap().is_empty());
}

#[test]
fn auto_pause_writer_failure_never_becomes_a_successful_cut() {
    for disconnected in [false, true] {
        let (mut engine, typed) = auto_pause_engine();
        make_idle_due(&mut engine);
        engine.auto_pause.supported = true;
        if disconnected {
            let (tx, rx) = mpsc::unbounded_channel();
            drop(rx);
            engine.writer = Some(tx);
        }
        let (_tx, rx) = unbounded();
        assert_eq!(
            engine.begin_idle_drain(&rx, &mut None).unwrap(),
            Some(Flow::DrainedPause)
        );
        assert!(!engine.shared.is_capturing());
        assert!(!engine.auto_pause.draining());
        assert!(typed.lock().unwrap().is_empty());
    }
}

#[test]
fn auto_pause_hold_arriving_with_ack_keeps_capture_active() {
    let (mut engine, _) = auto_pause_engine();
    let auto = engine.auto_pause.runtime.as_mut().unwrap();
    for _ in 0..16 {
        auto.control.observe(0.0).unwrap();
    }
    let id = auto
        .control
        .begin(engine.shared.capture_generation(), Instant::now())
        .unwrap();
    engine
        .handle_session_message(&serde_json::json!({"type":"final", "text":""}))
        .unwrap();
    engine.shared.set_ptt_held(true);
    assert_eq!(
        engine
            .handle_session_message(&serde_json::json!({"type":"drained", "request_id":id}))
            .unwrap(),
        Some(Flow::DrainedResume)
    );
    assert!(engine.shared.is_capturing());
}

#[test]
fn auto_pause_freeze_bound_and_manual_cancel_discard_without_dispatch() {
    let (mut engine, typed) = auto_pause_engine();
    let auto = engine.auto_pause.runtime.as_mut().unwrap();
    for _ in 0..16 {
        auto.control.observe(0.0).unwrap();
    }
    auto.control
        .begin(engine.shared.capture_generation(), Instant::now())
        .unwrap();
    assert!(engine.auto_pause.freeze(vec![0.0; 48_001]).is_err());
    engine.auto_pause.freeze(vec![0.3; 100]).unwrap();
    assert_eq!(
        engine.handle_command(DictationCommand::Cancel).unwrap(),
        Flow::Restart
    );
    assert!(engine.auto_pause.frozen.is_empty() && engine.auto_pause.replay.is_empty());
    assert!(typed.lock().unwrap().is_empty());
    assert!(!engine.shared.is_capturing());
}
