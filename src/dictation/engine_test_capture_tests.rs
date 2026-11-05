//! Settings test capture: recognised text reaches the UI only.
use super::*;
use crate::dictation::typing::LiveWordBuffer;
use crate::dictation::typing::tests::RecordingKeyboard;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn settings_test_capture_reaches_ui_only_and_restores_output() {
    let dir = std::env::temp_dir().join(format!("ears-test-capture-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = AppConfig::default();
    config.replacement.enabled = false;
    config.transcripts.enabled = false;
    let (events_tx, events_rx) = unbounded();
    let (commands_tx, commands_rx) = unbounded();
    let shared = Arc::new(Shared::new(&config, true, events_tx, commands_tx, false));
    let recording = RecordingKeyboard::default();
    let typed = recording.typed.clone();
    let mut inserter = Inserter::new(InsertionMode::Clipboard).unwrap();
    *inserter.keyboard() = Box::new(recording);
    let mut engine = Engine {
        target_reader: frontmost_app,
        secure_input_reader: |_| false,
        replacement: ReplacementReloader::new(config.replacement.clone()),
        recorder: TranscriptRecorder::new(config.transcripts.clone()),
        buffer: LiveWordBuffer::new(Duration::from_millis(700)),
        inserter,
        profiles: None,
        backend: BackendInfo::default(),
        writer: None,
        base_lang: None,
        effective_lang: None,
        base_insertion_mode: InsertionMode::Clipboard,
        active_profile: None,
        pinned_profile: None,
        last_profile_eval: None,
        was_capturing: false,
        utterance: Vec::new(),
        router: Router::new(VoiceMode::Dictation),
        auto_pause: Default::default(),
        registry: Registry::new(dir.join("commands.toml")),
        just_finished: false,
        ping_sent: Arc::new(Mutex::new(None)),
        history: None,
        last_level_at: None,
        input: Arc::new(input::InputSlot::detached()),
        config,
        options: DictationOptions::default(),
        shared: shared.clone(),
        commands: commands_rx,
    };
    engine
        .handle_command(DictationCommand::SetTestCapture(true))
        .unwrap();
    assert!(shared.is_capturing() && engine.inserter.is_preview());
    engine.on_capture_edge(true).unwrap();
    // A profile switching the target mid-test must not restore a real output.
    engine
        .apply_insertion_mode(InsertionMode::SendAsPrompt)
        .unwrap();
    assert!(engine.inserter.is_preview());
    for word in ["hello", "world"] {
        engine
            .handle_message(&serde_json::json!({"type":"word","word":word}))
            .unwrap();
    }
    engine
        .handle_message(&serde_json::json!({"type":"speech","active":false}))
        .unwrap();
    assert_eq!(
        &*typed.lock().unwrap(),
        "",
        "test speech never reaches the target"
    );
    let events: Vec<_> = events_rx.try_iter().collect();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, DictationEvent::Utterance { text } if text == "hello world"))
    );
    engine
        .handle_command(DictationCommand::SetTestCapture(false))
        .unwrap();
    assert!(!shared.is_capturing());
    engine.on_capture_edge(false).unwrap();
    assert!(!engine.inserter.is_preview());
    assert_eq!(engine.inserter.mode(), InsertionMode::SendAsPrompt);
    assert!(
        !events_rx
            .try_iter()
            .any(|e| matches!(e, DictationEvent::PromptReady { .. })),
        "a test never produces prompt output"
    );
    // Already dictating: the test is refused and dictation output is untouched.
    shared.set_capturing(true);
    engine
        .handle_command(DictationCommand::SetTestCapture(true))
        .unwrap();
    assert!(!engine.inserter.is_preview());
    assert!(events_rx.try_iter().any(
        |e| matches!(e, DictationEvent::Error { message } if message.contains("Stop dictation"))
    ));
}
