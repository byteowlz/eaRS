//! The engine loop: audio capture, WebSocket session, message handling,
//! command handling and context profile evaluation.

use super::BackendInfo;
use super::ConnectionState;
use super::DictationCommand;
use super::DictationEvent;
use super::DictationOptions;
use super::DictationState;
use super::FinalizationState;
use super::InsertionMode;
use super::Shared;
use super::VERBOSE;
use super::alias_for_url;
use super::audio_history::{AudioHistory, CommitContext, Skip};
use super::ensure_server_running;
use super::extract_local_port;
use super::hotkeys;
use super::is_server_port_open;
use super::remove_pid_file;
use super::typing::Inserter;
use super::typing::LiveWordBuffer;
use super::typing::ReplacementReloader;
use super::typing::TranscriptRecorder;
use super::vlog;
use super::voice_commands::{Registry, Route, Router, VoiceMode};
use super::write_pid_file;
use crate::audio;
use crate::config::AppConfig;
use crate::frontmost::frontmost_app;
use crate::profiles::ContextProfile;
use crate::profiles::ProfileReloader;
use crate::replacement::ReplacementDictionary;
use crate::replacement::dictionary_paths;
use anyhow::Context;
use anyhow::Result;
use crossbeam_channel::Receiver;
use crossbeam_channel::unbounded;
use futures_util::SinkExt;
use futures_util::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

/// How often the engine pings the server to measure latency.
const PING_INTERVAL: Duration = Duration::from_secs(5);
/// Minimum interval between frontmost-app lookups.
const PROFILE_EVAL_INTERVAL: Duration = Duration::from_millis(300);
/// Read timeout: bounds command latency and drives the phrase-hold flush.
const READ_TIMEOUT: Duration = Duration::from_millis(50);

enum WriterCommand {
    Audio(Vec<u8>),
    Text(String),
    Ping,
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Continue,
    Reconnect,
    Restart,
    DrainedPause,
    DrainedResume,
    Exit,
}

struct LocalPause {
    owner_generation: u64,
    request_id: u64,
    cancel_epoch: u64,
    paused_generation: u64,
    target: Option<crate::frontmost::FrontmostApp>,
    expect_ack: bool,
    deferred: Vec<String>,
}

impl LocalPause {
    fn accepts(&self, generation: u64) -> bool {
        generation == self.paused_generation
            || self.paused_generation.checked_add(1) == Some(generation)
    }
}

#[derive(Default)]
struct AutoPauseSession {
    runtime: Option<super::auto_pause::AutoPause>,
    local_pause: Option<LocalPause>,
    supported: bool,
    capabilities_known: bool,
    capture_generation: Option<u64>,
    warned: bool,
    frozen: std::collections::VecDeque<Vec<f32>>,
    frozen_samples: usize,
    replay: std::collections::VecDeque<Vec<f32>>,
    replay_observed: bool,
    words: Vec<String>,
    word_bytes: usize,
    word_overflow: bool,
}
impl AutoPauseSession {
    fn set_speech_threshold(&mut self, threshold: f32) -> Result<()> {
        super::auto_pause::validate_speech_threshold(threshold)?;
        if let Some(auto) = &mut self.runtime {
            auto.control.set_speech_threshold(threshold)?;
        }
        Ok(())
    }
    fn draining(&self) -> bool {
        self.runtime
            .as_ref()
            .is_some_and(|auto| auto.control.draining())
    }
    fn accepts(&self, generation: u64) -> bool {
        if let Some(pause) = &self.local_pause {
            return pause.accepts(generation);
        }
        self.runtime
            .as_ref()
            .is_none_or(|auto| auto.control.accepts(generation))
    }
    fn clear_audio(&mut self) {
        self.frozen.clear();
        self.replay.clear();
        self.frozen_samples = 0;
    }
    fn freeze(&mut self, chunk: Vec<f32>) -> Result<()> {
        let limit = self
            .runtime
            .as_ref()
            .context("auto-pause detector unavailable")?
            .buffer_limit;
        anyhow::ensure!(
            self.frozen_samples.saturating_add(chunk.len()) <= limit,
            "auto-pause replay buffer exceeded its bound"
        );
        self.frozen_samples += chunk.len();
        self.frozen.push_back(chunk);
        Ok(())
    }
    fn resume_audio(&mut self) {
        self.replay_observed = true;
        self.replay = std::mem::take(&mut self.frozen);
        self.frozen_samples = 0;
    }
    fn reset_words(&mut self) {
        self.words.clear();
        self.word_bytes = 0;
        self.word_overflow = false;
    }
    fn note_word(&mut self, text: &str) {
        if self.runtime.is_none() || self.word_overflow {
            return;
        }
        for word in text.split_whitespace() {
            if self.word_bytes.saturating_add(word.len() + 1) > 262_144 {
                self.word_overflow = true;
                return;
            }
            self.word_bytes += word.len() + 1;
            self.words.push(word.to_owned());
        }
    }
    fn missing_final_words(&self, text: &str) -> Result<Vec<String>> {
        anyhow::ensure!(
            !self.word_overflow && text.len() <= 262_144,
            "final transcript exceeds reconciliation bound"
        );
        let words: Vec<_> = text.split_whitespace().map(str::to_owned).collect();
        anyhow::ensure!(
            words.starts_with(&self.words),
            "final transcript does not match the stream word prefix; refusing to guess or retype"
        );
        Ok(words[self.words.len()..].to_vec())
    }
}

struct Engine {
    target_reader: fn() -> Option<crate::frontmost::FrontmostApp>,
    auto_pause: AutoPauseSession,
    config: AppConfig,
    options: DictationOptions,
    shared: Arc<Shared>,
    commands: Receiver<DictationCommand>,
    replacement: ReplacementReloader,
    recorder: TranscriptRecorder,
    buffer: LiveWordBuffer,
    inserter: Inserter,
    profiles: Option<ProfileReloader>,
    backend: BackendInfo,
    writer: Option<mpsc::UnboundedSender<WriterCommand>>,
    /// Language from options / SetLanguage, before profile overrides.
    base_lang: Option<String>,
    /// Language currently requested from the server.
    effective_lang: Option<String>,
    base_insertion_mode: InsertionMode,
    active_profile: Option<String>,
    pinned_profile: Option<String>,
    last_profile_eval: Option<Instant>,
    was_capturing: bool,
    /// Raw words of the current utterance, for the live partial line.
    utterance: Vec<String>,
    router: Router,
    registry: Registry,
    /// Suppress an authoritative final repeated after a word/VAD boundary.
    just_finished: bool,
    ping_sent: Arc<Mutex<Option<Instant>>>,
    /// Opt-in utterance audio retention; `None` when disabled.
    history: Option<AudioHistory>,
}

pub(crate) async fn run(
    config: AppConfig,
    options: DictationOptions,
    url: String,
    shared: Arc<Shared>,
    commands: Receiver<DictationCommand>,
) -> Result<()> {
    VERBOSE.store(options.verbose, Ordering::Relaxed);

    if config.replacement.enabled {
        if let Some(path) = dictionary_paths(&config.replacement).first() {
            let _ = ReplacementDictionary::load_or_create(path);
        }
    }

    let backend = BackendInfo {
        alias: alias_for_url(&url, &config),
        url: url.clone(),
        latency_ms: None,
    };
    shared.emit(DictationEvent::Backend(backend.clone()));

    // For local server URLs, ensure the server is running before we start
    // capturing audio and connecting. This avoids the confusing situation
    // where dictation appears to start successfully but silently fails to
    // transcribe because no server is available.
    if options.auto_start_server {
        if let Some(port) = extract_local_port(&url) {
            if !is_server_port_open(port) {
                if let Err(err) = ensure_server_running(&config) {
                    eprintln!("warning: could not auto-start ears server: {}", err);
                    eprintln!("dictation will keep retrying the connection to {}", url);
                    shared.emit(DictationEvent::Error {
                        message: format!("could not auto-start ears server: {err}"),
                    });
                }
            }
        }
    }

    if options.write_pid_file {
        write_pid_file(&url)?;
    }

    let inserter = match Inserter::new(options.insertion_mode) {
        Ok(inserter) => inserter,
        Err(err) => {
            if options.write_pid_file {
                remove_pid_file();
            }
            return Err(err);
        }
    };

    let hotkeys_enabled = options.hotkeys.unwrap_or(config.hotkeys.enable_internal);
    let escape_cancels = options
        .escape_cancels
        .unwrap_or(config.hotkeys.escape_cancels)
        || config.voice_commands.mode != VoiceMode::Dictation
        || config.voice_commands.speaker_policy.mode != crate::speaker_gate::Mode::Disabled;
    shared.set_escape_cancels(escape_cancels);
    if hotkeys_enabled {
        hotkeys::spawn_listener(config.hotkeys.clone(), escape_cancels, shared.clone());
    }

    let starts_listening = !config.dictation.start_paused;
    shared.set_state(if starts_listening {
        DictationState::Listening
    } else {
        DictationState::Suspended
    });

    let (audio_tx, audio_rx) = unbounded();
    // Single microphone owner: capture feeds the engine, and a wake tee when configured.
    let capture_tx = if config.dictation.wake.active() {
        let (capture_tx, capture_rx) = unbounded::<Vec<f32>>();
        super::wake::spawn_tee(&config.dictation.wake, capture_rx, audio_tx, shared.clone())?;
        capture_tx
    } else {
        audio_tx
    };
    let audio_shared = shared.clone();
    thread::Builder::new()
        .name("ears-audio".to_string())
        .spawn(move || {
            if let Err(e) = audio::start_audio_capture(capture_tx, None) {
                eprintln!("Audio capture error: {}", e);
                audio_shared.emit(DictationEvent::Error {
                    message: format!("microphone unavailable: {e}"),
                });
            }
        })
        .context("failed to spawn audio capture thread")?;

    vlog!("ears-dictation started");
    vlog!("Connecting to {}...", url);
    vlog!(
        "Hotkey: {} ({})",
        config.hotkeys.toggle,
        hotkeys::mode_description(config.hotkeys.mode)
    );
    vlog!(
        "Startup state: {}",
        if config.dictation.start_paused {
            "paused"
        } else {
            "live"
        }
    );

    let history = match AudioHistory::from_config(
        &config.dictation.audio_history,
        crate::audio::default_input_device_name(),
        backend.alias.clone().or_else(|| Some(backend.url.clone())),
    ) {
        Ok(Some(mut history)) => {
            history.set_engine(options.engine.clone());
            shared.audio_history.store(true, Ordering::SeqCst);
            eprintln!(
                "audio history ON: keeping utterance audio in {}",
                history.dir().display()
            );
            shared.emit(DictationEvent::AudioHistory {
                active: true,
                note: None,
            });
            Some(history)
        }
        Ok(None) => None,
        Err(err) => {
            eprintln!("audio history disabled: {err:#}");
            shared.emit(DictationEvent::AudioHistory {
                active: false,
                note: Some(format!("retention unavailable: {err:#}")),
            });
            shared.emit(DictationEvent::Error {
                message: format!("audio history disabled: {err:#}"),
            });
            None
        }
    };

    let auto_pause = match super::auto_pause::AutoPause::from_config(
        &config.dictation.auto_pause,
        config.dictation.audio_history.client_vad.as_ref(),
    ) {
        Ok(runtime) => AutoPauseSession {
            runtime,
            ..Default::default()
        },
        Err(err) => {
            shared.emit(DictationEvent::Error {
                message: format!("auto-pause unavailable: {err:#}"),
            });
            AutoPauseSession::default()
        }
    };
    let mut engine = Engine {
        target_reader: frontmost_app,
        auto_pause,
        history,
        router: {
            let mut router = Router::new(config.voice_commands.mode);
            router.set_stop_phrases(&config.dictation.wake.stop_phrases);
            router
        },
        registry: Registry::new(
            config
                .voice_commands
                .path
                .as_ref()
                .map(std::path::PathBuf::from)
                .unwrap_or_else(super::voice_commands::default_path),
        ),
        just_finished: false,
        replacement: ReplacementReloader::new(config.replacement.clone()),
        recorder: TranscriptRecorder::new(config.transcripts.clone()),
        buffer: LiveWordBuffer::new(Duration::from_millis(700)),
        inserter,
        profiles: options.profiles.then(ProfileReloader::new),
        backend,
        writer: None,
        base_lang: options.lang.clone(),
        effective_lang: options.lang.clone(),
        base_insertion_mode: options.insertion_mode,
        active_profile: None,
        pinned_profile: None,
        last_profile_eval: None,
        was_capturing: false,
        utterance: Vec::new(),
        ping_sent: Arc::new(Mutex::new(None)),
        config,
        options,
        shared,
        commands,
    };

    engine.shared.emit(DictationEvent::VoiceMode {
        status: engine.router.status(),
    });
    if engine.speaker_gate_enabled() {
        engine.shared.emit(DictationEvent::Error {
            message: "speaker_gate: configured but unavailable (unsupported_audio_binding); no instruction dispatch".into(),
        });
    }
    let result = engine.main_loop(&url, audio_rx).await;

    let _ = engine.recorder.flush();
    if engine.options.write_pid_file {
        remove_pid_file();
    }
    engine.shared.stop();
    engine.shared.set_state(DictationState::Inactive);
    engine.shared.emit(DictationEvent::Stopped);
    vlog!("ears-dictation stopped");
    result
}

impl Engine {
    async fn main_loop(&mut self, url: &str, audio_rx: Receiver<Vec<f32>>) -> Result<()> {
        loop {
            if !self.shared.is_running() {
                break;
            }
            self.shared.emit(DictationEvent::Connection {
                state: ConnectionState::Connecting,
                message: None,
            });

            let connection = connect_async(url);
            tokio::pin!(connection);
            let connected = loop {
                if !self.service_disconnected_capture(&audio_rx)? {
                    return Ok(());
                }
                tokio::select! {
                    result = &mut connection => break result,
                    _ = tokio::time::sleep(READ_TIMEOUT) => {}
                }
            };
            match connected {
                Ok((ws_stream, _)) => {
                    vlog!("Connected to transcription server");
                    self.shared.emit(DictationEvent::Connection {
                        state: ConnectionState::Connected,
                        message: None,
                    });
                    let flow = self.session(ws_stream, audio_rx.clone()).await?;
                    self.shared.emit(DictationEvent::Connection {
                        state: ConnectionState::Disconnected,
                        message: None,
                    });
                    if matches!(flow, Flow::Exit) || !self.shared.is_running() {
                        break;
                    }
                    if matches!(
                        flow,
                        Flow::Restart | Flow::DrainedPause | Flow::DrainedResume
                    ) {
                        continue;
                    }
                    vlog!("Disconnected, reconnecting in 2s...");
                }
                Err(e) => {
                    vlog!("Failed to connect: {} (retrying in 2s)", e);
                    self.shared.emit(DictationEvent::Connection {
                        state: ConnectionState::Disconnected,
                        message: Some(e.to_string()),
                    });
                }
            }

            // Retry delay that still honours commands and shutdown.
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                if !self.shared.is_running() {
                    return Ok(());
                }
                if !self.service_disconnected_capture(&audio_rx)? {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        Ok(())
    }

    async fn session(
        &mut self,
        ws_stream: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        audio_rx: Receiver<Vec<f32>>,
    ) -> Result<Flow> {
        self.auto_pause.supported = false;
        self.auto_pause.capabilities_known = false;
        self.auto_pause.warned = false;
        self.auto_pause.reset_words();
        let generation = self.shared.capture_generation();
        if self.shared.is_capturing() {
            if let Some(auto) = &mut self.auto_pause.runtime {
                if !auto.control.active(generation) {
                    auto.start(generation);
                }
            }
        }
        let (mut write, mut read) = ws_stream.split();
        let (writer_tx, mut writer_rx) = mpsc::unbounded_channel::<WriterCommand>();
        self.writer = Some(writer_tx.clone());

        let codec = self
            .options
            .codec
            .clone()
            .unwrap_or_else(|| self.config.dictation.codec.clone());
        let use_opus = codec.eq_ignore_ascii_case("opus");

        if use_opus {
            vlog!("Using Opus audio transport");
            self.send_text(serde_json::json!({"type": "setcodec", "codec": "opus"}).to_string());
        }
        if let Some(lang) = self.effective_lang.clone() {
            vlog!("Setting language to: {}", lang);
            self.send_text(serde_json::json!({"type": "setlanguage", "lang": lang}).to_string());
        }
        if let Some(engine) = self.options.engine.clone() {
            vlog!("Selecting engine: {}", engine);
            self.send_text(serde_json::json!({"type": "setengine", "engine": engine}).to_string());
        }

        let mut opus_encoder = if use_opus {
            Some(kaudio::ogg_opus::Encoder::new(24_000).context("failed to create opus encoder")?)
        } else {
            None
        };
        if let Some(enc) = opus_encoder.as_ref() {
            let _ = writer_tx.send(WriterCommand::Audio(enc.header_data().to_vec()));
        }
        // One consumer owns capture ordering, retention and transport for this
        // session. No detached receiver may steal chunks after reconnect.

        let ping_sent = self.ping_sent.clone();
        let mut writer_handle = tokio::spawn(async move {
            while let Some(cmd) = writer_rx.recv().await {
                match cmd {
                    WriterCommand::Audio(bytes) => {
                        if write.send(Message::binary(bytes)).await.is_err() {
                            break;
                        }
                    }
                    WriterCommand::Text(text) => {
                        if write.send(Message::text(text)).await.is_err() {
                            break;
                        }
                    }
                    WriterCommand::Ping => {
                        *ping_sent.lock().unwrap() = Some(Instant::now());
                        if write.send(Message::Ping(Vec::new().into())).await.is_err() {
                            break;
                        }
                    }
                    WriterCommand::Stop => {
                        // Send close frame to properly terminate the WebSocket
                        let _ = write.send(Message::Close(None)).await;
                        break;
                    }
                }
            }
        });

        let mut last_ping = Instant::now();
        let _ = writer_tx.send(WriterCommand::Ping);

        let flow = loop {
            if !self.shared.is_running() {
                break Flow::Exit;
            }
            if let Some(flow) = self.poll_commands()? {
                break flow;
            }
            if let Some(flow) = self.advance_capture(&audio_rx)? {
                break flow;
            }
            if let Some(flow) = self.capture_batch(&audio_rx, &mut opus_encoder)? {
                break flow;
            }
            if let Some(flow) = self.begin_idle_drain(&audio_rx, &mut opus_encoder)? {
                break flow;
            }
            if last_ping.elapsed() >= PING_INTERVAL && !self.auto_pause.draining() {
                last_ping = Instant::now();
                let _ = writer_tx.send(WriterCommand::Ping);
            }

            let incoming = tokio::time::timeout(READ_TIMEOUT, read.next()).await;
            if self.auto_pause.draining() {
                if let Some(flow) = self.poll_commands()? {
                    break flow;
                }
                if let Some(flow) = self.interrupted_drain()? {
                    break flow;
                }
            }
            match incoming {
                Ok(Some(message)) => match message {
                    Ok(Message::Text(text)) => {
                        vlog!("[WS RECEIVED] {}", text);
                        if let Ok(json) = serde_json::from_str::<Value>(&text) {
                            if let Some(flow) = self.handle_session_message(&json)? {
                                break flow;
                            }
                        } else {
                            vlog!("[ERROR] Failed to parse JSON");
                        }
                    }
                    Ok(Message::Binary(data)) => {
                        vlog!("[WS BINARY] {} bytes", data.len());
                    }
                    Ok(Message::Pong(_)) => {
                        let sent = self.ping_sent.lock().unwrap().take();
                        if let Some(sent) = sent {
                            let latency = sent.elapsed().as_millis() as u64;
                            if self.backend.latency_ms != Some(latency) {
                                self.backend.latency_ms = Some(latency);
                                self.shared
                                    .emit(DictationEvent::Backend(self.backend.clone()));
                            }
                        }
                    }
                    Ok(Message::Close(_)) => {
                        vlog!("WebSocket closed");
                        if self.auto_pause.draining()
                            && self.auto_pause.accepts(self.shared.capture_generation())
                        {
                            break self.auto_pause_failure(
                                "server closed without an owned finalization acknowledgement",
                            );
                        }
                        break Flow::Reconnect;
                    }
                    Err(e) => {
                        if self.auto_pause.draining() {
                            break self.auto_pause_failure(
                                "server connection failed before finalization acknowledgement",
                            );
                        }
                        vlog!("WebSocket error: {}", e);
                        self.shared.emit(DictationEvent::Error {
                            message: format!("connection error: {e}"),
                        });
                        break Flow::Reconnect;
                    }
                    _ => {}
                },
                Ok(None) => {
                    if self.auto_pause.draining()
                        && self.auto_pause.accepts(self.shared.capture_generation())
                    {
                        break self.auto_pause_failure(
                            "server disconnected before finalization acknowledgement",
                        );
                    }
                    break Flow::Reconnect;
                }
                Err(_) => {
                    // Timed out waiting for WS data; loop back so commands are
                    // observed promptly. Only retain text that could begin a
                    // multi-word dictionary phrase, and cap that hold so
                    // standalone words still type.
                    if let Some(typed) = self.buffer.flush_if_expired(
                        self.inserter.keyboard(),
                        &mut self.replacement,
                        &mut self.recorder,
                    )? {
                        self.emit_typed(typed);
                    }
                    self.recorder.flush_if_due()?;
                }
            }
        };

        // A connection loss is not an utterance boundary. Never classify a
        // half-heard command or let it join the first words after reconnect.
        self.router.discard_pending();
        self.utterance.clear();
        if flow != Flow::DrainedResume {
            self.discard_history();
            self.auto_pause.clear_audio();
            if let Some(auto) = &mut self.auto_pause.runtime {
                auto.stop();
            }
        }
        self.buffer.discard();
        let _ = writer_tx.send(WriterCommand::Stop);
        if tokio::time::timeout(Duration::from_secs(1), &mut writer_handle)
            .await
            .is_err()
        {
            writer_handle.abort();
            let _ = writer_handle.await;
        }
        self.writer = None;
        self.backend.latency_ms = None;
        if let Some(typed) = self.buffer.flush_all(
            self.inserter.keyboard(),
            &mut self.replacement,
            &mut self.recorder,
        )? {
            self.emit_typed(typed);
        }
        self.recorder.flush()?;
        Ok(flow)
    }

    fn abort_auto_pause(&mut self) {
        if let Some(pause) = self.auto_pause.local_pause.take() {
            self.shared.emit(DictationEvent::Finalization {
                capture_generation: pause.owner_generation,
                state: FinalizationState::Unconfirmed,
            });
        }
        if let Some(auto) = &mut self.auto_pause.runtime {
            auto.stop();
        }
        self.auto_pause.clear_audio();
        self.was_capturing = false; // New capture generation starts a fresh insertion session.
        self.auto_pause.capture_generation = None;
    }

    fn auto_pause_failure(&mut self, message: &str) -> Flow {
        if self.auto_pause.local_pause.is_some() {
            return self.finish_local_pause(false, Some(message));
        }
        self.buffer.discard();
        self.router.discard_pending();
        self.utterance.clear();
        self.discard_history();
        self.abort_auto_pause();
        self.shared.set_capturing(false);
        self.shared.emit(DictationEvent::Error {
            message: format!("auto-pause: {message}"),
        });
        Flow::Restart
    }

    fn handle_session_message(&mut self, json: &Value) -> Result<Option<Flow>> {
        use super::auto_pause::{Completion, DRAIN_CAPABILITY};
        if self.auto_pause.local_pause.is_some() {
            return self.handle_pending_finalization(json);
        }
        let generation = self.shared.capture_generation();
        if !self.auto_pause.accepts(generation) {
            return Ok(None);
        }
        match json.get("type").and_then(Value::as_str) {
            Some("capabilities") => {
                self.auto_pause.capabilities_known = true;
                self.auto_pause.supported = json
                    .get("features")
                    .and_then(Value::as_array)
                    .is_some_and(|features| {
                        features
                            .iter()
                            .any(|value| value.as_str() == Some(DRAIN_CAPABILITY))
                    });
                if self.auto_pause.draining() && !self.auto_pause.supported {
                    return Ok(Some(self.auto_pause_failure(
                        "server withdrew ordered-drain support during finalization",
                    )));
                }
            }
            Some("enginechanged") => {
                if self.auto_pause.draining() {
                    return Ok(Some(
                        self.auto_pause_failure("engine changed during finalization"),
                    ));
                }
                self.auto_pause.reset_words();
                self.handle_message(json)?;
            }
            Some("drained") if self.auto_pause.draining() => {
                let Some(id) = json.get("request_id").and_then(Value::as_u64) else {
                    return Ok(Some(
                        self.auto_pause_failure("invalid finalization acknowledgement"),
                    ));
                };
                if self.shared.ptt_held() {
                    self.auto_pause
                        .runtime
                        .as_mut()
                        .unwrap()
                        .control
                        .resume(generation);
                }
                let outcome = self
                    .auto_pause
                    .runtime
                    .as_mut()
                    .unwrap()
                    .control
                    .acknowledge(id, generation);
                match outcome {
                    Ok(Completion::Stale) => {}
                    Ok(Completion::Resume) => {
                        self.auto_pause.resume_audio();
                        eprintln!(
                            "auto-pause: speech resumed during finalization; replaying on a new stream"
                        );
                        return Ok(Some(Flow::DrainedResume));
                    }
                    Ok(Completion::Paused) => {
                        self.shared.set_capturing(false);
                        self.on_capture_edge(false)?;
                        eprintln!("auto-pause: paused after finalized capture");
                        return Ok(Some(Flow::DrainedPause));
                    }
                    Err(err) => return Ok(Some(self.auto_pause_failure(&err.to_string()))),
                }
            }
            Some("final") if self.auto_pause.draining() => {
                let Some(text) = json.get("text").and_then(Value::as_str) else {
                    return Ok(Some(
                        self.auto_pause_failure("final result is missing text"),
                    ));
                };
                let suffix = match self.auto_pause.missing_final_words(text) {
                    Ok(suffix) => suffix,
                    Err(err) => return Ok(Some(self.auto_pause_failure(&err.to_string()))),
                };
                for word in suffix {
                    if !self.shared.is_capturing() || self.shared.capture_generation() != generation
                    {
                        return Ok(None);
                    }
                    self.auto_pause.note_word(&word);
                    self.on_word(&word)?;
                }
                if !self.shared.is_capturing() || self.shared.capture_generation() != generation {
                    return Ok(None);
                }
                if !self.utterance.is_empty() {
                    self.complete_utterance()?;
                    self.just_finished = true;
                }
                if self.shared.capture_generation() == generation {
                    if let Some(auto) = &mut self.auto_pause.runtime {
                        auto.control.final_seen(generation);
                    }
                }
            }
            _ => {
                if json.get("type").and_then(Value::as_str) == Some("word") {
                    if let Some(word) = json.get("word").and_then(Value::as_str) {
                        self.auto_pause.note_word(word);
                    }
                }
                self.handle_message(json)?;
            }
        }
        Ok(None)
    }

    fn send_text(&self, text: String) {
        if let Some(writer) = &self.writer {
            let _ = writer.send(WriterCommand::Text(text));
        }
    }

    fn poll_commands(&mut self) -> Result<Option<Flow>> {
        while let Ok(command) = self.commands.try_recv() {
            let flow = self.handle_command(command)?;
            if matches!(flow, Flow::Exit | Flow::Restart) {
                return Ok(Some(flow));
            }
        }
        Ok(None)
    }

    fn handle_command(&mut self, command: DictationCommand) -> Result<Flow> {
        let draining = self.auto_pause.draining();
        let resume = match &command {
            DictationCommand::SetCapturing(true) => true,
            DictationCommand::Toggle => !self.shared.is_capturing(),
            _ => false,
        };
        match command {
            DictationCommand::SetCapturing(on) => {
                self.shared.set_capturing(on);
            }
            DictationCommand::Toggle => {
                self.shared.toggle();
            }
            DictationCommand::Cancel => {
                self.buffer.discard();
                self.router.cancel();
                self.router.mode = VoiceMode::Dictation; // Escape is always a way out.
                self.sync_escape_policy();
                self.utterance.clear();
                self.discard_history();
                self.shared.emit(DictationEvent::Cancelled);
                self.shared.emit(DictationEvent::VoiceMode {
                    status: self.router.status(),
                });
                self.shared.set_capturing(false);
            }
            DictationCommand::SetVoiceMode(mode) => {
                self.buffer.discard();
                self.router.cancel();
                self.router.mode = mode;
                self.sync_escape_policy();
                self.utterance.clear();
                self.discard_history();
                self.shared.emit(DictationEvent::VoiceMode {
                    status: self.router.status(),
                });
            }
            DictationCommand::SetInsertionMode(mode) => {
                self.base_insertion_mode = mode;
                self.apply_insertion_mode(mode)?;
            }
            DictationCommand::SetProfile(name) => {
                self.pinned_profile = name;
                self.last_profile_eval = None;
                self.evaluate_profiles(true)?;
            }
            DictationCommand::CycleProfile => {
                let names: Vec<String> = self
                    .profiles
                    .as_ref()
                    .map(|p| p.set().config().names_by_priority())
                    .unwrap_or_default();
                let next = match &self.pinned_profile {
                    None => names.first().cloned(),
                    Some(current) => names
                        .iter()
                        .position(|n| n.eq_ignore_ascii_case(current))
                        .and_then(|idx| names.get(idx + 1).cloned()),
                };
                self.pinned_profile = next;
                self.last_profile_eval = None;
                self.evaluate_profiles(true)?;
            }
            DictationCommand::SetLanguage(lang) => {
                self.base_lang = if lang.eq_ignore_ascii_case("auto") {
                    None
                } else {
                    Some(lang)
                };
                self.last_profile_eval = None;
                self.evaluate_profiles(true)?;
            }
            DictationCommand::SetEscapeCancels(on) => {
                self.shared.set_escape_cancels(
                    on || self.router.mode != VoiceMode::Dictation || self.speaker_gate_enabled(),
                );
            }
            DictationCommand::SetHotkeyMode(mode) => {
                self.shared.set_hotkey_mode(mode);
            }
            DictationCommand::SetAutoPauseSpeechThreshold(threshold) => {
                if let Err(error) = self.auto_pause.set_speech_threshold(threshold) {
                    self.shared.emit(DictationEvent::Error {
                        message: error.to_string(),
                    });
                }
                // A sensitivity edit is not a capture edge: preserve the owned
                // Final/ACK barrier, buffered PCM, generation and held-key state.
                return Ok(Flow::Continue);
            }
            DictationCommand::Reload => {
                self.registry.reload();
                self.replacement.reload_if_changed();
                if let Some(profiles) = self.profiles.as_mut() {
                    profiles.refresh();
                }
                self.last_profile_eval = None;
                self.evaluate_profiles(true)?;
            }
            DictationCommand::Shutdown => {
                self.shared.stop();
                return Ok(Flow::Exit);
            }
        }
        if draining {
            if resume
                && self.shared.is_capturing()
                && self.auto_pause.accepts(self.shared.capture_generation())
            {
                let generation = self
                    .auto_pause
                    .local_pause
                    .as_ref()
                    .map_or(self.shared.capture_generation(), |pause| {
                        pause.owner_generation
                    });
                if let Some(auto) = &mut self.auto_pause.runtime {
                    auto.control.resume(generation);
                }
            } else {
                self.buffer.discard();
                self.router.discard_pending();
                self.utterance.clear();
                if !self.shared.is_capturing() && self.was_capturing {
                    self.on_capture_edge(false)?;
                }
                self.abort_auto_pause();
                return Ok(Flow::Restart);
            }
        }
        Ok(Flow::Continue)
    }

    fn sync_escape_policy(&self) {
        let configured = self
            .options
            .escape_cancels
            .unwrap_or(self.config.hotkeys.escape_cancels);
        self.shared.set_escape_cancels(
            configured || self.router.mode != VoiceMode::Dictation || self.speaker_gate_enabled(),
        );
    }

    fn on_capture_edge(&mut self, capturing: bool) -> Result<()> {
        self.was_capturing = capturing;
        self.auto_pause.capture_generation = Some(self.shared.capture_generation());
        if capturing {
            self.auto_pause.clear_audio();
            if let Some(auto) = &mut self.auto_pause.runtime {
                auto.start(self.shared.capture_generation());
            }
            self.router.discard_pending();
            self.utterance.clear();
            if let Some(history) = self.history.as_mut() {
                history.begin_session();
            }
            self.inserter.begin_session()?;
            self.last_profile_eval = None;
            self.evaluate_profiles(true)?;
        } else {
            self.finish_capture_session(self.auto_pause.local_pause.is_none())?;
        }
        Ok(())
    }

    fn finish_capture_session(&mut self, emit_prompt: bool) -> Result<()> {
        self.was_capturing = false;
        if let Some(auto) = &mut self.auto_pause.runtime {
            auto.stop();
        }
        self.auto_pause.clear_audio();
        let context = if self.history.is_some() {
            capture_context(self.effective_lang.clone())
        } else {
            CommitContext::default()
        };
        if let Some(history) = self
            .history
            .as_mut()
            .filter(|_| self.auto_pause.local_pause.is_none())
        {
            if let Err(err) = history.end_capture(&context) {
                self.shared.emit(DictationEvent::Error {
                    message: format!("audio history: {err:#}"),
                });
            }
        }
        // PTT release is not Speech(false): incomplete commands are
        // discarded, but already-safe live dictionary words still flush.
        if let Some(typed) = self.buffer.flush_all(
            self.inserter.keyboard(),
            &mut self.replacement,
            &mut self.recorder,
        )? {
            self.emit_typed(typed);
        }
        // No audio follows a release, so no end-of-speech boundary will
        // arrive: what was said up to here is the utterance.
        if !self.utterance.is_empty() && !self.speaker_gate_enabled() {
            let raw = self.utterance.join(" ");
            let text =
                super::typing::normalize_punctuation_spacing(&self.replacement.replace(&raw));
            self.commit_history(&raw, &text);
        } else {
            self.discard_history();
        }
        self.router.discard_pending();
        self.utterance.clear();
        if emit_prompt {
            self.inserter.end_session(&self.shared);
        }
        Ok(())
    }

    fn apply_insertion_mode(&mut self, mode: InsertionMode) -> Result<()> {
        if self.inserter.mode() == mode {
            return Ok(());
        }
        self.inserter.set_mode(mode)?;
        self.shared.emit(DictationEvent::InsertionMode(mode));
        Ok(())
    }

    fn set_effective_language(&mut self, lang: Option<String>) {
        if lang == self.effective_lang {
            return;
        }
        if let Some(lang) = &lang {
            self.send_text(serde_json::json!({"type": "setlanguage", "lang": lang}).to_string());
            self.shared.emit(DictationEvent::Language(lang.clone()));
        }
        self.effective_lang = lang;
    }

    /// Re-evaluate the active profile. Cheap when called often: frontmost
    /// lookups are throttled unless `force` is set.
    fn evaluate_profiles(&mut self, force: bool) -> Result<()> {
        let Some(profiles) = self.profiles.as_mut() else {
            return Ok(());
        };
        if !force
            && self
                .last_profile_eval
                .is_some_and(|at| at.elapsed() < PROFILE_EVAL_INTERVAL)
        {
            return Ok(());
        }
        self.last_profile_eval = Some(Instant::now());
        profiles.refresh();

        let (profile, pinned): (Option<ContextProfile>, bool) = match &self.pinned_profile {
            Some(name) => match profiles.set().get(name) {
                Some(profile) => (Some(profile.clone()), true),
                None => {
                    self.shared.emit(DictationEvent::Error {
                        message: format!("profile '{name}' not found"),
                    });
                    self.pinned_profile = None;
                    (None, false)
                }
            },
            None => {
                if profiles.set().is_empty() {
                    (None, false)
                } else {
                    let resolved =
                        frontmost_app().and_then(|app| profiles.set().resolve(&app).cloned());
                    (resolved, false)
                }
            }
        };

        let name = profile.as_ref().map(|p| p.name.clone());
        let lang = profile
            .as_ref()
            .and_then(|p| p.language.clone())
            .or_else(|| self.base_lang.clone());
        let mode = profile
            .as_ref()
            .and_then(|p| p.insertion_mode)
            .unwrap_or(self.base_insertion_mode);

        if name != self.active_profile {
            self.active_profile = name.clone();
            let dictionaries = profile
                .as_ref()
                .map(|p| p.dictionaries.clone())
                .unwrap_or_default();
            self.replacement.set_override(&dictionaries);
            vlog!(
                "[profile] active profile: {}",
                name.as_deref().unwrap_or("(default)")
            );
            self.shared.emit(DictationEvent::Profile { name, pinned });
        }
        self.set_effective_language(lang);
        self.apply_insertion_mode(mode)?;
        Ok(())
    }

    fn handle_message(&mut self, json: &Value) -> Result<()> {
        if !self.auto_pause.accepts(self.shared.capture_generation()) {
            return Ok(());
        }
        let is_capturing = self.shared.is_capturing() || self.auto_pause.local_pause.is_some();
        let Some(event_type) = json.get("type").and_then(|v| v.as_str()) else {
            return Ok(());
        };
        match event_type {
            "word" if is_capturing => {
                if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
                    self.on_word(word)?;
                }
            }
            "speech" => {
                let active = json.get("active").and_then(|v| v.as_bool()).unwrap_or(true);
                self.shared.emit(DictationEvent::Speech { active });
                if active {
                    self.just_finished = false;
                }
                if is_capturing && !active && !self.utterance.is_empty() {
                    self.complete_utterance()?;
                    self.just_finished = true;
                    self.evaluate_profiles(true)?;
                }
            }
            "final" if is_capturing && !self.just_finished => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    self.on_final(text)?;
                }
            }
            "enginechanged" => {
                if let (Some(history), Some(engine)) = (
                    self.history.as_mut(),
                    json.get("engine").and_then(|v| v.as_str()),
                ) {
                    history.set_engine(Some(engine.to_string()));
                }
            }
            "error" => {
                let message = json
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("server error")
                    .to_string();
                self.shared.emit(DictationEvent::Error { message });
            }
            _ => {}
        }
        Ok(())
    }

    fn on_word(&mut self, word: &str) -> Result<()> {
        if word.is_empty() {
            return Ok(());
        }
        self.just_finished = false;
        if self.speaker_gate_enabled()
            && (self.utterance.len() >= crate::speaker_gate::MAX_TEXT_BYTES
                || self.utterance.iter().map(String::len).sum::<usize>()
                    + self.utterance.len()
                    + word.len()
                    > crate::speaker_gate::MAX_TEXT_BYTES)
        {
            // No gated text ever reaches typing buffers, including oversized
            // partials. The legacy protocol cannot bind this PCM to a final.
            self.utterance.clear();
            self.shared.emit(DictationEvent::Error {
                message: "speaker_gate: payload_limit (no dispatch)".into(),
            });
            return Ok(());
        }
        self.utterance.push(word.to_owned());
        if self.utterance.len() == 1 {
            if let Some(history) = self.history.as_mut() {
                history.begin_utterance();
            }
        }
        // Preview is independent of OS keystrokes, including held prefixes.
        self.emit_partial();
        if self.speaker_gate_enabled() {
            // Preview only. Never send an enabled-but-unavailable gate through
            // the live dictation/router fallback, even in Dictation mode.
            return Ok(());
        }
        if self.auto_pause.local_pause.is_none() && self.shared.take_wake_started() {
            self.router
                .arm_wake_strip(&self.config.dictation.wake.strip_words, 3);
        }
        if let Some(safe) = self.router.word(word) {
            self.buffer.push(safe);
            if let Some(typed) = self.buffer.flush_ready(
                self.inserter.keyboard(),
                &mut self.replacement,
                &mut self.recorder,
            )? {
                self.emit_typed(typed);
            }
        }
        Ok(())
    }

    fn on_final(&mut self, text: &str) -> Result<()> {
        if text.trim().is_empty() {
            return Ok(());
        }
        // Some backends emit final-only text; final after Speech(false) is
        // suppressed by just_finished. Never re-insert a streamed partial.
        if self.utterance.is_empty() {
            for word in text.split_whitespace() {
                self.on_word(word)?;
            }
        }
        self.complete_utterance()?;
        self.just_finished = true;
        Ok(())
    }

    fn emit_partial(&mut self) {
        let raw = self.utterance.join(" ");
        let text = super::typing::normalize_punctuation_spacing(&self.replacement.replace(&raw));
        self.shared.emit(DictationEvent::Partial { text });
    }

    /// Report typed text to the UI and to the utterance being retained.
    fn emit_typed(&mut self, text: String) {
        if let Some(history) = self.history.as_mut() {
            history.note_inserted(&text);
        }
        self.shared.emit(DictationEvent::Typed { text });
    }

    fn discard_history(&mut self) {
        if let Some(history) = self.history.as_mut() {
            history.discard();
        }
    }

    /// Keep the audio of a finished utterance when retention is on. Failures
    /// are reported but never interrupt dictation.
    fn commit_history(&mut self, raw: &str, replaced: &str) {
        let Some(history) = self.history.as_mut() else {
            return;
        };
        let app = frontmost_app();
        let ctx = CommitContext {
            raw: raw.to_string(),
            replaced: replaced.to_string(),
            language: self.effective_lang.clone(),
            app_bundle_id: app.as_ref().and_then(|a| a.bundle_id.clone()),
            app_binary: app.as_ref().and_then(|a| a.binary.clone()),
            app_name: app.as_ref().and_then(|a| a.name.clone()),
            secure_input: crate::frontmost::secure_input_active(),
        };
        if history.client_vad_enabled() {
            if let Err(err) = history.record_recognition(&ctx) {
                self.shared.emit(DictationEvent::Error {
                    message: format!("audio history recognition: {err:#}"),
                });
            }
            return;
        }
        match history.commit(&ctx) {
            Ok(Ok(record)) => vlog!(
                "[audio-history] kept {} ({:.1}s)",
                record.id,
                record.duration_secs
            ),
            Ok(Err(Skip::NoAudio)) => {}
            Ok(Err(skip)) => {
                let note = match skip {
                    Skip::ExcludedApp(app) => format!("not kept: excluded app {app}"),
                    Skip::SecureInput => "not kept: secure input active".to_string(),
                    Skip::NoAudio => unreachable!(),
                };
                self.shared.emit(DictationEvent::AudioHistory {
                    active: true,
                    note: Some(note),
                });
            }
            Err(err) => self.shared.emit(DictationEvent::Error {
                message: format!("audio history: {err:#}"),
            }),
        }
    }

    fn speaker_gate_enabled(&self) -> bool {
        self.config.voice_commands.speaker_policy.mode != crate::speaker_gate::Mode::Disabled
    }

    fn complete_utterance(&mut self) -> Result<()> {
        if self.utterance.is_empty() {
            return Ok(());
        }
        if self.speaker_gate_enabled() {
            // Legacy word/speech/final events contain no authoritative PCM span,
            // utterance ID or revision. Arrival-time audio slicing is NOT evidence.
            // Do not emit Utterance/PromptReady, record the private transcript, run
            // controls, or type rejected instructions into an ungated target.
            self.utterance.clear();
            self.buffer.discard();
            self.router.discard_pending();
            self.discard_history();
            self.shared.emit(DictationEvent::VoiceDecision {
                raw: String::new(),
                normalized: String::new(),
                outcome: "speaker_gate_unavailable_audio_binding".into(),
            });
            self.shared.emit(DictationEvent::Error {
                message: "speaker_gate: unsupported_audio_binding (no dispatch)".into(),
            });
            return Ok(());
        }
        self.registry.reload_if_changed();
        let raw = self.utterance.join(" ");
        let text = super::typing::normalize_punctuation_spacing(&self.replacement.replace(&raw));
        let route = self.router.finish(&text, &self.registry);
        match &route {
            Route::Type(text) | Route::Action(text) => {
                // Live words are already in the buffer. Buffered utterances
                // enter here only after the entire VAD-bounded phrase matches.
                if let Some(typed) = self.buffer.flush_all(
                    self.inserter.keyboard(),
                    &mut self.replacement,
                    &mut self.recorder,
                )? {
                    self.emit_typed(typed);
                }
                let formatted = self.buffer.type_formatted(self.inserter.keyboard(), text)?;
                self.emit_typed(formatted);
            }
            Route::Discard => self.buffer.discard(),
            Route::Streamed => {
                if let Some(typed) = self.buffer.flush_all(
                    self.inserter.keyboard(),
                    &mut self.replacement,
                    &mut self.recorder,
                )? {
                    self.emit_typed(typed);
                }
            }
            Route::Control(key) => {
                self.sync_escape_policy();
                self.buffer.discard();
                self.shared.emit(DictationEvent::VoiceMode {
                    status: self.router.status(),
                });
                if key == "ears cancel" || key == super::voice_commands::STOP_KEY {
                    self.shared.emit(DictationEvent::Cancelled);
                    self.shared.set_capturing(false);
                }
            }
        }
        let outcome = match route {
            Route::Type(_) => "typed",
            Route::Action(_) => "literal_text",
            Route::Control(_) => "control",
            Route::Discard => "discarded",
            Route::Streamed => "streamed",
        };
        self.recorder
            .record(&format!("voice_decision_{outcome}"), &raw, &text)?;
        self.commit_history(&raw, &text);
        self.shared.emit(DictationEvent::VoiceDecision {
            raw,
            normalized: text.clone(),
            outcome: outcome.into(),
        });
        self.utterance.clear();
        self.shared.emit(DictationEvent::Utterance { text });
        Ok(())
    }
}

fn capture_context(language: Option<String>) -> CommitContext {
    let app = crate::frontmost::frontmost_app_identity();
    CommitContext {
        language,
        app_bundle_id: app.as_ref().and_then(|app| app.bundle_id.clone()),
        app_binary: app.as_ref().and_then(|app| app.binary.clone()),
        app_name: app.as_ref().and_then(|app| app.name.clone()),
        secure_input: crate::frontmost::secure_input_active(),
        ..CommitContext::default()
    }
}

fn flush_opus_for_drain(encoder: &mut kaudio::ogg_opus::Encoder) -> Result<Vec<u8>> {
    // kaudio 0.2.1 uses 960-sample encoder frames with no flush API. One full
    // zero frame emits every retained source sample; any residual is only pad.
    // Padding is transport-only, never part of retained PCM or the VAD clock.
    encoder
        .encode_page(&[0.0; 960])
        .context("flushing retained Opus samples")
}

fn encode_chunk(chunk: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(chunk.len() * 4);
    for sample in chunk {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

#[path = "engine_finalization.rs"]
mod finalization;

#[path = "engine_capture.rs"]
mod capture;

#[cfg(all(test, feature = "client-vad"))]
#[path = "engine_auto_pause_tests.rs"]
mod auto_pause_tests;

#[cfg(all(test, feature = "client-vad"))]
#[path = "engine_local_pause_tests.rs"]
mod local_pause_tests;

#[cfg(all(test, feature = "client-vad"))]
#[path = "engine_auto_pause_progress_tests.rs"]
mod auto_pause_progress_tests;

#[cfg(test)]
mod tests {
    use super::super::typing::LiveWordBuffer;
    use super::super::typing::tests::RecordingKeyboard;
    use super::super::typing::tests::silent_recorder;
    use super::super::typing::tests::test_reloader;
    use super::*;
    use crate::virtual_keyboard::VirtualKeyboard;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn wake_started_utterance_drops_the_phrase_and_stop_phrase_pauses() {
        let dir = std::env::temp_dir().join(format!("ears-engine-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = AppConfig::default();
        config.replacement.enabled = false;
        config.transcripts.enabled = false;
        config.dictation.wake.stop_phrases = vec!["okto stopp".into()];
        let (events_tx, events_rx) = unbounded();
        let (commands_tx, commands_rx) = unbounded();
        let shared = Arc::new(Shared::new(&config, true, events_tx, commands_tx, true));
        let recording = RecordingKeyboard::default();
        let typed = recording.typed.clone();
        let mut inserter = Inserter::new(InsertionMode::Clipboard).unwrap();
        *inserter.keyboard() = Box::new(recording);
        let mut router = Router::new(VoiceMode::Dictation);
        router.set_stop_phrases(&config.dictation.wake.stop_phrases);
        let mut engine = Engine {
            target_reader: frontmost_app,
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
            was_capturing: true,
            utterance: Vec::new(),
            router,
            auto_pause: Default::default(),
            registry: Registry::new(dir.join("commands.toml")),
            just_finished: false,
            ping_sent: Arc::new(Mutex::new(None)),
            history: None,
            config,
            options: DictationOptions::default(),
            shared: shared.clone(),
            commands: commands_rx,
        };
        let say = |engine: &mut Engine, words: &[&str]| {
            for word in words {
                engine
                    .handle_message(&serde_json::json!({"type":"word","word":word}))
                    .unwrap();
            }
            engine
                .handle_message(&serde_json::json!({"type":"speech","active":false}))
                .unwrap();
        };
        // the wake path set capture on and marked the pre-roll
        shared.mark_wake_started();
        say(&mut engine, &["Hey", "Okto,", "schreib", "bitte"]);
        assert_eq!(&*typed.lock().unwrap(), "schreib bitte ");
        // a stop phrase pauses capture and types nothing
        say(&mut engine, &["Okto,", "stopp."]);
        assert_eq!(&*typed.lock().unwrap(), "schreib bitte ");
        assert!(!shared.is_capturing(), "stop phrase paused capture");
        assert!(
            events_rx
                .try_iter()
                .any(|e| matches!(e, DictationEvent::Cancelled))
        );
        // a hotkey start does not strip anything
        shared.set_capturing(true);
        say(&mut engine, &["Hey", "Anna"]);
        assert_eq!(&*typed.lock().unwrap(), "schreib bitte Hey Anna ");
    }

    #[test]
    fn message_path_previews_each_word_without_partial_execution() {
        let dir = std::env::temp_dir().join(format!("ears-engine-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("commands.toml");
        std::fs::write(
            &path,
            "[[command]]\nphrase='open notes'\nrisk='safe'\naction={type='literal_text',text='NOTES'}\n",
        )
        .unwrap();
        let mut config = AppConfig::default();
        config.voice_commands.mode = VoiceMode::CommandAware;
        config.replacement.enabled = false;
        config.transcripts.enabled = false;
        let (events_tx, events_rx) = unbounded();
        let (commands_tx, commands_rx) = unbounded();
        let shared = Arc::new(Shared::new(&config, true, events_tx, commands_tx, true));
        let recording = RecordingKeyboard::default();
        let typed = recording.typed.clone();
        let mut inserter = Inserter::new(InsertionMode::Clipboard).unwrap();
        *inserter.keyboard() = Box::new(recording);
        let mut engine = Engine {
            target_reader: frontmost_app,
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
            was_capturing: true,
            utterance: Vec::new(),
            router: Router::new(VoiceMode::CommandAware),
            auto_pause: Default::default(),
            registry: Registry::new(path),
            just_finished: false,
            ping_sent: Arc::new(Mutex::new(None)),
            history: None,
            config,
            options: DictationOptions::default(),
            shared,
            commands: commands_rx,
        };
        for (word, expected) in [("open", "open"), ("notes", "open notes")] {
            engine
                .handle_message(&serde_json::json!({"type":"word","word":word}))
                .unwrap();
            assert_eq!(&*typed.lock().unwrap(), "");
            assert!(
                events_rx
                    .try_iter()
                    .any(|e| matches!(e, DictationEvent::Partial { text } if text == expected))
            );
        }
        engine
            .handle_message(&serde_json::json!({"type":"speech","active":false}))
            .unwrap();
        assert_eq!(&*typed.lock().unwrap(), "NOTES ");
        assert!(events_rx.try_iter().any(|e| matches!(e, DictationEvent::VoiceDecision { outcome, .. } if outcome == "literal_text")));
        engine.router.mode = VoiceMode::Dictation;
        engine
            .handle_message(&serde_json::json!({"type":"word","word":"hello"}))
            .unwrap();
        assert_eq!(&*typed.lock().unwrap(), "NOTES hello ");
        engine
            .handle_message(&serde_json::json!({"type":"speech","active":false}))
            .unwrap();
        engine
            .handle_message(&serde_json::json!({"type":"word","word":"eaRS"}))
            .unwrap();
        assert_eq!(&*typed.lock().unwrap(), "NOTES hello ");
        engine
            .handle_message(&serde_json::json!({"type":"word","word":"not"}))
            .unwrap();
        assert_eq!(&*typed.lock().unwrap(), "NOTES hello eaRS not ");
        engine
            .handle_message(&serde_json::json!({"type":"speech","active":false}))
            .unwrap();
        engine.router.mode = VoiceMode::CommandOnly;
        for word in ["open", "note"] {
            engine
                .handle_message(&serde_json::json!({"type":"word","word":word}))
                .unwrap();
        }
        assert_eq!(&*typed.lock().unwrap(), "NOTES hello eaRS not ");
        engine
            .handle_message(&serde_json::json!({"type":"speech","active":false}))
            .unwrap();
        assert_eq!(&*typed.lock().unwrap(), "NOTES hello eaRS not ");
        for word in ["eaRS", "dictation", "mode"] {
            engine
                .handle_message(&serde_json::json!({"type":"word","word":word}))
                .unwrap();
        }
        engine
            .handle_message(&serde_json::json!({"type":"speech","active":false}))
            .unwrap();
        assert_eq!(engine.router.mode, VoiceMode::Dictation);
        assert_eq!(&*typed.lock().unwrap(), "NOTES hello eaRS not ");
        // A completed final after the VAD boundary must not repeat insertion.
        engine
            .handle_message(&serde_json::json!({"type":"final","text":"eaRS dictation mode"}))
            .unwrap();
        assert_eq!(&*typed.lock().unwrap(), "NOTES hello eaRS not ");
        engine
            .handle_command(DictationCommand::SetVoiceMode(VoiceMode::CommandOnly))
            .unwrap();
        assert!(engine.shared.escape_cancels());
        engine
            .handle_command(DictationCommand::SetEscapeCancels(false))
            .unwrap();
        assert!(engine.shared.escape_cancels());
        engine.handle_command(DictationCommand::Cancel).unwrap();
        assert_eq!(engine.router.mode, VoiceMode::Dictation);
        assert!(!engine.shared.escape_cancels());

        // Enabling a gate on the legacy WS adapter must never silently type,
        // run a control, emit an eligible utterance or fall back after cancel.
        for mode in [
            crate::speaker_gate::Mode::Allowlist,
            crate::speaker_gate::Mode::Annotate,
        ] {
            engine.config.voice_commands.speaker_policy.mode = mode;
            engine.shared.set_capturing(true);
            engine.router.mode = VoiceMode::Dictation; // Explicitly exercise live fallback.
            engine
                .handle_command(DictationCommand::SetEscapeCancels(false))
                .unwrap();
            assert!(engine.shared.escape_cancels()); // Keyboard escape stays a way out.
            let before = typed.lock().unwrap().clone();
            let _ = events_rx.try_iter().collect::<Vec<_>>();
            for word in ["eaRS", "command-only", "mode"] {
                engine
                    .handle_message(&serde_json::json!({"type":"word","word":word}))
                    .unwrap();
            }
            engine
                .handle_message(&serde_json::json!({"type":"speech","active":false}))
                .unwrap();
            engine
                .handle_message(&serde_json::json!({"type":"speech","active":false}))
                .unwrap();
            engine
                .handle_message(&serde_json::json!({"type":"final","text":"open notes"}))
                .unwrap();
            // Physical PTT release remains independent of voice evidence.
            engine
                .handle_command(DictationCommand::SetCapturing(false))
                .unwrap();
            engine.on_capture_edge(false).unwrap();
            assert!(!engine.shared.is_capturing());
            engine.shared.set_capturing(true);
            engine.handle_command(DictationCommand::Cancel).unwrap();
            assert!(!engine.shared.is_capturing());
            assert!(engine.shared.escape_cancels());
            engine
                .handle_message(&serde_json::json!({"type":"final","text":"late final"}))
                .unwrap();
            assert_eq!(engine.router.mode, VoiceMode::Dictation);
            assert_eq!(*typed.lock().unwrap(), before);
            let events: Vec<_> = events_rx.try_iter().collect();
            assert!(events.iter().any(|event| matches!(event,
                DictationEvent::VoiceDecision { raw, normalized, outcome }
                if raw.is_empty() && normalized.is_empty() && outcome == "speaker_gate_unavailable_audio_binding")));
            assert!(!events.iter().any(|event| matches!(
                event,
                DictationEvent::Typed { .. }
                    | DictationEvent::Utterance { .. }
                    | DictationEvent::PromptReady { .. }
            )));
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    pub(super) fn retention_engine(
        config: AppConfig,
        history: Option<AudioHistory>,
    ) -> (Engine, Arc<Mutex<String>>) {
        let (events_tx, _events_rx) = unbounded();
        let (commands_tx, commands_rx) = unbounded();
        let shared = Arc::new(Shared::new(&config, true, events_tx, commands_tx, true));
        let recording = RecordingKeyboard::default();
        let typed = recording.typed.clone();
        let mut inserter = Inserter::new(InsertionMode::Clipboard).unwrap();
        *inserter.keyboard() = Box::new(recording);
        let engine = Engine {
            target_reader: frontmost_app,
            replacement: ReplacementReloader::new(config.replacement.clone()),
            recorder: TranscriptRecorder::new(config.transcripts.clone()),
            buffer: LiveWordBuffer::new(Duration::from_millis(700)),
            inserter,
            profiles: None,
            backend: BackendInfo::default(),
            writer: None,
            base_lang: None,
            effective_lang: Some("de".into()),
            base_insertion_mode: InsertionMode::Clipboard,
            active_profile: None,
            pinned_profile: None,
            last_profile_eval: None,
            was_capturing: true,
            utterance: Vec::new(),
            router: Router::new(VoiceMode::Dictation),
            auto_pause: Default::default(),
            registry: Registry::new(std::env::temp_dir().join("ears-no-commands.toml")),
            just_finished: false,
            ping_sent: Arc::new(Mutex::new(None)),
            history,
            config,
            options: DictationOptions::default(),
            shared,
            commands: commands_rx,
        };
        (engine, typed)
    }

    /// Synthetic audio through the dictation message path, as the session's
    /// audio consumer would feed it, without a microphone or server.
    fn dictate(
        engine: &mut Engine,
        tap: Option<&super::super::audio_history::AudioTap>,
        words: &[&str],
    ) {
        let chunk: Vec<f32> = (0..1920).map(|i| (i as f32 * 0.07).sin() * 0.2).collect();
        let feed = |n: usize| {
            if let Some(tap) = tap {
                for _ in 0..n {
                    tap.push(&chunk);
                }
            }
        };
        feed(10);
        for word in words {
            feed(5);
            engine
                .handle_message(&serde_json::json!({"type":"word","word":word}))
                .unwrap();
        }
        feed(5);
        engine
            .handle_message(&serde_json::json!({"type":"speech","active":false}))
            .unwrap();
    }

    #[test]
    fn audio_history_keeps_one_wav_per_utterance_only_when_enabled() {
        use super::super::audio_history::{self, AudioHistoryConfig};
        let root =
            std::env::temp_dir().join(format!("ears-engine-history-{}", uuid::Uuid::new_v4()));

        // Off (the default): a whole dictation session writes nothing.
        let mut config = AppConfig::default();
        config.replacement.enabled = false;
        config.transcripts.enabled = false;
        config.dictation.audio_history.path = root.join("off").to_string_lossy().to_string();
        assert!(!config.dictation.audio_history.enabled);
        let history =
            AudioHistory::from_config(&config.dictation.audio_history, None, None).unwrap();
        assert!(history.is_none());
        let (mut engine, typed) = retention_engine(config, history);
        dictate(&mut engine, None, &["hallo", "welt"]);
        assert_eq!(&*typed.lock().unwrap(), "hallo welt ");
        assert!(!root.join("off").exists());

        // On: one WAV per committed utterance; cancelled words keep nothing.
        let dir = root.join("on");
        let mut config = AppConfig::default();
        config.replacement.enabled = false;
        config.transcripts.enabled = false;
        config.dictation.audio_history = AudioHistoryConfig {
            enabled: true,
            path: dir.to_string_lossy().to_string(),
            excluded_apps: Vec::new(),
            pause_on_secure_input: false,
            ..AudioHistoryConfig::default()
        };
        let history = AudioHistory::from_config(
            &config.dictation.audio_history,
            Some("Test Mic".into()),
            None,
        )
        .unwrap()
        .unwrap();
        let tap = history.tap();
        let (mut engine, _typed) = retention_engine(config, Some(history));
        engine
            .handle_message(&serde_json::json!({"type":"enginechanged","engine":"parakeet-rs"}))
            .unwrap();
        dictate(&mut engine, Some(&tap), &["erste", "zeile"]);
        dictate(&mut engine, Some(&tap), &["zweite"]);
        engine
            .handle_message(&serde_json::json!({"type":"word","word":"verworfen"}))
            .unwrap();
        engine.handle_command(DictationCommand::Cancel).unwrap();

        let records = audio_history::load_records(&dir).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].raw_transcript, "erste zeile");
        assert_eq!(records[0].inserted_text, "erste zeile");
        assert_eq!(records[1].raw_transcript, "zweite");
        assert_eq!(records[0].session_id, records[1].session_id);
        assert_eq!(records[0].engine.as_deref(), Some("parakeet-rs"));
        assert_eq!(records[0].language.as_deref(), Some("de"));
        for record in &records {
            assert!(dir.join(&record.audio_path).exists());
            assert!(record.duration_secs > 0.5);
        }
        let wavs: usize = std::fs::read_dir(dir.join("audio"))
            .unwrap()
            .map(|day| std::fs::read_dir(day.unwrap().path()).unwrap().count())
            .sum();
        assert_eq!(wavs, 2);

        // Export keeps IDs aligned with the audio and index.
        let out = root.join("export");
        assert_eq!(audio_history::export(&dir, &out).unwrap(), 2);
        let manifest = std::fs::read_to_string(out.join("manifest.jsonl")).unwrap();
        for record in &records {
            assert!(manifest.contains(&record.id));
            assert!(out.join(&record.audio_path).exists());
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn speech_boundary_flushes_held_words() {
        let dir =
            std::env::temp_dir().join(format!("ears-live-dict-test-{}", uuid::Uuid::new_v4()));
        let mut reloader = test_reloader(&dir);
        let mut recorder = silent_recorder();
        let mut buffer = LiveWordBuffer::new(Duration::from_millis(700));
        let recording_keyboard = RecordingKeyboard::default();
        let typed = Arc::clone(&recording_keyboard.typed);
        let mut keyboard: Box<dyn VirtualKeyboard> = Box::new(recording_keyboard);

        // A trailing word that starts a dictionary phrase is held...
        buffer.push("tricks".to_string());
        buffer
            .flush_ready(&mut keyboard, &mut reloader, &mut recorder)
            .unwrap();
        assert_eq!(buffer.chunks, vec!["tricks"]);

        // ...but an end-of-utterance boundary must type it immediately, which
        // is what the engine does on `speech {active:false}`.
        buffer
            .flush_all(&mut keyboard, &mut reloader, &mut recorder)
            .unwrap();
        assert!(buffer.chunks.is_empty());
        assert_eq!(&*typed.lock().unwrap(), "tricks ");
        let _ = std::fs::remove_dir_all(dir);
    }
}
