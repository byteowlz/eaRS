//! The engine loop: audio capture, WebSocket session, message handling,
//! command handling and context profile evaluation.

use super::BackendInfo;
use super::ConnectionState;
use super::DictationCommand;
use super::DictationEvent;
use super::DictationOptions;
use super::DictationState;
use super::InsertionMode;
use super::Shared;
use super::VERBOSE;
use super::alias_for_url;
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

enum Flow {
    Continue,
    Reconnect,
    Exit,
}

struct Engine {
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
    ping_sent: Arc<Mutex<Option<Instant>>>,
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
        .unwrap_or(config.hotkeys.escape_cancels);
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
    let audio_shared = shared.clone();
    thread::Builder::new()
        .name("ears-audio".to_string())
        .spawn(move || {
            if let Err(e) = audio::start_audio_capture(audio_tx, None) {
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

    let mut engine = Engine {
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

            match connect_async(url).await {
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
                while let Ok(command) = self.commands.try_recv() {
                    if matches!(self.handle_command(command)?, Flow::Exit) {
                        return Ok(());
                    }
                }
                let capturing = self.shared.is_capturing();
                if capturing != self.was_capturing {
                    self.on_capture_edge(capturing)?;
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
        let audio_writer = writer_tx.clone();
        let audio_shared = self.shared.clone();
        thread::spawn(move || {
            if let Some(enc) = opus_encoder.as_ref() {
                let _ = audio_writer.send(WriterCommand::Audio(enc.header_data().to_vec()));
            }
            while let Ok(chunk) = audio_rx.recv() {
                if !audio_shared.is_capturing() {
                    continue;
                }
                let bytes = match opus_encoder.as_mut() {
                    Some(enc) => match enc.encode_page(&chunk) {
                        Ok(bytes) => bytes,
                        Err(err) => {
                            eprintln!("opus encode failed: {err}");
                            break;
                        }
                    },
                    None => encode_chunk(&chunk),
                };
                if bytes.is_empty() {
                    continue;
                }
                if audio_writer.send(WriterCommand::Audio(bytes)).is_err() {
                    break;
                }
            }
        });

        let ping_sent = self.ping_sent.clone();
        let writer_handle = tokio::spawn(async move {
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
            let mut exit = false;
            while let Ok(command) = self.commands.try_recv() {
                if matches!(self.handle_command(command)?, Flow::Exit) {
                    exit = true;
                }
            }
            if exit {
                break Flow::Exit;
            }

            let capturing = self.shared.is_capturing();
            if capturing != self.was_capturing {
                self.on_capture_edge(capturing)?;
            }
            if capturing {
                self.evaluate_profiles(false)?;
            }
            if last_ping.elapsed() >= PING_INTERVAL {
                last_ping = Instant::now();
                let _ = writer_tx.send(WriterCommand::Ping);
            }

            match tokio::time::timeout(READ_TIMEOUT, read.next()).await {
                Ok(Some(message)) => match message {
                    Ok(Message::Text(text)) => {
                        vlog!("[WS RECEIVED] {}", text);
                        if let Ok(json) = serde_json::from_str::<Value>(&text) {
                            self.handle_message(&json)?;
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
                        break Flow::Reconnect;
                    }
                    Err(e) => {
                        vlog!("WebSocket error: {}", e);
                        self.shared.emit(DictationEvent::Error {
                            message: format!("connection error: {e}"),
                        });
                        break Flow::Reconnect;
                    }
                    _ => {}
                },
                Ok(None) => break Flow::Reconnect,
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
                        self.shared.emit(DictationEvent::Typed { text: typed });
                    }
                    self.recorder.flush_if_due()?;
                }
            }
        };

        let _ = writer_tx.send(WriterCommand::Stop);
        let _ = writer_handle.await;
        self.writer = None;
        self.backend.latency_ms = None;
        if let Some(typed) = self.buffer.flush_all(
            self.inserter.keyboard(),
            &mut self.replacement,
            &mut self.recorder,
        )? {
            self.shared.emit(DictationEvent::Typed { text: typed });
        }
        self.recorder.flush()?;
        Ok(flow)
    }

    fn send_text(&self, text: String) {
        if let Some(writer) = &self.writer {
            let _ = writer.send(WriterCommand::Text(text));
        }
    }

    fn handle_command(&mut self, command: DictationCommand) -> Result<Flow> {
        match command {
            DictationCommand::SetCapturing(on) => {
                self.shared.set_capturing(on);
            }
            DictationCommand::Toggle => {
                self.shared.toggle();
            }
            DictationCommand::Cancel => {
                self.buffer.discard();
                self.utterance.clear();
                self.shared.emit(DictationEvent::Cancelled);
                self.shared.set_capturing(false);
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
                self.shared.set_escape_cancels(on);
            }
            DictationCommand::SetHotkeyMode(mode) => {
                self.shared.set_hotkey_mode(mode);
            }
            DictationCommand::Reload => {
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
        Ok(Flow::Continue)
    }

    fn on_capture_edge(&mut self, capturing: bool) -> Result<()> {
        self.was_capturing = capturing;
        if capturing {
            self.utterance.clear();
            self.inserter.begin_session()?;
            self.last_profile_eval = None;
            self.evaluate_profiles(true)?;
        } else {
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
        let is_capturing = self.shared.is_capturing();
        let Some(event_type) = json.get("type").and_then(|v| v.as_str()) else {
            return Ok(());
        };
        match event_type {
            "word" if is_capturing => {
                if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
                    if !word.is_empty() {
                        self.buffer.push(word.to_string());
                        self.utterance.push(word.to_string());
                        self.emit_partial();
                        if let Some(typed) = self.buffer.flush_ready(
                            self.inserter.keyboard(),
                            &mut self.replacement,
                            &mut self.recorder,
                        )? {
                            self.shared.emit(DictationEvent::Typed { text: typed });
                        }
                    }
                }
            }
            "speech" => {
                let active = json.get("active").and_then(|v| v.as_bool()).unwrap_or(true);
                self.shared.emit(DictationEvent::Speech { active });
                if is_capturing && !active {
                    // End of an utterance: no continuation can complete a
                    // pending dictionary phrase, so type everything immediately
                    // instead of waiting out the phrase-hold timer. This is
                    // what keeps the last word of an utterance from lagging.
                    if let Some(typed) = self.buffer.flush_all(
                        self.inserter.keyboard(),
                        &mut self.replacement,
                        &mut self.recorder,
                    )? {
                        self.shared.emit(DictationEvent::Typed { text: typed });
                    }
                    self.finish_utterance();
                    self.evaluate_profiles(true)?;
                }
            }
            "final" if is_capturing => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        if let Some(typed) = self.buffer.flush_all(
                            self.inserter.keyboard(),
                            &mut self.replacement,
                            &mut self.recorder,
                        )? {
                            self.shared.emit(DictationEvent::Typed { text: typed });
                        }
                        let replaced = self.replacement.replace(text);
                        let formatted = self
                            .buffer
                            .type_formatted(self.inserter.keyboard(), &replaced)?;
                        vlog!("[TYPING FINAL] {}", formatted);
                        self.recorder.record("final", text, &formatted)?;
                        self.shared.emit(DictationEvent::Typed {
                            text: formatted.clone(),
                        });
                        self.utterance.clear();
                        self.shared
                            .emit(DictationEvent::Utterance { text: formatted });
                    }
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

    fn emit_partial(&mut self) {
        let raw = self.utterance.join(" ");
        let text = super::typing::normalize_punctuation_spacing(&self.replacement.replace(&raw));
        self.shared.emit(DictationEvent::Partial { text });
    }

    fn finish_utterance(&mut self) {
        if self.utterance.is_empty() {
            return;
        }
        let raw = self.utterance.join(" ");
        let text = super::typing::normalize_punctuation_spacing(&self.replacement.replace(&raw));
        self.utterance.clear();
        self.shared.emit(DictationEvent::Utterance { text });
    }
}

fn encode_chunk(chunk: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(chunk.len() * 4);
    for sample in chunk {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::super::typing::LiveWordBuffer;
    use super::super::typing::tests::RecordingKeyboard;
    use super::super::typing::tests::silent_recorder;
    use super::super::typing::tests::test_reloader;
    use crate::virtual_keyboard::VirtualKeyboard;
    use std::sync::Arc;
    use std::time::Duration;

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
