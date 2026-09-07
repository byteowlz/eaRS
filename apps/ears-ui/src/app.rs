//! Application state shared by every window: a GPUI entity that mirrors the
//! headless dictation engine. Views observe it and re-render; user actions
//! turn into [`DictationCommand`]s. No dictation logic lives here.

use crate::config::UiConfig;
use ears::config::AppConfig;
use ears::config::DictationHotkeyMode;
use ears::dictation::BackendInfo;
use ears::dictation::ConnectionState;
use ears::dictation::DictationCommand;
use ears::dictation::DictationEvent;
use ears::dictation::DictationHandle;
use ears::dictation::DictationState;
use ears::dictation::InsertionMode;
use ears::profiles::ProfilesConfig;
use gpui::Context;
use gpui::Entity;
use gpui::Global;
use std::time::Duration;
use std::time::Instant;

/// How long the check mark / spinner phase lasts after an utterance ends.
pub const FINALIZE_MS: u64 = 650;
/// How long a transient error is shown in the island.
pub const ERROR_MS: u64 = 4000;

/// Visual phase of the island, derived from engine state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Dictation paused or engine not running.
    Dormant,
    /// Capture on, nobody talking yet.
    Armed,
    /// Speech detected / partial words arriving.
    Listening,
    /// Utterance boundary reached; words are being flushed.
    Finalizing,
    /// Something to tell the user (server unreachable, mic denied, ...).
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    ServerUnreachable,
    MicrophoneDenied,
    ModelLoading,
    Other,
}

#[derive(Debug, Clone)]
pub struct UiError {
    pub kind: ErrorKind,
    pub message: String,
    pub since: Instant,
}

pub struct AppModel {
    pub core: AppConfig,
    pub ui: UiConfig,
    pub engine: Option<DictationHandle>,
    pub engine_stopped: bool,
    pub state: DictationState,
    pub connection: ConnectionState,
    pub connection_message: Option<String>,
    pub backend: Option<BackendInfo>,
    pub partial: String,
    pub last_utterance: Option<String>,
    pub last_prompt: Option<String>,
    pub speech_active: bool,
    pub finalizing_since: Option<Instant>,
    pub error: Option<UiError>,
    pub profile: Option<String>,
    pub profile_pinned: bool,
    pub profiles: ProfilesConfig,
    pub insertion_mode: InsertionMode,
    pub language: Option<String>,
    pub hotkey_mode: DictationHotkeyMode,
    pub reduce_motion: bool,
    /// Last time the island had something to show; drives the linger timer.
    pub last_activity: Instant,
    /// Bumped whenever the settings window should refresh derived data.
    pub generation: u64,
}

/// Global pointer to the single [`AppModel`] entity.
pub struct AppHandle(pub Entity<AppModel>);
impl Global for AppHandle {}

impl AppModel {
    pub fn new(core: AppConfig, ui: UiConfig) -> Self {
        let hotkey_mode = ui.dictation.hotkey_mode.unwrap_or(core.hotkeys.mode);
        let profiles = ProfilesConfig::load().unwrap_or_default();
        Self {
            insertion_mode: ui.dictation.insertion_mode,
            hotkey_mode,
            core,
            ui,
            engine: None,
            engine_stopped: false,
            state: DictationState::Inactive,
            connection: ConnectionState::Connecting,
            connection_message: None,
            backend: None,
            partial: String::new(),
            last_utterance: None,
            last_prompt: None,
            speech_active: false,
            finalizing_since: None,
            error: None,
            profile: None,
            profile_pinned: false,
            profiles,
            language: None,
            reduce_motion: false,
            last_activity: Instant::now(),
            generation: 0,
        }
    }

    pub fn is_capturing(&self) -> bool {
        self.state == DictationState::Listening
    }

    pub fn engine_running(&self) -> bool {
        self.engine.is_some() && !self.engine_stopped
    }

    pub fn phase(&self) -> Phase {
        if let Some(err) = &self.error {
            let sticky = matches!(
                err.kind,
                ErrorKind::ServerUnreachable | ErrorKind::ModelLoading
            ) && self.is_capturing();
            if sticky || err.since.elapsed() < Duration::from_millis(ERROR_MS) {
                return Phase::Error;
            }
        }
        if self.is_capturing() && self.connection == ConnectionState::Disconnected {
            return Phase::Error;
        }
        if let Some(since) = self.finalizing_since {
            if since.elapsed() < Duration::from_millis(FINALIZE_MS) {
                return Phase::Finalizing;
            }
        }
        if !self.is_capturing() {
            return Phase::Dormant;
        }
        if self.speech_active || !self.partial.trim().is_empty() {
            Phase::Listening
        } else {
            Phase::Armed
        }
    }

    /// Text the island shows in the error phase.
    pub fn error_text(&self) -> String {
        if let Some(err) = &self.error {
            return match err.kind {
                ErrorKind::ServerUnreachable => "Server unreachable".to_string(),
                ErrorKind::MicrophoneDenied => "Microphone unavailable".to_string(),
                ErrorKind::ModelLoading => "Model loading…".to_string(),
                ErrorKind::Other => err.message.clone(),
            };
        }
        if self.connection == ConnectionState::Disconnected {
            return "Server unreachable".to_string();
        }
        String::new()
    }

    pub fn apply_event(&mut self, event: DictationEvent, cx: &mut Context<Self>) {
        match event {
            DictationEvent::State(state) => {
                let was = self.state;
                self.state = state;
                if state == DictationState::Listening {
                    self.partial.clear();
                    self.speech_active = false;
                    self.finalizing_since = None;
                    if matches!(self.error.as_ref().map(|e| e.kind), Some(ErrorKind::Other)) {
                        self.error = None;
                    }
                } else if was == DictationState::Listening {
                    self.speech_active = false;
                }
                self.touch();
            }
            DictationEvent::Connection { state, message } => {
                self.connection = state;
                self.connection_message = message;
                if state == ConnectionState::Connected {
                    if matches!(
                        self.error.as_ref().map(|e| e.kind),
                        Some(ErrorKind::ServerUnreachable | ErrorKind::ModelLoading)
                    ) {
                        self.error = None;
                    }
                } else if state == ConnectionState::Disconnected {
                    let msg = self.connection_message.clone().unwrap_or_default();
                    let kind = if msg.to_lowercase().contains("load") {
                        ErrorKind::ModelLoading
                    } else {
                        ErrorKind::ServerUnreachable
                    };
                    self.set_error(kind, msg);
                }
                self.touch();
            }
            DictationEvent::Backend(info) => {
                self.backend = Some(info);
            }
            DictationEvent::Partial { text } => {
                self.partial = text;
                self.finalizing_since = None;
                self.touch();
            }
            DictationEvent::Typed { .. } => {
                self.touch();
            }
            DictationEvent::Utterance { text } => {
                if !text.trim().is_empty() {
                    self.last_utterance = Some(text);
                    self.finalizing_since = Some(Instant::now());
                }
                self.partial.clear();
                self.touch();
            }
            DictationEvent::Speech { active } => {
                self.speech_active = active;
                if active {
                    self.finalizing_since = None;
                }
                self.touch();
            }
            DictationEvent::Cancelled => {
                self.partial.clear();
                self.speech_active = false;
                self.finalizing_since = None;
                self.touch();
            }
            DictationEvent::Profile { name, pinned } => {
                self.profile = name;
                self.profile_pinned = pinned;
            }
            DictationEvent::InsertionMode(mode) => {
                self.insertion_mode = mode;
            }
            DictationEvent::Language(lang) => {
                self.language = Some(lang);
            }
            DictationEvent::PromptReady { text } => {
                self.last_prompt = Some(text);
            }
            DictationEvent::Error { message } => {
                let lower = message.to_lowercase();
                let kind = if lower.contains("microphone") || lower.contains("audio input") {
                    ErrorKind::MicrophoneDenied
                } else if lower.contains("loading") {
                    ErrorKind::ModelLoading
                } else {
                    ErrorKind::Other
                };
                self.set_error(kind, message);
                self.touch();
            }
            DictationEvent::Stopped => {
                self.engine_stopped = true;
                self.state = DictationState::Inactive;
                self.speech_active = false;
                self.touch();
            }
        }
        cx.notify();
    }

    fn set_error(&mut self, kind: ErrorKind, message: String) {
        self.error = Some(UiError {
            kind,
            message,
            since: Instant::now(),
        });
    }

    fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Called from the UI timer loop so time-based phases (finalizing, error
    /// timeout) expire even without new events. Returns true if the phase
    /// might have changed.
    pub fn tick(&mut self, cx: &mut Context<Self>) -> bool {
        let mut changed = false;
        if let Some(since) = self.finalizing_since {
            if since.elapsed() >= Duration::from_millis(FINALIZE_MS) {
                self.finalizing_since = None;
                changed = true;
            }
        }
        if let Some(err) = &self.error {
            let sticky = matches!(
                err.kind,
                ErrorKind::ServerUnreachable | ErrorKind::ModelLoading
            ) && self.is_capturing();
            if !sticky && err.since.elapsed() >= Duration::from_millis(ERROR_MS) {
                self.error = None;
                changed = true;
            }
        }
        if changed {
            cx.notify();
        }
        changed
    }

    // ---- commands -------------------------------------------------------

    pub fn send(&self, command: DictationCommand) {
        if let Some(engine) = &self.engine {
            engine.send(command);
        }
    }

    #[allow(dead_code)] // API parity with DictationHandle; used by tests/future keybindings
    pub fn toggle_dictation(&self) {
        self.send(DictationCommand::Toggle);
    }

    pub fn set_capturing(&self, on: bool) {
        self.send(DictationCommand::SetCapturing(on));
    }

    #[allow(dead_code)]
    pub fn cancel(&self) {
        self.send(DictationCommand::Cancel);
    }

    pub fn set_insertion_mode(&mut self, mode: InsertionMode, cx: &mut Context<Self>) {
        self.insertion_mode = mode;
        self.ui.dictation.insertion_mode = mode;
        self.send(DictationCommand::SetInsertionMode(mode));
        self.save_ui();
        cx.notify();
    }

    pub fn set_hotkey_mode(&mut self, mode: DictationHotkeyMode, cx: &mut Context<Self>) {
        self.hotkey_mode = mode;
        self.ui.dictation.hotkey_mode = Some(mode);
        self.send(DictationCommand::SetHotkeyMode(mode));
        self.save_ui();
        cx.notify();
    }

    pub fn set_profile(&mut self, name: Option<String>, cx: &mut Context<Self>) {
        self.profile_pinned = name.is_some();
        self.profile = name.clone();
        self.send(DictationCommand::SetProfile(name));
        cx.notify();
    }

    pub fn set_language(&mut self, lang: String, cx: &mut Context<Self>) {
        self.language = Some(lang.clone());
        self.send(DictationCommand::SetLanguage(lang));
        cx.notify();
    }

    /// Ask the engine to re-read dictionaries and profiles, and refresh the
    /// in-memory profile list used by menus.
    pub fn reload(&mut self, cx: &mut Context<Self>) {
        self.profiles = ProfilesConfig::load().unwrap_or_default();
        self.send(DictationCommand::Reload);
        self.generation += 1;
        cx.notify();
    }

    pub fn save_ui(&self) {
        if let Err(err) = self.ui.save() {
            eprintln!("ears-ui: cannot save ui.toml: {err:#}");
        }
    }

    /// Mutate the UI config, persist it and notify observers.
    pub fn update_ui(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&mut UiConfig)) {
        f(&mut self.ui);
        self.save_ui();
        self.generation += 1;
        cx.notify();
    }

    pub fn profile_names(&self) -> Vec<String> {
        self.profiles.names_by_priority()
    }
}
