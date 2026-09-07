//! Dictation engine.
//!
//! This module hosts the logic that used to live inside the `ears-dictation`
//! binary: audio capture, the WebSocket client, live word buffering, dictionary
//! replacement, keyboard insertion, hotkeys and transcript history. It is
//! exposed as a library so several front ends can drive it:
//!
//! - `ears-dictation` (and therefore `ears dictation start`) runs it headless.
//! - `ears-ui` embeds it in-process and renders its [`DictationEvent`] stream.
//!
//! The engine is observable through a [`DictationEvent`] channel and controlled
//! through [`DictationHandle`]. All behaviour lives here; front ends only render
//! state and forward user intent.

mod engine;
pub mod hotkeys;
pub mod typing;

use crate::config::AppConfig;
use crate::config::DictationNotificationConfig;
use crate::server;
use anyhow::Context;
use anyhow::Result;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use notifica::notify;
use serde::Deserialize;
use serde::Serialize;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

const PID_FILE_NAME: &str = "dictation.pid";

/// Where recognised text goes.
///
/// The enum is deliberately open-ended: new variants (for example an agent
/// hand-off) can be added without touching the front ends, which only need to
/// know the name of the mode the user picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsertionMode {
    /// Type into the focused application through the virtual keyboard.
    InsertAtCursor,
    /// Accumulate the current session's text on the clipboard instead of typing.
    Clipboard,
    /// Like [`InsertionMode::Clipboard`], but additionally emits
    /// [`DictationEvent::PromptReady`] when capture stops so an agent front end
    /// can pick the text up as a prompt.
    SendAsPrompt,
}

impl InsertionMode {
    pub const ALL: [InsertionMode; 3] = [
        InsertionMode::InsertAtCursor,
        InsertionMode::Clipboard,
        InsertionMode::SendAsPrompt,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            InsertionMode::InsertAtCursor => "insert_at_cursor",
            InsertionMode::Clipboard => "clipboard",
            InsertionMode::SendAsPrompt => "send_as_prompt",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            InsertionMode::InsertAtCursor => "Insert at cursor",
            InsertionMode::Clipboard => "Copy to clipboard",
            InsertionMode::SendAsPrompt => "Send as prompt",
        }
    }
}

impl Default for InsertionMode {
    fn default() -> Self {
        InsertionMode::InsertAtCursor
    }
}

impl fmt::Display for InsertionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for InsertionMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().replace('-', "_").as_str() {
            "insert" | "insert_at_cursor" | "cursor" | "type" => Ok(InsertionMode::InsertAtCursor),
            "clipboard" | "copy" | "copy_to_clipboard" => Ok(InsertionMode::Clipboard),
            "prompt" | "send_as_prompt" | "agent" => Ok(InsertionMode::SendAsPrompt),
            other => anyhow::bail!(
                "unknown insertion mode '{other}' (expected insert_at_cursor|clipboard|send_as_prompt)"
            ),
        }
    }
}

/// Capture state of the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DictationState {
    /// Audio is streamed to the server and words are inserted.
    Listening,
    /// Engine is running but audio is not forwarded.
    Suspended,
    /// Engine is not running.
    Inactive,
}

/// Connection state towards the transcription server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionState {
    Connecting,
    Connected,
    Disconnected,
}

/// Identity of the backend the engine talks to.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BackendInfo {
    /// WebSocket URL.
    pub url: String,
    /// Server alias from `config.toml`, when the URL came from one.
    pub alias: Option<String>,
    /// Last measured WebSocket ping round trip.
    pub latency_ms: Option<u64>,
}

/// Everything a front end can observe.
#[derive(Debug, Clone, PartialEq)]
pub enum DictationEvent {
    /// Capture state changed.
    State(DictationState),
    /// Connection state changed. `message` carries the error for failures.
    Connection {
        state: ConnectionState,
        message: Option<String>,
    },
    /// Backend identity / latency update.
    Backend(BackendInfo),
    /// Words recognised since the last utterance boundary, after dictionary
    /// replacement. This is the "live partial" line the overlay shows.
    Partial { text: String },
    /// Text that was actually inserted (or copied) just now.
    Typed { text: String },
    /// An utterance ended (VAD boundary or final transcript).
    Utterance { text: String },
    /// Server-side voice activity: `true` while the speaker is talking.
    Speech { active: bool },
    /// The pending words were discarded (Escape or explicit cancel).
    Cancelled,
    /// Active context profile changed (`None` = default settings).
    Profile { name: Option<String>, pinned: bool },
    /// Insertion mode changed.
    InsertionMode(InsertionMode),
    /// Language changed.
    Language(String),
    /// In [`InsertionMode::SendAsPrompt`], capture stopped and this is the
    /// accumulated text.
    PromptReady { text: String },
    /// Non-fatal error worth showing to the user.
    Error { message: String },
    /// The engine loop exited.
    Stopped,
}

/// Commands a front end can send.
#[derive(Debug, Clone, PartialEq)]
pub enum DictationCommand {
    /// Start (`true`) or pause (`false`) capture.
    SetCapturing(bool),
    /// Flip capture.
    Toggle,
    /// Discard pending words and pause.
    Cancel,
    SetInsertionMode(InsertionMode),
    /// Pin a profile by name (`None` returns to automatic matching).
    SetProfile(Option<String>),
    /// Cycle automatic -> first profile -> ... -> automatic.
    CycleProfile,
    SetLanguage(String),
    /// Switch how the toggle hotkey behaves (applied immediately).
    SetHotkeyMode(crate::config::DictationHotkeyMode),
    /// Switch whether Escape cancels pending words (applied immediately).
    SetEscapeCancels(bool),
    /// Re-read dictionaries and profiles from disk immediately.
    Reload,
    /// Stop the engine.
    Shutdown,
}

/// Options that used to be command line flags of `ears-dictation`.
#[derive(Debug, Clone)]
pub struct DictationOptions {
    /// Server alias or `ws://` URL. `None` uses `dictation.default_server`.
    pub server: Option<String>,
    /// Transcription language sent on connect.
    pub lang: Option<String>,
    /// Engine name sent on connect (`kyutai`, `parakeet`, ...).
    pub engine: Option<String>,
    /// Audio codec override (`pcm` / `opus`).
    pub codec: Option<String>,
    /// Initial insertion mode.
    pub insertion_mode: InsertionMode,
    /// Override `hotkeys.escape_cancels` from the config.
    pub escape_cancels: Option<bool>,
    /// Override `hotkeys.enable_internal`.
    pub hotkeys: Option<bool>,
    /// Start a local `ears server run` when the target is localhost and the
    /// port is closed.
    pub auto_start_server: bool,
    /// Write `$XDG_STATE_HOME/ears/dictation.pid` so `ears dictation status`
    /// and `ears server stop` can see this session.
    pub write_pid_file: bool,
    /// Evaluate context profiles against the frontmost application.
    pub profiles: bool,
    /// Print diagnostic output to stderr.
    pub verbose: bool,
}

impl Default for DictationOptions {
    fn default() -> Self {
        Self {
            server: None,
            lang: None,
            engine: None,
            codec: None,
            insertion_mode: InsertionMode::InsertAtCursor,
            escape_cancels: None,
            hotkeys: None,
            auto_start_server: true,
            write_pid_file: true,
            profiles: true,
            verbose: false,
        }
    }
}

static VERBOSE: AtomicBool = AtomicBool::new(false);

pub(crate) fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

macro_rules! vlog {
    ($($arg:tt)*) => {
        if $crate::dictation::verbose() {
            eprintln!($($arg)*);
        }
    };
}
pub(crate) use vlog;

/// State shared between the engine loop, the hotkey listener and handles.
pub(crate) struct Shared {
    capturing: Mutex<bool>,
    state: Mutex<DictationState>,
    hotkey_mode: Mutex<crate::config::DictationHotkeyMode>,
    escape_cancels: Mutex<bool>,
    running: AtomicBool,
    notifications: DictationNotificationConfig,
    #[cfg(feature = "hooks")]
    hooks: crate::config::DictationHooksConfig,
    events: Sender<DictationEvent>,
    commands: Sender<DictationCommand>,
}

impl Shared {
    fn new(
        config: &AppConfig,
        escape_cancels: bool,
        events: Sender<DictationEvent>,
        commands: Sender<DictationCommand>,
        capturing: bool,
    ) -> Self {
        Self {
            capturing: Mutex::new(capturing),
            state: Mutex::new(DictationState::Inactive),
            hotkey_mode: Mutex::new(config.hotkeys.mode),
            running: AtomicBool::new(true),
            escape_cancels: Mutex::new(escape_cancels),
            notifications: config.dictation.notifications.clone(),
            #[cfg(feature = "hooks")]
            hooks: config.dictation.hooks.clone(),
            events,
            commands,
        }
    }

    pub(crate) fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub(crate) fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub(crate) fn is_capturing(&self) -> bool {
        *self.capturing.lock().unwrap()
    }

    pub(crate) fn state(&self) -> DictationState {
        *self.state.lock().unwrap()
    }

    pub(crate) fn hotkey_mode(&self) -> crate::config::DictationHotkeyMode {
        *self.hotkey_mode.lock().unwrap()
    }

    pub(crate) fn set_hotkey_mode(&self, mode: crate::config::DictationHotkeyMode) {
        *self.hotkey_mode.lock().unwrap() = mode;
    }

    pub(crate) fn escape_cancels(&self) -> bool {
        *self.escape_cancels.lock().unwrap()
    }

    pub(crate) fn set_escape_cancels(&self, on: bool) {
        *self.escape_cancels.lock().unwrap() = on;
    }

    pub(crate) fn emit(&self, event: DictationEvent) {
        let _ = self.events.send(event);
    }

    pub(crate) fn command(&self, command: DictationCommand) {
        let _ = self.commands.send(command);
    }

    /// Set capture on/off. Returns whether the value changed.
    pub(crate) fn set_capturing(&self, on: bool) -> bool {
        let mut guard = self.capturing.lock().unwrap();
        if *guard == on {
            return false;
        }
        *guard = on;
        drop(guard);
        self.set_state(if on {
            DictationState::Listening
        } else {
            DictationState::Suspended
        });
        true
    }

    /// Flip capture. Returns the new value.
    pub(crate) fn toggle(&self) -> bool {
        let mut guard = self.capturing.lock().unwrap();
        *guard = !*guard;
        let on = *guard;
        drop(guard);
        self.set_state(if on {
            DictationState::Listening
        } else {
            DictationState::Suspended
        });
        on
    }

    /// Apply a state transition with its side effects (notification, hooks,
    /// event). No-op when the state is unchanged.
    pub(crate) fn set_state(&self, new_state: DictationState) {
        let mut guard = self.state.lock().unwrap();
        if *guard == new_state {
            return;
        }
        *guard = new_state;
        drop(guard);
        self.emit(DictationEvent::State(new_state));
        send_toggle_notification(new_state, &self.notifications);
        #[cfg(feature = "hooks")]
        if let Err(err) = run_hook_command(new_state, &self.hooks) {
            eprintln!("Failed to run dictation hook command: {}", err);
        }
    }
}

fn send_toggle_notification(state: DictationState, notifications: &DictationNotificationConfig) {
    if !notifications.enabled {
        return;
    }
    let message = match state {
        DictationState::Listening => notifications.start_message.as_str(),
        DictationState::Suspended => notifications.pause_message.as_str(),
        DictationState::Inactive => notifications.stop_message.as_str(),
    };
    if message.trim().is_empty() {
        return;
    }
    if let Err(err) = notify("eaRS Dictation", message) {
        eprintln!("Failed to send dictation notification: {}", err);
    }
}

#[cfg(feature = "hooks")]
fn run_hook_command(
    state: DictationState,
    hooks: &crate::config::DictationHooksConfig,
) -> Result<()> {
    let command = match state {
        DictationState::Listening => hooks.start_command.as_deref(),
        DictationState::Suspended => hooks.pause_command.as_deref(),
        DictationState::Inactive => hooks.stop_command.as_deref(),
    };
    let command = match command {
        Some(cmd) if !cmd.trim().is_empty() => cmd.trim(),
        _ => return Ok(()),
    };

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .arg("/C")
            .arg(command)
            .spawn()
            .with_context(|| format!("failed to spawn hook command '{}'", command))?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .spawn()
            .with_context(|| format!("failed to spawn hook command '{}'", command))?;
    }

    Ok(())
}

/// Control handle for a running engine. Cheap to clone.
#[derive(Clone)]
pub struct DictationHandle {
    shared: Arc<Shared>,
}

impl DictationHandle {
    pub fn send(&self, command: DictationCommand) {
        match command {
            DictationCommand::SetCapturing(on) => {
                self.shared.set_capturing(on);
            }
            DictationCommand::Toggle => {
                self.shared.toggle();
            }
            DictationCommand::SetHotkeyMode(mode) => {
                self.shared.set_hotkey_mode(mode);
            }
            other => self.shared.command(other),
        }
    }

    pub fn set_capturing(&self, on: bool) {
        self.send(DictationCommand::SetCapturing(on));
    }

    pub fn toggle(&self) {
        self.send(DictationCommand::Toggle);
    }

    pub fn set_escape_cancels(&self, on: bool) {
        self.send(DictationCommand::SetEscapeCancels(on));
    }

    pub fn cancel(&self) {
        self.send(DictationCommand::Cancel);
    }

    pub fn shutdown(&self) {
        self.send(DictationCommand::Shutdown);
    }

    pub fn is_capturing(&self) -> bool {
        self.shared.is_capturing()
    }

    pub fn state(&self) -> DictationState {
        self.shared.state()
    }

    pub fn hotkey_mode(&self) -> crate::config::DictationHotkeyMode {
        self.shared.hotkey_mode()
    }

    pub fn set_hotkey_mode(&self, mode: crate::config::DictationHotkeyMode) {
        self.shared.set_hotkey_mode(mode);
    }

    pub fn is_running(&self) -> bool {
        self.shared.is_running()
    }
}

/// A running engine: its handle, the event stream and the thread join handle.
pub struct DictationSession {
    pub handle: DictationHandle,
    pub events: Receiver<DictationEvent>,
    pub thread: thread::JoinHandle<Result<()>>,
}

/// Run the engine on a dedicated thread with its own Tokio runtime.
///
/// Fails early when the server URL cannot be resolved so front ends can show
/// a configuration error instead of an endless reconnect loop.
pub fn spawn(config: AppConfig, options: DictationOptions) -> Result<DictationSession> {
    let url = resolve_server_url(options.server.as_deref(), &config)?;
    let (events_tx, events_rx) = crossbeam_channel::unbounded();
    let (commands_tx, commands_rx) = crossbeam_channel::unbounded();
    let starts_listening = !config.dictation.start_paused;
    let escape_cancels = options
        .escape_cancels
        .unwrap_or(config.hotkeys.escape_cancels);
    let shared = Arc::new(Shared::new(
        &config,
        escape_cancels,
        events_tx,
        commands_tx,
        starts_listening,
    ));
    let handle = DictationHandle {
        shared: shared.clone(),
    };
    let thread = thread::Builder::new()
        .name("ears-dictation".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("failed to create tokio runtime for dictation")?;
            runtime.block_on(engine::run(config, options, url, shared, commands_rx))
        })
        .context("failed to spawn dictation thread")?;
    Ok(DictationSession {
        handle,
        events: events_rx,
        thread,
    })
}

/// Run the engine on the current Tokio runtime until [`DictationCommand::Shutdown`]
/// arrives or `Ctrl+C` is pressed. Used by the `ears-dictation` binary.
///
/// `on_event` receives every event; pass a no-op closure to ignore them.
pub async fn run(
    config: AppConfig,
    options: DictationOptions,
    on_event: impl FnMut(&DictationEvent) + Send + 'static,
) -> Result<()> {
    let url = resolve_server_url(options.server.as_deref(), &config)?;
    let (events_tx, events_rx) = crossbeam_channel::unbounded::<DictationEvent>();
    let (commands_tx, commands_rx) = crossbeam_channel::unbounded();
    let starts_listening = !config.dictation.start_paused;
    let escape_cancels = options
        .escape_cancels
        .unwrap_or(config.hotkeys.escape_cancels);
    let shared = Arc::new(Shared::new(
        &config,
        escape_cancels,
        events_tx,
        commands_tx,
        starts_listening,
    ));

    let ctrl_shared = shared.clone();
    ctrlc::set_handler(move || {
        ctrl_shared.stop();
        ctrl_shared.command(DictationCommand::Shutdown);
    })
    .context("Failed to set Ctrl+C handler")?;

    let mut on_event = on_event;
    thread::spawn(move || {
        while let Ok(event) = events_rx.recv() {
            on_event(&event);
        }
    });

    engine::run(config, options, url, shared, commands_rx).await
}

// ---------------------------------------------------------------------------
// PID file
// ---------------------------------------------------------------------------

/// `$XDG_STATE_HOME/ears/dictation.pid`.
pub fn pid_file_path() -> PathBuf {
    let state_dir = match std::env::var("XDG_STATE_HOME") {
        Ok(xdg_state) if !xdg_state.is_empty() => PathBuf::from(xdg_state),
        _ => {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".local/state")
        }
    };
    state_dir.join("ears").join(PID_FILE_NAME)
}

pub(crate) fn write_pid_file(server_url: &str) -> Result<()> {
    let pid_file = pid_file_path();
    if let Some(parent) = pid_file.parent() {
        fs::create_dir_all(parent)?;
    }
    // Store PID and server URL so `ears server stop` can determine whether
    // this dictation session depends on the local server.
    let contents = format!("{}\n{}", std::process::id(), server_url);
    fs::write(&pid_file, contents)?;
    Ok(())
}

pub fn remove_pid_file() {
    let _ = fs::remove_file(pid_file_path());
}

/// Read `(pid, server_url)` from the PID file, if present.
pub fn read_pid_info() -> Result<(Option<i32>, Option<String>)> {
    let pid_file = pid_file_path();
    if !pid_file.exists() {
        return Ok((None, None));
    }
    let contents = fs::read_to_string(&pid_file)?;
    let mut lines = contents.lines();
    let pid = lines.next().and_then(|s| s.trim().parse::<i32>().ok());
    let url = lines
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    Ok((pid, url))
}

/// PID of a live dictation process other than ourselves, if any.
pub fn running_instance() -> Option<i32> {
    let (pid, _) = read_pid_info().ok()?;
    let pid = pid?;
    if pid == std::process::id() as i32 {
        return None;
    }
    server::is_process_alive(pid).then_some(pid)
}

/// Ask a running dictation process to stop (SIGINT, then SIGTERM) and wait
/// for it to exit. Returns `Ok(false)` when nothing was running.
pub fn stop_running_instance() -> Result<bool> {
    let Some(pid) = running_instance() else {
        return Ok(false);
    };

    #[cfg(unix)]
    {
        unsafe {
            if libc::kill(pid, libc::SIGINT) != 0 {
                return Err(std::io::Error::last_os_error())
                    .context("failed to send SIGINT to ears-dictation");
            }
        }
        for _ in 0..60 {
            if !server::is_process_alive(pid) {
                return Ok(true);
            }
            thread::sleep(Duration::from_millis(100));
        }
        unsafe {
            let _ = libc::kill(pid, libc::SIGTERM);
        }
        for _ in 0..30 {
            if !server::is_process_alive(pid) {
                return Ok(true);
            }
            thread::sleep(Duration::from_millis(100));
        }
        anyhow::bail!("ears-dictation (pid {pid}) did not stop");
    }

    #[cfg(not(unix))]
    {
        anyhow::bail!("stopping dictation is currently supported only on unix platforms");
    }
}

// ---------------------------------------------------------------------------
// Server resolution / auto start
// ---------------------------------------------------------------------------

/// Resolve the server URL from an argument or the config.
///
/// Accepts a full `ws://` / `wss://` URL, a server alias from `config.toml`,
/// or `None` for the default server.
pub fn resolve_server_url(server_arg: Option<&str>, config: &AppConfig) -> Result<String> {
    match server_arg {
        Some(server) if server.starts_with("ws://") || server.starts_with("wss://") => {
            Ok(server.to_string())
        }
        Some(alias) => config.dictation.get_server_url(Some(alias)).ok_or_else(|| {
            let available: Vec<_> = config
                .dictation
                .servers
                .keys()
                .map(|s| s.as_str())
                .collect();
            anyhow::anyhow!(
                "Unknown server alias '{}'. Available: {}",
                alias,
                available.join(", ")
            )
        }),
        None => config.dictation.get_server_url(None).ok_or_else(|| {
            anyhow::anyhow!(
                "Default server '{}' not found in config",
                config.dictation.default_server
            )
        }),
    }
}

/// Alias whose URL equals `url`, if any.
pub fn alias_for_url(url: &str, config: &AppConfig) -> Option<String> {
    config
        .dictation
        .list_servers()
        .into_iter()
        .find(|(_, server)| server.ws_url() == url)
        .map(|(alias, _)| alias.to_string())
}

/// Check if a TCP port is accepting connections on localhost.
pub fn is_server_port_open(port: u16) -> bool {
    std::net::TcpStream::connect(("127.0.0.1", port)).is_ok()
}

/// Extract the port from a WebSocket URL. Returns `None` for non-localhost URLs.
pub fn extract_local_port(url: &str) -> Option<u16> {
    let url_lower = url.to_lowercase();
    if !url_lower.contains("localhost") && !url_lower.contains("127.0.0.1") {
        return None;
    }
    let host_port = url.split("://").nth(1)?;
    let host_port = host_port.split('/').next().unwrap_or(host_port);
    host_port.split(':').nth(1)?.parse::<u16>().ok()
}

/// Attempt to spawn the ears server process and wait for it to become ready.
/// Returns `Ok(true)` if the server is ready, `Ok(false)` if started but not
/// yet ready, and `Err` if the server binary could not be found or spawned.
pub fn ensure_server_running(config: &AppConfig) -> Result<bool> {
    if let Ok(Some(pid)) = server::read_pid_file() {
        if server::is_process_alive(pid) {
            let ready =
                wait_for_server_ready(config.server.websocket_port, 30, Duration::from_millis(500));
            return Ok(ready);
        }
        let _ = server::remove_pid_file();
    }

    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().context("failed to get exe directory")?;
    let ears_bin = find_ears_binary(exe_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "ears binary not found next to {}. Please start the server manually with: ears server start",
            exe.display()
        )
    })?;

    eprintln!("ears server is not running, starting it automatically...");

    let mut cmd = std::process::Command::new(&ears_bin);
    cmd.arg("server").arg("run");
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());

    let child = cmd.spawn().context("failed to spawn ears server process")?;
    let pid = child.id();

    eprint!("ears server starting (pid {})...", pid);
    std::io::stderr().flush().ok();

    let ready = wait_for_server_ready(config.server.websocket_port, 60, Duration::from_millis(500));

    if ready {
        eprintln!("\rears server started (pid {}) and ready        ", pid);
    } else {
        eprintln!(
            "\rears server started (pid {}) but not yet ready        ",
            pid
        );
        eprintln!("server may still be loading the model - dictation will connect once ready");
    }

    Ok(ready)
}

/// Locate the `ears` CLI: next to the current executable first, then on PATH.
fn find_ears_binary(exe_dir: &std::path::Path) -> Option<PathBuf> {
    let name = if cfg!(windows) { "ears.exe" } else { "ears" };
    let sibling = exe_dir.join(name);
    if sibling.exists() {
        return Some(sibling);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.exists())
}

/// Poll a local TCP port until it accepts connections or the attempts run out.
fn wait_for_server_ready(port: u16, max_attempts: usize, poll_interval: Duration) -> bool {
    for _ in 0..max_attempts {
        if is_server_port_open(port) {
            return true;
        }
        thread::sleep(poll_interval);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_mode_parses_aliases() {
        assert_eq!(
            "insert".parse::<InsertionMode>().unwrap(),
            InsertionMode::InsertAtCursor
        );
        assert_eq!(
            "copy-to-clipboard".parse::<InsertionMode>().unwrap(),
            InsertionMode::Clipboard
        );
        assert_eq!(
            "send_as_prompt".parse::<InsertionMode>().unwrap(),
            InsertionMode::SendAsPrompt
        );
        assert!("nope".parse::<InsertionMode>().is_err());
    }

    #[test]
    fn extracts_local_ports_only() {
        assert_eq!(extract_local_port("ws://localhost:8080"), Some(8080));
        assert_eq!(extract_local_port("ws://127.0.0.1:9000/ws"), Some(9000));
        assert_eq!(extract_local_port("ws://example.com:8080"), None);
    }
}
