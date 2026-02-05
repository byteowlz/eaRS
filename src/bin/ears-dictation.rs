use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, select, unbounded};
use ears::audio;
#[cfg(feature = "hooks")]
use ears::config::DictationHooksConfig;
use ears::config::{AppConfig, DictationNotificationConfig};
use ears::virtual_keyboard::{SpecialKey, VirtualKeyboard, create_virtual_keyboard};
use futures_util::{SinkExt, StreamExt};
use notifica::notify;
use rdev::{EventType, listen};
use serde_json::Value;
use std::fs;
#[cfg(feature = "hooks")]
use std::process::Command as ProcessCommand;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const PID_FILE_NAME: &str = "dictation.pid";

#[derive(Clone, Debug, clap::ValueEnum)]
enum EngineArg {
    Kyutai,
    #[cfg(feature = "parakeet")]
    Parakeet,
}

impl EngineArg {
    fn as_str(&self) -> &'static str {
        match self {
            EngineArg::Kyutai => "kyutai",
            #[cfg(feature = "parakeet")]
            EngineArg::Parakeet => "parakeet",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum DictationEvent {
    Started,
    Paused,
    Stopped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DictationState {
    Listening,
    Suspended,
    Inactive,
}

#[derive(Debug, Parser)]
#[command(name = "ears-dictation", about = "Dictation client for eaRS")]
struct Args {
    #[arg(
        long,
        help = "Set the transcription language (e.g., 'en', 'de', 'es', 'fr', 'ja')"
    )]
    lang: Option<String>,

    #[arg(
        long,
        value_enum,
        help = "Select transcription engine (kyutai|parakeet)"
    )]
    engine: Option<EngineArg>,

    #[arg(
        short,
        long,
        help = "Server alias from config (e.g., 'local', 'remote') or full WebSocket URL (ws://host:port)"
    )]
    server: Option<String>,

    #[arg(long, help = "List all configured servers and exit")]
    list_servers: bool,
}

fn get_pid_file() -> std::path::PathBuf {
    let state_dir = if let Ok(xdg_state) = std::env::var("XDG_STATE_HOME") {
        if !xdg_state.is_empty() {
            std::path::PathBuf::from(xdg_state)
        } else {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            std::path::PathBuf::from(home).join(".local/state")
        }
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(home).join(".local/state")
    };
    state_dir.join("ears").join(PID_FILE_NAME)
}

fn write_pid_file() -> Result<()> {
    let pid_file = get_pid_file();
    if let Some(parent) = pid_file.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&pid_file, std::process::id().to_string())?;
    Ok(())
}

fn remove_pid_file() {
    let pid_file = get_pid_file();
    let _ = fs::remove_file(pid_file);
}

/// Resolve the server URL from command-line argument or config.
/// Accepts either:
/// - A full WebSocket URL (ws:// or wss://)
/// - A server alias defined in config
/// - None to use the default server
fn resolve_server_url(server_arg: &Option<String>, config: &AppConfig) -> Result<String> {
    match server_arg {
        Some(server) if server.starts_with("ws://") || server.starts_with("wss://") => {
            // Direct WebSocket URL provided
            Ok(server.clone())
        }
        Some(alias) => {
            // Treat as server alias
            config.dictation.get_server_url(Some(alias)).ok_or_else(|| {
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
            })
        }
        None => {
            // Use default server
            config.dictation.get_server_url(None).ok_or_else(|| {
                anyhow::anyhow!(
                    "Default server '{}' not found in config",
                    config.dictation.default_server
                )
            })
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = AppConfig::load().unwrap_or_default();

    // Handle --list-servers flag
    if args.list_servers {
        println!("Configured dictation servers:");
        println!("  {:12} {:30} {}", "ALIAS", "URL", "DESCRIPTION");
        println!("  {:12} {:30} {}", "-----", "---", "-----------");
        let mut servers: Vec<_> = config.dictation.list_servers();
        servers.sort_by_key(|(alias, _)| *alias);
        for (alias, server) in servers {
            let is_default = alias == config.dictation.default_server;
            let marker = if is_default { "*" } else { " " };
            let desc = server.description.as_deref().unwrap_or("");
            println!("{} {:12} {:30} {}", marker, alias, server.ws_url(), desc);
        }
        println!("\n* = default server");
        return Ok(());
    }

    // Resolve server URL from --server argument or config
    let url = resolve_server_url(&args.server, &config)?;

    write_pid_file()?;

    let running = Arc::new(Mutex::new(true));
    let capturing = Arc::new(Mutex::new(true));
    let dictation_state = Arc::new(Mutex::new(DictationState::Inactive));

    let (stop_tx, stop_rx) = bounded::<()>(1);
    let running_clone = running.clone();

    ctrlc::set_handler(move || {
        *running_clone.lock().unwrap() = false;
        let _ = stop_tx.send(());
    })
    .context("Failed to set Ctrl+C handler")?;

    let hotkey_running = running.clone();
    let hotkey_capturing = capturing.clone();
    let hotkey_config = config.hotkeys.clone();
    let notification_config = config.dictation.notifications.clone();
    #[cfg(feature = "hooks")]
    let hook_config = config.dictation.hooks.clone();

    if hotkey_config.enable_internal {
        eprintln!("Initializing hotkey listener for: {}", hotkey_config.toggle);
        let dictation_state_thread = dictation_state.clone();
        #[cfg(feature = "hooks")]
        let hook_config_thread = hook_config.clone();
        let notification_config_thread = notification_config.clone();
        thread::spawn(move || {
            let toggle_combo = hotkey_config.toggle.to_lowercase();
            let (t_ctrl, t_shift, t_alt, t_key) = parse_combo(&toggle_combo);
            eprintln!(
                "Parsed combo - ctrl:{} shift:{} alt:{} key:{:?}",
                t_ctrl, t_shift, t_alt, t_key
            );

            if let Err(e) = listen(move |event| -> () {
                static mut CTRL: bool = false;
                static mut SHIFT: bool = false;
                static mut ALT: bool = false;

                match event.event_type {
                    EventType::KeyPress(rdev::Key::ControlLeft)
                    | EventType::KeyPress(rdev::Key::ControlRight) => unsafe {
                        CTRL = true;
                    },
                    EventType::KeyRelease(rdev::Key::ControlLeft)
                    | EventType::KeyRelease(rdev::Key::ControlRight) => unsafe {
                        CTRL = false;
                    },
                    EventType::KeyPress(rdev::Key::ShiftLeft)
                    | EventType::KeyPress(rdev::Key::ShiftRight) => unsafe {
                        SHIFT = true;
                    },
                    EventType::KeyRelease(rdev::Key::ShiftLeft)
                    | EventType::KeyRelease(rdev::Key::ShiftRight) => unsafe {
                        SHIFT = false;
                    },
                    EventType::KeyPress(rdev::Key::Alt) | EventType::KeyPress(rdev::Key::AltGr) => unsafe {
                        ALT = true;
                    },
                    EventType::KeyRelease(rdev::Key::Alt)
                    | EventType::KeyRelease(rdev::Key::AltGr) => unsafe {
                        ALT = false;
                    },
                    EventType::KeyRelease(k) => unsafe {
                        if !*hotkey_running.lock().unwrap() {
                            return;
                        }
                        if CTRL == t_ctrl && SHIFT == t_shift && ALT == t_alt && k == t_key {
                            let mut c = hotkey_capturing.lock().unwrap();
                            *c = !*c;
                            let is_active = *c;
                            eprintln!(
                                "Audio capture {}",
                                if is_active { "started" } else { "stopped" }
                            );
                            drop(c);
                            let event = if is_active {
                                DictationState::Listening
                            } else {
                                DictationState::Suspended
                            };
                            #[cfg(feature = "hooks")]
                            apply_state_change(
                                &dictation_state_thread,
                                event,
                                &notification_config_thread,
                                &hook_config_thread,
                            );
                            #[cfg(not(feature = "hooks"))]
                            apply_state_change(
                                &dictation_state_thread,
                                event,
                                &notification_config_thread,
                            );
                        }
                    },
                    _ => {}
                }
            }) {
                eprintln!("Hotkey listener error: {:?}", e);
            }
        });
    }

    #[cfg(feature = "hooks")]
    apply_state_change(
        &dictation_state,
        DictationState::Listening,
        &notification_config,
        &hook_config,
    );
    #[cfg(not(feature = "hooks"))]
    apply_state_change(
        &dictation_state,
        DictationState::Listening,
        &notification_config,
    );

    let (audio_tx, audio_rx) = unbounded();
    let device_index = None;

    thread::spawn(move || {
        if let Err(e) = audio::start_audio_capture(audio_tx, device_index) {
            eprintln!("Audio capture error: {}", e);
        }
    });

    eprintln!("ears-dictation started");
    eprintln!("Connecting to {}...", url);
    eprintln!("Hotkey: {} to toggle pause/resume", config.hotkeys.toggle);
    eprintln!("Press Ctrl+C to stop\n");

    loop {
        let is_running = *running.lock().unwrap();
        if !is_running {
            break;
        }

        match connect_async(&url).await {
            Ok((ws_stream, _)) => {
                eprintln!("Connected to transcription server");
                let (mut write, mut read) = ws_stream.split();
                let mut keyboard = create_virtual_keyboard().context(
                    "Failed to initialize virtual keyboard. \
                              On Linux/Wayland, ensure you are in the 'input' group.",
                )?;

                let (writer_tx, mut writer_rx) = mpsc::unbounded_channel::<WriterCommand>();

                if let Some(ref lang) = args.lang {
                    eprintln!("Setting language to: {}", lang);
                    let lang_cmd = serde_json::json!({
                        "type": "setlanguage",
                        "lang": lang
                    })
                    .to_string();
                    if let Err(e) = writer_tx.send(WriterCommand::Text(lang_cmd)) {
                        eprintln!("Failed to send language command: {}", e);
                    }
                }

                if let Some(ref engine) = args.engine {
                    eprintln!("Selecting engine: {}", engine.as_str());
                    let engine_cmd = serde_json::json!({
                        "type": "setengine",
                        "engine": engine.as_str(),
                    })
                    .to_string();
                    let _ = writer_tx.send(WriterCommand::Text(engine_cmd));
                }

                let audio_writer = writer_tx.clone();
                let audio_rx_clone = audio_rx.clone();
                let audio_capturing = capturing.clone();
                thread::spawn(move || {
                    while let Ok(chunk) = audio_rx_clone.recv() {
                        if *audio_capturing.lock().unwrap() {
                            if audio_writer
                                .send(WriterCommand::Audio(encode_chunk(&chunk)))
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                });

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
                            WriterCommand::Stop => {
                                // Send close frame to properly terminate the WebSocket
                                let _ = write.send(Message::Close(None)).await;
                                break;
                            }
                        }
                    }
                });

                loop {
                    select! {
                        recv(stop_rx) -> _ => {
                            break;
                        }
                        default => {
                            if let Some(message) = read.next().await {
                                match message {
                                    Ok(Message::Text(text)) => {
                                        eprintln!("[WS RECEIVED] {}", text);
                                        if let Ok(json) = serde_json::from_str::<Value>(&text) {
                                            handle_message(&json, &mut keyboard, &capturing)?;
                                        } else {
                                            eprintln!("[ERROR] Failed to parse JSON");
                                        }
                                    }
                                    Ok(Message::Binary(data)) => {
                                        eprintln!("[WS BINARY] {} bytes", data.len());
                                    }
                                    Ok(Message::Close(_)) => {
                                        eprintln!("WebSocket closed");
                                        break;
                                    }
                                    Err(e) => {
                                        eprintln!("WebSocket error: {}", e);
                                        break;
                                    }
                                    _ => {}
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }

                let _ = writer_tx.send(WriterCommand::Stop);
                let _ = writer_handle.await;

                let is_running = *running.lock().unwrap();
                if !is_running {
                    break;
                }

                eprintln!("Disconnected, reconnecting in 2s...");
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
            Err(e) => {
                eprintln!("Failed to connect: {} (retrying in 2s)", e);
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    }

    remove_pid_file();
    #[cfg(feature = "hooks")]
    apply_state_change(
        &dictation_state,
        DictationState::Inactive,
        &notification_config,
        &hook_config,
    );
    #[cfg(not(feature = "hooks"))]
    apply_state_change(
        &dictation_state,
        DictationState::Inactive,
        &notification_config,
    );
    eprintln!("ears-dictation stopped");
    Ok(())
}

fn handle_message(
    json: &Value,
    keyboard: &mut Box<dyn VirtualKeyboard>,
    capturing: &Arc<Mutex<bool>>,
) -> Result<()> {
    let is_capturing = *capturing.lock().unwrap();

    if let Some(event_type) = json.get("type").and_then(|v| v.as_str()) {
        match event_type {
            "word" if is_capturing => {
                if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
                    if !word.is_empty() {
                        eprintln!("[TYPING WORD] {}", word);
                        keyboard.type_text(word)?;
                        keyboard.press_key(SpecialKey::Space)?;
                    }
                }
            }
            "final" if is_capturing => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        eprintln!("[TYPING FINAL] {}", text);
                        keyboard.type_text(text)?;
                        keyboard.press_key(SpecialKey::Space)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(not(feature = "hooks"))]
fn apply_state_change(
    state: &Arc<Mutex<DictationState>>,
    new_state: DictationState,
    notifications: &DictationNotificationConfig,
) {
    let event = match new_state {
        DictationState::Listening => DictationEvent::Started,
        DictationState::Suspended => DictationEvent::Paused,
        DictationState::Inactive => DictationEvent::Stopped,
    };

    let mut guard = state.lock().unwrap();
    if *guard == new_state {
        return;
    }
    *guard = new_state;
    drop(guard);

    handle_toggle_side_effects(event, notifications);
}

#[cfg(feature = "hooks")]
fn apply_state_change(
    state: &Arc<Mutex<DictationState>>,
    new_state: DictationState,
    notifications: &DictationNotificationConfig,
    hooks: &DictationHooksConfig,
) {
    let event = match new_state {
        DictationState::Listening => DictationEvent::Started,
        DictationState::Suspended => DictationEvent::Paused,
        DictationState::Inactive => DictationEvent::Stopped,
    };

    let mut guard = state.lock().unwrap();
    if *guard == new_state {
        return;
    }
    *guard = new_state;
    drop(guard);

    handle_toggle_side_effects(event, notifications, hooks);
}

fn send_toggle_notification(event: DictationEvent, notifications: &DictationNotificationConfig) {
    if !notifications.enabled {
        return;
    }

    let message = match event {
        DictationEvent::Started => notifications.start_message.as_str(),
        DictationEvent::Paused => notifications.pause_message.as_str(),
        DictationEvent::Stopped => notifications.stop_message.as_str(),
    };

    if message.trim().is_empty() {
        return;
    }

    if let Err(err) = notify("eaRS Dictation", message) {
        eprintln!("Failed to send dictation notification: {}", err);
    }
}

#[cfg(not(feature = "hooks"))]
fn handle_toggle_side_effects(event: DictationEvent, notifications: &DictationNotificationConfig) {
    send_toggle_notification(event, notifications);
}

#[cfg(feature = "hooks")]
fn handle_toggle_side_effects(
    event: DictationEvent,
    notifications: &DictationNotificationConfig,
    hooks: &DictationHooksConfig,
) {
    send_toggle_notification(event, notifications);
    if let Err(err) = run_hook_command(event, hooks) {
        eprintln!("Failed to run dictation hook command: {}", err);
    }
}

#[cfg(feature = "hooks")]
fn run_hook_command(event: DictationEvent, hooks: &DictationHooksConfig) -> Result<()> {
    let command = match event {
        DictationEvent::Started => hooks.start_command.as_deref(),
        DictationEvent::Paused => hooks.pause_command.as_deref(),
        DictationEvent::Stopped => hooks.stop_command.as_deref(),
    };

    let command = match command {
        Some(cmd) if !cmd.trim().is_empty() => cmd.trim(),
        _ => return Ok(()),
    };

    #[cfg(target_os = "windows")]
    {
        ProcessCommand::new("cmd")
            .arg("/C")
            .arg(command)
            .spawn()
            .with_context(|| format!("failed to spawn hook command '{}'", command))?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        ProcessCommand::new("sh")
            .arg("-c")
            .arg(command)
            .spawn()
            .with_context(|| format!("failed to spawn hook command '{}'", command))?;
    }

    Ok(())
}

fn parse_combo(s: &str) -> (bool, bool, bool, rdev::Key) {
    let mut ctrl = false;
    let mut shift = false;
    let mut alt = false;
    let mut key = rdev::Key::Unknown(0);

    for part in s.split('+') {
        match part.trim() {
            "ctrl" | "control" => ctrl = true,
            "shift" => shift = true,
            "alt" => alt = true,
            k if k.len() == 1 => {
                if let Some(ch) = k.chars().next() {
                    key = match ch {
                        'a' => rdev::Key::KeyA,
                        'b' => rdev::Key::KeyB,
                        'c' => rdev::Key::KeyC,
                        'd' => rdev::Key::KeyD,
                        'e' => rdev::Key::KeyE,
                        'f' => rdev::Key::KeyF,
                        'g' => rdev::Key::KeyG,
                        'h' => rdev::Key::KeyH,
                        'i' => rdev::Key::KeyI,
                        'j' => rdev::Key::KeyJ,
                        'k' => rdev::Key::KeyK,
                        'l' => rdev::Key::KeyL,
                        'm' => rdev::Key::KeyM,
                        'n' => rdev::Key::KeyN,
                        'o' => rdev::Key::KeyO,
                        'p' => rdev::Key::KeyP,
                        'q' => rdev::Key::KeyQ,
                        'r' => rdev::Key::KeyR,
                        's' => rdev::Key::KeyS,
                        't' => rdev::Key::KeyT,
                        'u' => rdev::Key::KeyU,
                        'v' => rdev::Key::KeyV,
                        'w' => rdev::Key::KeyW,
                        'x' => rdev::Key::KeyX,
                        'y' => rdev::Key::KeyY,
                        'z' => rdev::Key::KeyZ,
                        _ => rdev::Key::Unknown(0),
                    }
                }
            }
            _ => {}
        }
    }
    (ctrl, shift, alt, key)
}

enum WriterCommand {
    Audio(Vec<u8>),
    Text(String),
    Stop,
}

fn encode_chunk(chunk: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(chunk.len() * 4);
    for sample in chunk {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}
