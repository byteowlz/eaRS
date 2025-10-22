use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, select, unbounded};
use ears::audio;
use ears::config::AppConfig;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use futures_util::{SinkExt, StreamExt};
use rdev::{EventType, listen};
use serde_json::Value;
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const PID_FILE_NAME: &str = "dictation.pid";

#[derive(Debug, Parser)]
#[command(name = "ears-dictation", about = "Dictation client for eaRS")]
struct Args {
    #[arg(long, help = "Set the transcription language (e.g., 'en', 'de', 'es', 'fr', 'ja')")]
    lang: Option<String>,
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = AppConfig::load().unwrap_or_default();
    let port = config.server.websocket_port;
    let url = format!("ws://127.0.0.1:{}", port);

    write_pid_file()?;

    let running = Arc::new(Mutex::new(true));
    let capturing = Arc::new(Mutex::new(true));

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

    if hotkey_config.enable_internal {
        eprintln!("Initializing hotkey listener for: {}", hotkey_config.toggle);
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
                            eprintln!("Audio capture {}", if *c { "started" } else { "stopped" });
                        }
                    },
                    _ => {}
                }
            }) {
                eprintln!("Hotkey listener error: {:?}", e);
            }
        });
    }

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
                let mut enigo = Enigo::new(&Settings::default())
                    .context("Failed to initialize keyboard controller")?;

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
                                if write.send(Message::Binary(bytes)).await.is_err() {
                                    break;
                                }
                            }
                            WriterCommand::Text(text) => {
                                if write.send(Message::Text(text)).await.is_err() {
                                    break;
                                }
                            }
                            WriterCommand::Stop => {
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
                                            handle_message(&json, &mut enigo, &capturing)?;
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
    eprintln!("ears-dictation stopped");
    Ok(())
}

fn handle_message(json: &Value, enigo: &mut Enigo, capturing: &Arc<Mutex<bool>>) -> Result<()> {
    let is_capturing = *capturing.lock().unwrap();

    if let Some(event_type) = json.get("type").and_then(|v| v.as_str()) {
        match event_type {
            "word" if is_capturing => {
                if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
                    if !word.is_empty() {
                        eprintln!("[TYPING WORD] {}", word);
                        enigo.text(word)?;
                        enigo.key(Key::Space, Direction::Click)?;
                    }
                }
            }
            "final" if is_capturing => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        eprintln!("[TYPING FINAL] {}", text);
                        enigo.text(text)?;
                        enigo.key(Key::Space, Direction::Click)?;
                    }
                }
            }
            _ => {}
        }
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
