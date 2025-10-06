use anyhow::Result;
use clap::Parser;
use ears::config::AppConfig;
use futures_util::{SinkExt, StreamExt};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[derive(Debug, Parser)]
#[command(name = "ears-daemon")]
#[command(about = "Global hotkey controller for eaRS WebSocket server")] 
struct Args {
    /// Hotkey: ctrl+shift+v toggles pause/resume
    #[arg(long, default_value = "true")]
    enable_toggle: bool,

    /// Hotkey: ctrl+shift+l cycles language among configured list
    #[arg(long, default_value = "true")]
    enable_lang_cycle: bool,

    /// WebSocket host
    #[arg(long, default_value = "localhost")]
    host: String,

    /// WebSocket port (default from config)
    #[arg(long)]
    port: Option<u16>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = AppConfig::load().unwrap_or_default();
    let port = args.port.unwrap_or(config.server.websocket_port);
    let url = format!("ws://{}:{}/", args.host, port);
    let hk_cfg = config.hotkeys.clone();

    let (evt_tx, mut evt_rx) = mpsc::unbounded_channel::<Event>();

    // Track language cycle
    let langs = Arc::new(Mutex::new(vec![
        String::from("en"),
        String::from("de"),
        String::from("fr"),
        String::from("es"),
        String::from("ja"),
    ]));
    let lang_index = Arc::new(Mutex::new(0usize));

    // Spawn global keyboard listener
    if args.enable_toggle || args.enable_lang_cycle {
        let evt_tx_cl = evt_tx.clone();
        let enable_toggle = args.enable_toggle;
        let enable_lang_cycle = args.enable_lang_cycle;
        let toggle_combo = hk_cfg.toggle.to_lowercase();
        let lang_combo = hk_cfg.language_cycle.to_lowercase();
        std::thread::spawn(move || {
            if let Err(e) = rdev::listen(move |event| {
                use rdev::{EventType, Key};
                // Track modifier state
                static mut CTRL: bool = false;
                static mut SHIFT: bool = false;
                static mut ALT: bool = false;

                fn parse_combo(s: &str) -> (bool, bool, bool, Key) {
                    let mut c = false; let mut sft = false; let mut a = false; let mut key = Key::Unknown(0);
                    for part in s.split('+') {
                        match part.trim() {
                            "ctrl" | "control" => c = true,
                            "shift" => sft = true,
                            "alt" => a = true,
                            k if k.len()==1 => {
                                key = match k.chars().next().unwrap() {
                                    'a'..='z' => {
                                        let ch = k.chars().next().unwrap();
                                        match ch { 'a'=>Key::KeyA,'b'=>Key::KeyB,'c'=>Key::KeyC,'d'=>Key::KeyD,'e'=>Key::KeyE,'f'=>Key::KeyF,'g'=>Key::KeyG,'h'=>Key::KeyH,'i'=>Key::KeyI,'j'=>Key::KeyJ,'k'=>Key::KeyK,'l'=>Key::KeyL,'m'=>Key::KeyM,'n'=>Key::KeyN,'o'=>Key::KeyO,'p'=>Key::KeyP,'q'=>Key::KeyQ,'r'=>Key::KeyR,'s'=>Key::KeyS,'t'=>Key::KeyT,'u'=>Key::KeyU,'v'=>Key::KeyV,'w'=>Key::KeyW,'x'=>Key::KeyX,'y'=>Key::KeyY,'z'=>Key::KeyZ,_=>Key::Unknown(0)}
                                    }
                                    _ => Key::Unknown(0)
                                }
                            }
                            _ => {}
                        }
                    }
                    (c,sft,a,key)
                }

                let (t_ctrl,t_shift,t_alt,t_key) = parse_combo(&toggle_combo);
                let (l_ctrl,l_shift,l_alt,l_key) = parse_combo(&lang_combo);

                match event.event_type {
                    EventType::KeyPress(Key::ControlLeft) | EventType::KeyPress(Key::ControlRight) => unsafe { CTRL = true; },
                    EventType::KeyRelease(Key::ControlLeft) | EventType::KeyRelease(Key::ControlRight) => unsafe { CTRL = false; },
                    EventType::KeyPress(Key::ShiftLeft) | EventType::KeyPress(Key::ShiftRight) => unsafe { SHIFT = true; },
                    EventType::KeyRelease(Key::ShiftLeft) | EventType::KeyRelease(Key::ShiftRight) => unsafe { SHIFT = false; },
                    EventType::KeyPress(Key::Alt) | EventType::KeyPress(Key::AltGr) => unsafe { ALT = true; },
                    EventType::KeyRelease(Key::Alt) | EventType::KeyRelease(Key::AltGr) => unsafe { ALT = false; },
                    EventType::KeyRelease(k) if enable_toggle => unsafe {
                        if CTRL==t_ctrl && SHIFT==t_shift && ALT==t_alt && k == t_key { let _ = evt_tx_cl.send(Event::Toggle); }
                    },
                    EventType::KeyRelease(k) if enable_lang_cycle => unsafe {
                        if CTRL==l_ctrl && SHIFT==l_shift && ALT==l_alt && k == l_key { let _ = evt_tx_cl.send(Event::CycleLang); }
                    },
                    _ => {}
                }
            }) { eprintln!("Global hotkey listener error: {:?}", e); }
        });
    }

    loop {
        match connect_async(&url).await {
            Ok((ws_stream, _)) => {
                eprintln!("ears-daemon connected to {}", url);
                let (mut write, mut read) = ws_stream.split();

                // Immediately request status
                let _ = write.send(Message::Text(json!({"type":"get_status"}).to_string())).await;

                let mut paused = true;
                let langs = langs.clone();
                let lang_index = lang_index.clone();

                loop {
                    tokio::select! {
                        Some(evt) = evt_rx.recv() => {
                            match evt {
                                Event::Toggle => {
                                    let cmd = if paused { "resume" } else { "pause" };
                                    let _ = write.send(Message::Text(json!({"type": cmd}).to_string())).await;
                                    paused = !paused;
                                }
                                Event::CycleLang => {
                                    let list = langs.lock().unwrap();
                                    let mut idx = lang_index.lock().unwrap();
                                    *idx = (*idx + 1) % list.len();
                                    let lang = &list[*idx];
                                    let _ = write.send(Message::Text(json!({"type":"set_language","lang":lang}).to_string())).await;
                                }
                            }
                        }
                        msg = read.next() => {
                            match msg {
                                Some(Ok(Message::Text(text))) => {
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                                        if let Some(t) = v.get("type").and_then(|s| s.as_str()) {
                                            match t {
                                                "status" => { paused = v.get("paused").and_then(|b| b.as_bool()).unwrap_or(paused); },
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                Some(Ok(Message::Close(_))) | None => break,
                                _ => {}
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("ears-daemon failed to connect: {} (retrying)", e);
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    }
}

#[derive(Debug)]
enum Event { Toggle, CycleLang }

use serde_json::json;
