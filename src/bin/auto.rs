use anyhow::Result;
use crossbeam_channel::unbounded;
use ears::{Model, TranscriptionOptions, audio, config::{AppConfig, ensure_ref_audio}};
use std::thread;

#[tokio::main]
async fn main() -> Result<()> {
    // Load config and assets
    let config = AppConfig::load()?;
    ensure_ref_audio(&config).await?;

    // Defaults
    let hf_repo = "kyutai/stt-1b-en_fr-candle";
    let cpu = false;
    let timestamps = false;
    let vad = true;
    let vad_timeout = None;
    let save_audio: Option<String> = None;

    let options = TranscriptionOptions {
        timestamps,
        vad,
        save_audio: save_audio.clone(),
        vad_timeout,
        whisper_enabled: false,
        whisper_model: None,
        whisper_quantization: None,
        whisper_languages: None,
        whisper_force_lang: None,
    };

    // Load model
    let mut model = if config.storage.model_dir == "default" {
        Model::load_from_hf(hf_repo, cpu, options, None).await?
    } else {
        let model_dir = config.model_dir_path();
        Model::load_from_hf(hf_repo, cpu, options, Some(&model_dir)).await?
    };

    // Setup audio capture
    let (audio_tx, audio_rx) = unbounded();
    let device_index = None;
    let _audio_handle = thread::spawn(move || {
        if let Err(e) = audio::start_audio_capture(audio_tx, device_index) {
            eprintln!("Audio capture error: {}", e);
        }
    });

    // WebSocket port from config
    let ws_port = config.server.websocket_port;
    let url = format!("ws://localhost:{}/", ws_port);

    // Dictation client (type words immediately)
    {
        let url1 = url.clone();
        std::thread::spawn(move || {
            use enigo::{Enigo, Keyboard, Key, Settings};
            use futures_util::StreamExt;
            use tokio_tungstenite::{connect_async, tungstenite::Message};
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            rt.block_on(async move {
                let (ws, _) = match connect_async(&url1).await { Ok(x) => x, Err(_) => return };
                let (_write, mut read) = ws.split();
                let mut enigo = Enigo::new(&Settings::default()).ok();
                while let Some(msg) = read.next().await {
                    if let Ok(Message::Text(txt)) = msg {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                            match v.get("type").and_then(|s| s.as_str()) {
                                Some("word") => {
                                    if let Some(word) = v.get("word").and_then(|t| t.as_str()) {
                                        if let Some(ref mut e) = enigo {
                                            let _ = e.text(word);
                                            let _ = e.key(Key::Space, enigo::Direction::Click);
                                        }
                                    }
                                }
                                Some("final") => {
                                    if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                                        if let Some(ref mut e) = enigo {
                                            // Ensure any final chunk is also typed
                                            let _ = e.text(text);
                                            let _ = e.key(Key::Space, enigo::Direction::Click);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            });
        });
    }

    // Hotkeys
    {
        let url2 = url.clone();
        std::thread::spawn(move || {
            use futures_util::{SinkExt, StreamExt};
            use rdev::{listen, EventType, Key};
            use tokio_tungstenite::{connect_async, tungstenite::Message};
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            rt.block_on(async move {
                let (mut write, mut read) = match connect_async(&url2).await { Ok((ws,_)) => ws.split(), Err(_) => return };
                let _ = write.send(Message::Text(serde_json::json!({"type":"get_status"}).to_string())).await;
                let mut paused = true;
                tokio::spawn(async move {
                    while let Some(Ok(Message::Text(txt))) = read.next().await {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                            if v.get("type").and_then(|s| s.as_str()) == Some("status") {
                                if let Some(b) = v.get("paused").and_then(|b| b.as_bool()) { let _ = b; }
                            }
                        }
                    }
                });
                let _ = listen(move |ev| {
                    static mut CTRL: bool = false; static mut SHIFT: bool = false; static mut IDX: usize = 0; const LIST: [&str;5] = ["en","de","fr","es","ja"];
                    match ev.event_type {
                        EventType::KeyPress(Key::ControlLeft) | EventType::KeyPress(Key::ControlRight) => unsafe { CTRL = true; },
                        EventType::KeyRelease(Key::ControlLeft) | EventType::KeyRelease(Key::ControlRight) => unsafe { CTRL = false; },
                        EventType::KeyPress(Key::ShiftLeft) | EventType::KeyPress(Key::ShiftRight) => unsafe { SHIFT = true; },
                        EventType::KeyRelease(Key::ShiftLeft) | EventType::KeyRelease(Key::ShiftRight) => unsafe { SHIFT = false; },
                        EventType::KeyRelease(Key::KeyV) => unsafe {
                            if CTRL && SHIFT {
                                let cmd = if paused { "resume" } else { "pause" };
                                let _ = futures::executor::block_on(write.send(Message::Text(serde_json::json!({"type": cmd}).to_string())));
                                paused = !paused;
                            }
                        },
                        EventType::KeyRelease(Key::KeyL) => unsafe {
                            if CTRL && SHIFT {
                                IDX = (IDX+1)%LIST.len();
                                let lang = LIST[IDX];
                                let _ = futures::executor::block_on(write.send(Message::Text(serde_json::json!({"type":"set_language","lang":lang}).to_string())));
                            }
                        },
                        _ => {}
                    }
                });
            });
        });
    }

    // Tray (disabled on macOS due to UI thread constraints)
    #[cfg(not(target_os = "macos"))]
    {
        use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
        use tray_icon::{Icon, TrayIconBuilder};
        use serde_json::json;

        let menu = Menu::new();
        let toggle = MenuItem::new("Toggle Dictation", true, None);
        let _ = menu.append(&toggle);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let langs = ["en", "de", "fr", "es", "ja"];
        let lang_items: Vec<MenuItem> = langs
            .iter()
            .map(|&l| MenuItem::new(l.to_uppercase(), true, None))
            .collect();
        for item in &lang_items {
            let _ = menu.append(item);
        }
        let _ = menu.append(&PredefinedMenuItem::separator());
        let quit = MenuItem::new("Quit", true, None);
        let _ = menu.append(&quit);

        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/logo/ears_logo_rounded_white_on_black.png"
        ));
        let img = image::load_from_memory(bytes).expect("icon").into_rgba8();
        let (w, h) = img.dimensions();
        let icon = Icon::from_rgba(img.into_raw(), w, h).expect("icon rgba");
        let _tray = TrayIconBuilder::new()
            .with_tooltip("eaRS")
            .with_icon(icon)
            .with_menu(Box::new(menu))
            .build()
            .expect("tray");

        // Background thread handles menu events and WS I/O
        let url_t = url.clone();
        std::thread::spawn(move || {
            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::{connect_async, tungstenite::Message};

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let (mut write, _read) = match connect_async(&url_t).await {
                    Ok((ws, _)) => ws.split(),
                    Err(_) => return,
                };
                let _ = write
                    .send(Message::Text(json!({ "type": "get_status" }).to_string()))
                    .await;

                let rx = MenuEvent::receiver();
                loop {
                    if let Ok(event) = rx.recv() {
                        let id = event.id;
                        if id == toggle.id() {
                            let _ = write
                                .send(Message::Text(json!({ "type": "restart" }).to_string()))
                                .await;
                        } else if id == quit.id() {
                            std::process::exit(0);
                        } else {
                            for (i, item) in lang_items.iter().enumerate() {
                                if id == item.id() {
                                    let lang = langs[i];
                                    let _ = write
                                        .send(Message::Text(
                                            json!({ "type": "set_language", "lang": lang }).to_string(),
                                        ))
                                        .await;
                                }
                            }
                        }
                    }
                }
            });
        });
    }

    eprintln!("eaRS Auto: WS on {}, hotkeys, dictation, tray running", ws_port);
    model.transcribe_live_ws(audio_rx, save_audio.as_deref(), ws_port).await?;

    Ok(())
}
