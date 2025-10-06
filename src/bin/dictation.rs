use anyhow::{Result, Context};
use clap::Parser;
use ears::config::AppConfig;
use enigo::{Enigo, Key, Keyboard, Settings};
use futures_util::StreamExt;
use serde_json::Value;
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[derive(Debug, Parser)]
#[command(name = "ears-dictation")]
#[command(about = "Dictation client that types transcribed text using keyboard simulation")]
struct Args {
    #[arg(long, default_value = "localhost")]
    host: String,

    #[arg(long)]
    port: Option<u16>,

    #[arg(long)]
    only_final: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    let config = AppConfig::load().unwrap_or_default();
    let port = args.port.unwrap_or(config.server.websocket_port);
    
    let url = format!("ws://{}:{}", args.host, port);
    eprintln!("Connecting to {}...", url);
    
    let (ws_stream, _) = connect_async(&url)
        .await
        .context("Failed to connect to WebSocket server")?;
    
    eprintln!("Connected! Transcribed text will be typed automatically.");
    eprintln!("Make sure the target text field is focused.");
    eprintln!("Press Ctrl+C to stop.\n");
    
    let (mut _write, mut read) = ws_stream.split();
    let mut enigo = Enigo::new(&Settings::default()).context("Failed to initialize keyboard controller")?;
    
    while let Some(message) = read.next().await {
        match message {
            Ok(Message::Text(text)) => {
                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                    if let Some(event_type) = json.get("type").and_then(|v| v.as_str()) {
                        match event_type {
                            "partial" if !args.only_final => {
                                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                                    if !text.is_empty() {
                                        eprintln!("[PARTIAL] {}", text);
                                    }
                                }
                            }
                            "final" => {
                                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                                    if !text.is_empty() {
                                        eprintln!("[FINAL] {}", text);
                                        let _ = enigo.text(text);
                                        let _ = enigo.key(Key::Space, enigo::Direction::Click);
                                    }
                                }
                            }
                            "complete" => {
                                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                                    if !text.is_empty() {
                                        eprintln!("[COMPLETE] {}", text);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                eprintln!("WebSocket connection closed");
                break;
            }
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
    
    Ok(())
}
