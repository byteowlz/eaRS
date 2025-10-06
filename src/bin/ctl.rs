use anyhow::Result;
use clap::{Parser, Subcommand};
use ears::config::AppConfig;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tokio::time::{timeout, Duration};

#[derive(Debug, Parser)]
#[command(name = "ears-ctl")]
#[command(about = "Control eaRS over WebSocket for WM bindings (Hyprland, etc.)")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    Toggle,
    Pause,
    Resume,
    Lang { lang: String },
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = AppConfig::load()?;
    let url = format!("ws://127.0.0.1:{}/", cfg.server.websocket_port);
    let (mut ws, _) = match timeout(Duration::from_secs(2), connect_async(&url)).await {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => return Err(e.into()),
        Err(_) => return Err(anyhow::anyhow!("Timed out connecting to {}", url)),
    };

    match args.cmd {
        Cmd::Toggle => {
            // Request status (best-effort), then flip
            let _ = ws
                .send(Message::Text(serde_json::json!({"type":"get_status"}).to_string()))
                .await;
            let mut paused = true;
            if let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(
                std::time::Duration::from_millis(800),
                ws.next(),
            )
            .await
            {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    if v.get("type").and_then(|s| s.as_str()) == Some("status") {
                        paused = v
                            .get("paused")
                            .and_then(|b| b.as_bool())
                            .unwrap_or(true);
                    }
                }
            }
            let cmd = if paused { "resume" } else { "pause" };
            let _ = ws
                .send(Message::Text(serde_json::json!({"type": cmd}).to_string()))
                .await;
        }
        Cmd::Pause => { let _ = ws.send(Message::Text("{\"type\":\"pause\"}".into())).await; }
        Cmd::Resume => { let _ = ws.send(Message::Text("{\"type\":\"resume\"}".into())).await; }
        Cmd::Lang { lang } => { let _ = ws.send(Message::Text(serde_json::json!({"type":"set_language","lang":lang}).to_string())).await; }
        Cmd::Status => {
            let _ = ws
                .send(Message::Text("{\"type\":\"get_status\"}".into()))
                .await;
            match tokio::time::timeout(std::time::Duration::from_secs(1), ws.next()).await {
                Ok(Some(Ok(Message::Text(txt)))) => println!("{}", txt),
                _ => {
                    eprintln!("Timed out waiting for status");
                    std::process::exit(1);
                }
            }
        }
    }
    Ok(())
}
