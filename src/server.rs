use anyhow::{Result, anyhow};
use futures::{SinkExt, StreamExt};
#[cfg(unix)]
use libc::{EPERM, ESRCH, kill};
use serde_json::json;
use std::{fs, io, path::PathBuf};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::{accept_async, tungstenite::Message};

use crate::config::{AppConfig, ensure_ref_audio};
use crate::{Model, TranscriptionOptions, TranscriptionSink, WebSocketMessage};

mod parallel;

#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub bind_addr: String,
    pub hf_repo: String,
    pub cpu: bool,
    pub transcription: TranscriptionOptions,
    pub max_parallel_sessions: usize,
}

pub async fn run(options: ServerOptions) -> Result<()> {
    let _pid_guard = create_pid_guard()?;
    let config = AppConfig::load()?;
    ensure_ref_audio(&config).await?;

    let model_dir = if config.storage.model_dir == "default" {
        None
    } else {
        Some(config.model_dir_path())
    };

    let batch_size = options.max_parallel_sessions.max(1);
    let model = if let Some(dir) = model_dir.as_ref() {
        Model::load_from_hf_with_batch(
            &options.hf_repo,
            options.cpu,
            options.transcription.clone(),
            Some(dir),
            batch_size,
        )
        .await?
    } else {
        Model::load_from_hf_with_batch(
            &options.hf_repo,
            options.cpu,
            options.transcription.clone(),
            None,
            batch_size,
        )
        .await?
    };

    let prime_languages = config.model.prime_languages.clone();
    let engine = parallel::spawn_parallel_engine(model, prime_languages);
    let listener = TcpListener::bind(&options.bind_addr).await?;

    eprintln!(
        "Server listening on {} (max {} concurrent sessions)",
        options.bind_addr,
        engine.capacity()
    );

    loop {
        let (stream, addr) = listener.accept().await?;
        eprintln!("[ears-server] new connection from {}", addr);
        let engine_clone = engine.clone();
        tokio::spawn(async move {
            eprintln!("[ears-server] handling connection from {}", addr);
            if let Err(err) = handle_connection(stream, engine_clone).await {
                eprintln!("[ears-server] connection {} error: {}", addr, err);
            }
            eprintln!("[ears-server] connection from {} closed", addr);
        });
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    engine: parallel::ParallelEngine,
) -> Result<()> {
    // Set TCP keepalive to detect dead connections
    let _ = stream.set_nodelay(true);

    let ws_stream = accept_async(stream).await?;
    let (mut ws_writer, mut ws_reader) = ws_stream.split();

    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<Message>();
    let sink = SessionSink::new(msg_tx.clone());
    let session = match engine.allocate_session(sink) {
        Ok(Some(handle)) => handle,
        Ok(None) => {
            let _ = msg_tx.send(Message::Text(
                json!({
                    "type": "error",
                    "message": "server busy - maximum concurrent sessions reached"
                })
                .to_string(),
            ));
            let _ = msg_tx.send(Message::Close(None));
            while let Some(msg) = msg_rx.recv().await {
                let _ = ws_writer.send(msg).await;
            }
            return Ok(());
        }
        Err(err) => {
            eprintln!("[ears-server] failed to allocate session: {err}");
            let _ = msg_tx.send(Message::Text(
                json!({ "type": "error", "message": "internal server error" }).to_string(),
            ));
            let _ = msg_tx.send(Message::Close(None));
            while let Some(msg) = msg_rx.recv().await {
                let _ = ws_writer.send(msg).await;
            }
            return Ok(());
        }
    };

    let writer = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            if ws_writer.send(msg).await.is_err() {
                break;
            }
        }
    });

    let session_reader = session.clone();
    let reader = tokio::spawn(async move {
        let session = session_reader;
        eprintln!("[ears-server] reader task started");
        while let Some(msg) = ws_reader.next().await {
            match msg {
                Ok(Message::Binary(data)) => {
                    if data.is_empty() {
                        continue;
                    }
                    let chunk = decode_audio_chunk(&data);
                    if chunk.is_empty() {
                        continue;
                    }
                    if session.send_audio(chunk).is_err() {
                        eprintln!("[ears-server] audio send failed, breaking reader");
                        break;
                    }
                }
                Ok(Message::Text(text)) => {
                    if should_stop(&text) {
                        eprintln!("[ears-server] stop command received, breaking reader");
                        session.request_stop();
                        break;
                    }

                    if let Ok(cmd) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(cmd_type) = cmd.get("type").and_then(|v| v.as_str()) {
                            if cmd_type == "setlanguage" {
                                if let Some(lang) = cmd.get("lang").and_then(|v| v.as_str()) {
                                    eprintln!(
                                        "[ears-server] received language change command: {}",
                                        lang
                                    );
                                    if let Err(err) = session.set_language(lang.to_string()) {
                                        eprintln!(
                                            "[ears-server] failed to forward language change: {err}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    eprintln!("[ears-server] WebSocket close received, breaking reader");
                    session.request_stop();
                    break;
                }
                Err(e) => {
                    eprintln!("[ears-server] WebSocket error: {}, breaking reader", e);
                    session.request_stop();
                    break;
                }
                _ => {}
            }
        }
        eprintln!("[ears-server] reader task finished");
    });

    let _ = reader.await;
    drop(session);

    let _ = writer.await;

    Ok(())
}

pub(crate) struct SessionSink {
    sender: mpsc::UnboundedSender<Message>,
}

impl SessionSink {
    pub(crate) fn new(sender: mpsc::UnboundedSender<Message>) -> Self {
        Self { sender }
    }

    pub(crate) fn close(&self) {
        let _ = self.sender.send(Message::Close(None));
    }
}

impl TranscriptionSink for SessionSink {
    fn handle_message(&mut self, message: WebSocketMessage) {
        if let Ok(json) = serde_json::to_string(&message) {
            if self.sender.send(Message::Text(json)).is_err() {
                eprintln!("[ears-server] failed to forward message to websocket writer");
            }
        }
    }
}

fn decode_audio_chunk(data: &[u8]) -> Vec<f32> {
    let mut samples = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        samples.push(sample);
    }
    samples
}

fn should_stop(text: &str) -> bool {
    if text.trim().eq_ignore_ascii_case("stop") {
        return true;
    }

    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.eq_ignore_ascii_case("stop"))
        })
        .unwrap_or(false)
}

fn create_pid_guard() -> Result<PidFileGuard> {
    let path = pid_file_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    if path.exists() {
        if let Some(existing_pid) = read_pid_file()? {
            if is_process_alive(existing_pid) {
                return Err(anyhow!(
                    "ears server already running (pid {})",
                    existing_pid
                ));
            }
        }
        let _ = fs::remove_file(&path);
    }

    fs::write(&path, std::process::id().to_string())?;
    Ok(PidFileGuard(path))
}

pub fn pid_file_path() -> Result<PathBuf> {
    let base = if let Ok(dir) = std::env::var("XDG_STATE_HOME") {
        if !dir.trim().is_empty() {
            PathBuf::from(dir)
        } else {
            default_state_home()?
        }
    } else {
        default_state_home()?
    };

    Ok(base.join("ears").join("server.pid"))
}

pub fn read_pid_file() -> Result<Option<i32>> {
    let path = pid_file_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(&path)?;
    let pid = contents
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow!("invalid pid file: {e}"))?;
    Ok(Some(pid))
}

pub fn remove_pid_file() -> Result<()> {
    let path = pid_file_path()?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub fn is_process_alive(pid: i32) -> bool {
    #[cfg(unix)]
    unsafe {
        if kill(pid, 0) == 0 {
            return true;
        }
        match io::Error::last_os_error().raw_os_error() {
            Some(code) if code == EPERM => true,
            Some(code) if code == ESRCH => false,
            _ => false,
        }
    }

    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

fn default_state_home() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("HOME directory not found"))?;
    Ok(home.join(".local").join("state"))
}

struct PidFileGuard(PathBuf);

impl Drop for PidFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
