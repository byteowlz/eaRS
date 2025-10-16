use anyhow::{Result, anyhow};
use futures::{SinkExt, StreamExt};
#[cfg(unix)]
use libc::{EPERM, ESRCH, kill};
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::{fs, io, path::PathBuf};
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, mpsc};
use tokio::task;
use tokio_tungstenite::{accept_async, tungstenite::Message};

use crate::config::{AppConfig, ensure_ref_audio};
use crate::{Model, TranscriptionOptions, TranscriptionSink, WebSocketMessage};

#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub bind_addr: String,
    pub hf_repo: String,
    pub cpu: bool,
    pub transcription: TranscriptionOptions,
}

pub async fn run(options: ServerOptions) -> Result<()> {
    let _pid_guard = create_pid_guard()?;
    let config = AppConfig::load()?;
    ensure_ref_audio(&config).await?;

    let prime_languages = Arc::new(config.model.prime_languages.clone());

    let model_dir = if config.storage.model_dir == "default" {
        None
    } else {
        Some(config.model_dir_path())
    };

    let mut model = if let Some(dir) = model_dir.as_ref() {
        Model::load_from_hf(
            &options.hf_repo,
            options.cpu,
            options.transcription.clone(),
            Some(dir),
        )
        .await?
    } else {
        Model::load_from_hf(
            &options.hf_repo,
            options.cpu,
            options.transcription.clone(),
            None,
        )
        .await?
    };

    for lang in prime_languages.iter() {
        eprintln!("Priming model with language: {}", lang);
        if let Err(e) = model.prime_with_lang_code(lang) {
            eprintln!("Failed to prime language {}: {}", lang, e);
        }
    }

    let model = Arc::new(Mutex::new(model));
    let listener = TcpListener::bind(&options.bind_addr).await?;
    let session_limit = Arc::new(Semaphore::new(5));

    eprintln!(
        "Server listening on {} (max 5 concurrent sessions)",
        options.bind_addr
    );

    loop {
        let (stream, addr) = listener.accept().await?;
        eprintln!("[ears-server] new connection from {}", addr);
        let permit = match session_limit.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                eprintln!(
                    "[ears-server] rejecting connection from {} (server busy)",
                    addr
                );
                tokio::spawn(async move {
                    if let Ok(mut ws) = accept_async(stream).await {
                        let _ = ws
                            .send(Message::Text(
                                json!({ "type": "error", "message": "server busy - maximum concurrent sessions reached" }).to_string(),
                            ))
                            .await;
                        let _ = ws.close(None).await;
                    }
                });
                continue;
            }
        };

        let model = model.clone();
        let prime_languages = prime_languages.clone();
        tokio::spawn(async move {
            eprintln!("[ears-server] handling connection from {}", addr);
            if let Err(err) = handle_connection(stream, model, permit, prime_languages).await {
                eprintln!("[ears-server] connection {} error: {}", addr, err);
            }
            eprintln!("[ears-server] connection from {} closed", addr);
        });
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    model: Arc<Mutex<Model>>,
    permit: tokio::sync::OwnedSemaphorePermit,
    prime_languages: Arc<Vec<String>>,
) -> Result<()> {
    // Set TCP keepalive to detect dead connections
    let _ = stream.set_nodelay(true);

    let ws_stream = accept_async(stream).await?;
    let (mut ws_writer, mut ws_reader) = ws_stream.split();

    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<Message>();
    let (audio_tx, audio_rx) = crossbeam_channel::unbounded::<Vec<f32>>();
    let (lang_cmd_tx, lang_cmd_rx) = crossbeam_channel::unbounded::<String>();

    let writer = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            if ws_writer.send(msg).await.is_err() {
                break;
            }
        }
    });

    let audio_sender = audio_tx.clone();
    let lang_sender = lang_cmd_tx.clone();
    let reader = tokio::spawn(async move {
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
                    if audio_sender.send(chunk).is_err() {
                        eprintln!("[ears-server] audio send failed, breaking reader");
                        break;
                    }
                }
                Ok(Message::Text(text)) => {
                    if should_stop(&text) {
                        eprintln!("[ears-server] stop command received, breaking reader");
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
                                    let _ = lang_sender.send(lang.to_string());
                                }
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    eprintln!("[ears-server] WebSocket close received, breaking reader");
                    break;
                }
                Err(e) => {
                    eprintln!("[ears-server] WebSocket error: {}, breaking reader", e);
                    break;
                }
                _ => {}
            }
        }
        eprintln!("[ears-server] reader task finished");
    });

    let msg_tx_clone = msg_tx.clone();
    let model_clone = model.clone();
    let prime_languages_clone = prime_languages.clone();
    let transcription = task::spawn_blocking(move || {
        eprintln!("[ears-server] transcription task started");
        let mut model = match model_clone.lock() {
            Ok(guard) => guard,
            Err(poison) => poison.into_inner(),
        };

        if let Err(e) = model.reset_stream_state() {
            eprintln!("[ears-server] failed to reset model state: {}", e);
            return Err(e);
        }

        if !prime_languages_clone.is_empty() {
            for lang in prime_languages_clone.iter() {
                eprintln!("[ears-server] re-priming model with language: {}", lang);
                if let Err(e) = model.prime_with_lang_code(lang) {
                    eprintln!("[ears-server] failed to prime language {}: {}", lang, e);
                } else {
                    eprintln!("[ears-server] language priming complete: {}", lang);
                }
            }
        }

        while let Ok(lang) = lang_cmd_rx.try_recv() {
            eprintln!("[ears-server] priming model with language: {}", lang);
            if let Err(e) = model.prime_with_lang_code(&lang) {
                eprintln!("[ears-server] failed to prime language {}: {}", lang, e);
            } else {
                eprintln!("[ears-server] language priming complete: {}", lang);
            }
        }

        eprintln!("[ears-server] starting transcription loop");
        let mut sink = SessionSink::new(msg_tx_clone);
        let result = model.transcribe_live_with_sink(audio_rx, None, Some(lang_cmd_rx), &mut sink);
        eprintln!(
            "[ears-server] transcription loop finished: {:?}",
            result.as_ref().map(|_| "Ok").map_err(|e| e.to_string())
        );
        result
    });

    let _ = reader.await;
    eprintln!("[ears-server] reader finished, closing audio channel");
    drop(audio_tx);

    eprintln!("[ears-server] waiting for transcription to complete");
    match transcription.await? {
        Ok(_) => {
            eprintln!("[ears-server] transcription completed successfully");
        }
        Err(err) => {
            eprintln!("[ears-server] transcription error: {}", err);
            let _ = msg_tx.send(Message::Text(
                json!({ "type": "error", "message": err.to_string() }).to_string(),
            ));
        }
    }

    // Release the session permit now that transcription is done
    // This allows new connections while we finish cleanup
    eprintln!("[ears-server] releasing session permit");
    drop(permit);

    let _ = msg_tx.send(Message::Close(None));
    let _ = writer.await;

    Ok(())
}

struct SessionSink {
    sender: mpsc::UnboundedSender<Message>,
}

impl SessionSink {
    fn new(sender: mpsc::UnboundedSender<Message>) -> Self {
        Self { sender }
    }
}

impl TranscriptionSink for SessionSink {
    fn handle_message(&mut self, message: WebSocketMessage) {
        if let Ok(json) = serde_json::to_string(&message) {
            let _ = self.sender.send(Message::Text(json));
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
