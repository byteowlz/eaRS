use anyhow::{Result, anyhow};
use futures::{SinkExt, StreamExt};
#[cfg(unix)]
use libc::{EPERM, ESRCH, kill};
use serde_json::json;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::{accept_async, tungstenite::Message};

use crate::config::{AppConfig, ensure_ref_audio};
use crate::{Model, TranscriptionOptions, TranscriptionSink, WebSocketMessage};
use engine::{EngineManager, EngineSession, send_engine_changed};
#[cfg(feature = "parakeet")]
use parakeet::{ParakeetEngine, ParakeetEngineConfig};
#[cfg(feature = "parakeet-rs")]
use parakeet_rs::ParakeetRsEngine;
#[cfg(feature = "transcribe-cpp")]
use transcribe_cpp::TranscribeCppEngine;

mod engine;
pub mod listener;
#[cfg(feature = "parakeet")]
mod parakeet;
#[cfg(feature = "parakeet-rs")]
mod parakeet_rs;
#[cfg(feature = "transcribe-cpp")]
mod transcribe_cpp;
mod parallel;
pub use engine::EngineKind;
#[cfg(feature = "parakeet")]
pub use parakeet::ParakeetDevice;
#[cfg(feature = "parakeet-rs")]
pub use parakeet_rs::ParakeetRsEngineConfig;
#[cfg(feature = "transcribe-cpp")]
pub use transcribe_cpp::TranscribeCppEngineConfig;

/// Per-connection audio transport codec. Binary WS frames are raw f32 LE PCM
/// by default; a `setcodec` command switches the connection to ogg-opus.
enum SessionCodec {
    Pcm,
    Opus(Box<kaudio::ogg_opus::Decoder>),
}

#[derive(Debug, Clone)]
struct WsMessageDebug {
    is_audio: bool,
    audio_samples: usize,
    audio_sum_squares: f64,
    is_text: bool,
    is_close: bool,
}

impl WsMessageDebug {
    fn from_message(msg: &Message) -> Self {
        match msg {
            Message::Binary(data) => {
                let mut audio_samples = 0usize;
                let mut audio_sum_squares = 0.0f64;
                for bytes in data.chunks_exact(4) {
                    let sample = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    let sample = f64::from(sample);
                    audio_sum_squares += sample * sample;
                    audio_samples += 1;
                }
                Self {
                    is_audio: !data.is_empty(),
                    audio_samples,
                    audio_sum_squares,
                    is_text: false,
                    is_close: false,
                }
            }
            Message::Text(_) => Self {
                is_audio: false,
                audio_samples: 0,
                audio_sum_squares: 0.0,
                is_text: true,
                is_close: false,
            },
            Message::Close(_) => Self {
                is_audio: false,
                audio_samples: 0,
                audio_sum_squares: 0.0,
                is_text: false,
                is_close: true,
            },
            _ => Self {
                is_audio: false,
                audio_samples: 0,
                audio_sum_squares: 0.0,
                is_text: false,
                is_close: false,
            },
        }
    }
}

struct WsDebugLog {
    file: File,
    session_id: u64,
    last_log: Instant,
    audio_msgs: usize,
    audio_samples: usize,
    audio_sum_squares: f64,
    send_audio_ok: usize,
    send_audio_err: usize,
    text_msgs: usize,
    close_msgs: usize,
    last_audio: Option<Instant>,
}

impl WsDebugLog {
    fn from_env(session_id: u64) -> Option<Self> {
        let enabled = std::env::var("EARS_DEBUG_ENGINE")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        if !enabled {
            return None;
        }

        let dir = std::env::var("EARS_DEBUG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_debug_dir());
        if let Err(err) = fs::create_dir_all(&dir) {
            eprintln!(
                "[ears-server] EARS_DEBUG_ENGINE=1 but failed to create {}: {}",
                dir.display(),
                err
            );
            return None;
        }

        let path = dir.join("ws-sessions.log");
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => {
                eprintln!("[ears-server] websocket debug log: {}", path.display());
                Some(Self {
                    file,
                    session_id,
                    last_log: Instant::now(),
                    audio_msgs: 0,
                    audio_samples: 0,
                    audio_sum_squares: 0.0,
                    send_audio_ok: 0,
                    send_audio_err: 0,
                    text_msgs: 0,
                    close_msgs: 0,
                    last_audio: None,
                })
            }
            Err(err) => {
                eprintln!(
                    "[ears-server] EARS_DEBUG_ENGINE=1 but failed to open {}: {}",
                    path.display(),
                    err
                );
                None
            }
        }
    }

    fn observe(&mut self, engine: EngineKind, msg: &WsMessageDebug, ok: bool) {
        if msg.is_audio {
            self.audio_msgs += 1;
            self.audio_samples += msg.audio_samples;
            self.audio_sum_squares += msg.audio_sum_squares;
            self.last_audio = Some(Instant::now());
            if ok {
                self.send_audio_ok += 1;
            } else {
                self.send_audio_err += 1;
            }
        }
        if msg.is_text {
            self.text_msgs += 1;
        }
        if msg.is_close {
            self.close_msgs += 1;
        }
        self.maybe_log(engine);
    }

    fn maybe_log(&mut self, engine: EngineKind) {
        if self.last_log.elapsed() < Duration::from_secs(1) {
            return;
        }
        let rms = if self.audio_samples == 0 {
            0.0
        } else {
            (self.audio_sum_squares / self.audio_samples as f64).sqrt()
        };
        let last_audio_ms = self
            .last_audio
            .map(|last| last.elapsed().as_millis() as u64)
            .unwrap_or(u64::MAX);
        let line = format!(
            "{{\"ts_ms\":{},\"kind\":\"ws\",\"session\":{},\"engine\":\"{}\",\"audio_msgs\":{},\"audio_samples\":{},\"rms\":{:.6},\"send_audio_ok\":{},\"send_audio_err\":{},\"text_msgs\":{},\"close_msgs\":{},\"last_audio_ms\":{}}}",
            unix_ms(),
            self.session_id,
            engine.as_str(),
            self.audio_msgs,
            self.audio_samples,
            rms,
            self.send_audio_ok,
            self.send_audio_err,
            self.text_msgs,
            self.close_msgs,
            last_audio_ms,
        );
        eprintln!("[ws-debug] {line}");
        let _ = writeln!(self.file, "{line}");
        let _ = self.file.flush();

        self.last_log = Instant::now();
        self.audio_msgs = 0;
        self.audio_samples = 0;
        self.audio_sum_squares = 0.0;
        self.send_audio_ok = 0;
        self.send_audio_err = 0;
        self.text_msgs = 0;
        self.close_msgs = 0;
    }
}

fn default_debug_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local")
        .join("state")
        .join("ears")
        .join("debug")
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub bind_addr: String,
    pub hf_repo: String,
    pub cpu: bool,
    pub transcription: TranscriptionOptions,
    pub max_parallel_sessions: usize,
    pub enable_listener_mode: bool,
    pub listener_tokens: Vec<String>,
    pub default_engine: EngineKind,
    #[cfg(feature = "parakeet")]
    pub parakeet_repo: String,
    #[cfg(feature = "parakeet")]
    pub parakeet_model_dir: Option<PathBuf>,
    #[cfg(feature = "parakeet")]
    pub parakeet_device: ParakeetDevice,
    #[cfg(feature = "parakeet")]
    pub parakeet_chunk_seconds: f32,
    #[cfg(feature = "parakeet")]
    pub parakeet_overlap_seconds: f32,
    #[cfg(feature = "parakeet")]
    pub parakeet_noise_gate_rms: f32,
    #[cfg(feature = "parakeet-rs")]
    pub parakeet_rs_model_dir: Option<PathBuf>,
    #[cfg(feature = "parakeet-rs")]
    pub parakeet_rs_lang: Option<String>,
    #[cfg(feature = "transcribe-cpp")]
    pub transcribe_cpp_model: Option<PathBuf>,
    #[cfg(feature = "transcribe-cpp")]
    pub transcribe_cpp_lang: Option<String>,
}

pub async fn run(options: ServerOptions) -> Result<()> {
    let _pid_guard = create_pid_guard()?;
    let config = AppConfig::load()?;
    ensure_ref_audio(&config).await?;

    let batch_size = options.max_parallel_sessions.max(1);
    let mut engine_manager = EngineManager::new();

    let model_dir = if config.storage.model_dir == "default" {
        None
    } else {
        Some(config.model_dir_path())
    };

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
    let kyutai_engine = parallel::spawn_parallel_engine(model, prime_languages);
    let kyutai_capacity = kyutai_engine.capacity();
    engine_manager.register(Arc::new(kyutai_engine.clone()));

    #[cfg(feature = "parakeet")]
    {
        let model_dir = if let Some(dir) = options.parakeet_model_dir.clone() {
            Some(PathBuf::from(dir))
        } else if config.storage.model_dir != "default" {
            Some(config.model_dir_path())
        } else {
            None
        };

        let parakeet_cfg = ParakeetEngineConfig {
            model_repo: options.parakeet_repo.clone(),
            model_dir,
            device: options.parakeet_device,
            chunk_seconds: options.parakeet_chunk_seconds,
            overlap_seconds: options.parakeet_overlap_seconds,
            noise_gate_rms: options.parakeet_noise_gate_rms,
        };

        match ParakeetEngine::load(parakeet_cfg, options.transcription.clone(), batch_size) {
            Ok(engine) => {
                engine_manager.register(Arc::new(engine));
            }
            Err(err) => {
                eprintln!("[ears-server] failed to initialize parakeet engine: {err:#}");
            }
        }
    }

    #[cfg(feature = "parakeet-rs")]
    {
        if let Some(model_dir) = options.parakeet_rs_model_dir.clone() {
            let cfg = ParakeetRsEngineConfig {
                model_dir,
                lang: options.parakeet_rs_lang.clone(),
            };
            match ParakeetRsEngine::load(cfg, options.transcription.clone(), batch_size) {
                Ok(engine) => engine_manager.register(Arc::new(engine)),
                Err(err) => {
                    eprintln!("[ears-server] failed to initialize parakeet-rs engine: {err:#}");
                }
            }
        }
    }

    #[cfg(feature = "transcribe-cpp")]
    {
        if let Some(model_path) = options.transcribe_cpp_model.clone() {
            let cfg = TranscribeCppEngineConfig {
                model_path,
                lang: options.transcribe_cpp_lang.clone(),
            };
            match TranscribeCppEngine::load(cfg) {
                Ok(engine) => engine_manager.register(Arc::new(engine)),
                Err(err) => {
                    eprintln!("[ears-server] failed to initialize transcribe-cpp engine: {err:#}");
                }
            }
        }
    }

    let default_engine = if engine_manager.has(options.default_engine) {
        options.default_engine
    } else {
        EngineKind::Kyutai
    };

    let listener = TcpListener::bind(&options.bind_addr).await?;

    let stream_registry = if options.enable_listener_mode {
        Some(listener::StreamRegistry::new())
    } else {
        None
    };

    let token_validator = if options.enable_listener_mode {
        Some(listener::TokenValidator::new(
            options.listener_tokens.clone(),
        ))
    } else {
        None
    };

    eprintln!(
        "Server listening on {} (max {} concurrent sessions)",
        options.bind_addr, kyutai_capacity
    );
    eprintln!(
        "[ears-server] engines available: {}",
        engine_manager
            .available()
            .iter()
            .map(|e| e.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    if options.enable_listener_mode {
        eprintln!(
            "[ears-server] Listener mode enabled with {} authorized token(s)",
            options.listener_tokens.len()
        );
    }

    let transcription_opts = options.transcription.clone();

    loop {
        let (stream, addr) = listener.accept().await?;
        eprintln!("[ears-server] new connection from {}", addr);
        let engine_clone = engine_manager.clone();
        let registry_clone = stream_registry.clone();
        let validator_clone = token_validator.clone();
        let default_engine = default_engine;
        let transcription_clone = transcription_opts.clone();
        tokio::spawn(async move {
            eprintln!("[ears-server] handling connection from {}", addr);
            if let Err(err) = handle_connection(
                stream,
                engine_clone,
                registry_clone,
                validator_clone,
                default_engine,
                transcription_clone,
            )
            .await
            {
                eprintln!("[ears-server] connection {} error: {}", addr, err);
            }
            eprintln!("[ears-server] connection from {} closed", addr);
        });
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    engine_manager: EngineManager,
    registry: Option<listener::StreamRegistry>,
    validator: Option<listener::TokenValidator>,
    default_engine: EngineKind,
    transcription: TranscriptionOptions,
) -> Result<()> {
    let _ = stream.set_nodelay(true);

    let ws_stream = accept_async(stream).await?;
    let (mut ws_writer, mut ws_reader) = ws_stream.split();

    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<Message>();

    let first_msg = match ws_reader.next().await {
        Some(Ok(Message::Text(text))) => {
            if let Ok(cmd) = serde_json::from_str::<listener::ListenerCommand>(&text) {
                return handle_listener_connection(
                    ws_writer, ws_reader, msg_tx, msg_rx, cmd, registry, validator,
                )
                .await;
            } else {
                Some(Message::Text(text))
            }
        }
        Some(Ok(msg)) => Some(msg),
        Some(Err(e)) => return Err(e.into()),
        None => return Ok(()),
    };

    use std::sync::atomic::{AtomicU64, Ordering};
    static SESSION_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
    let session_id = SESSION_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

    let sink = if let Some(ref reg) = registry {
        reg.register_stream(session_id)?;
        SessionSink::with_broadcast(msg_tx.clone(), session_id, reg.clone())
    } else {
        SessionSink::new(msg_tx.clone())
    };

    let mut current_engine = default_engine;
    let session = allocate_session(&engine_manager, current_engine, sink.clone(), &msg_tx)?;
    if session.is_none() {
        return Ok(());
    }
    let mut sink_for_status = sink.clone();
    send_engine_changed(&mut sink_for_status, current_engine);
    eprintln!(
        "[ears-server] session {} allocated with engine {:?}",
        session_id, current_engine
    );

    let writer = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            if ws_writer.send(msg).await.is_err() {
                break;
            }
        }
    });

    let engine_for_reader = engine_manager.clone();
    let registry_clone = registry.clone();
    let mut sink_for_reader = sink.clone();
    let mut session_opt = session;
    let transcription_options = transcription.clone();

    let reader = tokio::spawn(async move {
        let mut ws_debug = WsDebugLog::from_env(session_id);
        let mut codec = SessionCodec::Pcm;
        if let Some(msg) = first_msg {
            let msg_debug = WsMessageDebug::from_message(&msg);
            let result = handle_client_message(
                msg,
                &mut session_opt,
                &mut sink_for_reader,
                &engine_for_reader,
                &msg_tx,
                &mut current_engine,
                &transcription_options,
                &mut codec,
            );
            if let Some(debug) = ws_debug.as_mut() {
                debug.observe(current_engine, &msg_debug, result.is_ok());
            }
            if let Err(err) = result {
                eprintln!("[ears-server] failed to process initial message: {err}");
                return;
            }
        }

        loop {
            match tokio::time::timeout(Duration::from_secs(1), ws_reader.next()).await {
                Ok(Some(Ok(message))) => {
                    let msg_debug = WsMessageDebug::from_message(&message);
                    let result = handle_client_message(
                        message,
                        &mut session_opt,
                        &mut sink_for_reader,
                        &engine_for_reader,
                        &msg_tx,
                        &mut current_engine,
                        &transcription_options,
                        &mut codec,
                    );
                    if let Some(debug) = ws_debug.as_mut() {
                        debug.observe(current_engine, &msg_debug, result.is_ok());
                    }
                    if let Err(err) = result {
                        eprintln!("[ears-server] reader error: {err}");
                        break;
                    }
                }
                Ok(Some(Err(e))) => {
                    eprintln!("[ears-server] WebSocket error: {e}");
                    if let Some(sess) = session_opt.take() {
                        sess.request_stop();
                    }
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    if let Some(debug) = ws_debug.as_mut() {
                        debug.maybe_log(current_engine);
                    }
                }
            }
        }
    });

    let _ = reader.await;
    let _ = writer.await;

    if let Some(ref reg) = registry_clone {
        reg.unregister_stream(session_id);
    }

    Ok(())
}

async fn handle_listener_connection(
    mut ws_writer: futures::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        Message,
    >,
    mut ws_reader: futures::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    >,
    msg_tx: mpsc::UnboundedSender<Message>,
    mut msg_rx: mpsc::UnboundedReceiver<Message>,
    first_cmd: listener::ListenerCommand,
    registry: Option<listener::StreamRegistry>,
    validator: Option<listener::TokenValidator>,
) -> Result<()> {
    use futures::{SinkExt, StreamExt};

    let registry = registry.ok_or_else(|| anyhow!("Listener mode not enabled"))?;
    let validator = validator.ok_or_else(|| anyhow!("Listener mode not configured"))?;

    eprintln!("[ears-server] handling listener connection");

    let mut authenticated = false;
    let mut _subscribed_stream: Option<u64> = None;

    // Process first command
    let response = match first_cmd {
        listener::ListenerCommand::Authenticate { token } => {
            if validator.validate(&token) {
                authenticated = true;
                eprintln!("[ears-server] listener authenticated (auth={authenticated})");
                Some(Message::text(
                    json!({
                        "type": "authenticated",
                        "success": true
                    })
                    .to_string(),
                ))
            } else {
                eprintln!("[ears-server] listener authentication failed");
                Some(Message::text(
                    json!({
                        "type": "error",
                        "message": "invalid token"
                    })
                    .to_string(),
                ))
            }
        }
        listener::ListenerCommand::ListStreams => {
            if !authenticated {
                Some(Message::text(
                    json!({
                        "type": "error",
                        "message": "not authenticated"
                    })
                    .to_string(),
                ))
            } else {
                let streams = registry.list_active_streams();
                Some(Message::text(
                    json!({
                        "type": "streams",
                        "stream_ids": streams
                    })
                    .to_string(),
                ))
            }
        }
        listener::ListenerCommand::Subscribe { stream_id } => {
            if !authenticated {
                Some(Message::text(
                    json!({
                        "type": "error",
                        "message": "not authenticated"
                    })
                    .to_string(),
                ))
            } else {
                match registry.add_listener(stream_id, msg_tx.clone()) {
                    Ok(_) => {
                        _subscribed_stream = Some(stream_id);
                        eprintln!("[ears-server] listener subscribed to stream {}", stream_id);
                        Some(Message::text(
                            json!({
                                "type": "subscribed",
                                "stream_id": stream_id
                            })
                            .to_string(),
                        ))
                    }
                    Err(e) => Some(Message::text(
                        json!({
                            "type": "error",
                            "message": format!("subscription failed: {}", e)
                        })
                        .to_string(),
                    )),
                }
            }
        }
    };

    if let Some(response) = response {
        ws_writer.send(response).await?;
    }

    let writer = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            if ws_writer.send(msg).await.is_err() {
                break;
            }
        }
    });

    let registry_clone = registry.clone();
    let validator_clone = validator.clone();
    let msg_tx_clone = msg_tx.clone();

    let reader = tokio::spawn(async move {
        let mut auth = false;
        let mut _sub: Option<u64> = None;

        while let Some(msg) = ws_reader.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(cmd) = serde_json::from_str::<listener::ListenerCommand>(&text) {
                        let response = match cmd {
                            listener::ListenerCommand::Authenticate { token } => {
                                if validator_clone.validate(&token) {
                                    auth = true;
                                    eprintln!("[ears-server] listener authenticated");
                                    Some(Message::text(
                                        json!({
                                            "type": "authenticated",
                                            "success": true
                                        })
                                        .to_string(),
                                    ))
                                } else {
                                    eprintln!("[ears-server] listener authentication failed");
                                    Some(Message::text(
                                        json!({
                                            "type": "error",
                                            "message": "invalid token"
                                        })
                                        .to_string(),
                                    ))
                                }
                            }
                            listener::ListenerCommand::ListStreams => {
                                if !auth {
                                    Some(Message::text(
                                        json!({
                                            "type": "error",
                                            "message": "not authenticated"
                                        })
                                        .to_string(),
                                    ))
                                } else {
                                    let streams = registry_clone.list_active_streams();
                                    Some(Message::text(
                                        json!({
                                            "type": "streams",
                                            "stream_ids": streams
                                        })
                                        .to_string(),
                                    ))
                                }
                            }
                            listener::ListenerCommand::Subscribe { stream_id } => {
                                if !auth {
                                    Some(Message::text(
                                        json!({
                                            "type": "error",
                                            "message": "not authenticated"
                                        })
                                        .to_string(),
                                    ))
                                } else {
                                    match registry_clone
                                        .add_listener(stream_id, msg_tx_clone.clone())
                                    {
                                        Ok(_) => {
                                            _sub = Some(stream_id);
                                            eprintln!(
                                                "[ears-server] listener subscribed to stream {}",
                                                stream_id
                                            );
                                            Some(Message::text(
                                                json!({
                                                    "type": "subscribed",
                                                    "stream_id": stream_id
                                                })
                                                .to_string(),
                                            ))
                                        }
                                        Err(e) => Some(Message::text(
                                            json!({
                                                "type": "error",
                                                "message": format!("subscription failed: {}", e)
                                            })
                                            .to_string(),
                                        )),
                                    }
                                }
                            }
                        };

                        if let Some(resp) = response {
                            let _ = msg_tx_clone.send(resp);
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {}
            }
        }
    });

    let _ = tokio::join!(writer, reader);

    eprintln!("[ears-server] listener connection closed");
    Ok(())
}

fn handle_client_message(
    msg: Message,
    session: &mut Option<Box<dyn EngineSession>>,
    sink: &mut SessionSink,
    engine_manager: &EngineManager,
    msg_tx: &mpsc::UnboundedSender<Message>,
    current_engine: &mut EngineKind,
    transcription: &TranscriptionOptions,
    codec: &mut SessionCodec,
) -> Result<()> {
    match msg {
        Message::Binary(data) => {
            if data.is_empty() {
                return Ok(());
            }
            let chunk = match codec {
                SessionCodec::Pcm => decode_audio_chunk(&data),
                SessionCodec::Opus(decoder) => match decoder.decode(&data) {
                    Ok(Some(pcm)) => pcm.to_vec(),
                    Ok(None) => return Ok(()),
                    Err(err) => {
                        eprintln!("[ears-server] opus decode failed: {err}");
                        return Ok(());
                    }
                },
            };
            if chunk.is_empty() {
                return Ok(());
            }
            if session.is_none() {
                *session = allocate_session(engine_manager, *current_engine, sink.clone(), msg_tx)?;
                if session.is_none() {
                    eprintln!("[ears-server] failed to allocate session when receiving audio");
                    return Ok(());
                } else {
                    eprintln!(
                        "[ears-server] allocated session on engine {:?} for audio",
                        current_engine
                    );
                }
            }
            if let Some(sess) = session.as_ref() {
                sess.send_audio(chunk)?;
            }
        }
        Message::Text(text) => {
            if should_stop(&text) {
                if let Some(sess) = session.take() {
                    sess.request_stop();
                }
                return Err(anyhow!("stop requested"));
            }

            if let Ok(cmd) = serde_json::from_str::<crate::WebSocketCommand>(&text) {
                match cmd {
                    crate::WebSocketCommand::SetLanguage { lang } => {
                        if let Some(sess) = session.as_mut() {
                            if sess.supports_language() {
                                let _ = sess.set_language(lang.clone());
                            } else {
                                send_status(sink, transcription, *current_engine);
                            }
                        }
                    }
                    crate::WebSocketCommand::Restart => {
                        if let Some(new_session) =
                            allocate_session(engine_manager, *current_engine, sink.clone(), msg_tx)?
                        {
                            if let Some(old) = session.take() {
                                old.request_stop();
                            }
                            *session = Some(new_session);
                            send_engine_changed(sink, *current_engine);
                        }
                    }
                    crate::WebSocketCommand::SetEngine { engine } => {
                        eprintln!(
                            "[ears-server] SetEngine requested: {engine}, current: {:?}",
                            *current_engine
                        );
                        if let Some(kind) = EngineKind::from_str(&engine) {
                            if !engine_manager.has(kind) {
                                eprintln!("[ears-server] SetEngine: engine {engine} not available");
                                send_error(msg_tx, "engine not available");
                            } else if kind != *current_engine {
                                eprintln!("[ears-server] SetEngine: switching to {kind:?}...");
                                match allocate_session(engine_manager, kind, sink.clone(), msg_tx) {
                                    Ok(Some(new_session)) => {
                                        if let Some(old) = session.take() {
                                            old.request_stop();
                                        }
                                        *session = Some(new_session);
                                        *current_engine = kind;
                                        send_engine_changed(sink, kind);
                                        eprintln!(
                                            "[ears-server] SetEngine: successfully switched to {kind:?}"
                                        );
                                    }
                                    Ok(None) => {
                                        eprintln!(
                                            "[ears-server] SetEngine: allocate_session returned None (busy?)"
                                        );
                                    }
                                    Err(err) => {
                                        eprintln!(
                                            "[ears-server] SetEngine: allocate_session error: {err}"
                                        );
                                    }
                                }
                            } else {
                                send_engine_changed(sink, kind);
                            }
                        } else {
                            send_error(msg_tx, "unknown engine");
                        }
                    }
                    crate::WebSocketCommand::GetStatus => {
                        send_status(sink, transcription, *current_engine);
                    }
                    crate::WebSocketCommand::SetVadTimeout { .. } => {
                        // Not adjustable per-session for now; ignore gracefully.
                    }
                    crate::WebSocketCommand::SetCodec { codec: name } => {
                        match name.to_ascii_lowercase().as_str() {
                            "opus" => match kaudio::ogg_opus::Decoder::new(24_000, 1) {
                                Ok(decoder) => {
                                    *codec = SessionCodec::Opus(Box::new(decoder));
                                    eprintln!("[ears-server] session codec set to opus");
                                }
                                Err(err) => {
                                    eprintln!("[ears-server] opus decoder init failed: {err}");
                                    send_error(msg_tx, "failed to initialize opus decoder");
                                }
                            },
                            "pcm" => *codec = SessionCodec::Pcm,
                            other => {
                                eprintln!("[ears-server] unknown codec requested: {other}");
                                send_error(msg_tx, "unknown codec (supported: pcm, opus)");
                            }
                        }
                    }
                    crate::WebSocketCommand::Pause | crate::WebSocketCommand::Resume => {}
                }
                return Ok(());
            }
        }
        Message::Close(_) => {
            if let Some(sess) = session.take() {
                sess.request_stop();
            }
            return Err(anyhow!("client closed connection"));
        }
        _ => {}
    }

    Ok(())
}

fn allocate_session(
    manager: &EngineManager,
    engine: EngineKind,
    sink: SessionSink,
    msg_tx: &mpsc::UnboundedSender<Message>,
) -> Result<Option<Box<dyn EngineSession>>> {
    match manager.allocate(engine, sink) {
        Ok(Some(session)) => Ok(Some(session)),
        Ok(None) => {
            send_error(msg_tx, "server busy - maximum concurrent sessions reached");
            Ok(None)
        }
        Err(err) => {
            eprintln!("[ears-server] failed to allocate session: {err}");
            send_error(msg_tx, "internal server error");
            Ok(None)
        }
    }
}

fn send_status(sink: &mut SessionSink, transcription: &TranscriptionOptions, engine: EngineKind) {
    let status = WebSocketMessage::Status {
        paused: false,
        vad: transcription.vad,
        timestamps: transcription.timestamps,
        vad_timeout: transcription.vad_timeout,
        lang: None,
        engine: Some(engine.as_str().to_string()),
    };
    sink.handle_message(status);
}

fn send_error(msg_tx: &mpsc::UnboundedSender<Message>, message: &str) {
    let _ = msg_tx.send(Message::text(
        json!({
            "type": "error",
            "message": message
        })
        .to_string(),
    ));
}

#[derive(Clone)]
pub(crate) struct SessionSink {
    inner: Arc<SessionSinkInner>,
}

impl SessionSink {
    pub(crate) fn new(sender: mpsc::UnboundedSender<Message>) -> Self {
        Self {
            inner: Arc::new(SessionSinkInner {
                sender,
                session_id: None,
                registry: None,
            }),
        }
    }

    pub(crate) fn with_broadcast(
        sender: mpsc::UnboundedSender<Message>,
        session_id: u64,
        registry: listener::StreamRegistry,
    ) -> Self {
        Self {
            inner: Arc::new(SessionSinkInner {
                sender,
                session_id: Some(session_id),
                registry: Some(registry),
            }),
        }
    }

    pub(crate) fn close(&self) {
        let _ = self.inner.sender.send(Message::Close(None));
    }
}

impl TranscriptionSink for SessionSink {
    fn handle_message(&mut self, message: WebSocketMessage) {
        if let Ok(json) = serde_json::to_string(&message) {
            if self.inner.sender.send(Message::text(json.clone())).is_err() {
                eprintln!("[ears-server] failed to forward message to websocket writer");
            }
        }

        if let (Some(session_id), Some(registry)) = (self.inner.session_id, &self.inner.registry) {
            let _ = registry.broadcast_message(session_id, &message);
        }
    }
}

#[derive(Clone)]
struct SessionSinkInner {
    sender: mpsc::UnboundedSender<Message>,
    session_id: Option<u64>,
    registry: Option<listener::StreamRegistry>,
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
