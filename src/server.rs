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
#[cfg(feature = "parakeet-rs")]
use parakeet_rs::ParakeetRsEngine;
#[cfg(feature = "transcribe-cpp")]
use transcribe_cpp::TranscribeCppEngine;

#[cfg(feature = "transcribe-cpp")]
pub mod catalog;
mod engine;
pub mod listener;
#[cfg(feature = "moonshine")]
mod moonshine;
#[cfg(feature = "parakeet-rs")]
mod parakeet_rs;
mod parallel;
#[cfg(feature = "transcribe-cpp")]
pub mod transcribe_cpp;
mod vad;
pub use engine::EngineKind;
#[cfg(feature = "moonshine")]
pub use moonshine::MoonshineEngineConfig;
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
    /// Load Kyutai even when it is not the default engine.
    pub with_kyutai: bool,
    #[cfg(feature = "moonshine")]
    pub moonshine_model_dir: Option<PathBuf>,
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

    // Kyutai is ~2 GB to download and several GB resident. Load it only when
    // it is the default engine or explicitly requested, so a parakeet-rs-only
    // deployment neither downloads nor holds it (trx-zef0).
    if options.default_engine == EngineKind::Kyutai || options.with_kyutai {
        let model_dir = if config.storage.model_dir == "default" {
            None
        } else {
            Some(config.model_dir_path())
        };
        let model = Model::load_from_hf_with_batch(
            &options.hf_repo,
            options.cpu,
            options.transcription.clone(),
            model_dir.as_deref(),
            batch_size,
        )
        .await?;
        let prime_languages = config.model.prime_languages.clone();
        let kyutai_engine = parallel::spawn_parallel_engine(model, prime_languages);
        engine_manager.register(Arc::new(kyutai_engine));
    }

    #[cfg(feature = "moonshine")]
    if options.default_engine == EngineKind::Moonshine || options.moonshine_model_dir.is_some() {
        let engine = moonshine::MoonshineEngine::load(MoonshineEngineConfig {
            model_dir: options.moonshine_model_dir.clone(),
        })?;
        engine_manager.register(Arc::new(engine));
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
        } else if options.default_engine == EngineKind::ParakeetRs {
            eprintln!(
                "[ears-server] parakeet-rs needs a model: pass --parakeet-rs-model <dir> or set [parakeet_rs].model_dir"
            );
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

    // Never fall back to another engine silently: a German deployment that
    // quietly serves an English model is worse than a startup error.
    if !engine_manager.has(options.default_engine) {
        return Err(anyhow!(
            "default engine {} is not available; see the error above, or pass its model \
             (e.g. --parakeet-rs-model / [parakeet_rs].model_dir) or choose another --engine",
            options.default_engine.as_str()
        ));
    }
    let default_engine = options.default_engine;

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
        options.bind_addr, batch_size
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
    run_connection(
        stream,
        engine_manager,
        registry,
        validator,
        default_engine,
        transcription,
        SESSION_FINISH_TIMEOUT,
    )
    .await
}

async fn run_connection(
    stream: tokio::net::TcpStream,
    engine_manager: EngineManager,
    registry: Option<listener::StreamRegistry>,
    validator: Option<listener::TokenValidator>,
    default_engine: EngineKind,
    transcription: TranscriptionOptions,
    finish_timeout: Duration,
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
        if let Some(ref reg) = registry {
            reg.unregister_stream(session_id);
        }
        // Allocation errors were queued before the writer exists.
        while let Ok(msg) = msg_rx.try_recv() {
            let _ = tokio::time::timeout(finish_timeout, ws_writer.send(msg)).await;
        }
        let _ = tokio::time::timeout(finish_timeout, ws_writer.close()).await;
        return Ok(());
    }
    let mut sink_for_status = sink.clone();
    send_engine_changed(&mut sink_for_status, current_engine);
    send_capabilities(&mut sink_for_status, session.as_deref());
    eprintln!(
        "[ears-server] session {} allocated with engine {:?}",
        session_id, current_engine
    );

    let mut writer = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            let closing = matches!(msg, Message::Close(_));
            if ws_writer.send(msg).await.is_err() || closing {
                break;
            }
        }
    });

    let engine_for_reader = engine_manager.clone();
    let registry_clone = registry.clone();
    let mut sink_for_reader = sink.clone();
    let mut session_opt = session;
    let transcription_options = transcription.clone();

    let mut reader = tokio::spawn(async move {
        let mut ws_debug = WsDebugLog::from_env(session_id);
        let mut codec = SessionCodec::Pcm;
        // Single engine-agnostic speech-boundary detector for this connection.
        let mut boundary_vad = vad::BoundaryVad::new(
            transcription_options.boundary_vad,
            24_000,
            transcription_options.boundary_vad_hangover_ms,
        );
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
                &mut boundary_vad,
            );
            if let Some(debug) = ws_debug.as_mut() {
                debug.observe(current_engine, &msg_debug, result.is_ok());
            }
            match result {
                Ok(ClientAction::Continue) => {}
                Ok(ClientAction::Draining) => return,
                Err(err) => {
                    eprintln!("[ears-server] failed to process initial message: {err}");
                    if let Some(sess) = session_opt.take() {
                        sess.request_stop();
                    }
                    return;
                }
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
                        &mut boundary_vad,
                    );
                    if let Some(debug) = ws_debug.as_mut() {
                        debug.observe(current_engine, &msg_debug, result.is_ok());
                    }
                    match result {
                        Ok(ClientAction::Continue) => {}
                        Ok(ClientAction::Draining) => return,
                        Err(err) => {
                            eprintln!("[ears-server] reader error: {err}");
                            break;
                        }
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
        if let Some(sess) = session_opt.take() {
            sess.request_stop();
        }
    });

    // An engine Close must release an idle reader. After reader exit, bound
    // finalization and writes; expiry is never evidence of a successful drain.
    tokio::select! {
        _ = &mut writer => {
            reader.abort();
            let _ = reader.await;
        }
        _ = &mut reader => {
            if tokio::time::timeout(finish_timeout, &mut writer).await.is_err() {
                eprintln!("[ears-server] session {session_id} finalization timed out; no ACK synthesized");
                writer.abort();
                let _ = writer.await;
            }
        }
    }

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
            let closing = matches!(msg, Message::Close(_));
            if ws_writer.send(msg).await.is_err() || closing {
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

/// Includes engine finalization and writes. Expiry closes without claiming success.
const SESSION_FINISH_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, PartialEq, Eq)]
enum ClientAction {
    Continue,
    Draining,
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
    boundary_vad: &mut vad::BoundaryVad,
) -> Result<ClientAction> {
    match msg {
        Message::Binary(data) => {
            if data.is_empty() {
                return Ok(ClientAction::Continue);
            }
            let chunk = match codec {
                SessionCodec::Pcm => {
                    if data.len() % 4 != 0 {
                        send_error(
                            msg_tx,
                            "invalid PCM sample framing; stream cannot be finalized safely",
                        );
                        return Err(anyhow!("incomplete PCM sample"));
                    }
                    decode_audio_chunk(&data)
                }
                SessionCodec::Opus(decoder) => match decoder.decode(&data) {
                    Ok(Some(pcm)) => pcm.to_vec(),
                    Ok(None) => return Ok(ClientAction::Continue),
                    Err(err) => {
                        send_error(
                            msg_tx,
                            "Opus decode failed; stream cannot be finalized safely",
                        );
                        return Err(anyhow!("Opus decode failed: {err}"));
                    }
                },
            };
            if chunk.is_empty() {
                return Ok(ClientAction::Continue);
            }
            if !chunk.iter().all(|sample| sample.is_finite()) {
                send_error(
                    msg_tx,
                    "nonfinite audio sample; stream cannot be finalized safely",
                );
                return Err(anyhow!("nonfinite audio sample"));
            }
            // Detect at the single engine-agnostic ingress. Engines may take
            // ownership of emitting a boundary when they must flush first.
            let speech_boundary = boundary_vad.observe(&chunk);
            if session.is_none() {
                *session = allocate_session(engine_manager, *current_engine, sink.clone(), msg_tx)?;
                if session.is_none() {
                    eprintln!("[ears-server] failed to allocate session when receiving audio");
                    return Ok(ClientAction::Continue);
                } else {
                    eprintln!(
                        "[ears-server] allocated session on engine {:?} for audio",
                        current_engine
                    );
                }
            }
            if let Some(sess) = session.as_ref() {
                sess.send_audio(chunk)?;
                if let Some(active) = speech_boundary {
                    let engine_will_emit = sess.send_speech_boundary(active)?;
                    if !engine_will_emit {
                        sink.handle_message(WebSocketMessage::Speech {
                            active,
                            timestamp: current_timestamp(),
                        });
                    }
                }
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
                    crate::WebSocketCommand::Drain { request_id } => {
                        if request_id == 0 {
                            send_error(msg_tx, "drain request_id must be nonzero");
                        } else if let Some(sess) = session.as_ref() {
                            if !sess.supports_ordered_drain() {
                                send_error(msg_tx, "ordered drain is not supported by this engine");
                            } else {
                                sess.request_drain(request_id).map_err(|err| {
                                    send_error(msg_tx, "failed to queue ordered drain");
                                    err
                                })?;
                                // Stop reading immediately. Do not enqueue ordinary Stop:
                                // only the queued barrier may ACK. Handle disconnection
                                // follows, rather than overtakes, the accepted Drain.
                                session.take();
                                return Ok(ClientAction::Draining);
                            }
                        } else {
                            send_error(msg_tx, "no active engine session to drain");
                        }
                    }
                    crate::WebSocketCommand::SetLanguage { lang } => {
                        if let Some(sess) = session.as_mut() {
                            if sess.supports_language() {
                                if let Err(err) = sess.set_language(lang.clone()) {
                                    eprintln!("[ears-server] language request rejected: {err}");
                                    send_error(msg_tx, &err.to_string());
                                }
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
                            send_capabilities(sink, session.as_deref());
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
                                        send_capabilities(sink, session.as_deref());
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
                                send_capabilities(sink, session.as_deref());
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
                    crate::WebSocketCommand::SetBoundaryVad { enabled } => {
                        boundary_vad.set_enabled(enabled);
                        eprintln!("[ears-server] boundary VAD set to {enabled} for session");
                    }
                    crate::WebSocketCommand::Pause | crate::WebSocketCommand::Resume => {}
                }
                return Ok(ClientAction::Continue);
            }
            if serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .is_some_and(|value| value["type"] == "drain")
            {
                send_error(msg_tx, "invalid drain request_id (expected a nonzero u64)");
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

    Ok(ClientAction::Continue)
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

pub(crate) fn current_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn send_capabilities(sink: &mut SessionSink, session: Option<&dyn EngineSession>) {
    let features = if session.is_some_and(EngineSession::supports_ordered_drain) {
        vec![crate::DICTATION_DRAIN_CAPABILITY.to_string()]
    } else {
        vec![]
    };
    sink.handle_message(WebSocketMessage::Capabilities { features });
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

#[cfg(test)]
mod drain_tests {
    use super::engine::Engine;
    use super::*;
    use crossbeam_channel::{Receiver, Sender, unbounded};
    use std::sync::Mutex;
    use tokio::sync::Semaphore;
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

    type Client = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Normal,
        RejectDrain,
        FailDrain,
        CloseOnAudio,
    }

    #[derive(Clone)]
    struct FakeEngine {
        kind: EngineKind,
        supported: bool,
        mode: Mode,
        gate: Option<Receiver<()>>,
        early_final: bool,
        quiet_stop: bool,
        accepted: Arc<Mutex<Vec<f32>>>,
        drains: Arc<Mutex<Vec<u64>>>,
        permits: Arc<Semaphore>,
    }

    impl FakeEngine {
        fn new(supported: bool) -> Self {
            Self {
                kind: EngineKind::Kyutai,
                supported,
                mode: Mode::Normal,
                gate: None,
                early_final: false,
                quiet_stop: false,
                accepted: Arc::new(Mutex::new(vec![])),
                drains: Arc::new(Mutex::new(vec![])),
                permits: Arc::new(Semaphore::new(1)),
            }
        }
    }

    enum Command {
        Audio(Vec<f32>),
        Drain(u64),
        Stop,
    }

    struct FakeSession {
        tx: Sender<Command>,
        supported: bool,
        reject: bool,
        kind: EngineKind,
    }

    impl EngineSession for FakeSession {
        fn engine(&self) -> EngineKind {
            self.kind
        }
        fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
            self.tx.send(Command::Audio(pcm)).map_err(Into::into)
        }
        fn set_language(&self, _: String) -> Result<()> {
            Ok(())
        }
        fn supports_language(&self) -> bool {
            false
        }
        fn supports_ordered_drain(&self) -> bool {
            self.supported
        }
        fn request_drain(&self, id: u64) -> Result<()> {
            if self.reject {
                return Err(anyhow!("fake queue rejected drain"));
            }
            self.tx.send(Command::Drain(id)).map_err(Into::into)
        }
        fn request_stop(&self) {
            let _ = self.tx.send(Command::Stop);
        }
    }

    impl Engine for FakeEngine {
        fn kind(&self) -> EngineKind {
            self.kind
        }
        fn allocate(&self, mut sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
            let Ok(permit) = self.permits.clone().try_acquire_owned() else {
                return Ok(None);
            };
            let (tx, rx) = unbounded();
            let engine = self.clone();
            std::thread::spawn(move || {
                let _permit = permit;
                let mut text = Vec::new();
                while let Ok(command) = rx.recv() {
                    match command {
                        Command::Audio(pcm) => {
                            engine.accepted.lock().unwrap().extend(&pcm);
                            text.extend(pcm.iter().map(|n| n.to_string()));
                            if engine.mode == Mode::CloseOnAudio {
                                sink.close();
                                return;
                            }
                            if engine.early_final {
                                sink.handle_message(WebSocketMessage::Final {
                                    text: "unrelated final".into(),
                                    words: vec![],
                                });
                            }
                        }
                        Command::Drain(request_id) => {
                            engine.drains.lock().unwrap().push(request_id);
                            if let Some(gate) = &engine.gate {
                                gate.recv_timeout(Duration::from_secs(3)).unwrap();
                            }
                            if engine.mode != Mode::FailDrain {
                                sink.handle_message(WebSocketMessage::Final {
                                    text: text.join(" "),
                                    words: vec![],
                                });
                                sink.handle_message(WebSocketMessage::Drained { request_id });
                            }
                            sink.close();
                            return;
                        }
                        Command::Stop => {
                            if !engine.quiet_stop {
                                sink.handle_message(WebSocketMessage::Final {
                                    text: text.join(" "),
                                    words: vec![],
                                });
                                sink.close();
                            }
                            return;
                        }
                    }
                }
                // A raw disconnect is not a drain either.
                if !engine.quiet_stop {
                    sink.close();
                }
            });
            Ok(Some(Box::new(FakeSession {
                tx,
                supported: self.supported,
                reject: self.mode == Mode::RejectDrain,
                kind: self.kind,
            })))
        }
    }

    async fn connect(
        manager: EngineManager,
        kind: EngineKind,
        finish: Duration,
    ) -> (
        Client,
        tokio::task::JoinHandle<Result<()>>,
        listener::StreamRegistry,
    ) {
        let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = tcp.local_addr().unwrap();
        let registry = listener::StreamRegistry::new();
        let reg = registry.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = tcp.accept().await?;
            run_connection(
                stream,
                manager,
                Some(reg),
                None,
                kind,
                TranscriptionOptions {
                    boundary_vad: false,
                    ..Default::default()
                },
                finish,
            )
            .await
        });
        let (client, _) = connect_async(format!("ws://{address}")).await.unwrap();
        (client, server, registry)
    }

    async fn setup(
        engine: &FakeEngine,
    ) -> (
        Client,
        tokio::task::JoinHandle<Result<()>>,
        listener::StreamRegistry,
    ) {
        let mut manager = EngineManager::new();
        manager.register(Arc::new(engine.clone()));
        let (mut client, server, registry) =
            connect(manager, engine.kind, Duration::from_millis(500)).await;
        client
            .send(Message::text(r#"{"type":"getstatus"}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "enginechanged");
        let caps = next_json(&mut client).await;
        assert_eq!(caps["type"], "capabilities");
        assert_eq!(
            caps["features"],
            if engine.supported {
                json!(["dictation-drain/1"])
            } else {
                json!([])
            }
        );
        assert_eq!(next_json(&mut client).await["type"], "status");
        (client, server, registry)
    }

    async fn next_json(client: &mut Client) -> serde_json::Value {
        let message = tokio::time::timeout(Duration::from_secs(2), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        match message {
            Message::Text(text) => serde_json::from_str(&text).unwrap(),
            other => panic!("expected JSON, got {other:?}"),
        }
    }

    async fn closed(client: &mut Client) {
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(2), client.next())
                .await
                .unwrap(),
            Some(Ok(Message::Close(_))) | None
        ));
    }

    async fn cleaned(
        server: tokio::task::JoinHandle<Result<()>>,
        registry: &listener::StreamRegistry,
        engine: &FakeEngine,
    ) {
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(registry.list_active_streams().is_empty());
        let permit = tokio::time::timeout(
            Duration::from_secs(2),
            engine.permits.clone().acquire_owned(),
        )
        .await
        .unwrap()
        .unwrap();
        drop(permit);
    }

    #[tokio::test]
    async fn ws_bad_audio_closes_without_acknowledging_dropped_input() {
        for (codec, data) in [
            ("pcm", vec![1, 2, 3]),
            ("pcm", f32::NAN.to_le_bytes().to_vec()),
            ("opus", vec![0; 64]),
        ] {
            let engine = FakeEngine::new(true);
            let (mut client, server, registry) = setup(&engine).await;
            client
                .send(Message::text(
                    json!({"type":"setcodec", "codec":codec}).to_string(),
                ))
                .await
                .unwrap();
            client.send(Message::binary(data)).await.unwrap();
            let _ = client
                .send(Message::text(r#"{"type":"drain","request_id":17}"#))
                .await;
            assert_eq!(next_json(&mut client).await["type"], "error");
            assert_eq!(next_json(&mut client).await["type"], "final"); // Legacy stop only, never ACK.
            closed(&mut client).await;
            cleaned(server, &registry, &engine).await;
            assert!(engine.accepted.lock().unwrap().is_empty());
            assert!(engine.drains.lock().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn ws_drain_orders_audio_final_ack_close_and_stops_reader() {
        let (release, gate) = unbounded();
        let mut engine = FakeEngine::new(true);
        engine.gate = Some(gate);
        engine.early_final = true;
        let (mut client, server, registry) = setup(&engine).await;
        for n in [1f32, 2f32] {
            client
                .send(Message::binary(n.to_le_bytes().to_vec()))
                .await
                .unwrap();
        }
        // An ordinary Final does not authorize an ACK.
        assert_eq!(next_json(&mut client).await["text"], "unrelated final");
        assert_eq!(next_json(&mut client).await["text"], "unrelated final");
        assert!(
            tokio::time::timeout(Duration::from_millis(30), client.next())
                .await
                .is_err()
        );

        client
            .send(Message::text(
                r#"{"type":"drain","request_id":18446744073709551615}"#,
            ))
            .await
            .unwrap();
        // These frames follow the barrier and must not reach the engine.
        let _ = client
            .send(Message::binary(3f32.to_le_bytes().to_vec()))
            .await;
        let _ = client
            .send(Message::text(r#"{"type":"drain","request_id":7}"#))
            .await;
        assert!(
            tokio::time::timeout(Duration::from_millis(30), client.next())
                .await
                .is_err()
        );
        release.send(()).unwrap();

        let final_message = next_json(&mut client).await;
        assert_eq!(final_message["type"], "final");
        assert_eq!(final_message["text"], "1 2");
        assert_eq!(
            next_json(&mut client).await,
            json!({"type":"drained","request_id":u64::MAX})
        );
        closed(&mut client).await;
        cleaned(server, &registry, &engine).await;
        assert_eq!(*engine.accepted.lock().unwrap(), [1.0, 2.0]);
        assert_eq!(*engine.drains.lock().unwrap(), [u64::MAX]);
    }

    #[tokio::test]
    async fn ws_drain_as_first_frame_finalizes_empty_session() {
        let engine = FakeEngine::new(true);
        let mut manager = EngineManager::new();
        manager.register(Arc::new(engine.clone()));
        let (mut client, server, registry) =
            connect(manager, engine.kind, Duration::from_millis(500)).await;
        client
            .send(Message::text(r#"{"type":"drain","request_id":1}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "enginechanged");
        assert_eq!(
            next_json(&mut client).await["features"],
            json!(["dictation-drain/1"])
        );
        assert_eq!(next_json(&mut client).await["type"], "final");
        assert_eq!(
            next_json(&mut client).await,
            json!({"type":"drained","request_id":1})
        );
        closed(&mut client).await;
        cleaned(server, &registry, &engine).await;
    }

    #[tokio::test]
    async fn ws_busy_allocation_does_not_advertise_or_leak_registration() {
        let engine = FakeEngine::new(true);
        let permit = engine.permits.clone().try_acquire_owned().unwrap();
        let mut manager = EngineManager::new();
        manager.register(Arc::new(engine.clone()));
        let (mut client, server, registry) =
            connect(manager, engine.kind, Duration::from_millis(500)).await;
        client
            .send(Message::text(r#"{"type":"drain","request_id":1}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "error");
        closed(&mut client).await;
        drop(permit);
        cleaned(server, &registry, &engine).await;
    }

    #[tokio::test]
    async fn ws_invalid_and_stale_messages_never_ack_or_poison_valid_drain() {
        let engine = FakeEngine::new(true);
        let (mut client, server, registry) = setup(&engine).await;
        for raw_id in ["0", "-1", "1.0", "\"1\"", "null", "18446744073709551616"] {
            client
                .send(Message::text(format!(
                    r#"{{"type":"drain","request_id":{raw_id}}}"#
                )))
                .await
                .unwrap();
            assert_eq!(next_json(&mut client).await["type"], "error");
        }
        client
            .send(Message::text(r#"{"type":"drain"}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "error");
        // An unsolicited/stale ACK-shaped client message is not a command.
        client
            .send(Message::text(r#"{"type":"drained","request_id":99}"#))
            .await
            .unwrap();
        client
            .send(Message::text(r#"{"type":"getstatus"}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "status");
        assert!(engine.drains.lock().unwrap().is_empty());
        client
            .send(Message::text(r#"{"type":"drain","request_id":5}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "final");
        assert_eq!(
            next_json(&mut client).await,
            json!({"type":"drained","request_id":5})
        );
        closed(&mut client).await;
        cleaned(server, &registry, &engine).await;
    }

    #[tokio::test]
    async fn ws_unsupported_engine_and_legacy_stop_never_ack() {
        let engine = FakeEngine::new(false);
        let (mut client, server, registry) = setup(&engine).await;
        client
            .send(Message::text(r#"{"type":"drain","request_id":1}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "error");
        client
            .send(Message::binary(9f32.to_le_bytes().to_vec()))
            .await
            .unwrap();
        client.send(Message::text("stop")).await.unwrap();
        let final_message = next_json(&mut client).await;
        assert_eq!(final_message["type"], "final");
        assert_eq!(final_message["text"], "9");
        closed(&mut client).await;
        cleaned(server, &registry, &engine).await;
        assert!(engine.drains.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ws_queue_and_worker_failures_never_ack() {
        for mode in [Mode::RejectDrain, Mode::FailDrain] {
            let mut engine = FakeEngine::new(true);
            engine.mode = mode;
            let (mut client, server, registry) = setup(&engine).await;
            client
                .send(Message::text(r#"{"type":"drain","request_id":1}"#))
                .await
                .unwrap();
            if mode == Mode::RejectDrain {
                assert_eq!(next_json(&mut client).await["type"], "error");
                assert_eq!(next_json(&mut client).await["type"], "final");
            }
            closed(&mut client).await;
            cleaned(server, &registry, &engine).await;
        }
    }

    #[tokio::test]
    async fn ws_engine_close_releases_idle_reader_and_registry() {
        let mut engine = FakeEngine::new(true);
        engine.mode = Mode::CloseOnAudio;
        let (mut client, server, registry) = setup(&engine).await;
        client
            .send(Message::binary(1f32.to_le_bytes().to_vec()))
            .await
            .unwrap();
        closed(&mut client).await;
        cleaned(server, &registry, &engine).await;
    }

    #[tokio::test]
    async fn ws_peer_disconnect_releases_worker_and_registry() {
        let engine = FakeEngine::new(true);
        let (client, server, registry) = setup(&engine).await;
        drop(client);
        cleaned(server, &registry, &engine).await;
    }

    #[tokio::test]
    async fn ws_finalization_deadline_never_synthesizes_ack() {
        let (release, gate) = unbounded();
        let mut engine = FakeEngine::new(true);
        engine.gate = Some(gate);
        let (mut client, server, registry) = setup(&engine).await;
        client
            .send(Message::text(r#"{"type":"drain","request_id":1}"#))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(registry.list_active_streams().is_empty());
        // The socket is gone; a late successful worker must not deliver any ACK.
        assert!(matches!(client.next().await, None | Some(Err(_))));
        release.send(()).unwrap();
        let _permit = tokio::time::timeout(
            Duration::from_secs(2),
            engine.permits.clone().acquire_owned(),
        )
        .await
        .unwrap()
        .unwrap();
    }

    #[cfg(feature = "moonshine")]
    #[tokio::test]
    async fn ws_engine_change_revokes_and_restores_capabilities() {
        let mut supported = FakeEngine::new(true);
        supported.kind = EngineKind::Moonshine;
        // This fixture intentionally does not close the shared sink on an old
        // engine Stop. The unrelated production SetEngine shared-sink bug is
        // not fixed or claimed tested here.
        supported.quiet_stop = true;
        let mut unsupported = FakeEngine::new(false);
        unsupported.quiet_stop = true;
        let mut manager = EngineManager::new();
        manager.register(Arc::new(supported.clone()));
        manager.register(Arc::new(unsupported.clone()));
        let (mut client, server, registry) =
            connect(manager, supported.kind, Duration::from_millis(500)).await;
        client
            .send(Message::text(r#"{"type":"getstatus"}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "enginechanged");
        assert_eq!(
            next_json(&mut client).await["features"],
            json!(["dictation-drain/1"])
        );
        assert_eq!(next_json(&mut client).await["type"], "status");
        // Same-engine selection also refreshes capabilities.
        client
            .send(Message::text(
                r#"{"type":"setengine","engine":"moonshine"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "enginechanged");
        assert_eq!(
            next_json(&mut client).await["features"],
            json!(["dictation-drain/1"])
        );
        client
            .send(Message::text(r#"{"type":"setengine","engine":"kyutai"}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "enginechanged");
        assert_eq!(next_json(&mut client).await["features"], json!([]));
        client
            .send(Message::text(r#"{"type":"drain","request_id":1}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "error");
        // Wait for the previous fixture worker's permit before reallocation.
        let permit = tokio::time::timeout(
            Duration::from_secs(2),
            supported.permits.clone().acquire_owned(),
        )
        .await
        .unwrap()
        .unwrap();
        drop(permit);
        client
            .send(Message::text(
                r#"{"type":"setengine","engine":"moonshine"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "enginechanged");
        assert_eq!(
            next_json(&mut client).await["features"],
            json!(["dictation-drain/1"])
        );
        client
            .send(Message::text(r#"{"type":"drain","request_id":2}"#))
            .await
            .unwrap();
        assert_eq!(next_json(&mut client).await["type"], "final");
        assert_eq!(
            next_json(&mut client).await,
            json!({"type":"drained","request_id":2})
        );
        closed(&mut client).await;
        cleaned(server, &registry, &supported).await;
    }
}
