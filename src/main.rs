use anyhow::{Context, Result, anyhow};
use clap::{Args, Parser, Subcommand, ValueEnum};
use crossbeam_channel::unbounded;
use ears::{
    TranscriptionOptions, WebSocketMessage, WordTimestamp, audio, config::AppConfig, server,
};
use futures::{SinkExt, StreamExt};
use serde_json::json;
use std::{
    fs::OpenOptions,
    io::{self, Write},
    net::{TcpStream, ToSocketAddrs},
    process::{Command as ProcessCommand, Stdio},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Message, http::Uri},
};

use ears::server::EngineKind;
#[cfg(feature = "parakeet")]
use ears::server::ParakeetDevice;
#[cfg(feature = "parakeet")]
use std::path::PathBuf;

#[cfg(unix)]
use libc::{SIGINT, SIGTERM, kill};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "eaRS client and server controller",
    after_long_help = "Tip: for system-wide dictation, run `ears dictation start` and toggle it with your hotkey."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    client: ClientArgs,
}

#[derive(Subcommand)]
enum Commands {
    #[command(subcommand)]
    Server(ServerCommand),
    #[command(subcommand)]
    Dictation(DictationCommand),
}

#[derive(Subcommand)]
enum ServerCommand {
    Start(ServerStartArgs),
    Stop,
    Status,
    #[command(hide = true)]
    Run(ServerStartArgs),
}

#[derive(Subcommand)]
enum DictationCommand {
    Start(DictationStartArgs),
    Stop,
    Status,
    #[command(about = "Run dictation in foreground with debug output")]
    Debug(DictationStartArgs),
}

#[derive(Args, Clone)]
struct DictationStartArgs {
    /// Server alias from config (e.g., 'local', 'remote') or full WebSocket URL (ws://host:port)
    #[arg(short, long)]
    server: Option<String>,

    /// Set the transcription language (e.g., 'en', 'de', 'es', 'fr', 'ja')
    #[arg(long)]
    lang: Option<String>,

    /// Select transcription engine (kyutai|parakeet)
    #[arg(long)]
    engine: Option<String>,
}

#[derive(Clone, Debug, ValueEnum)]
enum EngineArg {
    Kyutai,
    #[cfg(feature = "parakeet")]
    Parakeet,
}

impl EngineArg {
    fn to_engine_kind(&self) -> EngineKind {
        match self {
            EngineArg::Kyutai => EngineKind::Kyutai,
            #[cfg(feature = "parakeet")]
            EngineArg::Parakeet => EngineKind::Parakeet,
        }
    }
}

#[cfg(feature = "parakeet")]
#[derive(Clone, Debug, ValueEnum)]
enum ParakeetDeviceArg {
    Cpu,
    #[cfg(feature = "nvidia")]
    Cuda,
    #[cfg(feature = "apple")]
    Coreml,
    #[cfg(feature = "directml")]
    Directml,
    #[cfg(feature = "amd")]
    Rocm,
}

#[cfg(feature = "parakeet")]
impl From<ParakeetDeviceArg> for ParakeetDevice {
    fn from(value: ParakeetDeviceArg) -> Self {
        match value {
            ParakeetDeviceArg::Cpu => ParakeetDevice::Cpu,
            #[cfg(feature = "nvidia")]
            ParakeetDeviceArg::Cuda => ParakeetDevice::Cuda,
            #[cfg(feature = "apple")]
            ParakeetDeviceArg::Coreml => ParakeetDevice::CoreML,
            #[cfg(feature = "directml")]
            ParakeetDeviceArg::Directml => ParakeetDevice::DirectML,
            #[cfg(feature = "amd")]
            ParakeetDeviceArg::Rocm => ParakeetDevice::ROCm,
        }
    }
}

#[derive(Args, Clone)]
struct ClientArgs {
    /// List available audio input devices
    #[arg(long)]
    list_devices: bool,

    /// Select audio input device by index
    #[arg(long)]
    device: Option<usize>,

    /// Override transcription server WebSocket URL
    #[arg(long)]
    server: Option<String>,

    /// Print final output with per-word timestamps
    #[arg(long, default_value_t = false)]
    timestamps: bool,

    /// Show verbose metadata output (debug messages)
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Enable VAD with silence timeout in seconds (auto-terminate after silence)
    #[arg(long)]
    vad: Option<f64>,

    /// Set language for transcription (e.g., "de", "ja", "it", "es", "pt")
    #[arg(long)]
    lang: Option<String>,

    /// Path to audio file to transcribe (instead of live capture). Use '-' to read from stdin
    #[arg(long, short = 'f')]
    file: Option<String>,
}

#[derive(Args, Clone)]
struct ServerStartArgs {
    /// Address to bind the server to (default: 0.0.0.0:<config port>)
    #[arg(long)]
    bind: Option<String>,

    /// Select default STT engine (kyutai or parakeet)
    #[arg(long, value_enum, default_value = "kyutai")]
    engine: EngineArg,

    /// Hugging Face repository containing the Kyutai model
    #[arg(long, default_value = "kyutai/stt-1b-en_fr-candle")]
    hf_repo: String,

    /// Force CPU execution instead of GPU/Metal
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Emit word-level timestamps in server events
    #[arg(long, default_value_t = false)]
    timestamps: bool,

    /// Enable voice-activity detection in the server session
    #[arg(long, default_value_t = false)]
    vad: bool,

    /// Maximum number of concurrent streaming sessions handled in parallel
    #[arg(long, default_value_t = 8)]
    max_sessions: usize,

    /// Force-enable Whisper enhancements (requires --features whisper)
    #[cfg(feature = "whisper")]
    #[arg(long, default_value_t = false)]
    whisper: bool,

    /// Parakeet Hugging Face repository (requires --features parakeet)
    #[cfg(feature = "parakeet")]
    #[arg(long, default_value = "istupakov/parakeet-tdt-0.6b-v3-onnx")]
    parakeet_repo: String,

    /// Optional Parakeet model directory override
    #[cfg(feature = "parakeet")]
    #[arg(long)]
    parakeet_model_dir: Option<String>,

    /// Execution device for Parakeet (compile-time providers only)
    #[cfg(feature = "parakeet")]
    #[arg(long, value_enum)]
    parakeet_device: Option<ParakeetDeviceArg>,

    /// Chunk duration for Parakeet streaming (seconds)
    #[cfg(feature = "parakeet")]
    #[arg(long, default_value_t = 3.0)]
    parakeet_chunk_seconds: f32,

    /// Overlap duration between Parakeet chunks (seconds)
    #[cfg(feature = "parakeet")]
    #[arg(long, default_value_t = 1.0)]
    parakeet_overlap_seconds: f32,

    /// Noise gate RMS for Parakeet when VAD is off (set 0 to disable)
    #[cfg(feature = "parakeet")]
    #[arg(long, default_value_t = 0.0015)]
    parakeet_noise_gate_rms: f32,
}

impl Default for ServerStartArgs {
    fn default() -> Self {
        Self {
            bind: None,
            engine: EngineArg::Kyutai,
            hf_repo: "kyutai/stt-1b-en_fr-candle".to_string(),
            cpu: false,
            timestamps: false,
            vad: false,
            max_sessions: 8,
            #[cfg(feature = "whisper")]
            whisper: false,
            #[cfg(feature = "parakeet")]
            parakeet_repo: "istupakov/parakeet-tdt-0.6b-v3-onnx".to_string(),
            #[cfg(feature = "parakeet")]
            parakeet_model_dir: None,
            #[cfg(feature = "parakeet")]
            parakeet_device: None,
            #[cfg(feature = "parakeet")]
            parakeet_chunk_seconds: 3.0,
            #[cfg(feature = "parakeet")]
            parakeet_overlap_seconds: 1.0,
            #[cfg(feature = "parakeet")]
            parakeet_noise_gate_rms: 0.0015,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let Cli { command, client } = cli;

    match command {
        Some(Commands::Server(server_cmd)) => {
            handle_server_command(server_cmd).await?;
            return Ok(());
        }
        Some(Commands::Dictation(dictation_cmd)) => {
            handle_dictation_command(dictation_cmd)?;
            return Ok(());
        }
        None => {}
    }

    run_client(client).await
}

async fn handle_server_command(command: ServerCommand) -> Result<()> {
    match command {
        ServerCommand::Start(args) => start_server(args),
        ServerCommand::Stop => stop_server(),
        ServerCommand::Status => check_server_status(),
        ServerCommand::Run(args) => run_server(args).await,
    }
}

fn handle_dictation_command(command: DictationCommand) -> Result<()> {
    match command {
        DictationCommand::Start(args) => start_dictation(&args),
        DictationCommand::Stop => stop_dictation(),
        DictationCommand::Status => check_dictation_status(),
        DictationCommand::Debug(args) => run_dictation_foreground(&args),
    }
}

fn run_dictation_foreground(args: &DictationStartArgs) -> Result<()> {
    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().context("failed to get exe directory")?;
    let dictation_bin = exe_dir.join("ears-dictation");

    let mut cmd = ProcessCommand::new(&dictation_bin);
    append_dictation_args(&mut cmd, args);

    let status = cmd.status().context("failed to run ears-dictation")?;

    if !status.success() {
        return Err(anyhow!("ears-dictation exited with error"));
    }

    Ok(())
}

fn start_server(args: ServerStartArgs) -> Result<()> {
    let pid = spawn_server_process(&args)?;

    print!("ears server starting (pid {})...", pid);
    io::stdout().flush().ok();

    let config = AppConfig::load()?;
    let ready = wait_for_server_ready(
        config.server.websocket_port,
        30,
        Duration::from_millis(500),
        true,
    );
    report_server_ready(pid, ready);
    Ok(())
}

fn check_server_status() -> Result<()> {
    let config = AppConfig::load()?;

    match server::read_pid_file()? {
        Some(pid) => {
            if server::is_process_alive(pid) {
                println!("ears server is running (pid {})", pid);

                if is_server_port_open(config.server.websocket_port) {
                    println!(
                        "server is accepting connections on port {}",
                        config.server.websocket_port
                    );
                } else {
                    println!(
                        "server process exists but not accepting connections (may be loading model)"
                    );
                }
            } else {
                println!("ears server is not running (stale pid file exists)");
            }
        }
        None => {
            println!("ears server is not running");
        }
    }

    Ok(())
}

fn stop_server() -> Result<()> {
    match server::read_pid_file()? {
        Some(pid) => {
            if server::is_process_alive(pid) {
                #[cfg(unix)]
                unsafe {
                    if kill(pid, SIGTERM) != 0 {
                        return Err(io::Error::last_os_error())
                            .context("failed to send SIGTERM to ears server");
                    }
                }

                #[cfg(not(unix))]
                {
                    return Err(anyhow!(
                        "stopping the server is currently supported only on unix platforms"
                    ));
                }

                for _ in 0..50 {
                    if !server::is_process_alive(pid) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            } else {
                println!(
                    "No running server found for PID {} (removing stale pid file).",
                    pid
                );
            }

            server::remove_pid_file()?;
            println!("ears server stopped.");
        }
        None => {
            println!("ears server is not running.");
        }
    }

    // Only stop dictation if it is connected to the local server.
    // If it is connected to a remote server, leave it running.
    if let (Some(dictation_pid), dictation_url) = read_dictation_pid_info()? {
        if server::is_process_alive(dictation_pid) {
            let is_local = match dictation_url.as_deref() {
                Some(url) => {
                    let lower = url.to_lowercase();
                    lower.contains("localhost")
                        || lower.contains("127.0.0.1")
                        || lower.contains("0.0.0.0")
                        || lower.contains("::1")
                }
                // Legacy pid file without URL -- assume local
                None => true,
            };

            if is_local {
                println!(
                    "stopping associated dictation process (pid {})...",
                    dictation_pid
                );
                #[cfg(unix)]
                unsafe {
                    let _ = kill(dictation_pid, SIGTERM);
                }

                for _ in 0..50 {
                    if !server::is_process_alive(dictation_pid) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }

                let _ = std::fs::remove_file(get_dictation_pid_file());
                println!("ears dictation stopped.");
            } else {
                println!(
                    "ears dictation (pid {}) is connected to remote server {}, leaving it running.",
                    dictation_pid,
                    dictation_url.as_deref().unwrap_or("unknown")
                );
            }
        }
    }

    Ok(())
}

async fn run_server(args: ServerStartArgs) -> Result<()> {
    let options = build_server_options(&args)?;
    server::run(options).await
}

fn append_server_args(cmd: &mut ProcessCommand, args: &ServerStartArgs) {
    if let Some(bind) = &args.bind {
        cmd.arg("--bind").arg(bind);
    }
    cmd.arg("--engine")
        .arg(args.engine.to_possible_value().unwrap().get_name());
    if args.cpu {
        cmd.arg("--cpu");
    }
    if args.timestamps {
        cmd.arg("--timestamps");
    }
    if args.vad {
        cmd.arg("--vad");
    }
    cmd.arg("--max-sessions").arg(args.max_sessions.to_string());
    if args.hf_repo != "kyutai/stt-1b-en_fr-candle" {
        cmd.arg("--hf-repo").arg(&args.hf_repo);
    }
    #[cfg(feature = "whisper")]
    if args.whisper {
        cmd.arg("--whisper");
    }
    #[cfg(feature = "parakeet")]
    {
        if args.parakeet_repo != "istupakov/parakeet-tdt-0.6b-v3-onnx" {
            cmd.arg("--parakeet-repo").arg(&args.parakeet_repo);
        }
        if let Some(dir) = &args.parakeet_model_dir {
            cmd.arg("--parakeet-model-dir").arg(dir);
        }
        if let Some(device) = args.parakeet_device.as_ref() {
            cmd.arg("--parakeet-device")
                .arg(device.to_possible_value().unwrap().get_name());
        }
        cmd.arg("--parakeet-chunk-seconds")
            .arg(args.parakeet_chunk_seconds.to_string());
        cmd.arg("--parakeet-overlap-seconds")
            .arg(args.parakeet_overlap_seconds.to_string());
        cmd.arg("--parakeet-noise-gate-rms")
            .arg(args.parakeet_noise_gate_rms.to_string());
    }
}

fn build_server_options(args: &ServerStartArgs) -> Result<server::ServerOptions> {
    let config = AppConfig::load()?;

    let bind_addr = args
        .bind
        .clone()
        .unwrap_or_else(|| format!("0.0.0.0:{}", config.server.websocket_port));

    let mut transcription = TranscriptionOptions::default();
    transcription.timestamps = args.timestamps;
    transcription.vad = args.vad;

    #[cfg(feature = "whisper")]
    {
        let whisper_enabled = if args.whisper {
            true
        } else {
            config.whisper.enabled
        };
        transcription.whisper_enabled = whisper_enabled;
        if whisper_enabled {
            transcription.whisper_model = Some(config.whisper.default_model.clone());
            transcription.whisper_quantization = Some(config.whisper.quantization.clone());
            transcription.whisper_languages = Some(config.whisper.languages.clone());
        }
    }

    Ok(server::ServerOptions {
        bind_addr,
        hf_repo: args.hf_repo.clone(),
        cpu: args.cpu,
        transcription,
        max_parallel_sessions: args.max_sessions.max(1),
        enable_listener_mode: config.server.enable_listener_mode,
        listener_tokens: config.server.listener_tokens.clone(),
        default_engine: args.engine.to_engine_kind(),
        #[cfg(feature = "parakeet")]
        parakeet_repo: args.parakeet_repo.clone(),
        #[cfg(feature = "parakeet")]
        parakeet_model_dir: args.parakeet_model_dir.clone().map(PathBuf::from),
        #[cfg(feature = "parakeet")]
        parakeet_device: args
            .parakeet_device
            .as_ref()
            .map(|d| d.clone().into())
            .unwrap_or_else(ParakeetDevice::default_for_build),
        #[cfg(feature = "parakeet")]
        parakeet_chunk_seconds: args.parakeet_chunk_seconds,
        #[cfg(feature = "parakeet")]
        parakeet_overlap_seconds: args.parakeet_overlap_seconds,
        #[cfg(feature = "parakeet")]
        parakeet_noise_gate_rms: args.parakeet_noise_gate_rms,
    })
}

async fn run_client(args: ClientArgs) -> Result<()> {
    if args.list_devices {
        return audio::list_audio_devices();
    }

    if let Some(file_path) = &args.file {
        return transcribe_file(file_path, &args).await;
    }

    let config = AppConfig::load()?;
    let server_url = args
        .server
        .clone()
        .unwrap_or_else(|| format!("ws://127.0.0.1:{}/", config.server.websocket_port));

    let should_autostart = args.server.is_none();
    let (ws_stream, _) = match connect_async(&server_url).await {
        Ok(result) => result,
        Err(err) => {
            let connect_err = anyhow!(err).context(format!("failed to connect to {}", server_url));
            if should_autostart {
                eprintln!(
                    "No local server detected at {}. Starting one now...",
                    server_url
                );
                let ready = ensure_server_running(&config)?;
                if !ready {
                    eprintln!("Server is still loading, trying to connect anyway...");
                }
                connect_async(&server_url)
                    .await
                    .map_err(|err| anyhow!(err))
                    .with_context(|| format!("failed to connect to {}", server_url))?
            } else {
                return Err(connect_err);
            }
        }
    };

    if atty::is(atty::Stream::Stdout) {
        eprintln!("Tip: for system-wide dictation, run `ears dictation start`.");
    }

    let (audio_tx, audio_rx) = unbounded::<Vec<f32>>();
    let device_index = args.device;
    if args.verbose {
        eprintln!("Connected to server at {}", server_url);
    }
    let (mut ws_writer, mut ws_reader) = ws_stream.split();

    let (writer_tx, mut writer_rx) = mpsc::unbounded_channel::<WriterCommand>();

    let lang_to_send = args.lang.clone();
    let writer_handle = tokio::spawn(async move {
        if let Some(lang) = lang_to_send {
            let set_lang_cmd = json!({ "type": "setlanguage", "lang": lang }).to_string();
            if ws_writer.send(Message::text(set_lang_cmd)).await.is_err() {
                eprintln!("Failed to send language change command");
            }
        }

        let mut should_close = false;
        while let Some(cmd) = writer_rx.recv().await {
            match cmd {
                WriterCommand::Audio(bytes) => {
                    if should_close {
                        continue;
                    }
                    if ws_writer.send(Message::binary(bytes)).await.is_err() {
                        break;
                    }
                }
                WriterCommand::Stop => {
                    let _ = ws_writer
                        .send(Message::text(json!({ "type": "stop" }).to_string()))
                        .await;
                    should_close = true;
                }
                WriterCommand::Close => {
                    break;
                }
            }
        }
        let _ = ws_writer.close().await;
    });

    let audio_writer = writer_tx.clone();
    thread::spawn(move || {
        while let Ok(chunk) = audio_rx.recv() {
            if audio_writer
                .send(WriterCommand::Audio(encode_chunk(&chunk)))
                .is_err()
            {
                break;
            }
        }
    });

    thread::spawn(move || {
        if let Err(err) = audio::start_audio_capture(audio_tx, device_index) {
            eprintln!("Audio capture error: {err}");
        }
    });

    let mut printed_live = false;
    let mut final_result: Option<(String, Vec<WordTimestamp>)> = None;
    let vad_timeout = args.vad.map(|seconds| Duration::from_secs_f64(seconds));
    let (silence_tx, mut silence_rx) = mpsc::unbounded_channel::<()>();
    let mut vad_timeout_triggered = false;
    let mut stop_requested = false;

    'client: loop {
        let timeout_future = async {
            if vad_timeout.is_some() {
                silence_rx.recv().await
            } else {
                std::future::pending().await
            }
        };

        let final_timeout = if stop_requested {
            tokio::time::sleep(Duration::from_secs(5))
        } else {
            tokio::time::sleep(Duration::from_secs(3600))
        };

        tokio::select! {
            _ = final_timeout => {
                if stop_requested {
                    eprintln!("Timeout waiting for Final message from server");
                    let _ = writer_tx.send(WriterCommand::Close);
                    break;
                }
            }
            _ = tokio::signal::ctrl_c() => {
                if stop_requested {
                    if args.verbose {
                        eprintln!("Force quit");
                    }
                    let _ = writer_tx.send(WriterCommand::Close);
                    break;
                }
                stop_requested = true;
                let _ = writer_tx.send(WriterCommand::Stop);
            }
            _ = timeout_future => {
                if args.verbose {
                    eprintln!("VAD timeout reached, stopping...");
                }
                vad_timeout_triggered = true;
                stop_requested = true;
                let _ = writer_tx.send(WriterCommand::Stop);
            }
            maybe_msg = ws_reader.next() => {
                match maybe_msg {
                    Some(Ok(Message::Text(payload))) => {
                        if args.verbose {
                            eprintln!("Received message: {}", payload);
                        }
                        if let Ok(event) = serde_json::from_str::<WebSocketMessage>(&payload) {
                            match event {
                                WebSocketMessage::Word { word, end_time, .. } => {
                                    if let Some(timeout_duration) = vad_timeout {
                                        let silence_notifier = silence_tx.clone();
                                        tokio::spawn(async move {
                                            tokio::time::sleep(timeout_duration).await;
                                            let _ = silence_notifier.send(());
                                        });
                                    }
                                    if !args.timestamps && end_time.is_none() {
                                        print!(" {}", word);
                                        io::stdout().flush().ok();
                                        printed_live = true;
                                    }
                                }
                                WebSocketMessage::Final { text, words } => {
                                    final_result = Some((text, words));
                                    let _ = writer_tx.send(WriterCommand::Close);
                                    break 'client;
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        if args.verbose {
                            eprintln!("WebSocket connection closed");
                        }
                        let _ = writer_tx.send(WriterCommand::Close);
                        break 'client;
                    }
                    _ => {}
                }
            }
        }
    }

    let _ = writer_tx.send(WriterCommand::Stop);
    let _ = writer_handle.await;

    if let Some((text, words)) = final_result {
        if args.timestamps {
            for word in words {
                if let Some(end_time) = word.end_time {
                    println!("[{:.2}-{:.2}] {}", word.start_time, end_time, word.word);
                } else {
                    println!("[{:.2}-     ] {}", word.start_time, word.word);
                }
            }
        } else {
            if printed_live {
                println!();
            }
            println!("{text}");
        }
    } else {
        if vad_timeout_triggered {
            eprintln!("\nVAD timeout reached");
        } else {
            eprintln!("\nNo transcription received from server.");
        }
    }

    Ok(())
}

enum WriterCommand {
    Audio(Vec<u8>),
    Stop,
    Close,
}

async fn transcribe_file(file_path: &str, args: &ClientArgs) -> Result<()> {
    use ears::kaudio;
    use std::io::Read;

    let (pcm, sample_rate) = if file_path == "-" {
        eprintln!("Reading audio from stdin...");
        let mut buffer = Vec::new();
        io::stdin()
            .read_to_end(&mut buffer)
            .context("Failed to read from stdin")?;

        let temp_file = std::env::temp_dir().join("ears_stdin_audio");
        std::fs::write(&temp_file, &buffer).context("Failed to write temp file")?;

        let result = kaudio::pcm_decode(&temp_file).context("Failed to decode audio from stdin");

        let _ = std::fs::remove_file(&temp_file);
        result?
    } else {
        eprintln!("Loading audio file: {}", file_path);
        kaudio::pcm_decode(file_path)
            .with_context(|| format!("Failed to load audio file: {}", file_path))?
    };

    eprintln!(
        "Sample rate: {}, samples: {}, duration: {:.2}s",
        sample_rate,
        pcm.len(),
        pcm.len() as f64 / sample_rate as f64
    );

    let pcm = if sample_rate != 24_000 {
        eprintln!("Resampling from {}Hz to 24000Hz", sample_rate);
        kaudio::resample(&pcm, sample_rate as usize, 24_000)?
    } else {
        pcm
    };

    let config = AppConfig::load()?;
    let server_url = args
        .server
        .clone()
        .unwrap_or_else(|| format!("ws://127.0.0.1:{}/", config.server.websocket_port));

    let (ws_stream, _) = connect_async(&server_url)
        .await
        .with_context(|| format!("Failed to connect to {}", server_url))?;

    if args.verbose {
        eprintln!("Connected to server at {}", server_url);
    }

    let (mut ws_writer, mut ws_reader) = ws_stream.split();
    let (writer_tx, mut writer_rx) = mpsc::unbounded_channel::<WriterCommand>();

    let lang_to_send = args.lang.clone();
    let writer_handle = tokio::spawn(async move {
        if let Some(lang) = lang_to_send {
            let set_lang_cmd = json!({ "type": "setlanguage", "lang": lang }).to_string();
            if ws_writer.send(Message::text(set_lang_cmd)).await.is_err() {
                eprintln!("Failed to send language change command");
            }
        }

        while let Some(cmd) = writer_rx.recv().await {
            match cmd {
                WriterCommand::Audio(bytes) => {
                    if ws_writer.send(Message::binary(bytes)).await.is_err() {
                        break;
                    }
                }
                WriterCommand::Stop => {
                    let _ = ws_writer
                        .send(Message::text(json!({ "type": "stop" }).to_string()))
                        .await;
                }
                WriterCommand::Close => {
                    break;
                }
            }
        }
        let _ = ws_writer.close().await;
    });

    eprintln!("Streaming audio to server...");
    let chunk_size = 1920;
    for chunk in pcm.chunks(chunk_size) {
        if writer_tx
            .send(WriterCommand::Audio(encode_chunk(chunk)))
            .is_err()
        {
            break;
        }
    }

    let _ = writer_tx.send(WriterCommand::Stop);

    let mut final_result: Option<(String, Vec<WordTimestamp>)> = None;
    let timeout_duration = Duration::from_secs(10);

    let timeout = tokio::time::sleep(timeout_duration);
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            _ = &mut timeout => {
                eprintln!("Timeout waiting for transcription result");
                break;
            }
            maybe_msg = ws_reader.next() => {
                match maybe_msg {
                    Some(Ok(Message::Text(payload))) => {
                        if args.verbose {
                            eprintln!("Received: {}", payload);
                        }
                        if let Ok(event) = serde_json::from_str::<WebSocketMessage>(&payload) {
                            match event {
                                WebSocketMessage::Word { word, start_time, end_time } => {
                                    if args.timestamps {
                                        if let Some(end) = end_time {
                                            println!("[{:.2}-{:.2}] {}", start_time, end, word);
                                        } else {
                                            println!("[{:.2}-     ] {}", start_time, word);
                                        }
                                    } else if !args.verbose {
                                        print!(" {}", word);
                                        io::stdout().flush().ok();
                                    }
                                }
                                WebSocketMessage::Final { text, words } => {
                                    final_result = Some((text, words));
                                    let _ = writer_tx.send(WriterCommand::Close);
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        if args.verbose {
                            eprintln!("WebSocket closed");
                        }
                        let _ = writer_tx.send(WriterCommand::Close);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    let _ = writer_handle.await;

    if let Some((text, words)) = final_result {
        if !args.timestamps {
            println!("\n{}", text);
        }
        if args.verbose {
            eprintln!("Transcription complete: {} words", words.len());
        }
    } else {
        eprintln!("No transcription received from server");
    }

    Ok(())
}

fn encode_chunk(chunk: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(chunk.len() * 4);
    for sample in chunk {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn spawn_server_process(args: &ServerStartArgs) -> Result<u32> {
    if let Some(pid) = server::read_pid_file()? {
        if server::is_process_alive(pid) {
            return Err(anyhow!("ears server already running (pid {})", pid));
        }
        server::remove_pid_file()?;
    }

    let mut cmd = ProcessCommand::new(std::env::current_exe()?);
    cmd.arg("server").arg("run");
    append_server_args(&mut cmd, args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());

    let child = cmd.spawn().context("failed to spawn ears server process")?;
    Ok(child.id())
}

fn ensure_server_running(config: &AppConfig) -> Result<bool> {
    if let Some(pid) = server::read_pid_file()? {
        if server::is_process_alive(pid) {
            let ready = wait_for_server_ready(
                config.server.websocket_port,
                30,
                Duration::from_millis(500),
                false,
            );
            return Ok(ready);
        }
        server::remove_pid_file()?;
    }

    let pid = spawn_server_process(&ServerStartArgs::default())?;
    print!("ears server starting (pid {})...", pid);
    io::stdout().flush().ok();

    let ready = wait_for_server_ready(
        config.server.websocket_port,
        30,
        Duration::from_millis(500),
        true,
    );
    report_server_ready(pid, ready);

    Ok(ready)
}

fn report_server_ready(pid: u32, ready: bool) {
    if ready {
        println!("\rears server started (pid {}) and ready", pid);
    } else {
        println!("\rears server started (pid {}) but not yet ready", pid);
        println!("server may still be loading the model - try connecting in a few seconds");
    }
}

fn is_server_port_open(port: u16) -> bool {
    std::net::TcpStream::connect(("127.0.0.1", port)).is_ok()
}

fn wait_for_server_ready(
    port: u16,
    max_attempts: usize,
    poll_interval: Duration,
    show_progress: bool,
) -> bool {
    for attempt in 0..max_attempts {
        if is_server_port_open(port) {
            return true;
        }
        if show_progress && attempt % 2 == 1 {
            print!(".");
            io::stdout().flush().ok();
        }
        thread::sleep(poll_interval);
    }
    false
}

fn get_dictation_pid_file() -> std::path::PathBuf {
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
    state_dir.join("ears").join("dictation.pid")
}

fn get_dictation_log_file() -> std::path::PathBuf {
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
    state_dir.join("ears").join("dictation.log")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn wait_for_server_ready_detects_listener() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let port = listener.local_addr().expect("local addr").port();
        assert!(wait_for_server_ready(
            port,
            10,
            Duration::from_millis(10),
            false
        ));
    }

    #[test]
    fn wait_for_server_ready_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);
        assert!(!wait_for_server_ready(
            port,
            5,
            Duration::from_millis(10),
            false
        ));
    }

    #[test]
    fn server_start_args_defaults_match_expected() {
        let args = ServerStartArgs::default();
        match args.engine {
            EngineArg::Kyutai => {}
            #[cfg(feature = "parakeet")]
            EngineArg::Parakeet => panic!("unexpected default engine"),
        }
        assert_eq!(args.hf_repo, "kyutai/stt-1b-en_fr-candle");
        assert_eq!(args.max_sessions, 8);
        assert!(!args.cpu);
        assert!(!args.timestamps);
        assert!(!args.vad);
    }
}

/// Read the dictation PID file. Returns (pid, optional server URL).
/// The file format is either just a PID (legacy) or PID\nURL.
fn read_dictation_pid_info() -> Result<(Option<i32>, Option<String>)> {
    let pid_file = get_dictation_pid_file();
    if !pid_file.exists() {
        return Ok((None, None));
    }
    let contents = std::fs::read_to_string(&pid_file)?;
    let mut lines = contents.lines();
    let pid = lines.next().and_then(|s| s.trim().parse::<i32>().ok());
    let url = lines
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    Ok((pid, url))
}

fn read_dictation_pid() -> Result<Option<i32>> {
    let (pid, _) = read_dictation_pid_info()?;
    Ok(pid)
}

fn start_dictation(args: &DictationStartArgs) -> Result<()> {
    if let Some(pid) = read_dictation_pid()? {
        if server::is_process_alive(pid) {
            return Err(anyhow!("ears dictation already running (pid {})", pid));
        }
        let _ = std::fs::remove_file(get_dictation_pid_file());
    }

    let config = AppConfig::load()?;
    let target = resolve_dictation_target(args.server.as_deref(), &config)?;
    let probe_host = dictation_probe_host(&target.host, target.is_local);

    if target.is_local {
        let server_running = match server::read_pid_file()? {
            Some(pid) if server::is_process_alive(pid) => {
                is_tcp_endpoint_open(&probe_host, target.port)
            }
            _ => false,
        };

        if !server_running {
            println!("ears server is not running, starting it first...");
            let _ = ensure_server_running(&config)?;
        }

        if !wait_for_tcp_endpoint_ready(&probe_host, target.port, 40, Duration::from_millis(250)) {
            return Err(anyhow!(
                "ears server is not reachable at {}:{}",
                probe_host,
                target.port
            ));
        }

        println!("ears server is ready at {}", target.url);
    } else {
        println!("checking remote dictation server at {}...", target.url);
        if !wait_for_tcp_endpoint_ready(&probe_host, target.port, 12, Duration::from_millis(250)) {
            return Err(anyhow!(
                "remote dictation server is not reachable at {}",
                target.url
            ));
        }
        println!("remote dictation server is reachable");
    }

    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().context("failed to get exe directory")?;
    let dictation_bin = exe_dir.join("ears-dictation");

    let log_path = get_dictation_log_file();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .with_context(|| {
            format!(
                "failed to open dictation log file at {}",
                log_path.display()
            )
        })?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone dictation log file handle")?;

    let mut cmd = ProcessCommand::new(&dictation_bin);
    append_dictation_args(&mut cmd, args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::from(log_file));
    cmd.stderr(Stdio::from(log_file_err));

    let mut child = cmd
        .spawn()
        .context("failed to spawn ears-dictation process")?;
    let pid = child.id();

    let startup_deadline = Instant::now()
        + if target.is_local {
            Duration::from_secs(5)
        } else {
            Duration::from_secs(3)
        };
    let required_consecutive_successes = if target.is_local { 2 } else { 3 };
    let mut consecutive_successes = 0usize;
    let mut endpoint_ready = false;

    while Instant::now() < startup_deadline {
        if let Some(status) = child
            .try_wait()
            .context("failed to check ears-dictation process status")?
        {
            return Err(anyhow!(
                "ears dictation exited early with status {}. See log at {}",
                status,
                log_path.display()
            ));
        }

        if is_tcp_endpoint_open(&probe_host, target.port) {
            consecutive_successes += 1;
            if consecutive_successes >= required_consecutive_successes {
                endpoint_ready = true;
                break;
            }
        } else {
            consecutive_successes = 0;
        }

        if target.is_local {
            match server::read_pid_file()? {
                Some(server_pid) if server::is_process_alive(server_pid) => {}
                _ => {
                    return Err(anyhow!(
                        "ears server failed to start. See log at {}",
                        log_path.display()
                    ));
                }
            }
        }

        thread::sleep(Duration::from_millis(200));
    }

    if !endpoint_ready {
        if target.is_local {
            return Err(anyhow!(
                "ears server did not become stably reachable at {}:{}",
                probe_host,
                target.port
            ));
        }
        return Err(anyhow!(
            "remote dictation server at {} became unavailable during startup",
            target.url
        ));
    }

    println!("ears dictation started (pid {})", pid);
    println!("Use keyboard shortcut to toggle pause/resume (see config)");

    Ok(())
}

fn append_dictation_args(cmd: &mut ProcessCommand, args: &DictationStartArgs) {
    if let Some(ref server) = args.server {
        cmd.arg("--server").arg(server);
    }
    if let Some(ref lang) = args.lang {
        cmd.arg("--lang").arg(lang);
    }
    if let Some(ref engine) = args.engine {
        cmd.arg("--engine").arg(engine);
    }
}

#[derive(Debug, Clone)]
struct DictationTarget {
    url: String,
    host: String,
    port: u16,
    is_local: bool,
}

fn resolve_dictation_target(
    server_arg: Option<&str>,
    config: &AppConfig,
) -> Result<DictationTarget> {
    match server_arg {
        Some(raw) if raw.starts_with("ws://") || raw.starts_with("wss://") => {
            let (host, port) = parse_ws_host_port(raw)?;
            Ok(DictationTarget {
                url: raw.to_string(),
                is_local: is_local_host(&host),
                host,
                port,
            })
        }
        Some(alias) => {
            let server_cfg = config.dictation.servers.get(alias).ok_or_else(|| {
                let mut available: Vec<_> = config.dictation.servers.keys().cloned().collect();
                available.sort();
                anyhow!(
                    "Unknown server alias '{}'. Available: {}",
                    alias,
                    available.join(", ")
                )
            })?;
            Ok(DictationTarget {
                url: server_cfg.ws_url(),
                host: server_cfg.host.clone(),
                port: server_cfg.port,
                is_local: is_local_host(&server_cfg.host),
            })
        }
        None => {
            let alias = &config.dictation.default_server;
            let server_cfg = config
                .dictation
                .servers
                .get(alias)
                .ok_or_else(|| anyhow!("Default server '{}' not found in config", alias))?;
            Ok(DictationTarget {
                url: server_cfg.ws_url(),
                host: server_cfg.host.clone(),
                port: server_cfg.port,
                is_local: is_local_host(&server_cfg.host),
            })
        }
    }
}

fn parse_ws_host_port(url: &str) -> Result<(String, u16)> {
    let uri: Uri = url
        .parse()
        .with_context(|| format!("invalid server URL: {}", url))?;

    let scheme = uri
        .scheme_str()
        .ok_or_else(|| anyhow!("missing URL scheme in {}", url))?;
    if scheme != "ws" && scheme != "wss" {
        return Err(anyhow!("unsupported URL scheme '{}' in {}", scheme, url));
    }

    let host = uri
        .host()
        .ok_or_else(|| anyhow!("missing host in {}", url))?
        .to_string();
    let port = uri
        .port_u16()
        .unwrap_or_else(|| if scheme == "wss" { 443 } else { 80 });

    Ok((host, port))
}

fn is_local_host(host: &str) -> bool {
    matches!(
        host.to_ascii_lowercase().as_str(),
        "localhost" | "127.0.0.1" | "::1" | "0.0.0.0" | "::" | "[::1]" | "[::]"
    )
}

fn dictation_probe_host(host: &str, is_local: bool) -> String {
    if !is_local {
        return host.to_string();
    }

    match host.to_ascii_lowercase().as_str() {
        "::" | "[::]" | "::1" | "[::1]" | "0.0.0.0" => "127.0.0.1".to_string(),
        _ => host.to_string(),
    }
}

fn is_tcp_endpoint_open(host: &str, port: u16) -> bool {
    let timeout = Duration::from_millis(800);
    let addr = format!("{}:{}", host, port);
    let addrs = match addr.to_socket_addrs() {
        Ok(addrs) => addrs,
        Err(_) => return false,
    };

    addrs
        .into_iter()
        .any(|sock_addr| TcpStream::connect_timeout(&sock_addr, timeout).is_ok())
}

fn wait_for_tcp_endpoint_ready(
    host: &str,
    port: u16,
    max_attempts: usize,
    poll_interval: Duration,
) -> bool {
    for _ in 0..max_attempts {
        if is_tcp_endpoint_open(host, port) {
            return true;
        }
        thread::sleep(poll_interval);
    }
    false
}

fn stop_dictation() -> Result<()> {
    match read_dictation_pid()? {
        Some(pid) => {
            if server::is_process_alive(pid) {
                #[cfg(unix)]
                unsafe {
                    // Send SIGINT first so ears-dictation can run graceful
                    // shutdown logic (including stop hooks). If it does not
                    // exit quickly, fall back to SIGTERM.
                    if kill(pid, SIGINT) != 0 {
                        return Err(io::Error::last_os_error())
                            .context("failed to send SIGINT to ears-dictation");
                    }
                }

                #[cfg(not(unix))]
                {
                    return Err(anyhow!(
                        "stopping dictation is currently supported only on unix platforms"
                    ));
                }

                let mut stopped = false;
                for _ in 0..20 {
                    if !server::is_process_alive(pid) {
                        stopped = true;
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }

                if !stopped {
                    #[cfg(unix)]
                    unsafe {
                        if kill(pid, SIGTERM) != 0 {
                            return Err(io::Error::last_os_error())
                                .context("failed to send SIGTERM to ears-dictation");
                        }
                    }

                    for _ in 0..30 {
                        if !server::is_process_alive(pid) {
                            break;
                        }
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            } else {
                println!(
                    "No running dictation process found for PID {} (removing stale pid file).",
                    pid
                );
            }

            let _ = std::fs::remove_file(get_dictation_pid_file());
            println!("ears dictation stopped.");
        }
        None => {
            println!("ears dictation is not running.");
        }
    }

    Ok(())
}

fn check_dictation_status() -> Result<()> {
    match read_dictation_pid()? {
        Some(pid) => {
            if server::is_process_alive(pid) {
                println!("ears dictation is running (pid {})", pid);
            } else {
                println!("ears dictation is not running (stale pid file exists)");
            }
        }
        None => {
            println!("ears dictation is not running");
        }
    }

    Ok(())
}
