use anyhow::Result;
use clap::Parser;
use ears::{TranscriptionOptions, config::AppConfig, server};
#[cfg(any(feature = "parakeet-rs", feature = "transcribe-cpp"))]
use std::path::PathBuf;

// Use the server's engine registry for CLI choices as well.
type EngineArg = ears::server::EngineKind;

#[derive(Debug, Parser)]
#[command(
    name = "ears-server",
    about = "Standalone transcription server for eaRS"
)]
struct Args {
    /// Address to bind the transcription server to (default: <config host>:<config port>)
    #[arg(long)]
    bind: Option<String>,

    /// Select default STT engine
    #[arg(long, value_enum, default_value = "kyutai")]
    engine: EngineArg,

    /// Hugging Face repository for the speech-to-text model
    #[arg(long, default_value = "kyutai/stt-1b-en_fr-candle")]
    hf_repo: String,

    /// Force CPU execution instead of GPU/Metal
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Enable word-level timestamps in transcriptions
    #[arg(long, default_value_t = false)]
    timestamps: bool,

    /// Enable voice-activity detection during transcription
    #[arg(long, default_value_t = false)]
    vad: bool,

    /// Disable the engine-agnostic ingress boundary VAD that emits
    /// `Speech { active }` events. Boundary detection is on by default.
    #[arg(long, default_value_t = false)]
    no_boundary_vad: bool,

    /// Silence before closing an utterance and flushing its trailing word.
    #[arg(long, default_value_t = 300)]
    boundary_vad_hangover_ms: usize,

    /// Maximum number of concurrent streaming sessions handled in parallel
    #[arg(long, default_value_t = 8)]
    max_sessions: usize,

    /// Also load the Kyutai model when another engine is the default, so
    /// clients can switch to it. Kyutai is always loaded when it is the default.
    #[arg(long, default_value_t = false)]
    with_kyutai: bool,

    /// Log transcriptions from language injection audio (for debugging)
    #[arg(long, default_value_t = false)]
    verbose_injection: bool,

    /// Local Moonshine Tiny Streaming ORT bundle, including a fine-tuned
    /// decoder export. Unset downloads the public German base into HF_HOME/moonshine.
    #[cfg(feature = "moonshine")]
    #[arg(long)]
    moonshine_model: Option<String>,

    /// parakeet-rs (Nemotron) model directory. Enables `--engine parakeet-rs`.
    #[cfg(feature = "parakeet-rs")]
    #[arg(long)]
    parakeet_rs_model: Option<String>,

    /// Target language for the multilingual parakeet-rs model (e.g. "de", "auto").
    #[cfg(feature = "parakeet-rs")]
    #[arg(long)]
    parakeet_rs_lang: Option<String>,

    /// transcribe.cpp streaming GGUF model path. Enables `--engine transcribe-cpp`.
    #[cfg(feature = "transcribe-cpp")]
    #[arg(long)]
    transcribe_cpp_model: Option<String>,

    /// Language hint for multilingual transcribe.cpp models (e.g. "de", "auto").
    #[cfg(feature = "transcribe-cpp")]
    #[arg(long)]
    transcribe_cpp_lang: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let options = build_server_options(&args)?;
    server::run(options).await
}

fn build_server_options(args: &Args) -> Result<server::ServerOptions> {
    let config = AppConfig::load()?;

    let bind_addr = args
        .bind
        .clone()
        .unwrap_or_else(|| format!("{}:{}", config.server.host, config.server.websocket_port));

    let mut transcription = TranscriptionOptions::default();
    transcription.timestamps = args.timestamps;
    transcription.vad = args.vad;
    transcription.boundary_vad = !args.no_boundary_vad;
    transcription.boundary_vad_hangover_ms = args.boundary_vad_hangover_ms;
    transcription.verbose_injection = args.verbose_injection;

    Ok(server::ServerOptions {
        bind_addr,
        hf_repo: args.hf_repo.clone(),
        cpu: args.cpu,
        transcription,
        max_parallel_sessions: args.max_sessions.max(1),
        enable_listener_mode: config.server.enable_listener_mode,
        listener_tokens: config.server.listener_tokens.clone(),
        default_engine: args.engine,
        with_kyutai: args.with_kyutai,
        #[cfg(feature = "moonshine")]
        moonshine_model_dir: args.moonshine_model.clone()
            .or_else(|| config.moonshine.model_dir.clone())
            .map(|path| ears::replacement::expand_tilde(&path)),
        #[cfg(feature = "parakeet-rs")]
        parakeet_rs_model_dir: args
            .parakeet_rs_model
            .clone()
            .or_else(|| config.parakeet_rs.model_dir.clone())
            .map(PathBuf::from),
        #[cfg(feature = "parakeet-rs")]
        parakeet_rs_lang: args
            .parakeet_rs_lang
            .clone()
            .or_else(|| Some(config.parakeet_rs.language.clone())),
        #[cfg(feature = "transcribe-cpp")]
        transcribe_cpp_model: args
            .transcribe_cpp_model
            .clone()
            .or_else(|| config.transcribe_cpp.model_path.clone())
            .map(PathBuf::from),
        #[cfg(feature = "transcribe-cpp")]
        transcribe_cpp_lang: args
            .transcribe_cpp_lang
            .clone()
            .or_else(|| config.transcribe_cpp.language.clone()),
    })
}
