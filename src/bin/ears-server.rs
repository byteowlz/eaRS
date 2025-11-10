use anyhow::Result;
use clap::Parser;
use ears::{TranscriptionOptions, config::AppConfig, server};

#[derive(Debug, Parser)]
#[command(
    name = "ears-server",
    about = "Standalone transcription server for eaRS"
)]
struct Args {
    /// Address to bind the transcription server to (default: <config host>:<config port>)
    #[arg(long)]
    bind: Option<String>,

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

    /// Maximum number of concurrent streaming sessions handled in parallel
    #[arg(long, default_value_t = 8)]
    max_sessions: usize,

    /// Force Whisper enhancement, overriding config defaults (requires `--features whisper`)
    #[cfg(feature = "whisper")]
    #[arg(long, default_value_t = false)]
    whisper: bool,

    /// Log transcriptions from language injection audio (for debugging)
    #[arg(long, default_value_t = false)]
    verbose_injection: bool,
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
    transcription.verbose_injection = args.verbose_injection;

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
    })
}
