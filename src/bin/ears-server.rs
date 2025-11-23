use anyhow::Result;
use clap::{Parser, ValueEnum};
use ears::{TranscriptionOptions, config::AppConfig, server};
use ears::server::EngineKind;
#[cfg(feature = "parakeet")]
use ears::server::ParakeetDevice;
use std::path::PathBuf;

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

#[derive(Debug, Parser)]
#[command(
    name = "ears-server",
    about = "Standalone transcription server for eaRS"
)]
struct Args {
    /// Address to bind the transcription server to (default: <config host>:<config port>)
    #[arg(long)]
    bind: Option<String>,

    /// Select default STT engine (kyutai or parakeet)
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

    /// Parakeet Hugging Face repository (requires `--features parakeet`)
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
    })
}
