use anyhow::Result;
use clap::{Parser, ValueEnum};
use ears::server::EngineKind;
#[cfg(feature = "parakeet")]
use ears::server::ParakeetDevice;
use ears::{TranscriptionOptions, config::AppConfig, server};
#[cfg(any(feature = "parakeet", feature = "sherpa", feature = "parakeet-rs"))]
use std::path::PathBuf;

#[derive(Clone, Debug, ValueEnum)]
enum EngineArg {
    Kyutai,
    #[cfg(feature = "parakeet")]
    Parakeet,
    #[cfg(feature = "sherpa")]
    Sherpa,
}

impl EngineArg {
    fn to_engine_kind(&self) -> EngineKind {
        match self {
            EngineArg::Kyutai => EngineKind::Kyutai,
            #[cfg(feature = "parakeet")]
            EngineArg::Parakeet => EngineKind::Parakeet,
            #[cfg(feature = "sherpa")]
            EngineArg::Sherpa => EngineKind::Sherpa,
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

    /// Noise gate RMS for Parakeet when VAD is off (set 0 to disable)
    #[cfg(feature = "parakeet")]
    #[arg(long, default_value_t = 0.0015)]
    parakeet_noise_gate_rms: f32,

    /// Path to a sherpa-onnx streaming model directory (encoder/decoder/joiner/tokens).
    /// Loaded under language code "default". For multilingual setups use
    /// repeated `--sherpa-model LANG=PATH` instead.
    #[cfg(feature = "sherpa")]
    #[arg(long)]
    sherpa_model_dir: Option<String>,

    /// Per-language sherpa-onnx model in `LANG=PATH` form (repeatable).
    /// Example: `--sherpa-model en=/models/en --sherpa-model de=/models/de`.
    /// The first one becomes the default. Clients pick a language via
    /// `set_language` over the WebSocket protocol.
    #[cfg(feature = "sherpa")]
    #[arg(long = "sherpa-model", value_parser = parse_sherpa_model)]
    sherpa_models: Vec<(String, String)>,

    /// Number of CPU threads for sherpa-onnx inference
    #[cfg(feature = "sherpa")]
    #[arg(long, default_value_t = 1)]
    sherpa_num_threads: i32,

    /// Provider for sherpa-onnx (cpu, cuda, coreml, ...)
    #[cfg(feature = "sherpa")]
    #[arg(long, default_value = "cpu")]
    sherpa_provider: String,

    /// parakeet-rs (Nemotron) model directory. Enables `--engine parakeet-rs`.
    #[cfg(feature = "parakeet-rs")]
    #[arg(long)]
    parakeet_rs_model: Option<String>,

    /// Target language for the multilingual parakeet-rs model (e.g. "de", "auto").
    #[cfg(feature = "parakeet-rs")]
    #[arg(long)]
    parakeet_rs_lang: Option<String>,
}

#[cfg(feature = "sherpa")]
fn parse_sherpa_model(s: &str) -> Result<(String, String), String> {
    let (lang, path) = s.split_once('=').ok_or_else(|| {
        format!("expected `LANG=PATH`, got `{s}` (e.g., `--sherpa-model en=/models/en`)")
    })?;
    let lang = lang.trim();
    let path = path.trim();
    if lang.is_empty() || path.is_empty() {
        return Err(format!("LANG and PATH must be non-empty in `{s}`"));
    }
    Ok((lang.to_string(), path.to_string()))
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
        #[cfg(feature = "parakeet")]
        parakeet_noise_gate_rms: args.parakeet_noise_gate_rms,
        #[cfg(feature = "sherpa")]
        sherpa_models: build_sherpa_models(args),
        #[cfg(feature = "sherpa")]
        sherpa_num_threads: args.sherpa_num_threads,
        #[cfg(feature = "sherpa")]
        sherpa_provider: args.sherpa_provider.clone(),
        #[cfg(feature = "parakeet-rs")]
        parakeet_rs_model_dir: args.parakeet_rs_model.as_ref().map(PathBuf::from),
        #[cfg(feature = "parakeet-rs")]
        parakeet_rs_lang: args.parakeet_rs_lang.clone(),
    })
}

#[cfg(feature = "sherpa")]
fn build_sherpa_models(args: &Args) -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    if let Some(dir) = args.sherpa_model_dir.clone() {
        out.push(("default".to_string(), PathBuf::from(dir)));
    }
    for (lang, path) in &args.sherpa_models {
        out.push((lang.clone(), PathBuf::from(path)));
    }
    out
}
