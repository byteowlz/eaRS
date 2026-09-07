//! Headless dictation client for eaRS.
//!
//! All behaviour lives in [`ears::dictation`]; this binary only parses
//! arguments, loads the config and runs the engine in the foreground.

use anyhow::Result;
use clap::Parser;
use ears::config::AppConfig;
use ears::dictation::DictationEvent;
use ears::dictation::DictationOptions;
use ears::dictation::InsertionMode;

#[derive(Clone, Debug, clap::ValueEnum)]
enum EngineArg {
    Kyutai,
    #[cfg(feature = "parakeet")]
    Parakeet,
}

impl EngineArg {
    fn as_str(&self) -> &'static str {
        match self {
            EngineArg::Kyutai => "kyutai",
            #[cfg(feature = "parakeet")]
            EngineArg::Parakeet => "parakeet",
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "ears-dictation", about = "Dictation client for eaRS")]
struct Args {
    #[arg(
        long,
        help = "Set the transcription language (e.g., 'en', 'de', 'es', 'fr', 'ja')"
    )]
    lang: Option<String>,

    #[arg(
        long,
        value_enum,
        help = "Select transcription engine (kyutai|parakeet)"
    )]
    engine: Option<EngineArg>,

    #[arg(
        short,
        long,
        help = "Server alias from config (e.g., 'local', 'remote') or full WebSocket URL (ws://host:port)"
    )]
    server: Option<String>,

    #[arg(
        long,
        value_parser = ["pcm", "opus"],
        help = "Audio transport codec (pcm|opus); overrides dictation.codec"
    )]
    codec: Option<String>,

    #[arg(
        long,
        value_name = "MODE",
        help = "Where text goes: insert_at_cursor (default) | clipboard | send_as_prompt"
    )]
    insertion_mode: Option<String>,

    #[arg(
        long,
        help = "Discard pending words and pause when Escape is pressed (overrides hotkeys.escape_cancels)"
    )]
    escape_cancels: bool,

    #[arg(long, help = "Do not evaluate context profiles")]
    no_profiles: bool,

    #[arg(long, help = "Suppress per-word diagnostic output")]
    quiet: bool,

    #[arg(long, help = "List all configured servers and exit")]
    list_servers: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = AppConfig::load().unwrap_or_default();

    if args.list_servers {
        println!("Configured dictation servers:");
        println!("  {:12} {:30} {}", "ALIAS", "URL", "DESCRIPTION");
        println!("  {:12} {:30} {}", "-----", "---", "-----------");
        let mut servers: Vec<_> = config.dictation.list_servers();
        servers.sort_by_key(|(alias, _)| *alias);
        for (alias, server) in servers {
            let is_default = alias == config.dictation.default_server;
            let marker = if is_default { "*" } else { " " };
            let desc = server.description.as_deref().unwrap_or("");
            println!("{} {:12} {:30} {}", marker, alias, server.ws_url(), desc);
        }
        println!("\n* = default server");
        return Ok(());
    }

    let insertion_mode = match args.insertion_mode.as_deref() {
        Some(mode) => mode.parse::<InsertionMode>()?,
        None => InsertionMode::InsertAtCursor,
    };

    let options = DictationOptions {
        server: args.server.clone(),
        lang: args.lang.clone(),
        engine: args.engine.as_ref().map(|e| e.as_str().to_string()),
        codec: args.codec.clone(),
        insertion_mode,
        escape_cancels: args.escape_cancels.then_some(true),
        hotkeys: None,
        auto_start_server: true,
        write_pid_file: true,
        profiles: !args.no_profiles,
        verbose: !args.quiet,
    };

    eprintln!("Press Ctrl+C to stop\n");
    ears::dictation::run(config, options, |event| {
        if let DictationEvent::PromptReady { text } = event {
            println!("{text}");
        }
    })
    .await
}
