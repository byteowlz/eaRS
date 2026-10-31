mod fsutil;
mod launcher;
mod payload;
mod settings;
mod store;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "Per-user eaRS payload installer; execution requires explicit launch"
)]
struct Cli {
    /// Override the per-user installation prefix.
    #[arg(long, global = true)]
    prefix: Option<PathBuf>,
    /// Explicit TOML config (above environment, below --prefix).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Install {
        #[arg(long)]
        package: PathBuf,
        /// Trusted out-of-band SHA-256 of the exact manifest.json bytes.
        #[arg(long)]
        manifest_sha256: String,
    },
    Rollback,
    Uninstall,
    Status,
    /// Launch one whitelisted program from the verified active receipt.
    Launch {
        #[arg(value_enum)]
        program: launcher::Program,
        /// Literal program arguments; accepted only after --.
        #[arg(last = true)]
        args: Vec<OsString>,
    },
}

fn execute(cli: Cli) -> Result<serde_json::Value> {
    let prefix = settings::resolve(cli.prefix, cli.config)?;
    match cli.command {
        Command::Install {
            package,
            manifest_sha256,
        } => {
            // Validate source before claiming/creating a destination prefix.
            let payload = payload::Payload::load(&package, &manifest_sha256)?;
            store::Store::open(&prefix, true)?
                .expect("create returns store")
                .install(&package, payload)
        }
        Command::Launch { program, args } => {
            let Some(store) = store::Store::open(&prefix, false)? else {
                bail!("no managed active installation to launch");
            };
            store.launch(program, &args)
        }
        Command::Rollback => {
            let Some(mut store) = store::Store::open(&prefix, false)? else {
                bail!("no managed installation to roll back");
            };
            store.rollback()
        }
        Command::Uninstall => match store::Store::open(&prefix, false)? {
            Some(mut store) => store.uninstall(),
            None => Ok(store::absent_status(&prefix, "uninstall")),
        },
        Command::Status => match store::Store::open(&prefix, false)? {
            Some(store) => store.status("status", false),
            None => Ok(store::absent_status(&prefix, "status")),
        },
    }
}

fn main() {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            let _ = error.print();
            return;
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({"ok": false, "error": error.to_string()})
            );
            std::process::exit(2);
        }
    };
    match execute(cli) {
        Ok(value) => println!("{value}"),
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({"ok": false, "error": format!("{error:#}")})
            );
            std::process::exit(1);
        }
    }
}
