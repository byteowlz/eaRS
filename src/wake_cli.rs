//! `ears wake`: list published wake heads, install them, bind a phrase to an action.
//!
//! Everything a Settings UI would do is a command here first, so an agent or script can do
//! it too. Changes apply the next time dictation starts.

use anyhow::{Context, Result, bail};
use clap::{Subcommand, ValueEnum};
use ears::dictation::wake::{WakeAction, WakeConfig};
use ears::wake_models::{self, ACTIVE_HEADS_WARNING, Catalog, Store};

#[derive(Subcommand)]
pub enum WakeCommand {
    /// List heads in the published catalog (and whether they are installed or bound).
    Catalog {
        #[arg(long)]
        json: bool,
        /// Catalog URL or path (default: [dictation.wake].catalog, else the waqy Hugging Face repo).
        #[arg(long)]
        catalog: Option<String>,
    },
    /// Download heads (shared front end once), verify every sha256, record them as installed.
    Install {
        #[arg(required = true)]
        heads: Vec<String>,
        /// Take the head from this release instead of the newest one carrying it.
        #[arg(long)]
        release: Option<String>,
        #[arg(long)]
        catalog: Option<String>,
        /// Also bind every installed head to this action.
        #[arg(long, value_enum)]
        bind: Option<ActionArg>,
    },
    /// Installed heads, their actions, and how many run.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Bind an installed head to an action; only bound heads run.
    Bind {
        head: String,
        #[arg(value_enum)]
        action: ActionArg,
    },
    /// Remove a head's action (it stays installed but stops running).
    Unbind { head: String },
    /// Uninstall a head (and unbind it).
    Remove { head: String },
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ActionArg {
    /// Start dictation (the toggle hotkey while paused).
    StartDictation,
    /// Discard pending words and pause (Escape); the detector also listens during dictation.
    StopDictation,
    /// Detected and logged, nothing happens; the head does not run.
    Ignore,
}

impl From<ActionArg> for WakeAction {
    fn from(arg: ActionArg) -> Self {
        match arg {
            ActionArg::StartDictation => WakeAction::StartDictation,
            ActionArg::StopDictation => WakeAction::StopDictation,
            ActionArg::Ignore => WakeAction::Ignore,
        }
    }
}

/// `[dictation.wake]` straight from config.toml (missing fields take their defaults).
fn wake_config() -> Result<WakeConfig> {
    let path = ears::config::AppConfig::path()?;
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(WakeConfig::default());
    };
    let value: toml::Value =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let table = value
        .get("dictation")
        .and_then(|d| d.get("wake"))
        .cloned()
        .unwrap_or_else(|| toml::Value::Table(Default::default()));
    serde::Deserialize::deserialize(table).context("[dictation.wake]")
}

fn catalog_source(explicit: Option<String>, config: &WakeConfig) -> String {
    explicit
        .or_else(|| (!config.catalog.is_empty()).then(|| config.catalog.clone()))
        .unwrap_or_else(|| wake_models::DEFAULT_CATALOG.to_string())
}

/// Apply one binding change to config.toml, then rebuild `active/` from the resulting actions.
fn rebind(store: &Store, head: &str, action: Option<WakeAction>) -> Result<()> {
    let path = ears::config::AppConfig::path()?;
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let edited = wake_models::edit_config(&text, head, action, &store.active_dir())?;
    // Validate only [dictation.wake]: user configs are partial and get defaults on load.
    let value: toml::Value =
        toml::from_str(&edited).context("the edited config no longer parses")?;
    let wake_table = value
        .get("dictation")
        .and_then(|d| d.get("wake"))
        .cloned()
        .unwrap_or_else(|| toml::Value::Table(Default::default()));
    let wake: WakeConfig =
        serde::Deserialize::deserialize(wake_table).context("[dictation.wake] no longer parses")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, edited)?;
    let bound = wake_models::bound_heads(&wake.actions);
    let installed = store.installed()?;
    let runnable: Vec<String> = bound
        .iter()
        .filter(|h| installed.heads.contains_key(*h))
        .cloned()
        .collect();
    for missing in bound.iter().filter(|h| !installed.heads.contains_key(*h)) {
        eprintln!(
            "note: {missing} is bound but not installed here (a custom detector command may provide it)"
        );
    }
    if installed.frontend.is_some() {
        store.compose_active(&runnable)?;
    }
    println!("{} active head(s): {}", runnable.len(), runnable.join(", "));
    if runnable.len() > ACTIVE_HEADS_WARNING {
        println!(
            "warning: every active head adds its own false triggers; consider at most {ACTIVE_HEADS_WARNING}"
        );
    }
    println!(
        "config: {} (applies when dictation restarts; needs `waqy` on PATH)",
        path.display()
    );
    Ok(())
}

pub fn handle(command: WakeCommand) -> Result<()> {
    let config = wake_config()?;
    let store = Store::new(wake_models::store_dir());
    match command {
        WakeCommand::Catalog { json, catalog } => {
            let source = catalog_source(catalog, &config);
            let catalog = wake_models::load_catalog(&source)?;
            let installed = store.installed()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&catalog)?);
                return Ok(());
            }
            println!("{source}");
            for release in &catalog.releases {
                println!("\n{} (threshold {})", release.name, release.threshold);
                for head in &release.heads {
                    let mut state = String::new();
                    if installed
                        .heads
                        .get(&head.name)
                        .is_some_and(|h| h.release == release.name)
                    {
                        state.push_str(" installed");
                    }
                    if let Some(action) = config.actions.get(&head.name) {
                        state.push_str(&format!(
                            " -> {}",
                            serde_json::to_value(action)?.as_str().unwrap_or("")
                        ));
                    }
                    // Release manifests carry hashes, not sizes; show none rather than 0 KB.
                    let size = if head.bytes > 0 {
                        format!("{} KB", head.bytes / 1024)
                    } else {
                        "-".into()
                    };
                    println!(
                        "  {:<14} {:<16} {:<6} {:>7}{state}",
                        head.name,
                        head.phrase,
                        head.languages.join(","),
                        size
                    );
                }
            }
        }
        WakeCommand::Install {
            heads,
            release,
            catalog,
            bind,
        } => {
            let source = catalog_source(catalog, &config);
            let done = store.install(&source, &heads, release.as_deref())?;
            println!(
                "installed {} into {}",
                done.join(", "),
                store.root.display()
            );
            if let Some(action) = bind {
                for head in &done {
                    rebind(&store, head, Some(action.into()))?;
                }
            } else {
                println!(
                    "bind one to make it run: ears wake bind <head> start-dictation|stop-dictation"
                );
            }
        }
        WakeCommand::List { json } => {
            let installed = store.installed()?;
            let bound = wake_models::bound_heads(&config.actions);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "store": store.root, "installed": installed.heads, "actions": config.actions, "active": bound,
                        "detector_command": config.command, "active_heads_warning": ACTIVE_HEADS_WARNING,
                    }))?
                );
                return Ok(());
            }
            if installed.heads.is_empty() {
                println!("no heads installed; see `ears wake catalog`");
            }
            for (name, head) in &installed.heads {
                let action = config.actions.get(name).map(|a| {
                    serde_json::to_value(a)
                        .ok()
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_default()
                });
                println!(
                    "  {:<14} {:<16} {:<10} {}",
                    name,
                    head.phrase,
                    head.release,
                    action.as_deref().unwrap_or("(not bound, not running)")
                );
            }
            println!(
                "{} active head(s){}",
                bound.len(),
                if bound.len() > ACTIVE_HEADS_WARNING {
                    " - more than recommended"
                } else {
                    ""
                }
            );
        }
        WakeCommand::Bind { head, action } => {
            if !store.installed()?.heads.contains_key(&head) {
                bail!("{head} is not installed; run `ears wake install {head}` first");
            }
            rebind(&store, &head, Some(action.into()))?;
        }
        WakeCommand::Unbind { head } => rebind(&store, &head, None)?,
        WakeCommand::Remove { head } => {
            let removed = store.remove(&head)?;
            rebind(&store, &head, None)?;
            println!(
                "{}",
                if removed {
                    format!("removed {head}")
                } else {
                    format!("{head} was not installed")
                }
            );
        }
    }
    Ok(())
}
