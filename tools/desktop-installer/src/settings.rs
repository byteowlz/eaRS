use crate::fsutil;
use anyhow::{bail, Context, Result};
use config::{Config, Environment, File};
use serde::Deserialize;
use std::env;
use std::path::{Path, PathBuf};

const CONFIG_SCHEMA: &str = include_str!("../schemas/config.schema.json");

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    prefix: Option<PathBuf>,
}

fn home() -> Result<PathBuf> {
    env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .context("home directory is unavailable")
}

pub fn default_prefix() -> Result<PathBuf> {
    if cfg!(windows) {
        return Ok(PathBuf::from(
            env::var_os("LOCALAPPDATA")
                .filter(|v| !v.is_empty())
                .context("LOCALAPPDATA is unavailable")?,
        )
        .join("Programs/eaRS"));
    }
    if cfg!(target_os = "linux") {
        return Ok(home()?.join(".local/opt/ears"));
    }
    bail!("no default prefix on this OS; native macOS installation is parent-owned (explicit --prefix for contract tests)")
}

fn config_dir() -> Result<PathBuf> {
    let root = match env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
        Some(path) => PathBuf::from(path),
        None => home()?.join(".config"),
    };
    Ok(root.join("ears-installer"))
}

fn initialize(directory: &Path) -> Result<()> {
    std::fs::create_dir_all(directory)?;
    fsutil::directory(directory)?;
    // Defaults are resolved at runtime so moving home directories does not leave
    // a stale absolute prefix. The only setting is an optional prefix override.
    for (name, bytes) in [
        ("config.toml", b"#:schema ./config.schema.json\n# prefix defaults to the per-user platform install directory.\n# prefix = \"/custom/ears\"\n".as_slice()),
        ("config.schema.json", CONFIG_SCHEMA.as_bytes()),
    ] {
        let path = directory.join(name);
        if !fsutil::exists(&path)? {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        } else {
            fsutil::open_regular(&path)?;
        }
    }
    Ok(())
}

pub fn resolve(cli: Option<PathBuf>, explicit: Option<PathBuf>) -> Result<PathBuf> {
    let directory = config_dir()?;
    initialize(&directory)?;
    let global = directory.join("config.toml");
    let local = env::current_dir()?.join("ears-installer.toml");
    let mut builder = Config::builder()
        .add_source(File::from(global))
        .add_source(File::from(local).required(false))
        .add_source(
            Environment::with_prefix("EARS_INSTALLER")
                .prefix_separator("_")
                .separator("__"),
        );
    if let Some(path) = explicit {
        // Explicit --config takes precedence over the environment.
        fsutil::open_regular(&path)?;
        builder = builder.add_source(File::from(path).required(true));
    }
    if let Some(path) = cli {
        builder = builder.set_override("prefix", path.to_str().context("prefix must be UTF-8")?)?;
    }
    let settings: Settings = builder.build()?.try_deserialize()?;
    let path = match settings.prefix {
        Some(path) => path,
        None => default_prefix()?,
    };
    anyhow::ensure!(!path.as_os_str().is_empty(), "prefix must not be empty");
    Ok(if path.is_absolute() {
        path
    } else {
        env::current_dir()?.join(path)
    })
}
