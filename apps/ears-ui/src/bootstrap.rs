//! First-run setup is a persisted, fail-closed state, not permission to capture.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

fn marker(config: &Path) -> Result<PathBuf> {
    Ok(config
        .parent()
        .context("configuration has no parent directory")?
        .join(".setup-pending"))
}

pub fn setup_required(config: &Path, requested: bool) -> Result<bool> {
    let pending = marker(config)?;
    if !config.is_file() {
        std::fs::create_dir_all(pending.parent().unwrap())?;
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)
        {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error).context("could not preserve first-run setup state"),
        }
    }
    Ok(requested || pending.is_file())
}

pub fn finish_setup(config: &Path) -> Result<()> {
    match std::fs::remove_file(marker(config)?) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("could not save setup completion"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fresh_setup_survives_relaunch_and_requires_explicit_completion() {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("ears/config.toml");
        assert!(setup_required(&config, false).unwrap());
        std::fs::write(&config, "sentinel preferences").unwrap();
        assert!(setup_required(&config, false).unwrap());
        finish_setup(&config).unwrap();
        assert!(!setup_required(&config, false).unwrap());
        assert_eq!(
            std::fs::read_to_string(&config).unwrap(),
            "sentinel preferences"
        );
        assert!(setup_required(&config, true).unwrap());
    }
    #[test]
    fn existing_install_does_not_create_a_marker_or_restart_into_setup() {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("config.toml");
        std::fs::write(&config, "existing preferences").unwrap();
        assert!(!setup_required(&config, false).unwrap());
        assert!(!marker(&config).unwrap().exists());
    }
}
