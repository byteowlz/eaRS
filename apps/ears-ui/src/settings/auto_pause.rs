//! Atomic narrow config edits; never save a stale copy of the entire engine config.
use anyhow::{Context, Result, ensure};
use std::{fs, io::Write, path::Path};

pub(super) fn persist_edit(path: &Path, edit: impl FnOnce(&str) -> Result<String>) -> Result<()> {
    // Follow an intentional dotfile symlink rather than replacing it.
    let resolved = if path.exists() {
        path.canonicalize()?
    } else {
        path.to_path_buf()
    };
    let original = match fs::read_to_string(&resolved) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };
    let updated = edit(&original)?;
    let parent = resolved.parent().context("Config path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)?;
    if let Ok(metadata) = fs::metadata(&resolved) {
        staged.as_file().set_permissions(metadata.permissions())?;
    }
    staged.write_all(updated.as_bytes())?;
    staged.as_file().sync_all()?;
    let current = match fs::read_to_string(&resolved) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };
    ensure!(
        current == original,
        "config.toml changed during this edit; retry"
    );
    staged
        .persist(&resolved)
        .context("Cannot replace config.toml")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::live_settings::{LiveSetting, persist};
    use super::*;

    #[test]
    fn atomic_save_failure_and_symlink_preserve_existing_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "# keep\n[dictation.auto_pause]\nidle_ms=3000\n").unwrap();
        persist(&path, LiveSetting::SpeechThreshold, 0.75).unwrap();
        let saved = fs::read_to_string(&path).unwrap();
        assert!(saved.contains("# keep") && saved.contains("0.75"));
        assert!(persist(&path, LiveSetting::SpeechThreshold, 1.0).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), saved);
        #[cfg(unix)]
        {
            let link = dir.path().join("linked.toml");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            persist(&link, LiveSetting::PauseAfterSilence, 2.5).unwrap();
            assert!(link.is_symlink());
            let parsed: toml::Value = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(
                parsed["dictation"]["auto_pause"]["idle_ms"].as_integer(),
                Some(2500)
            );
        }
    }
}
