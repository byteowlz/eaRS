//! A narrow config edit; never save a stale copy of the entire engine config.
use anyhow::{Context, Result, ensure};
use ears::dictation::auto_pause::validate_speech_threshold;
use std::{fs, io::Write, path::Path};
use toml_edit::{DocumentMut, value};

fn updated_document(original: &str, threshold: f32) -> Result<String> {
    validate_speech_threshold(threshold)?;
    let mut document: DocumentMut = original.parse().context("Cannot edit config.toml")?;
    ensure!(
        document
            .get("dictation")
            .is_none_or(|item| item.as_table_like().is_some()),
        "dictation must be a TOML table"
    );
    ensure!(
        document
            .get("dictation")
            .and_then(|item| item.get("auto_pause"))
            .is_none_or(|item| item.as_table_like().is_some()),
        "dictation.auto_pause must be a TOML table"
    );
    let item = &mut document["dictation"]["auto_pause"]["speech_threshold"];
    let decor = item.as_value().map(|v| v.decor().clone());
    *item = value(threshold.to_string().parse::<f64>()?);
    if let Some(decor) = decor {
        *item.as_value_mut().unwrap().decor_mut() = decor;
    }
    Ok(document.to_string())
}

pub fn persist_threshold(path: &Path, threshold: f32) -> Result<()> {
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
    let updated = updated_document(&original, threshold)?;
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
    use super::*;
    #[test]
    fn narrow_edit_retains_comments_and_unrelated_configuration() {
        let source = "# owner comment\n[dictation.auto_pause]\nidle_ms = 3000 # keep\nspeech_threshold = 0.5 # speech\n[future]\nuntouched = \"yes\"\n";
        let edited = updated_document(source, 0.75).unwrap();
        assert!(edited.contains("idle_ms = 3000 # keep"));
        assert!(edited.contains("# owner comment"));
        assert!(edited.contains(" # speech"));
        assert!(edited.contains("untouched = \"yes\""));
        let parsed: toml::Value = toml::from_str(&edited).unwrap();
        assert_eq!(
            parsed["dictation"]["auto_pause"]["speech_threshold"].as_float(),
            Some(0.75)
        );
        assert!(updated_document(source, f32::NAN).is_err());
        assert!(updated_document("not valid TOML", 0.75).is_err());
        assert!(updated_document("dictation=1", 0.75).is_err());
        assert!(updated_document("[dictation]\nauto_pause=false", 0.75).is_err());
        assert!(updated_document("", 0.8).unwrap().contains("0.8"));
    }
    #[test]
    fn atomic_save_failure_and_symlink_preserve_existing_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "# keep\n[dictation.auto_pause]\nidle_ms=3000\n").unwrap();
        persist_threshold(&path, 0.75).unwrap();
        let saved = fs::read_to_string(&path).unwrap();
        assert!(persist_threshold(&path, 1.0).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), saved);
        #[cfg(unix)]
        {
            let link = dir.path().join("linked.toml");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            persist_threshold(&link, 0.625).unwrap();
            assert!(link.is_symlink());
            assert!(fs::read_to_string(&path).unwrap().contains("0.625"));
        }
    }
}
