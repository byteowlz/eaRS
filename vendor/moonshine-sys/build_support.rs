//! Verified runtime dependency cache. No stock SDK fallback or model downloads.
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

#[derive(Clone, Debug, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Deserialize)]
pub struct TargetArtifacts {
    pub mode: String,
    pub files: Vec<Artifact>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    pub version: String,
    pub source_sha256: String,
    pub critical_sources: std::collections::BTreeMap<String, String>,
}

pub fn digest(path: &Path) -> Result<String> {
    let mut input = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut digest = Sha256::new();
    std::io::copy(&mut input, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

pub fn verify_artifact(path: &Path, artifact: &Artifact) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.len() != artifact.bytes || digest(path)? != artifact.sha256 {
        bail!(
            "Corrupt pinned runtime dependency: {}; remove it explicitly before retrying",
            path.display()
        );
    }
    Ok(())
}

pub fn validate_artifact(artifact: &Artifact) -> Result<()> {
    if artifact.name.is_empty()
        || Path::new(&artifact.name)
            .file_name()
            .and_then(|s| s.to_str())
            != Some(&artifact.name)
        || artifact.sha256.len() != 64
        || !artifact.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        || artifact.bytes == 0
        || artifact.bytes > 512 * 1024 * 1024
        || !(artifact
            .url
            .starts_with("https://raw.githubusercontent.com/moonshine-ai/moonshine/")
            || artifact
                .url
                .starts_with("https://media.githubusercontent.com/media/moonshine-ai/moonshine/"))
    {
        bail!("Invalid pinned runtime dependency descriptor");
    }
    Ok(())
}

pub fn artifact_path(cache: &Path, artifact: &Artifact, offline: bool) -> Result<PathBuf> {
    validate_artifact(artifact)?;
    let directory = cache.join(&artifact.sha256);
    let destination = directory.join(&artifact.name);
    if destination.try_exists()? {
        verify_artifact(&destination, artifact)?;
        return Ok(destination);
    }
    if offline {
        bail!(
            "Pinned ONNX Runtime dependency missing offline: {}",
            destination.display()
        );
    }
    fs::create_dir_all(&directory)?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temporary = directory.join(format!(
        ".partial-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let response = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(180))
            .build()
            .get(&artifact.url)
            .call()
            .with_context(|| format!("fetching pinned ONNX Runtime {}", artifact.name))?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let count = std::io::copy(
            &mut response.into_reader().take(artifact.bytes + 1),
            &mut file,
        )?;
        file.flush()?;
        drop(file);
        if count != artifact.bytes {
            bail!("ONNX Runtime dependency size mismatch");
        }
        verify_artifact(&temporary, artifact)?;
        if destination.exists() {
            verify_artifact(&destination, artifact)?;
            fs::remove_file(&temporary)?;
        } else {
            fs::rename(&temporary, &destination)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(destination)
}

pub fn verify_source(root: &Path, provenance: &Provenance) -> Result<()> {
    for (relative, expected) in &provenance.critical_sources {
        if digest(&root.join(relative))? != *expected {
            bail!("Source does not match the pinned final-audio repair: {relative}");
        }
    }
    Ok(())
}

pub fn cache_root() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MOONSHINE_ORT_CACHE") {
        return Ok(path.into());
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path).join("ears-moonshine/ort"));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("HOME/USERPROFILE or MOONSHINE_ORT_CACHE is required")?;
    Ok(PathBuf::from(home).join(".cache/ears-moonshine/ort"))
}

pub fn offline() -> bool {
    ["MOONSHINE_OFFLINE", "CARGO_NET_OFFLINE", "HF_HUB_OFFLINE"]
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|s| matches!(s.as_str(), "1" | "true")))
}
