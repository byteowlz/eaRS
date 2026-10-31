use crate::fsutil;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

pub const SCHEMA: &str = "ears.desktop-payload/v1";
pub const MAX_FILES: usize = 1024;
pub const MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_MANIFEST: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub os: String,
    pub arch: String,
    pub version: String,
    pub source_commit: String,
    pub client_vad: bool,
    pub files: Vec<PayloadFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadFile {
    pub path: String,
    pub sha256: String,
    pub executable: bool,
}

pub struct Payload {
    pub manifest: Manifest,
    pub hash: String,
    raw: Vec<u8>,
    directories: BTreeSet<String>,
}

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn validate_path(path: &str) -> Result<()> {
    ensure!(
        !path.is_empty() && path.len() <= 1024,
        "invalid path length"
    );
    ensure!(
        !path
            .chars()
            .any(|c| c.is_control() || "\\:<>\"|?*".contains(c)),
        "invalid path characters"
    );
    let parts: Vec<_> = path.split('/').collect();
    ensure!(parts.len() <= 16, "path depth exceeds limit");
    for part in parts {
        ensure!(
            !part.is_empty() && part != "." && part != "..",
            "unsafe path component"
        );
        ensure!(
            part.len() <= 128 && !part.ends_with(['.', ' ']),
            "nonportable path"
        );
        let stem = part
            .split('.')
            .next()
            .unwrap_or("")
            .trim_end_matches(' ')
            .to_ascii_uppercase();
        let reserved = matches!(
            stem.as_str(),
            "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
        ) || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'));
        ensure!(!reserved, "reserved device path");
        // Conservative portable names avoid Unicode/case-normalization aliases.
        ensure!(part.is_ascii(), "payload paths must be portable ASCII");
    }
    ensure!(
        !path.eq_ignore_ascii_case("manifest.json"),
        "reserved manifest path"
    );
    Ok(())
}

impl Payload {
    pub fn load(root: &Path, expected: &str) -> Result<Self> {
        Self::load_inner(root, expected, false)
    }

    pub fn load_for_cleanup(root: &Path, expected: &str) -> Result<Self> {
        Self::load_inner(root, expected, true)
    }

    fn load_inner(root: &Path, expected: &str, missing_ok: bool) -> Result<Self> {
        ensure!(
            valid_hash(expected),
            "expected SHA-256 must be 64 lowercase hex characters"
        );
        fsutil::directory(root)?;
        let raw = fsutil::read_bounded(&root.join("manifest.json"), MAX_MANIFEST)?;
        ensure!(hash(&raw) == expected, "manifest SHA-256 mismatch");
        let manifest: Manifest =
            serde_json::from_slice(&raw).context("invalid payload manifest")?;
        ensure!(manifest.schema == SCHEMA, "unsupported manifest schema");
        ensure!(manifest.os == std::env::consts::OS, "host OS mismatch");
        ensure!(
            manifest.arch == std::env::consts::ARCH,
            "host architecture mismatch"
        );
        ensure!(manifest.client_vad, "client_vad must be true");
        for text in [&manifest.version, &manifest.source_commit] {
            ensure!(
                !text.is_empty() && text.len() <= 256 && !text.chars().any(char::is_control),
                "invalid version/source_commit"
            );
        }
        ensure!(
            !manifest.files.is_empty() && manifest.files.len() <= MAX_FILES,
            "invalid file count"
        );
        let mut names = BTreeMap::new();
        let mut directories = BTreeSet::new();
        for entry in &manifest.files {
            validate_path(&entry.path)?;
            ensure!(valid_hash(&entry.sha256), "invalid file SHA-256");
            ensure!(
                names
                    .insert(entry.path.to_ascii_lowercase(), &entry.path)
                    .is_none(),
                "duplicate path"
            );
            let mut parent = entry.path.as_str();
            while let Some((dir, _)) = parent.rsplit_once('/') {
                directories.insert(dir.to_owned());
                parent = dir;
            }
        }
        let mut dir_aliases = BTreeMap::new();
        for dir in &directories {
            let key = dir.to_ascii_lowercase();
            ensure!(!names.contains_key(&key), "file/directory collision");
            if let Some(prior) = dir_aliases.insert(key, dir) {
                ensure!(prior == dir, "case alias in directory paths");
            }
        }
        let payload = Self {
            manifest,
            hash: expected.to_owned(),
            raw,
            directories,
        };
        payload.verify(root, missing_ok)?;
        Ok(payload)
    }

    fn inventory(&self, root: &Path) -> Result<()> {
        let files: BTreeSet<_> = self
            .manifest
            .files
            .iter()
            .map(|e| e.path.as_str())
            .collect();
        let mut pending = vec![String::new()];
        while let Some(relative) = pending.pop() {
            for entry in fs::read_dir(root.join(&relative))? {
                let entry = entry?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("non-UTF8 path"))?;
                let path = if relative.is_empty() {
                    name
                } else {
                    format!("{relative}/{name}")
                };
                let meta = fsutil::metadata(&entry.path())?;
                if meta.is_dir() {
                    ensure!(self.directories.contains(&path), "unlisted directory");
                    pending.push(path);
                } else {
                    ensure!(
                        path == "manifest.json" || files.contains(path.as_str()),
                        "unlisted file"
                    );
                }
            }
        }
        Ok(())
    }

    fn verify(&self, root: &Path, missing_ok: bool) -> Result<()> {
        self.inventory(root)?;
        let mut total = self.raw.len() as u64;
        for entry in &self.manifest.files {
            let path = root.join(&entry.path);
            if missing_ok && !fsutil::exists(&path)? {
                continue;
            }
            let mut input = fsutil::open_regular(&path)?;
            ensure!(
                input.metadata()?.len() <= MAX_BYTES - total,
                "payload exceeds 2 GiB limit"
            );
            ensure!(
                stream(&mut input, None, &mut total)? == entry.sha256,
                "payload SHA-256 mismatch"
            );
        }
        Ok(())
    }

    pub fn copy_to(&self, source: &Path, destination: &Path) -> Result<()> {
        // The destination is a private, newly created staging directory.
        for dir in &self.directories {
            fs::create_dir_all(destination.join(dir))?;
        }
        let mut total = self.raw.len() as u64;
        for entry in &self.manifest.files {
            let mut input = fsutil::open_regular(&source.join(&entry.path))?;
            ensure!(
                input.metadata()?.len() <= MAX_BYTES - total,
                "payload exceeds 2 GiB limit"
            );
            let mut output =
                fsutil::create_payload_file(&destination.join(&entry.path), entry.executable)?;
            ensure!(
                stream(&mut input, Some(&mut output), &mut total)? == entry.sha256,
                "source changed while staging"
            );
            output.sync_all()?;
        }
        let mut manifest = fsutil::create_payload_file(&destination.join("manifest.json"), false)?;
        manifest.write_all(&self.raw)?;
        manifest.sync_all()?;
        Self::load(destination, &self.hash)?;
        Ok(())
    }

    pub fn verify_modes(&self, root: &Path) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for entry in &self.manifest.files {
                let mode = fsutil::metadata(&root.join(&entry.path))?
                    .permissions()
                    .mode();
                ensure!(
                    mode & 0o7000 == 0,
                    "special permission bits on installed file"
                );
                ensure!(
                    (mode & 0o111 != 0) == entry.executable,
                    "installed executable flag mismatch"
                );
            }
        }
        #[cfg(not(unix))]
        let _ = root;
        Ok(())
    }

    // Called only after all managed trees were verified. Never recursively remove
    // an installed directory: an unexpected file must survive even a late failure.
    pub fn remove(&self, root: &Path) -> Result<()> {
        for entry in &self.manifest.files {
            let path = root.join(&entry.path);
            if fsutil::exists(&path)? {
                fs::remove_file(path)?;
            }
        }
        let mut dirs: Vec<_> = self.directories.iter().collect();
        dirs.sort_by_key(|d| std::cmp::Reverse(d.len()));
        for dir in dirs {
            let path = root.join(dir);
            if fsutil::exists(&path)? {
                fs::remove_dir(path)?;
            }
        }
        // Keep the authenticated inventory until all other content is gone.
        fs::remove_file(root.join("manifest.json"))?;
        fs::remove_dir(root)?;
        Ok(())
    }
}

fn stream(
    input: &mut impl Read,
    mut output: Option<&mut dyn Write>,
    total: &mut u64,
) -> Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        *total += count as u64;
        ensure!(*total <= MAX_BYTES, "payload exceeds 2 GiB limit");
        digest.update(&buffer[..count]);
        if let Some(writer) = output.as_mut() {
            writer.write_all(&buffer[..count])?;
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nonportable_paths() {
        for path in [
            "",
            "/bin/x",
            "../x",
            "a/../b",
            "a/./b",
            "a//b",
            "C:/x",
            "a\\b",
            "a\nb",
            "a\u{7f}",
            "nul.txt",
            "COM1",
            "x.",
            "x ",
            "Manifest.JSON",
            "é",
        ] {
            assert!(validate_path(path).is_err(), "{path:?}");
        }
        validate_path("bin/ears.exe").unwrap();
    }

    #[test]
    fn enforces_stream_bound_without_large_fixture() {
        let mut reader = &b"ab"[..];
        let mut total = MAX_BYTES - 1;
        assert!(stream(&mut reader, None, &mut total).is_err());
    }

    #[test]
    fn hash_is_strict() {
        assert!(valid_hash(&hash(b"synthetic")));
        assert!(!valid_hash(&"A".repeat(64)));
        assert!(!valid_hash("../receipt.json"));
    }

    #[test]
    fn stream_hashes_multiple_buffers_and_copies_exact_bytes() {
        let bytes = vec![42u8; 150_000];
        let mut output = Vec::new();
        let mut total = 0;
        let digest = stream(&mut bytes.as_slice(), Some(&mut output), &mut total).unwrap();
        assert_eq!(output, bytes);
        assert_eq!(digest, hash(&bytes));
        assert_eq!(total, bytes.len() as u64);
    }

    #[test]
    fn changed_source_is_detected_during_staging() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        let stage = root.path().join("stage");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&stage).unwrap();
        fs::write(source.join("data"), b"original").unwrap();
        let raw = serde_json::to_vec(&serde_json::json!({
            "schema": SCHEMA, "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
            "version": "fixture", "source_commit": "synthetic", "client_vad": true,
            "files": [{"path":"data", "sha256":hash(b"original"), "executable":false}]
        }))
        .unwrap();
        fs::write(source.join("manifest.json"), &raw).unwrap();
        let payload = Payload::load(&source, &hash(&raw)).unwrap();
        fs::write(source.join("data"), b"mutated!").unwrap();
        assert!(payload.copy_to(&source, &stage).is_err());
        assert!(!stage.join("manifest.json").exists());
    }
}
