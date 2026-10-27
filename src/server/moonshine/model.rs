//! Official German Tiny runtime assets, cached under HF_HOME/moonshine.
//! The HF training checkpoint is PyTorch; these native ORT exports are published
//! on Moonshine's CDN. Local model selection never downloads or modifies files.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

const VERSION: &str = "tiny-streaming-de/quantized_26_08_24";
const FILES: &[(&str, &str)] = &[
    (
        "adapter.ort",
        "23b91142a35cfbc533a7f03ad8387d0b6d32856b465c6fe7a14ede44a254a345",
    ),
    (
        "cross_kv.ort",
        "25e6d9d4f16569c2ede7e23a381fe7c60cc4b0f141adc88580557d4530634f17",
    ),
    (
        "decoder_kv.ort",
        "768b1464074fc4fb7327e8b863fc13dcd12cb7bbe8d8e9428fa7292f7fd9ca6e",
    ),
    (
        "encoder.ort",
        "74610b818058e396f4b9d9ffd730cf8e36772122464d989dd3011dfa76bd6baf",
    ),
    (
        "frontend.model.ort",
        "4445bb4c2f41b0dd993999e06dc9367a0dde0d42a2e777116e7d93be102dc43d",
    ),
    (
        "frontend.weights.ort",
        "233fda61d583403bbf13ea71b6ba2335fdf57eac71be85a97dcbb90a6c2f4186",
    ),
    (
        "streaming_config.json",
        "d31afd7c2dd5ee9ae3fa232312c5b42cb3e4fe1c6e9c53d885f2bea63023790a",
    ),
    (
        "tokenizer.bin",
        "5725ad3771a1339802f8cebeda0fa8f7723f5282b1ee726c2544774d7399b9a7",
    ),
];

fn cache_dir() -> PathBuf {
    let hub = hf_hub::Cache::from_env();
    hub.path()
        .parent()
        .unwrap_or(hub.path())
        .join("moonshine")
        .join(VERSION)
}

pub(super) fn resolve(local: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = local {
        validate_local(path)?;
        return Ok(path.to_path_buf());
    }
    let dir = cache_dir();
    let offline = std::env::var("HF_HUB_OFFLINE").is_ok_and(|v| matches!(v.as_str(), "1" | "true"));
    fs::create_dir_all(&dir)?;
    for (name, expected) in FILES {
        let path = dir.join(name);
        if digest(&path).is_ok_and(|actual| actual == *expected) {
            continue;
        }
        if offline {
            bail!(
                "Moonshine cache file missing or invalid: {} (HF_HUB_OFFLINE is set)",
                path.display()
            );
        }
        download(&path, name, expected)?;
    }
    validate_local(&dir)?;
    Ok(dir)
}

fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn download(path: &Path, name: &str, expected: &str) -> Result<()> {
    let url = format!("https://download.moonshine.ai/model/{VERSION}/{name}");
    eprintln!("[moonshine] downloading {url}");
    let tmp = path.with_extension(format!("partial-{}", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let response = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .get(&url)
            .call()
            .with_context(|| format!("downloading {url}"))?;
        let mut file = File::create(&tmp)?;
        let limit = 128 * 1024 * 1024;
        let count = std::io::copy(&mut response.into_reader().take(limit + 1), &mut file)?;
        file.flush()?;
        if count > limit || digest(&tmp)? != expected {
            bail!("Moonshine download failed size/SHA-256 verification: {url}");
        }
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Validate only; never repair, quantize or download into an explicit path.
fn validate_local(path: &Path) -> Result<()> {
    if !path.is_dir() {
        bail!("Moonshine model directory not found: {}", path.display());
    }
    let missing: Vec<_> = FILES
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !path.join(name).is_file())
        .collect();
    if !missing.is_empty() {
        bail!(
            "Moonshine native model {} is missing {}. Supply a native Tiny Streaming ORT export bundle, not a PyTorch/LoRA directory. No base-model fallback was attempted.",
            path.display(),
            missing.join(", ")
        );
    }
    for (name, _) in FILES.iter().filter(|(name, _)| name.ends_with(".ort")) {
        let mut header = [0u8; 8];
        File::open(path.join(name))?.read_exact(&mut header)?;
        if &header[4..] != b"ORTM" {
            bail!(
                "{} is not an ORT-format model; do not rename an ONNX file to .ort",
                path.join(name).display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_missing_model_never_falls_back() {
        let path =
            std::env::temp_dir().join(format!("ears-moonshine-missing-{}", uuid::Uuid::new_v4()));
        assert!(
            resolve(Some(&path))
                .unwrap_err()
                .to_string()
                .contains("not found")
        );
        assert!(!path.exists());
    }

    #[test]
    fn local_export_validation_is_read_only_and_accepts_custom_decoder() {
        let dir =
            std::env::temp_dir().join(format!("ears-moonshine-model-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        assert!(
            resolve(Some(&dir))
                .unwrap_err()
                .to_string()
                .contains("not a PyTorch/LoRA")
        );
        for (name, _) in FILES {
            fs::write(
                dir.join(name),
                if name.ends_with(".ort") {
                    b"\x18\0\0\0ORTMcustom".as_slice()
                } else {
                    b"test"
                },
            )
            .unwrap();
        }
        assert_eq!(resolve(Some(&dir)).unwrap(), dir);
        // Local fine-tunes are not compared against the public decoder hash.
        assert_eq!(
            fs::read(dir.join("decoder_kv.ort")).unwrap(),
            b"\x18\0\0\0ORTMcustom"
        );
        fs::write(dir.join("decoder_kv.ort"), b"not an ORT model").unwrap();
        assert!(
            resolve(Some(&dir))
                .unwrap_err()
                .to_string()
                .contains("not an ORT-format")
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
