//! transcribe.cpp streaming model catalog.
//!
//! The catalog is data, not compiled judgment: [`catalog.json`] is generated
//! from the `handy-computer` Hugging Face org (see `scripts/gen_catalog.py`)
//! and baked into the binary with `include_str!` so `ears models list` works
//! offline. `ears models refresh` (see [`refresh`]) fetches a newer
//! `catalog.json` into `~/.config/ears/models.json`, which [`load`] then
//! prefers over the baked copy — so a model published after this binary was
//! built shows up without a reinstall. The resolver also accepts any
//! `handy-computer/*-gguf` slug directly, catalog or not.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One catalog row: an HF repo slug plus display metadata.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogModel {
    /// HF repo slug under `handy-computer/{slug}-gguf`, also the model id.
    pub slug: String,
    /// Human-readable name.
    pub name: String,
    /// e.g. "English" or "Multilingual (28)".
    pub languages: String,
    /// Whether the model streams (required for live dictation).
    pub streaming: bool,
    /// Default quant fetched when the slug is used without `@QUANT`.
    pub default_quant: String,
    /// Approx download size of the default quant, in MB.
    pub default_size_mb: u32,
    /// One-line description.
    pub description: String,
    /// Part of the small curated "recommended" set.
    #[serde(default)]
    pub recommended: bool,
}

#[derive(Debug, Deserialize)]
struct CatalogFile {
    models: Vec<CatalogModel>,
}

const BAKED_JSON: &str = include_str!("catalog.json");

/// Where a refreshed catalog is fetched from. Override with `EARS_CATALOG_URL`.
const DEFAULT_CATALOG_URL: &str =
    "https://raw.githubusercontent.com/byteowlz/eaRS/main/src/server/catalog.json";

fn parse(json: &str) -> Result<Vec<CatalogModel>> {
    let file: CatalogFile = serde_json::from_str(json).context("parsing model catalog JSON")?;
    Ok(file.models)
}

/// The catalog compiled into this binary.
pub fn baked() -> Vec<CatalogModel> {
    parse(BAKED_JSON).expect("baked catalog.json is valid and matches the schema")
}

/// The user's refreshed catalog cache path (`~/.config/ears/models.json`).
pub fn cache_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ears")
        .join("models.json")
}

/// The active catalog: the refreshed cache if present and valid, else baked.
pub fn load() -> Vec<CatalogModel> {
    let path = cache_path();
    if path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(models) = parse(&text) {
                if !models.is_empty() {
                    return models;
                }
            }
        }
    }
    baked()
}

fn catalog_url() -> String {
    std::env::var("EARS_CATALOG_URL").unwrap_or_else(|_| DEFAULT_CATALOG_URL.to_string())
}

/// Fetch the latest catalog and write it to [`cache_path`].
///
/// Returns the source URL and the model count on success.
///
/// # Errors
///
/// Returns an error if the download, parse, or cache write fails.
pub fn refresh() -> Result<(String, usize)> {
    let url = catalog_url();
    let body = ureq::get(&url)
        .call()
        .with_context(|| format!("fetching catalog from {url}"))?
        .into_string()
        .context("reading catalog response body")?;
    // Validate before persisting so a bad response never poisons the cache.
    let models = parse(&body).context("downloaded catalog is not valid")?;
    let path = cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&path, &body).with_context(|| format!("writing {}", path.display()))?;
    Ok((url, models.len()))
}
