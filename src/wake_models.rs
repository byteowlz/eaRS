//! Wake-phrase heads from the published waqy catalog: list, install, bind to an action.
//!
//! The waqy Hugging Face repo keeps `catalog.json` (schema `waqy-catalog/1`): every published
//! release and head with its file, sha256 and size, plus the shared front end (mel +
//! embedding) every head runs on. eaRS downloads the front end once and each chosen head as a
//! content-addressed blob under `$XDG_DATA_HOME/ears/wake/blobs/<sha256>`, verifies every hash,
//! and records what is installed in `installed.json`.
//!
//! Only heads bound to an action run: binding composes `active/` (a normal waqy release folder
//! holding the front end plus the bound heads) and points `[dictation.wake].command` at it when
//! the command is empty or already managed here. Each active head adds its own false triggers,
//! so more than [`ACTIVE_HEADS_WARNING`] active heads earns a warning; CPU is not the limit
//! (measured: 2 heads 4.9 ms, 24 heads 11.4 ms per 80 ms frame on an M2 Ultra).
//!
//! A detection stays evidence only. Binding chooses which hotkey-equivalent action a phrase
//! may trigger; it never grants anything else.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::dictation::wake::WakeAction;

pub const CATALOG_SCHEMA: &str = "waqy-catalog/1";
pub const DEFAULT_CATALOG: &str = "https://huggingface.co/byteowlz/waqy/resolve/main/catalog.json";
pub const ACTIVE_HEADS_WARNING: usize = 8;
const INSTALLED_SCHEMA: &str = "ears-wake-installed/1";
const MEL_FILE: &str = "melspectrogram_raw.onnx";
const EMBEDDING_FILE: &str = "embedding_model.onnx";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileRef {
    pub file: String,
    pub sha256: String,
    #[serde(default)]
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frontend {
    pub mel: FileRef,
    pub embedding: FileRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogHead {
    pub name: String,
    pub file: String,
    pub sha256: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub phrase: String,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub measured: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogRelease {
    pub name: String,
    pub threshold: f32,
    pub frontend: Frontend,
    pub heads: Vec<CatalogHead>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub schema: String,
    #[serde(default)]
    pub repo: Option<String>,
    pub releases: Vec<CatalogRelease>,
}

impl Catalog {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let catalog: Catalog = serde_json::from_slice(bytes).context("parsing the wake catalog")?;
        if catalog.schema != CATALOG_SCHEMA {
            bail!(
                "unsupported catalog schema {:?} (expected {CATALOG_SCHEMA})",
                catalog.schema
            );
        }
        Ok(catalog)
    }

    /// The newest release carrying `head` (or exactly `release` when given).
    pub fn find(
        &self,
        head: &str,
        release: Option<&str>,
    ) -> Result<(&CatalogRelease, &CatalogHead)> {
        self.releases
            .iter()
            .rev()
            .filter(|r| release.is_none_or(|want| r.name == want))
            .find_map(|r| r.heads.iter().find(|h| h.name == head).map(|h| (r, h)))
            .ok_or_else(|| match release {
                Some(r) => anyhow!("head {head:?} is not in release {r:?}"),
                None => anyhow!("head {head:?} is not in the catalog"),
            })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstalledHead {
    pub release: String,
    pub sha256: String,
    pub phrase: String,
    pub languages: Vec<String>,
    pub kind: String,
    pub threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Installed {
    pub schema: String,
    /// One front end for every installed head; heads with another front end are refused.
    pub frontend: Option<Frontend>,
    pub heads: BTreeMap<String, InstalledHead>,
}

impl Default for Installed {
    fn default() -> Self {
        Self {
            schema: INSTALLED_SCHEMA.into(),
            frontend: None,
            heads: BTreeMap::new(),
        }
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Where installed heads live: `$XDG_DATA_HOME/ears/wake` (default `~/.local/share/ears/wake`).
pub fn store_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".local/share"));
    base.join("ears").join("wake")
}

/// The detector command for a managed active folder.
pub fn managed_command(active: &Path) -> Vec<String> {
    vec![
        "waqy".into(),
        "listen".into(),
        "--manifest".into(),
        active.display().to_string(),
    ]
}

/// The manifest for `active/`: front end plus exactly the bound heads. Heads from releases with
/// different thresholds use the strictest (highest) one, because loaders apply one threshold.
pub fn active_manifest(installed: &Installed, bound: &[String]) -> Result<serde_json::Value> {
    let frontend = installed
        .frontend
        .as_ref()
        .ok_or_else(|| anyhow!("no heads installed"))?;
    let mut files = serde_json::Map::new();
    files.insert(
        "mel".into(),
        serde_json::json!({"file": MEL_FILE, "sha256": frontend.mel.sha256}),
    );
    files.insert(
        "embedding".into(),
        serde_json::json!({"file": EMBEDDING_FILE, "sha256": frontend.embedding.sha256}),
    );
    let mut classifiers = Vec::new();
    let mut threshold = 0.0f32;
    for name in bound {
        let head = installed
            .heads
            .get(name)
            .ok_or_else(|| anyhow!("{name:?} is bound but not installed"))?;
        files.insert(
            name.clone(),
            serde_json::json!({"file": format!("{name}.onnx"), "sha256": head.sha256}),
        );
        classifiers.push(name.clone());
        threshold = threshold.max(head.threshold);
    }
    classifiers.sort();
    Ok(serde_json::json!({
        "release": "ears-active",
        // f32 -> JSON would print 0.6 as 0.6000000238; thresholds are set to three decimals.
        "threshold": if classifiers.is_empty() { 0.5 } else { (f64::from(threshold) * 1000.0).round() / 1000.0 },
        "classifiers": classifiers,
        "files": files,
        "composed_from": bound.iter().filter_map(|n| installed.heads.get(n).map(|h| (n.clone(), h.release.clone()))).collect::<BTreeMap<_, _>>(),
    }))
}

/// Set `[dictation.wake.actions].<head> = <action>` (or remove it), keeping comments and layout.
/// When the detector command is empty or already managed, point it at `active`.
pub fn edit_config(
    text: &str,
    head: &str,
    action: Option<WakeAction>,
    active: &Path,
) -> Result<String> {
    let mut doc: toml_edit::DocumentMut = text.parse().context("parsing config.toml")?;
    let dictation = doc
        .entry("dictation")
        .or_insert(toml_edit::table())
        .as_table_mut()
        .ok_or_else(|| anyhow!("[dictation] is not a table"))?;
    let wake = dictation
        .entry("wake")
        .or_insert(toml_edit::table())
        .as_table_mut()
        .ok_or_else(|| anyhow!("[dictation.wake] is not a table"))?;
    let managed = managed_command(active);
    let current: Vec<String> = wake
        .get("command")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let is_managed =
        current.len() == 4 && current[..3] == managed[..3] && Path::new(&current[3]) == active;
    if current.is_empty() || is_managed {
        let mut array = toml_edit::Array::new();
        for part in &managed {
            array.push(part.as_str());
        }
        wake["command"] = toml_edit::value(array);
        if wake.get("enabled").is_none() {
            wake["enabled"] = toml_edit::value(true);
        }
    }
    let actions = wake
        .entry("actions")
        .or_insert(toml_edit::table())
        .as_table_mut()
        .ok_or_else(|| anyhow!("[dictation.wake.actions] is not a table"))?;
    match action {
        Some(action) => {
            let name = serde_json::to_value(action)?
                .as_str()
                .unwrap_or_default()
                .to_string();
            actions[head] = toml_edit::value(name);
        }
        None => {
            actions.remove(head);
        }
    }
    Ok(doc.to_string())
}

/// Heads that run: everything bound to an action other than `ignore`.
pub fn bound_heads(actions: &std::collections::HashMap<String, WakeAction>) -> Vec<String> {
    let mut heads: Vec<String> = actions
        .iter()
        .filter(|(_, a)| **a != WakeAction::Ignore)
        .map(|(h, _)| h.clone())
        .collect();
    heads.sort();
    heads
}

/// Download or read a catalog/blob source: https URL, `file://` URL or a local path.
pub fn fetch(source: &str) -> Result<Vec<u8>> {
    if source.starts_with("http://") || source.starts_with("https://") {
        #[cfg(feature = "wake-models")]
        {
            use std::io::Read;
            let response = ureq::get(source)
                .timeout(std::time::Duration::from_secs(120))
                .call()
                .with_context(|| format!("downloading {source}"))?;
            let mut bytes = Vec::new();
            response
                .into_reader()
                .take(256 * 1024 * 1024)
                .read_to_end(&mut bytes)?;
            return Ok(bytes);
        }
        #[cfg(not(feature = "wake-models"))]
        bail!("this eaRS build has no downloader; rebuild with `--features wake-models`");
    }
    let path = source.strip_prefix("file://").unwrap_or(source);
    std::fs::read(path).with_context(|| format!("reading {path}"))
}

/// Base URL for files next to the catalog (a release folder sits beside catalog.json).
fn sibling(catalog_source: &str, relative: &str) -> String {
    match catalog_source.rfind('/') {
        Some(i) => format!("{}/{relative}", &catalog_source[..i]),
        None => relative.to_string(),
    }
}

pub struct Store {
    pub root: PathBuf,
}

impl Store {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
    pub fn active_dir(&self) -> PathBuf {
        self.root.join("active")
    }
    fn blob(&self, sha: &str) -> PathBuf {
        self.root.join("blobs").join(sha)
    }
    pub fn installed(&self) -> Result<Installed> {
        let path = self.root.join("installed.json");
        if !path.exists() {
            return Ok(Installed::default());
        }
        let installed: Installed =
            serde_json::from_slice(&std::fs::read(&path)?).context("reading installed.json")?;
        if installed.schema != INSTALLED_SCHEMA {
            bail!("unsupported {}: {}", path.display(), installed.schema);
        }
        Ok(installed)
    }
    fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
        std::fs::create_dir_all(path.parent().unwrap())?;
        let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
    /// Fetch one file into the blob store unless present; the hash must match.
    fn ensure_blob(&self, source: &str, sha: &str) -> Result<()> {
        let path = self.blob(sha);
        if path.exists() && sha256_hex(&std::fs::read(&path)?) == sha {
            return Ok(());
        }
        let bytes = fetch(source)?;
        let got = sha256_hex(&bytes);
        if got != sha {
            bail!("{source}: sha256 {got} does not match the catalog ({sha})");
        }
        Self::write_atomic(&path, &bytes)
    }
    /// Install heads from the catalog at `catalog_source`. Returns the installed head names.
    pub fn install(
        &self,
        catalog_source: &str,
        heads: &[String],
        release: Option<&str>,
    ) -> Result<Vec<String>> {
        let catalog = Catalog::parse(&fetch(catalog_source)?)?;
        let mut installed = self.installed()?;
        let mut done = Vec::new();
        for name in heads {
            let (rel, head) = catalog.find(name, release)?;
            if let Some(existing) = &installed.frontend
                && (existing.mel.sha256 != rel.frontend.mel.sha256
                    || existing.embedding.sha256 != rel.frontend.embedding.sha256)
            {
                bail!(
                    "{name} ({}) uses a different front end than the installed heads; remove them first",
                    rel.name
                );
            }
            for file in [&rel.frontend.mel, &rel.frontend.embedding] {
                self.ensure_blob(
                    &sibling(catalog_source, &format!("{}/{}", rel.name, file.file)),
                    &file.sha256,
                )?;
            }
            self.ensure_blob(
                &sibling(catalog_source, &format!("{}/{}", rel.name, head.file)),
                &head.sha256,
            )?;
            installed.frontend = Some(rel.frontend.clone());
            installed.heads.insert(
                name.clone(),
                InstalledHead {
                    release: rel.name.clone(),
                    sha256: head.sha256.clone(),
                    phrase: head.phrase.clone(),
                    languages: head.languages.clone(),
                    kind: head.kind.clone(),
                    threshold: rel.threshold,
                },
            );
            done.push(name.clone());
        }
        Self::write_atomic(
            &self.root.join("installed.json"),
            &serde_json::to_vec_pretty(&installed)?,
        )?;
        Ok(done)
    }
    pub fn remove(&self, head: &str) -> Result<bool> {
        let mut installed = self.installed()?;
        let removed = installed.heads.remove(head).is_some();
        if installed.heads.is_empty() {
            installed.frontend = None;
        }
        Self::write_atomic(
            &self.root.join("installed.json"),
            &serde_json::to_vec_pretty(&installed)?,
        )?;
        Ok(removed)
    }
    /// Rebuild `active/` with exactly the bound heads (copies of verified blobs).
    pub fn compose_active(&self, bound: &[String]) -> Result<PathBuf> {
        let installed = self.installed()?;
        let manifest = active_manifest(&installed, bound)?;
        let dir = self.active_dir();
        let staging = self.root.join(format!("active.tmp-{}", std::process::id()));
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }
        std::fs::create_dir_all(&staging)?;
        let frontend = installed.frontend.as_ref().unwrap();
        std::fs::copy(self.blob(&frontend.mel.sha256), staging.join(MEL_FILE))?;
        std::fs::copy(
            self.blob(&frontend.embedding.sha256),
            staging.join(EMBEDDING_FILE),
        )?;
        for name in bound {
            std::fs::copy(
                self.blob(&installed.heads[name].sha256),
                staging.join(format!("{name}.onnx")),
            )?;
        }
        std::fs::write(
            staging.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::rename(&staging, &dir)?;
        Ok(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(root: &Path) -> (String, String) {
        // A two-release catalog with real files, so install verifies actual hashes.
        let repo = root.join("repo");
        let write = |rel: &str, name: &str, body: &[u8]| -> String {
            let dir = repo.join(rel);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(name), body).unwrap();
            sha256_hex(body)
        };
        let (mel, emb) = (
            write("pub1", MEL_FILE, b"mel"),
            write("pub1", EMBEDDING_FILE, b"emb"),
        );
        write("pub2", MEL_FILE, b"mel");
        write("pub2", EMBEDDING_FILE, b"emb");
        let a = write("pub1", "hey_oqto.onnx", b"oqto-v1");
        let b = write("pub2", "hey_oqto.onnx", b"oqto-v2");
        let c = write("pub2", "jupp-stop.onnx", b"jupp-stop");
        let frontend = serde_json::json!({"mel": {"file": MEL_FILE, "sha256": mel}, "embedding": {"file": EMBEDDING_FILE, "sha256": emb}});
        let catalog = serde_json::json!({"schema": CATALOG_SCHEMA, "repo": "test", "releases": [
            {"name": "pub1", "threshold": 0.5, "frontend": frontend, "heads": [{"name": "hey_oqto", "file": "hey_oqto.onnx", "sha256": a, "phrase": "Hey Oqto"}]},
            {"name": "pub2", "threshold": 0.6, "frontend": frontend, "heads": [
                {"name": "hey_oqto", "file": "hey_oqto.onnx", "sha256": b, "phrase": "Hey Oqto"},
                {"name": "jupp-stop", "file": "jupp-stop.onnx", "sha256": c, "phrase": "Jupp stop!", "kind": "command"}]}]});
        std::fs::write(
            repo.join("catalog.json"),
            serde_json::to_vec(&catalog).unwrap(),
        )
        .unwrap();
        (repo.join("catalog.json").display().to_string(), b)
    }

    fn temp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ears-wake-models-{name}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn install_takes_the_newest_release_verifies_hashes_and_composes_only_bound_heads() {
        let root = temp("install");
        let (catalog, newest) = fixture(&root);
        let store = Store::new(root.join("store"));
        store
            .install(&catalog, &["hey_oqto".into(), "jupp-stop".into()], None)
            .unwrap();
        let installed = store.installed().unwrap();
        assert_eq!(installed.heads["hey_oqto"].sha256, newest);
        assert_eq!(installed.heads["hey_oqto"].release, "pub2");
        // Pinning an older release is explicit.
        store
            .install(&catalog, &["hey_oqto".into()], Some("pub1"))
            .unwrap();
        assert_eq!(store.installed().unwrap().heads["hey_oqto"].release, "pub1");
        let active = store.compose_active(&["jupp-stop".into()]).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(active.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["classifiers"], serde_json::json!(["jupp-stop"]));
        assert_eq!(manifest["threshold"], serde_json::json!(0.6));
        assert!(
            !active.join("hey_oqto.onnx").exists(),
            "unbound heads do not run"
        );
        assert_eq!(
            sha256_hex(&std::fs::read(active.join("jupp-stop.onnx")).unwrap()),
            manifest["files"]["jupp-stop"]["sha256"].as_str().unwrap()
        );
    }

    #[test]
    fn a_tampered_download_is_refused_and_nothing_is_recorded() {
        let root = temp("tamper");
        let (catalog, _) = fixture(&root);
        std::fs::write(
            root.join("repo/pub2/jupp-stop.onnx"),
            b"not the published head",
        )
        .unwrap();
        let store = Store::new(root.join("store"));
        let err = store
            .install(&catalog, &["jupp-stop".into()], None)
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("does not match the catalog"),
            "{err:#}"
        );
        assert!(store.installed().unwrap().heads.is_empty());
        assert!(store.install(&catalog, &["nope".into()], None).is_err());
    }

    #[test]
    fn binding_edits_config_keeps_comments_and_never_replaces_a_custom_command() {
        let active = PathBuf::from("/data/ears/wake/active");
        let fresh = edit_config(
            "# my config\n[dictation]\nlanguage = \"de\" # keep\n",
            "hey_jupp",
            Some(WakeAction::StartDictation),
            &active,
        )
        .unwrap();
        assert!(fresh.contains("# my config") && fresh.contains("# keep"));
        let parsed: toml::Value = toml::from_str(&fresh).unwrap();
        assert_eq!(
            parsed["dictation"]["wake"]["actions"]["hey_jupp"].as_str(),
            Some("start_dictation")
        );
        assert_eq!(
            parsed["dictation"]["wake"]["command"][3].as_str(),
            Some("/data/ears/wake/active")
        );
        let stop = edit_config(
            &fresh,
            "jupp-stop",
            Some(WakeAction::StopDictation),
            &active,
        )
        .unwrap();
        assert!(stop.contains("jupp-stop = \"stop_dictation\""));
        let custom =
            "[dictation.wake]\ncommand = [\"waqy\", \"listen\", \"--manifest\", \"/my/release\"]\n";
        let kept =
            edit_config(custom, "hey_hex", Some(WakeAction::StartDictation), &active).unwrap();
        assert!(kept.contains("/my/release") && !kept.contains("/data/ears/wake/active"));
        let unbound = edit_config(&stop, "hey_jupp", None, &active).unwrap();
        assert!(!unbound.contains("hey_jupp"));
    }

    #[test]
    fn ignore_bindings_do_not_run() {
        let mut actions = std::collections::HashMap::new();
        actions.insert("hey_jupp".into(), WakeAction::StartDictation);
        actions.insert("oqto".into(), WakeAction::Ignore);
        actions.insert("jupp-stop".into(), WakeAction::StopDictation);
        assert_eq!(
            bound_heads(&actions),
            vec!["hey_jupp".to_string(), "jupp-stop".to_string()]
        );
    }
}
