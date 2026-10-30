//! Read-only executable discovery for speaker setup. Compatibility is not model/profile readiness.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryStatus {
    NotFound,
    NotExecutable,
    ProbeFailed,
    Unsupported,
    CompatibleExecutable,
}

#[derive(Debug, Serialize)]
pub struct Discovery {
    pub status: DiscoveryStatus,
    pub executable: Option<PathBuf>,
    pub version: Option<String>,
    pub preprocessing: Vec<String>,
    pub model_state: &'static str,
    pub registry_state: &'static str,
    pub detail: Option<String>,
}

#[derive(Deserialize)]
struct Capabilities {
    protocol: String,
    version: String,
    recognition: RecognitionCapabilities,
}

#[derive(Deserialize)]
struct RecognitionCapabilities {
    compiled: bool,
    identify_json: bool,
    preprocessing: Vec<String>,
}

impl Discovery {
    fn new(status: DiscoveryStatus, executable: Option<PathBuf>) -> Self {
        Self {
            status,
            executable,
            version: None,
            preprocessing: vec![],
            model_state: "not_checked",
            registry_state: "not_checked",
            detail: None,
        }
    }
}

fn filename() -> &'static str {
    if cfg!(windows) {
        "voqx.exe"
    } else {
        "voqx"
    }
}

fn candidates(
    explicit: Option<&Path>,
    sibling: Option<&Path>,
    path: &[PathBuf],
    home: Option<&Path>,
) -> Vec<PathBuf> {
    let in_path = |name: &Path| {
        path.iter()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.join(name))
            .collect::<Vec<_>>()
    };
    if let Some(explicit) = explicit {
        if matches!(
            explicit.components().next(),
            Some(std::path::Component::Normal(_))
        ) && explicit.components().count() == 1
        {
            return in_path(explicit)
                .into_iter()
                .find(|p| p.is_file())
                .into_iter()
                .collect();
        }
        return vec![explicit.to_path_buf()];
    }
    let mut paths = Vec::new();
    if let Some(sibling) = sibling {
        paths.push(sibling.join(filename()));
    }
    paths.extend(in_path(Path::new(filename())));
    if let Some(home) = home {
        paths.push(home.join(".cargo/bin").join(filename()));
        paths.push(home.join(".local/bin").join(filename()));
    }
    #[cfg(target_os = "macos")]
    paths.extend([
        PathBuf::from("/opt/homebrew/bin/voqx"),
        PathBuf::from("/usr/local/bin/voqx"),
    ]);
    let mut unique = Vec::new();
    for p in paths {
        if !unique.contains(&p) {
            unique.push(p);
        }
    }
    unique
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    true
}

/// Probe an explicitly chosen executable, or known installation locations. Does not read/write
/// configuration, load models, open the registry, start a service or access the microphone.
/// Call inside a Tokio runtime (the CLI and future UI worker provide it).
pub async fn discover(explicit: Option<&Path>) -> Discovery {
    let executable = std::env::current_exe().ok();
    let home = dirs::home_dir();
    let path: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    let candidates = candidates(
        explicit,
        executable.as_deref().and_then(Path::parent),
        &path,
        home.as_deref(),
    );
    let selected = if explicit.is_some() {
        candidates.first().cloned()
    } else {
        candidates.into_iter().find(|p| p.is_file())
    };
    let Some(selected) = selected else {
        return Discovery::new(DiscoveryStatus::NotFound, None);
    };
    probe(&selected, Duration::from_secs(2)).await
}

async fn probe(selected: &Path, timeout: Duration) -> Discovery {
    if !selected.exists() {
        return Discovery::new(DiscoveryStatus::NotFound, Some(selected.to_path_buf()));
    }
    if !is_executable(selected) {
        return Discovery::new(DiscoveryStatus::NotExecutable, Some(selected.to_path_buf()));
    }
    let path = selected
        .canonicalize()
        .unwrap_or_else(|_| selected.to_path_buf());
    let mut result = Discovery::new(DiscoveryStatus::ProbeFailed, Some(path.clone()));
    let Some(program) = path.to_str() else {
        result.detail = Some("executable path is not UTF-8".into());
        return result;
    };
    let command = vec![program.into(), "capabilities".into(), "--json".into()];
    match super::wake_speaker::execute_command(&command, None, timeout).await {
        Ok(bytes) => apply_capabilities(result, &bytes),
        Err(reason) => {
            result.detail = Some(reason);
            result
        }
    }
}

fn apply_capabilities(mut result: Discovery, bytes: &[u8]) -> Discovery {
    result.status = DiscoveryStatus::Unsupported;
    if bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) != Some(b'{') {
        result.detail = Some("expected voqx-capabilities/1 JSON object".into());
        return result;
    }
    let Ok(capabilities) = serde_json::from_slice::<Capabilities>(bytes) else {
        result.detail = Some("invalid capability response".into());
        return result;
    };
    if capabilities.protocol != "voqx-capabilities/1"
        || capabilities.version.is_empty()
        || capabilities.version.len() > 128
    {
        result.detail = Some("unsupported capability protocol or version field".into());
        return result;
    }
    result.version = Some(capabilities.version);
    if !capabilities.recognition.compiled || !capabilities.recognition.identify_json {
        result.detail = Some("this build lacks JSON speaker identification".into());
        return result;
    }
    result.preprocessing = capabilities
        .recognition
        .preprocessing
        .into_iter()
        .filter(|p| matches!(p.as_str(), "uniform" | "valid-v1"))
        .collect();
    if result.preprocessing.is_empty() {
        result.detail = Some("no supported recognition preprocessing".into());
        return result;
    }
    result.status = DiscoveryStatus::CompatibleExecutable;
    result
}

#[cfg(test)]
mod tests;
