use crate::replacement::{ReplacementConfig, TranscriptHistoryConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub storage: StorageConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub parakeet_rs: ParakeetRsConfig,
    #[serde(default)]
    pub transcribe_cpp: TranscribeCppConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub dictation: DictationConfig,
    #[serde(default)]
    pub hotkeys: HotkeyConfig,
    #[serde(default)]
    pub subs: SubsConfig,
    #[serde(default)]
    pub replacement: ReplacementConfig,
    #[serde(default)]
    pub transcripts: TranscriptHistoryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub model_dir: String,
    pub ref_audio: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub prime_languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParakeetRsConfig {
    /// Directory containing encoder.onnx, encoder.onnx.data,
    /// decoder_joint.onnx, and tokenizer.model. None means the engine is not
    /// loaded unless --parakeet-rs-model is supplied.
    #[serde(default)]
    pub model_dir: Option<String>,
    /// Default language prompt for multilingual Nemotron sessions.
    #[serde(default = "default_parakeet_rs_language")]
    pub language: String,
}

fn default_parakeet_rs_language() -> String {
    "auto".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TranscribeCppConfig {
    /// Path to a streaming-capable GGUF model (e.g.
    /// multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf). None means the
    /// engine is not loaded unless --transcribe-cpp-model is supplied.
    #[serde(default)]
    pub model_path: Option<String>,
    /// Language hint for multilingual models (e.g. "de"). "auto" or unset
    /// lets the model autodetect; a dictation client --lang overrides per
    /// session.
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub websocket_port: u16,
    pub host: String,
    #[serde(default)]
    pub listener_tokens: Vec<String>,
    #[serde(default)]
    pub enable_listener_mode: bool,
}

/// Configuration for a single dictation server endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationServerConfig {
    /// WebSocket host address (e.g., "localhost", "192.168.1.100", "transcribe.example.com")
    pub host: String,
    /// WebSocket port number
    pub port: u16,
    /// Optional description for this server
    #[serde(default)]
    pub description: Option<String>,
}

impl DictationServerConfig {
    /// Build the WebSocket URL for this server
    pub fn ws_url(&self) -> String {
        format!("ws://{}:{}", self.host, self.port)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationConfig {
    pub enabled: bool,
    pub type_live_words: bool,
    #[serde(default)]
    pub start_paused: bool,
    /// Audio transport codec for dictation: "pcm" (default) or "opus".
    #[serde(default = "default_audio_codec")]
    pub codec: String,
    /// Default server alias to use when none is specified (defaults to "local")
    #[serde(default = "default_server_alias")]
    pub default_server: String,
    /// Named server configurations for quick switching
    #[serde(default = "default_servers")]
    pub servers: HashMap<String, DictationServerConfig>,
    #[serde(default)]
    pub notifications: DictationNotificationConfig,
    #[cfg(feature = "hooks")]
    #[serde(default)]
    pub hooks: DictationHooksConfig,
}

fn default_server_alias() -> String {
    "local".to_string()
}

fn default_audio_codec() -> String {
    "pcm".to_string()
}

fn default_servers() -> HashMap<String, DictationServerConfig> {
    let mut servers = HashMap::new();
    servers.insert(
        "local".to_string(),
        DictationServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8765,
            description: Some("Local eaRS server".to_string()),
        },
    );
    servers
}

impl DictationConfig {
    /// Get the WebSocket URL for a server by alias.
    /// If alias is None, uses the default_server.
    /// Returns None if the alias doesn't exist.
    pub fn get_server_url(&self, alias: Option<&str>) -> Option<String> {
        let alias = alias.unwrap_or(&self.default_server);
        self.servers.get(alias).map(|s| s.ws_url())
    }

    /// List all available server aliases
    pub fn list_servers(&self) -> Vec<(&str, &DictationServerConfig)> {
        self.servers.iter().map(|(k, v)| (k.as_str(), v)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DictationHotkeyMode {
    Toggle,
    PushToTalk,
    Hybrid,
}

impl Default for DictationHotkeyMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub enable_internal: bool,
    pub toggle: String,
    #[serde(default)]
    pub mode: DictationHotkeyMode,
    pub language_cycle: String,
    /// Optional combo that cycles through context profiles (auto -> each
    /// profile -> auto). Disabled when unset.
    #[serde(default)]
    pub profile_cycle: Option<String>,
    /// When true, pressing Escape while capturing discards the pending words
    /// and pauses dictation. Off by default so plain `ears dictation` keeps
    /// its historical behaviour; the companion UI turns it on.
    #[serde(default)]
    pub escape_cancels: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsConfig {
    pub font: String,
    pub x_position: u32,
    pub y_position: u32,
    pub width: u32,
    pub heigth: u32,
    pub border_radius: u32,
    pub border_thickness: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DictationNotificationConfig {
    pub enabled: bool,
    pub start_message: String,
    pub pause_message: String,
    pub stop_message: String,
}

impl Default for DictationNotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            start_message: "Dictation enabled".to_string(),
            pause_message: "Dictation paused".to_string(),
            stop_message: "Dictation disabled".to_string(),
        }
    }
}

#[cfg(feature = "hooks")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DictationHooksConfig {
    pub start_command: Option<String>,
    pub pause_command: Option<String>,
    pub stop_command: Option<String>,
}

#[cfg(feature = "hooks")]
impl Default for DictationHooksConfig {
    fn default() -> Self {
        Self {
            start_command: None,
            pause_command: None,
            stop_command: None,
        }
    }
}

impl Default for ParakeetRsConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            language: default_parakeet_rs_language(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            prime_languages: vec![],
        }
    }
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            enable_internal: true,
            toggle: "ctrl+shift+v".to_string(),
            mode: DictationHotkeyMode::Toggle,
            language_cycle: "ctrl+shift+l".to_string(),
            profile_cycle: None,
            escape_cancels: false,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            websocket_port: 8765,
            host: "localhost".to_string(),
            listener_tokens: vec![],
            enable_listener_mode: false,
        }
    }
}

impl Default for DictationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            type_live_words: true,
            start_paused: false,
            codec: default_audio_codec(),
            default_server: default_server_alias(),
            servers: default_servers(),
            notifications: DictationNotificationConfig::default(),
            #[cfg(feature = "hooks")]
            hooks: DictationHooksConfig::default(),
        }
    }
}

impl Default for SubsConfig {
    fn default() -> Self {
        Self {
            font: "JetBrainsMono NerdFont".to_string(),
            x_position: 50,
            y_position: 90,
            width: 90,
            heigth: 10,
            border_radius: 5,
            border_thickness: 1,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                model_dir: "default".to_string(),
                ref_audio: "~/.local/share/ears/ref_audio".to_string(),
            },
            model: ModelConfig::default(),
            parakeet_rs: ParakeetRsConfig::default(),
            transcribe_cpp: TranscribeCppConfig::default(),
            server: ServerConfig::default(),
            dictation: DictationConfig::default(),
            hotkeys: HotkeyConfig::default(),
            subs: SubsConfig::default(),
            replacement: ReplacementConfig::default(),
            transcripts: TranscriptHistoryConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let config_path = get_config_path()?;

        if !config_path.exists() {
            let default_config = Self::default();
            default_config.save()?;
            Ok(default_config)
        } else {
            let contents = fs::read_to_string(&config_path)?;

            let mut config: AppConfig = toml::from_str(&contents)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

            // Expand tilde paths
            config.storage.model_dir = expand_tilde(&config.storage.model_dir)?;
            config.storage.ref_audio = expand_tilde(&config.storage.ref_audio)?;
            if let Some(model_dir) = config.parakeet_rs.model_dir.as_mut() {
                *model_dir = expand_tilde(model_dir)?;
            }
            if let Some(model_path) = config.transcribe_cpp.model_path.as_mut() {
                *model_path = expand_tilde(model_path)?;
            }

            Ok(config)
        }
    }

    pub fn save(&self) -> Result<()> {
        let config_path = get_config_path()?;

        // Create config directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)?;
        fs::write(config_path, contents)?;
        Ok(())
    }

    pub fn ref_audio_path(&self) -> PathBuf {
        PathBuf::from(&self.storage.ref_audio)
    }

    pub fn model_dir_path(&self) -> PathBuf {
        PathBuf::from(&self.storage.model_dir)
    }
}

fn get_config_path() -> Result<PathBuf> {
    let config_dir = if let Some(xdg_config_home) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(xdg_config_home)
    } else {
        dirs::config_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine config directory"))?
    };

    Ok(config_dir.join("ears").join("config.toml"))
}

fn expand_tilde(path: &str) -> Result<String> {
    if path.starts_with('~') {
        let home_dir =
            dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;

        if path == "~" {
            Ok(home_dir.to_string_lossy().to_string())
        } else if path.starts_with("~/") {
            let rest = &path[2..];
            Ok(home_dir.join(rest).to_string_lossy().to_string())
        } else {
            Ok(path.to_string())
        }
    } else {
        Ok(path.to_string())
    }
}

pub async fn ensure_ref_audio(config: &AppConfig) -> Result<()> {
    let ref_audio_dir = config.ref_audio_path();

    // Create ref_audio directory if it doesn't exist
    fs::create_dir_all(&ref_audio_dir)?;

    // Only create custom model directory if it's not using the default
    if config.storage.model_dir != "default" {
        let model_dir = config.model_dir_path();
        fs::create_dir_all(&model_dir)?;
    }

    let required_files = ["esp.mp3", "ger.mp3", "jap.mp3", "ita.mp3", "por.mp3"];
    let repo_ref_audio_dir = PathBuf::from("ref_audio");

    for file in &required_files {
        let target_path = ref_audio_dir.join(file);
        if target_path.exists() {
            continue;
        }

        let source_path = repo_ref_audio_dir.join(file);
        if source_path.exists() {
            fs::copy(&source_path, &target_path)?;
            eprintln!("Copied {} from source to {}", file, target_path.display());
            continue;
        }

        // Fallback: write from embedded assets (no network)
        let bytes: Option<&'static [u8]> = match *file {
            "esp.mp3" => Some(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/ref_audio/esp.mp3"
            ))),
            "ger.mp3" => Some(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/ref_audio/ger.mp3"
            ))),
            "jap.mp3" => Some(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/ref_audio/jap.mp3"
            ))),
            "ita.mp3" => Some(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/ref_audio/ita.mp3"
            ))),
            "por.mp3" => Some(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/ref_audio/por.mp3"
            ))),
            _ => None,
        };
        if let Some(data) = bytes {
            fs::write(&target_path, data)?;
            eprintln!("Installed {} to {}", file, target_path.display());
        } else {
            eprintln!("Warning: missing embedded asset {}", file);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_rs_config_defaults_to_auto_without_model() {
        let config: AppConfig = toml::from_str(
            r#"
[storage]
model_dir = "default"
ref_audio = "~/.local/share/ears/ref_audio"
"#,
        )
        .expect("minimal config should deserialize");
        assert!(config.parakeet_rs.model_dir.is_none());
        assert_eq!(config.parakeet_rs.language, "auto");
    }

    #[test]
    fn example_config_deserializes() {
        let config: AppConfig = toml::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/config.toml"
        )))
        .expect("examples/config.toml must stay in sync with AppConfig");
        assert_eq!(config.parakeet_rs.language, "auto");
    }

    #[test]
    fn parakeet_rs_config_deserializes_model_and_language() {
        let config: AppConfig = toml::from_str(
            r#"
[storage]
model_dir = "default"
ref_audio = "~/.local/share/ears/ref_audio"

[parakeet_rs]
model_dir = "/models/nemotron"
language = "de"
"#,
        )
        .expect("parakeet-rs config should deserialize");
        assert_eq!(
            config.parakeet_rs.model_dir.as_deref(),
            Some("/models/nemotron")
        );
        assert_eq!(config.parakeet_rs.language, "de");
    }
}
