use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub storage: StorageConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub whisper: WhisperConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub dictation: DictationConfig,
    #[serde(default)]
    pub hotkeys: HotkeyConfig,
    #[serde(default)]
    pub subs: SubsConfig,
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
pub struct WhisperConfig {
    pub enabled: bool,
    pub default_model: String,
    pub model_format: String,
    pub quantization: String,
    pub languages: Vec<String>,
    pub confidence_threshold: f32,
    pub storage_dir: String,
    pub sentence_detection: SentenceDetectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceDetectionConfig {
    pub min_duration: f64,
    pub max_duration: f64,
    pub vad_pause_threshold: f32,
    pub silence_duration: f64,
    pub punctuation_markers: Vec<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationConfig {
    pub enabled: bool,
    pub type_live_words: bool,
    #[serde(default)]
    pub notifications: DictationNotificationConfig,
    #[cfg(feature = "hooks")]
    #[serde(default)]
    pub hooks: DictationHooksConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub enable_internal: bool,
    pub toggle: String,
    pub language_cycle: String,
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
            language_cycle: "ctrl+shift+l".to_string(),
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
            whisper: WhisperConfig::default(),
            server: ServerConfig::default(),
            dictation: DictationConfig::default(),
            hotkeys: HotkeyConfig::default(),
            subs: SubsConfig::default(),
        }
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_model: "large-v3-turbo".to_string(),
            model_format: "gguf".to_string(),
            quantization: "Q5_0".to_string(), // Use Q5_0 for whisper.cpp models
            languages: vec!["de".to_string(), "ja".to_string(), "it".to_string()],
            confidence_threshold: 0.7,
            storage_dir: "default".to_string(), // Use HF cache
            sentence_detection: SentenceDetectionConfig::default(),
        }
    }
}

impl Default for SentenceDetectionConfig {
    fn default() -> Self {
        Self {
            min_duration: 1.0,
            max_duration: 30.0,
            vad_pause_threshold: 0.8,
            silence_duration: 0.5,
            punctuation_markers: vec![
                ".".to_string(),
                "!".to_string(),
                "?".to_string(),
                "。".to_string(),
                "！".to_string(),
                "？".to_string(),
            ],
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

            // Try to parse the config
            let mut config: AppConfig = match toml::from_str(&contents) {
                Ok(c) => c,
                Err(e) => {
                    // If parsing fails due to missing whisper field, try to migrate
                    if contents.contains("[storage]") && !contents.contains("[whisper]") {
                        eprintln!("Migrating config file to include Whisper settings...");

                        // Parse just the storage section
                        #[derive(Deserialize)]
                        struct OldConfig {
                            storage: StorageConfig,
                        }

                        let old_config: OldConfig = toml::from_str(&contents)?;

                        let new_config = AppConfig {
                            storage: old_config.storage,
                            model: ModelConfig::default(),
                            whisper: WhisperConfig::default(),
                            server: ServerConfig::default(),
                            dictation: DictationConfig::default(),
                            hotkeys: HotkeyConfig::default(),
                            subs: SubsConfig::default(),
                        };

                        // Save the updated config
                        new_config.save()?;
                        eprintln!("Config file updated with Whisper defaults");

                        new_config
                    } else {
                        return Err(anyhow::anyhow!("Failed to parse config: {}", e));
                    }
                }
            };

            // Expand tilde paths
            config.storage.model_dir = expand_tilde(&config.storage.model_dir)?;
            config.storage.ref_audio = expand_tilde(&config.storage.ref_audio)?;
            if config.whisper.storage_dir != "default" {
                config.whisper.storage_dir = expand_tilde(&config.whisper.storage_dir)?;
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

    pub fn whisper_storage_path(&self) -> PathBuf {
        if self.whisper.storage_dir == "default" {
            // Use a subdirectory in the cache for whisper models
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("~/.cache"))
                .join("huggingface")
                .join("whisper-models")
        } else {
            PathBuf::from(&self.whisper.storage_dir)
        }
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
