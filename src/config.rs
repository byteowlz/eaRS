use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub storage: StorageConfig,
    #[serde(default)]
    pub whisper: WhisperConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub hotkeys: HotkeyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub model_dir: String,
    pub ref_audio: String,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub enable_internal: bool,
    pub toggle: String,
    pub language_cycle: String,
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
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                model_dir: "default".to_string(), // "default" means use HuggingFace default cache
                ref_audio: "~/.local/share/ears/ref_audio".to_string(),
            },
            whisper: WhisperConfig::default(),
            server: ServerConfig::default(),
            hotkeys: HotkeyConfig::default(),
        }
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_model: "large-v3-turbo".to_string(),
            model_format: "gguf".to_string(),
            quantization: "Q5_0".to_string(),  // Use Q5_0 for whisper.cpp models
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
                ".".to_string(), "!".to_string(), "?".to_string(),
                "。".to_string(), "！".to_string(), "？".to_string(),
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
                        
                        // Create new config with defaults for whisper
                        let new_config = AppConfig {
                            storage: old_config.storage,
                            whisper: WhisperConfig::default(),
                            server: ServerConfig::default(),
                            hotkeys: HotkeyConfig::default(),
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
        dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine config directory"))?
    };
    
    Ok(config_dir.join("ears").join("config.toml"))
}

fn expand_tilde(path: &str) -> Result<String> {
    if path.starts_with('~') {
        let home_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
        
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
        if target_path.exists() { continue; }
        
        let source_path = repo_ref_audio_dir.join(file);
        if source_path.exists() {
            fs::copy(&source_path, &target_path)?;
            eprintln!("Copied {} from source to {}", file, target_path.display());
            continue;
        }
        
        // Fallback: write from embedded assets (no network)
        let bytes: Option<&'static [u8]> = match *file {
            "esp.mp3" => Some(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/ref_audio/esp.mp3"))),
            "ger.mp3" => Some(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/ref_audio/ger.mp3"))),
            "jap.mp3" => Some(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/ref_audio/jap.mp3"))),
            "ita.mp3" => Some(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/ref_audio/ita.mp3"))),
            "por.mp3" => Some(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/ref_audio/por.mp3"))),
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

async fn download_file(url: &str, path: &PathBuf) -> Result<()> {
    let response = reqwest::get(url).await?;
    
    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Failed to download {}: {}", url, response.status()));
    }
    
    let bytes = response.bytes().await?;
    fs::write(path, bytes)?;
    Ok(())
}
