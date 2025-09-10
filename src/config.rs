use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub model_dir: String,
    pub ref_audio: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                model_dir: "default".to_string(), // "default" means use HuggingFace default cache
                ref_audio: "~/.local/share/ears/ref_audio".to_string(),
            },
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
            let mut config: AppConfig = toml::from_str(&contents)?;
            
            // Expand tilde paths
            config.storage.model_dir = expand_tilde(&config.storage.model_dir)?;
            config.storage.ref_audio = expand_tilde(&config.storage.ref_audio)?;
            
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
    
    let required_files = ["esp.mp3", "ger.mp3", "jap.mp3"];
    let repo_ref_audio_dir = PathBuf::from("ref_audio");
    
    for file in &required_files {
        let target_path = ref_audio_dir.join(file);
        
        // Skip if file already exists
        if target_path.exists() {
            continue;
        }
        
        let source_path = repo_ref_audio_dir.join(file);
        
        // Check if we're building from source (ref_audio directory exists)
        if source_path.exists() {
            // Copy from source
            fs::copy(&source_path, &target_path)?;
            eprintln!("Copied {} from source to {}", file, target_path.display());
        } else {
            // Download from GitHub
            let url = format!("https://raw.githubusercontent.com/byteowlz/eaRS/main/ref_audio/{}", file);
            download_file(&url, &target_path).await?;
            eprintln!("Downloaded {} from GitHub to {}", file, target_path.display());
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