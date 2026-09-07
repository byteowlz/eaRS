//! `~/.config/ears/ui.toml`: preferences that belong to the companion UI.
//!
//! Everything about *dictation* (hotkeys, servers, dictionaries, transcripts)
//! stays in the core `config.toml`; this file only holds overlay, tray and
//! theme preferences plus the few dictation options the UI exposes.

use anyhow::Context;
use anyhow::Result;
use ears::config::DictationHotkeyMode;
use ears::dictation::InsertionMode;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct UiConfig {
    pub overlay: OverlayConfig,
    pub dictation: DictationUiConfig,
    pub theme: ThemeConfig,
    pub tray: TrayConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverlayAnchor {
    #[default]
    Top,
    Bottom,
}

impl OverlayAnchor {
    pub const ALL: [OverlayAnchor; 2] = [OverlayAnchor::Top, OverlayAnchor::Bottom];

    pub fn label(self) -> &'static str {
        match self {
            OverlayAnchor::Top => "Top center",
            OverlayAnchor::Bottom => "Bottom center",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OverlayConfig {
    /// Show the floating island at all. The tray keeps working when off.
    pub enabled: bool,
    pub anchor: OverlayAnchor,
    /// Distance from the screen edge in logical pixels.
    pub margin: f32,
    /// How long the island stays after dictation stops.
    pub linger_ms: u64,
    /// Pill background opacity (0.2 - 1.0).
    pub opacity: f32,
    /// Morph/appear animations. Ignored when the OS asks for reduced motion.
    pub animation: bool,
    /// Let clicks pass through the island (disables drag + click-to-open).
    pub click_through: bool,
    /// Keep a tiny dot visible while dictation is paused.
    pub dormant_dot: bool,
    /// macOS only: hug the camera notch like a dynamic island.
    pub island_mode: bool,
    /// Drag offset relative to the anchor position, persisted after a drag.
    pub offset_x: f32,
    pub offset_y: f32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            anchor: OverlayAnchor::Top,
            margin: 12.0,
            linger_ms: 900,
            opacity: 0.96,
            animation: true,
            click_through: false,
            dormant_dot: false,
            island_mode: false,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DictationUiConfig {
    /// Server alias (from core `dictation.servers`) or `ws://` URL. Empty =
    /// core default.
    pub server: String,
    pub insertion_mode: InsertionMode,
    pub escape_cancels: bool,
    /// Overrides `hotkeys.mode` from the core config when set.
    pub hotkey_mode: Option<DictationHotkeyMode>,
    /// Evaluate context profiles.
    pub profiles: bool,
}

impl Default for DictationUiConfig {
    fn default() -> Self {
        Self {
            server: String::new(),
            insertion_mode: InsertionMode::InsertAtCursor,
            escape_cancels: true,
            hotkey_mode: None,
            profiles: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ThemeConfig {
    /// `auto` (follow the OS), a built-in id (`oqto-dark`, `oqto-light`) or a
    /// path to a design-system scheme JSON file.
    pub scheme: String,
    /// Radius dial in pixels. 0 = sharp everywhere.
    pub radius: f32,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            scheme: "auto".to_string(),
            radius: 8.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TrayConfig {
    pub launch_at_login: bool,
}

impl UiConfig {
    pub fn path() -> PathBuf {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(dirs::config_dir)
            .unwrap_or_else(|| PathBuf::from(".config"));
        base.join("ears").join("ui.toml")
    }

    pub fn load() -> Result<Self> {
        let path = Self::path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let mut config: Self =
            toml::from_str(&contents).with_context(|| format!("parsing {}", path.display()))?;
        config.sanitize();
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, toml::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    fn sanitize(&mut self) {
        self.overlay.opacity = self.overlay.opacity.clamp(0.2, 1.0);
        self.overlay.margin = self.overlay.margin.clamp(0.0, 400.0);
        self.overlay.linger_ms = self.overlay.linger_ms.min(60_000);
        self.theme.radius = self.theme.radius.clamp(0.0, 32.0);
    }
}
