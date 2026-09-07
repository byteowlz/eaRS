//! Context profiles: per-application dictation settings.
//!
//! A profile binds a set of match rules (bundle id, binary name, window-title
//! regex) to a dictionary set, a language and an insertion mode. Profiles are
//! evaluated engine-side against the frontmost application (see
//! [`crate::frontmost`]) so that both the CLI (`ears profile ...`) and the
//! companion UI see the same behaviour.
//!
//! Profiles live in `~/.config/ears/profiles.toml`:
//!
//! ```toml
//! version = 1
//!
//! [[profiles]]
//! name = "coding"
//! priority = 10
//! dictionaries = ["~/.config/ears/dictionaries/code.toml"]
//! language = "en"
//! insertion_mode = "insert_at_cursor"
//!
//! [profiles.match]
//! bundle_id = ["com.microsoft.VSCode", "dev.zed.Zed"]
//! binary = ["nvim"]
//! window_title = "(?i)\\.rs\\b"
//! ```
//!
//! All specified criteria of a rule must match (AND); list criteria match any
//! entry (OR). A rule without any criterion never matches. The highest
//! priority matching profile wins; ties keep file order.

use crate::dictation::InsertionMode;
use crate::frontmost::FrontmostApp;
use anyhow::Context;
use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

/// Match rules for a profile.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileMatch {
    /// macOS bundle identifiers (case-insensitive exact match).
    pub bundle_id: Vec<String>,
    /// Executable / application names (case-insensitive exact match against
    /// the binary name or the application name).
    pub binary: Vec<String>,
    /// Regular expression matched against the focused window title.
    pub window_title: Option<String>,
}

impl ProfileMatch {
    /// Whether this rule has at least one criterion.
    pub fn is_empty(&self) -> bool {
        self.bundle_id.is_empty() && self.binary.is_empty() && self.window_title.is_none()
    }

    /// Compile the window title regex, if any.
    pub fn compile_title(&self) -> Result<Option<Regex>> {
        match &self.window_title {
            Some(pattern) => Regex::new(pattern)
                .map(Some)
                .with_context(|| format!("invalid window_title regex '{pattern}'")),
            None => Ok(None),
        }
    }

    fn matches(&self, app: &FrontmostApp, title_regex: Option<&Regex>) -> bool {
        if self.is_empty() {
            return false;
        }
        if !self.bundle_id.is_empty() {
            let Some(bundle) = app.bundle_id.as_deref() else {
                return false;
            };
            if !self
                .bundle_id
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(bundle))
            {
                return false;
            }
        }
        if !self.binary.is_empty() {
            let names: Vec<&str> = app
                .binary
                .as_deref()
                .into_iter()
                .chain(app.name.as_deref())
                .collect();
            if names.is_empty() {
                return false;
            }
            let hit = self.binary.iter().any(|candidate| {
                names.iter().any(|name| {
                    name.eq_ignore_ascii_case(candidate)
                        || std::path::Path::new(name)
                            .file_stem()
                            .map(|stem| stem.to_string_lossy().eq_ignore_ascii_case(candidate))
                            .unwrap_or(false)
                })
            });
            if !hit {
                return false;
            }
        }
        if self.window_title.is_some() {
            let Some(regex) = title_regex else {
                return false;
            };
            let Some(title) = app.window_title.as_deref() else {
                return false;
            };
            if !regex.is_match(title) {
                return false;
            }
        }
        true
    }
}

/// A single context profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextProfile {
    /// Unique, human readable name.
    pub name: String,
    /// Higher priority wins when several profiles match.
    pub priority: i32,
    /// Match rules.
    #[serde(rename = "match")]
    pub matcher: ProfileMatch,
    /// Dictionary paths used while this profile is active. Empty keeps the
    /// global `replacement.dictionary_paths` from `config.toml`.
    pub dictionaries: Vec<String>,
    /// Transcription language override (e.g. `en`, `de`).
    pub language: Option<String>,
    /// Insertion mode override.
    pub insertion_mode: Option<InsertionMode>,
}

impl Default for ContextProfile {
    fn default() -> Self {
        Self {
            name: String::new(),
            priority: 0,
            matcher: ProfileMatch::default(),
            dictionaries: Vec::new(),
            language: None,
            insertion_mode: None,
        }
    }
}

/// On-disk profile list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfilesConfig {
    pub version: u32,
    pub profiles: Vec<ContextProfile>,
}

impl Default for ProfilesConfig {
    fn default() -> Self {
        Self {
            version: 1,
            profiles: Vec::new(),
        }
    }
}

/// Path of the profiles file (`$XDG_CONFIG_HOME/ears/profiles.toml`).
pub fn profiles_path() -> PathBuf {
    let config_dir = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(dirs::config_dir)
        .unwrap_or_else(|| PathBuf::from("."));
    config_dir.join("ears").join("profiles.toml")
}

impl ProfilesConfig {
    /// Load the profiles file, returning an empty list when it does not exist.
    pub fn load() -> Result<Self> {
        Self::load_from(&profiles_path())
    }

    pub fn load_from(path: &std::path::Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read profiles file {}", path.display()))?;
        let config: Self = toml::from_str(&contents)
            .with_context(|| format!("failed to parse profiles file {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&profiles_path())
    }

    pub fn save_to(&self, path: &std::path::Path) -> Result<()> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self).context("failed to serialize profiles")?;
        fs::write(path, contents)
            .with_context(|| format!("failed to write profiles file {}", path.display()))?;
        Ok(())
    }

    /// Check names are unique and non-empty and every regex compiles.
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for profile in &self.profiles {
            if profile.name.trim().is_empty() {
                anyhow::bail!("profile with empty name");
            }
            if !seen.insert(profile.name.to_lowercase()) {
                anyhow::bail!("duplicate profile name '{}'", profile.name);
            }
            profile
                .matcher
                .compile_title()
                .with_context(|| format!("profile '{}'", profile.name))?;
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ContextProfile> {
        self.profiles
            .iter()
            .find(|profile| profile.name.eq_ignore_ascii_case(name))
    }

    /// Insert or replace a profile by name.
    pub fn upsert(&mut self, profile: ContextProfile) {
        if let Some(existing) = self
            .profiles
            .iter_mut()
            .find(|p| p.name.eq_ignore_ascii_case(&profile.name))
        {
            *existing = profile;
        } else {
            self.profiles.push(profile);
        }
    }

    /// Remove a profile by name. Returns whether something was removed.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.profiles.len();
        self.profiles
            .retain(|profile| !profile.name.eq_ignore_ascii_case(name));
        before != self.profiles.len()
    }

    /// Names in priority order (highest first, then file order).
    pub fn names_by_priority(&self) -> Vec<String> {
        let mut indexed: Vec<(usize, &ContextProfile)> = self.profiles.iter().enumerate().collect();
        indexed.sort_by(|(ia, a), (ib, b)| b.priority.cmp(&a.priority).then(ia.cmp(ib)));
        indexed.into_iter().map(|(_, p)| p.name.clone()).collect()
    }
}

/// Compiled profile set ready for repeated evaluation.
#[derive(Debug, Clone)]
pub struct ProfileSet {
    config: ProfilesConfig,
    /// Indices into `config.profiles` sorted by priority (highest first).
    order: Vec<usize>,
    title_regexes: Vec<Option<Regex>>,
}

impl ProfileSet {
    pub fn new(config: ProfilesConfig) -> Result<Self> {
        config.validate()?;
        let title_regexes = config
            .profiles
            .iter()
            .map(|profile| profile.matcher.compile_title())
            .collect::<Result<Vec<_>>>()?;
        let mut order: Vec<usize> = (0..config.profiles.len()).collect();
        order.sort_by(|&a, &b| {
            config.profiles[b]
                .priority
                .cmp(&config.profiles[a].priority)
                .then(a.cmp(&b))
        });
        Ok(Self {
            config,
            order,
            title_regexes,
        })
    }

    pub fn empty() -> Self {
        Self {
            config: ProfilesConfig::default(),
            order: Vec::new(),
            title_regexes: Vec::new(),
        }
    }

    pub fn config(&self) -> &ProfilesConfig {
        &self.config
    }

    pub fn is_empty(&self) -> bool {
        self.config.profiles.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&ContextProfile> {
        self.config.get(name)
    }

    /// Profiles in priority order.
    pub fn ordered(&self) -> impl Iterator<Item = &ContextProfile> {
        self.order.iter().map(|&idx| &self.config.profiles[idx])
    }

    /// The highest priority profile matching `app`.
    pub fn resolve(&self, app: &FrontmostApp) -> Option<&ContextProfile> {
        self.order.iter().copied().find_map(|idx| {
            let profile = &self.config.profiles[idx];
            profile
                .matcher
                .matches(app, self.title_regexes[idx].as_ref())
                .then_some(profile)
        })
    }
}

/// Reloads the profile file when its modification time changes.
#[derive(Debug)]
pub struct ProfileReloader {
    path: PathBuf,
    set: ProfileSet,
    stamp: Option<SystemTime>,
}

impl ProfileReloader {
    pub fn new() -> Self {
        Self::at(profiles_path())
    }

    pub fn at(path: PathBuf) -> Self {
        let set = ProfilesConfig::load_from(&path)
            .and_then(ProfileSet::new)
            .unwrap_or_else(|err| {
                eprintln!("warning: failed to load context profiles: {err}");
                ProfileSet::empty()
            });
        let stamp = stamp_of(&path);
        Self { path, set, stamp }
    }

    /// Current set, reloading first if the file changed. Returns whether the
    /// set was reloaded.
    pub fn refresh(&mut self) -> bool {
        let stamp = stamp_of(&self.path);
        if stamp == self.stamp {
            return false;
        }
        self.stamp = stamp;
        match ProfilesConfig::load_from(&self.path).and_then(ProfileSet::new) {
            Ok(set) => {
                self.set = set;
                true
            }
            Err(err) => {
                eprintln!("warning: failed to hot-reload context profiles: {err}");
                false
            }
        }
    }

    pub fn set(&self) -> &ProfileSet {
        &self.set
    }
}

impl Default for ProfileReloader {
    fn default() -> Self {
        Self::new()
    }
}

fn stamp_of(path: &std::path::Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|meta| meta.modified()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(bundle: &str, binary: &str, title: &str) -> FrontmostApp {
        FrontmostApp {
            bundle_id: Some(bundle.to_string()),
            name: Some(binary.to_string()),
            binary: Some(binary.to_string()),
            window_title: Some(title.to_string()),
            pid: Some(1),
        }
    }

    fn profile(name: &str, priority: i32, matcher: ProfileMatch) -> ContextProfile {
        ContextProfile {
            name: name.to_string(),
            priority,
            matcher,
            ..Default::default()
        }
    }

    #[test]
    fn resolves_highest_priority_match() {
        let set = ProfileSet::new(ProfilesConfig {
            version: 1,
            profiles: vec![
                profile(
                    "editor",
                    1,
                    ProfileMatch {
                        bundle_id: vec!["com.microsoft.VSCode".into()],
                        ..Default::default()
                    },
                ),
                profile(
                    "rust",
                    5,
                    ProfileMatch {
                        bundle_id: vec!["com.microsoft.vscode".into()],
                        window_title: Some(r"\.rs\b".into()),
                        ..Default::default()
                    },
                ),
            ],
        })
        .unwrap();

        let rust = set
            .resolve(&app("com.microsoft.VSCode", "Code", "main.rs — eaRS"))
            .unwrap();
        assert_eq!(rust.name, "rust");
        let editor = set
            .resolve(&app("com.microsoft.VSCode", "Code", "README.md"))
            .unwrap();
        assert_eq!(editor.name, "editor");
        assert!(
            set.resolve(&app("com.apple.Safari", "Safari", "x"))
                .is_none()
        );
    }

    #[test]
    fn binary_matches_name_or_stem_case_insensitively() {
        let set = ProfileSet::new(ProfilesConfig {
            version: 1,
            profiles: vec![profile(
                "term",
                0,
                ProfileMatch {
                    binary: vec!["ghostty".into()],
                    ..Default::default()
                },
            )],
        })
        .unwrap();
        assert!(set.resolve(&app("", "Ghostty", "zsh")).is_some());
        assert!(set.resolve(&app("", "kitty", "zsh")).is_none());
    }

    #[test]
    fn empty_rule_never_matches_and_invalid_regex_is_rejected() {
        let set = ProfileSet::new(ProfilesConfig {
            version: 1,
            profiles: vec![profile("nothing", 0, ProfileMatch::default())],
        })
        .unwrap();
        assert!(set.resolve(&app("a", "b", "c")).is_none());

        let bad = ProfilesConfig {
            version: 1,
            profiles: vec![profile(
                "bad",
                0,
                ProfileMatch {
                    window_title: Some("(".into()),
                    ..Default::default()
                },
            )],
        };
        assert!(ProfileSet::new(bad).is_err());
    }

    #[test]
    fn roundtrips_through_toml() {
        let config = ProfilesConfig {
            version: 1,
            profiles: vec![ContextProfile {
                name: "coding".into(),
                priority: 10,
                matcher: ProfileMatch {
                    bundle_id: vec!["dev.zed.Zed".into()],
                    binary: vec![],
                    window_title: Some("(?i)rust".into()),
                },
                dictionaries: vec!["~/.config/ears/dictionaries/code.toml".into()],
                language: Some("en".into()),
                insertion_mode: Some(InsertionMode::Clipboard),
            }],
        };
        let text = toml::to_string_pretty(&config).unwrap();
        let parsed: ProfilesConfig = toml::from_str(&text).unwrap();
        assert_eq!(parsed, config);
    }
}
