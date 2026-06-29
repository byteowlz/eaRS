use std::fs;
use std::path::{Path, PathBuf};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplacementConfig {
    pub enabled: bool,
    pub dictionary_paths: Vec<String>,
    pub case_sensitive: bool,
}

impl Default for ReplacementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dictionary_paths: vec![default_dictionary_path().to_string_lossy().to_string()],
            case_sensitive: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TranscriptHistoryConfig {
    pub enabled: bool,
    pub path: String,
    pub store_raw: bool,
    pub store_replaced: bool,
    pub store_unchanged_replaced: bool,
    pub flush_interval_ms: u64,
    pub flush_max_events: usize,
}

impl Default for TranscriptHistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "default".to_string(),
            store_raw: true,
            store_replaced: true,
            store_unchanged_replaced: false,
            flush_interval_ms: 2000,
            flush_max_events: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplacementDictionary {
    pub version: u32,
    pub entries: Vec<ReplacementEntry>,
}

impl Default for ReplacementDictionary {
    fn default() -> Self {
        Self {
            version: 1,
            entries: starter_entries(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplacementEntry {
    pub replace: String,
    pub phrases: Vec<String>,
}

pub struct ReplacementEngine {
    matcher: Option<AhoCorasick>,
    replacements: Vec<String>,
}

impl ReplacementEngine {
    pub fn empty() -> Self {
        Self {
            matcher: None,
            replacements: Vec::new(),
        }
    }

    pub fn from_config(config: &ReplacementConfig) -> Result<Self> {
        if !config.enabled {
            return Ok(Self::empty());
        }

        let mut phrases = Vec::new();
        let mut replacements = Vec::new();
        let paths = dictionary_paths(config);
        if !paths.iter().any(|path| path.exists()) {
            if let Some(path) = paths.first() {
                ReplacementDictionary::default().save(path)?;
            }
        }
        for path in paths {
            if !path.exists() {
                continue;
            }
            let dictionary = ReplacementDictionary::load(&path)
                .with_context(|| format!("failed to load dictionary {}", path.display()))?;
            for entry in dictionary.entries {
                for phrase in entry.phrases {
                    if phrase.trim().is_empty() {
                        continue;
                    }
                    phrases.push(phrase);
                    replacements.push(entry.replace.clone());
                }
            }
        }

        if phrases.is_empty() {
            return Ok(Self::empty());
        }

        let matcher = AhoCorasickBuilder::new()
            .ascii_case_insensitive(!config.case_sensitive)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&phrases)?;

        Ok(Self {
            matcher: Some(matcher),
            replacements,
        })
    }

    pub fn replace(&self, text: &str) -> String {
        let Some(matcher) = &self.matcher else {
            return text.to_string();
        };
        let mut out = String::with_capacity(text.len());
        let mut last_end = 0;
        for mat in matcher.find_iter(text) {
            if !has_text_boundaries(text, mat.start(), mat.end()) {
                continue;
            }
            out.push_str(&text[last_end..mat.start()]);
            out.push_str(&self.replacements[mat.pattern().as_usize()]);
            last_end = mat.end();
        }
        out.push_str(&text[last_end..]);
        out
    }

    pub fn is_empty(&self) -> bool {
        self.matcher.is_none()
    }
}

impl ReplacementDictionary {
    pub fn load(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        Ok(toml::from_str(&contents)?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load_or_create(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            let dictionary = Self::default();
            dictionary.save(path)?;
            Ok(dictionary)
        }
    }

    pub fn add_entry(&mut self, replacement: String, phrases: Vec<String>) {
        let normalized_phrases = normalize_phrases(phrases);

        // A phrase must only have one owner. If the user explicitly adds
        // `-r "Octo" -p "oqto"`, remove `oqto` from any previous replacement
        // so the dictionary does not depend on file/order precedence.
        for entry in &mut self.entries {
            if entry.replace == replacement {
                continue;
            }
            entry.phrases.retain(|existing| {
                !normalized_phrases
                    .iter()
                    .any(|phrase| phrase.eq_ignore_ascii_case(existing))
            });
        }
        self.entries
            .retain(|entry| !entry.phrases.is_empty() || entry.replace == replacement);

        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.replace == replacement)
        {
            for phrase in normalized_phrases {
                if !entry
                    .phrases
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(&phrase))
                {
                    entry.phrases.push(phrase);
                }
            }
            return;
        }
        self.entries.push(ReplacementEntry {
            replace: replacement,
            phrases: normalized_phrases,
        });
    }

    pub fn remove_replacement(&mut self, replacement: &str) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|entry| !entry.replace.eq_ignore_ascii_case(replacement));
        before - self.entries.len()
    }

    pub fn remove_phrases(&mut self, phrases: Vec<String>, replacement: Option<&str>) -> usize {
        let normalized_phrases = normalize_phrases(phrases);
        let mut removed = 0;
        for entry in &mut self.entries {
            if replacement
                .is_some_and(|replacement| !entry.replace.eq_ignore_ascii_case(replacement))
            {
                continue;
            }
            let before = entry.phrases.len();
            entry.phrases.retain(|existing| {
                !normalized_phrases
                    .iter()
                    .any(|phrase| phrase.eq_ignore_ascii_case(existing))
            });
            removed += before - entry.phrases.len();
        }
        self.entries.retain(|entry| !entry.phrases.is_empty());
        removed
    }
}

fn normalize_phrases(phrases: Vec<String>) -> Vec<String> {
    phrases
        .into_iter()
        .map(|phrase| phrase.trim().to_string())
        .filter(|phrase| !phrase.is_empty())
        .collect()
}

pub fn dictionary_paths(config: &ReplacementConfig) -> Vec<PathBuf> {
    config
        .dictionary_paths
        .iter()
        .map(|path| expand_tilde(path))
        .collect()
}

pub fn default_dictionary_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(dirs::config_dir)
        .unwrap_or_else(|| PathBuf::from(".config"));
    base.join("ears").join("dictionaries").join("global.toml")
}

pub fn transcript_history_dir(configured_path: &str) -> PathBuf {
    if configured_path != "default" {
        return expand_tilde(configured_path);
    }
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(dirs::state_dir)
        .or_else(|| dirs::home_dir().map(|home| home.join(".local/state")))
        .unwrap_or_else(|| PathBuf::from(".local/state"));
    base.join("ears").join("transcripts")
}

fn has_text_boundaries(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    !is_word_char(before) && !is_word_char(after)
}

fn is_word_char(ch: Option<char>) -> bool {
    ch.is_some_and(|ch| ch.is_alphanumeric() || ch == '_')
}

fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn starter_entries() -> Vec<ReplacementEntry> {
    vec![
        ReplacementEntry {
            replace: "eaRS".to_string(),
            phrases: vec!["ear S".to_string(), "EARS".to_string()],
        },
        ReplacementEntry {
            replace: "Kyutai".to_string(),
            phrases: vec![
                "cute eye".to_string(),
                "q tie".to_string(),
                "kyu tai".to_string(),
            ],
        },
        ReplacementEntry {
            replace: "Parakeet".to_string(),
            phrases: vec!["pair a key".to_string(), "para keet".to_string()],
        },
        ReplacementEntry {
            replace: "pi agent".to_string(),
            phrases: vec!["pie agent".to_string(), "py agent".to_string()],
        },
        ReplacementEntry {
            replace: "tmux".to_string(),
            phrases: vec!["team ups".to_string(), "t mux".to_string()],
        },
        ReplacementEntry {
            replace: "ctx".to_string(),
            phrases: vec!["see TX".to_string(), "CTX".to_string()],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_multiple_phrases_with_canonical_text() {
        let dir = std::env::temp_dir().join(format!("ears-dict-test-{}", uuid::Uuid::new_v4()));
        let path = dir.join("global.toml");
        let dictionary = ReplacementDictionary {
            version: 1,
            entries: vec![ReplacementEntry {
                replace: "Kyutai".to_string(),
                phrases: vec!["cute eye".to_string(), "q tie".to_string()],
            }],
        };
        dictionary.save(&path).unwrap();
        let engine = ReplacementEngine::from_config(&ReplacementConfig {
            enabled: true,
            dictionary_paths: vec![path.to_string_lossy().to_string()],
            case_sensitive: false,
        })
        .unwrap();

        assert_eq!(engine.replace("cute eye and Q TIE"), "Kyutai and Kyutai");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn uses_leftmost_longest_matches() {
        let dir = std::env::temp_dir().join(format!("ears-dict-test-{}", uuid::Uuid::new_v4()));
        let path = dir.join("global.toml");
        let dictionary = ReplacementDictionary {
            version: 1,
            entries: vec![
                ReplacementEntry {
                    replace: "pi".to_string(),
                    phrases: vec!["pie".to_string()],
                },
                ReplacementEntry {
                    replace: "pi agent".to_string(),
                    phrases: vec!["pie agent".to_string()],
                },
            ],
        };
        dictionary.save(&path).unwrap();
        let engine = ReplacementEngine::from_config(&ReplacementConfig {
            enabled: true,
            dictionary_paths: vec![path.to_string_lossy().to_string()],
            case_sensitive: false,
        })
        .unwrap();

        assert_eq!(engine.replace("pie agent"), "pi agent");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn add_entry_moves_conflicting_phrases_to_new_replacement() {
        let mut dictionary = ReplacementDictionary {
            version: 1,
            entries: vec![ReplacementEntry {
                replace: "oqto".to_string(),
                phrases: vec!["octo".to_string(), "OQTO".to_string()],
            }],
        };

        dictionary.add_entry("Octo".to_string(), vec!["octo".to_string()]);

        assert_eq!(dictionary.entries[0].phrases, vec!["OQTO".to_string()]);
        assert_eq!(dictionary.entries[1].replace, "Octo");
        assert_eq!(dictionary.entries[1].phrases, vec!["octo".to_string()]);
    }

    #[test]
    fn does_not_replace_inside_larger_words() {
        let dir = std::env::temp_dir().join(format!("ears-dict-test-{}", uuid::Uuid::new_v4()));
        let path = dir.join("global.toml");
        let dictionary = ReplacementDictionary {
            version: 1,
            entries: vec![ReplacementEntry {
                replace: "eaRS".to_string(),
                phrases: vec!["ears".to_string()],
            }],
        };
        dictionary.save(&path).unwrap();
        let engine = ReplacementEngine::from_config(&ReplacementConfig {
            enabled: true,
            dictionary_paths: vec![path.to_string_lossy().to_string()],
            case_sensitive: false,
        })
        .unwrap();

        assert_eq!(engine.replace("ears hears appears"), "eaRS hears appears");
        let _ = fs::remove_dir_all(dir);
    }
}
