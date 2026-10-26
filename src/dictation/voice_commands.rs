//! Deterministic, client-side utterance routing. The STT server stays unaware of commands.
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::PathBuf, time::SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VoiceMode {
    #[default]
    Dictation,
    CommandAware,
    CommandOnly,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub phrase: String,
    pub risk: Risk,
    pub action: Action,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Risk {
    Safe,
    Consequential,
    Destructive,
}

/// Only a literal, config-authored text action is currently supported. No
/// capture groups, interpolation, shell, or executable lookup is possible.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    LiteralText { text: String },
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandFile {
    #[serde(default)]
    command: Vec<Definition>,
}

pub fn default_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(dirs::config_dir)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("ears").join("commands.toml")
}

/// Invalid or ambiguous registries fail closed: never retain old commands.
/// Reads are only performed at utterance boundaries and explicit reloads.
fn file_stamp(path: &std::path::Path) -> Option<(SystemTime, u64)> {
    let metadata = fs::metadata(path).ok()?;
    Some((metadata.modified().ok()?, metadata.len()))
}

pub struct Registry {
    path: PathBuf,
    stamp: Option<(SystemTime, u64)>,
    definitions: Vec<Definition>,
}

impl Registry {
    pub fn new(path: PathBuf) -> Self {
        let mut registry = Self {
            path,
            stamp: None,
            definitions: Vec::new(),
        };
        registry.reload();
        registry
    }

    pub fn reload(&mut self) {
        self.stamp = file_stamp(&self.path);
        self.definitions.clear();
        if !self.path.exists() {
            return;
        }
        let result = (|| -> anyhow::Result<Vec<Definition>> {
            let source = fs::read_to_string(&self.path)?;
            let definitions = toml::from_str::<CommandFile>(&source)?.command;
            let mut seen = HashSet::new();
            for entry in &definitions {
                anyhow::ensure!(
                    entry.risk == Risk::Safe,
                    "only safe actions are supported; confirmation is not implemented"
                );
                let phrase = canonical(&entry.phrase);
                anyhow::ensure!(
                    !phrase.is_empty() && !phrase.starts_with("ears ") && phrase != "ears",
                    "empty or reserved command phrase"
                );
                anyhow::ensure!(seen.insert(phrase), "ambiguous command phrase");
                match &entry.action {
                    Action::LiteralText { text } => {
                        anyhow::ensure!(
                            !text.is_empty() && !text.chars().any(char::is_control),
                            "empty or control-character action text"
                        )
                    }
                }
            }
            Ok(definitions)
        })();
        match result {
            Ok(definitions) => self.definitions = definitions,
            Err(err) => eprintln!(
                "eaRS command registry rejected ({}): {err:#}",
                self.path.display()
            ),
        }
    }

    pub fn reload_if_changed(&mut self) {
        let stamp = file_stamp(&self.path);
        if stamp != self.stamp {
            self.reload();
        }
    }

    fn match_phrase(&self, normalized: &str) -> Option<&Action> {
        let key = canonical(normalized);
        self.definitions
            .iter()
            .find(|def| canonical(&def.phrase) == key)
            .map(|def| &def.action)
    }
}

/// Canonical whole-utterance comparison; punctuation at the edges is ignored,
/// but arbitrary interior punctuation and word order are significant.
pub fn canonical(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| matches!(c, '.' | '!' | '?' | ','))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

const CONTROLS: [&str; 7] = [
    "ears dictation mode",
    "ears command-aware mode",
    "ears command-only mode",
    "ears start typing",
    "ears stop typing",
    "ears cancel",
    "ears what mode",
];

#[derive(Debug, PartialEq, Eq)]
pub enum Route {
    Type(String),
    Action(String),
    Control(String),
    Discard,
    /// Already inserted incrementally; never reclassify at the boundary.
    Streamed,
}

pub struct Router {
    pub mode: VoiceMode,
    /// Temporary live typing in command-only mode.
    typing: bool,
    pending: Vec<String>,
    streaming: bool,
}

impl Router {
    pub fn new(mode: VoiceMode) -> Self {
        Self {
            mode,
            typing: false,
            pending: Vec::new(),
            streaming: false,
        }
    }

    fn live(&self) -> bool {
        self.mode == VoiceMode::Dictation || self.typing
    }

    /// Returns words safe to insert now. A possible reserved prefix alone is
    /// held; all other words take the existing low-latency dictation path.
    pub fn word(&mut self, word: &str) -> Option<String> {
        if self.streaming {
            return Some(word.to_owned());
        }
        self.pending.push(word.to_owned());
        if !self.live() {
            return None;
        }
        let candidate = canonical(&self.pending.join(" "));
        if CONTROLS
            .iter()
            .any(|control| *control == candidate || control.starts_with(&(candidate.clone() + " ")))
        {
            None
        } else {
            self.streaming = true;
            Some(std::mem::take(&mut self.pending).join(" "))
        }
    }

    pub fn finish(&mut self, normalized: &str, registry: &Registry) -> Route {
        let raw = self.pending.join(" ");
        self.pending.clear();
        let streamed = std::mem::replace(&mut self.streaming, false);
        // A control is eligible only if the leading raw token was eaRS and
        // nothing has already been inserted. Never retract speculative keys.
        let key = canonical(normalized);
        if !streamed
            && canonical(&raw).split_whitespace().next() == Some("ears")
            && CONTROLS.contains(&key.as_str())
        {
            self.control(&key);
            return Route::Control(key);
        }
        if streamed {
            return Route::Streamed; // Already typed; never reclassify.
        }
        if self.live() || self.mode == VoiceMode::CommandAware {
            if !self.live() {
                if let Some(Action::LiteralText { text }) = registry.match_phrase(normalized) {
                    return Route::Action(text.clone());
                }
            }
            Route::Type(normalized.to_owned())
        } else {
            if let Some(Action::LiteralText { text }) = registry.match_phrase(normalized) {
                return Route::Action(text.clone());
            }
            Route::Discard
        }
    }

    fn control(&mut self, key: &str) {
        match key {
            "ears dictation mode" => {
                self.mode = VoiceMode::Dictation;
                self.typing = false;
            }
            "ears command-aware mode" => {
                self.mode = VoiceMode::CommandAware;
                self.typing = false;
            }
            "ears command-only mode" => {
                self.mode = VoiceMode::CommandOnly;
                self.typing = false;
            }
            "ears start typing" if self.mode == VoiceMode::CommandOnly => self.typing = true,
            "ears stop typing" => self.typing = false,
            "ears cancel" => self.cancel(),
            _ => {}
        }
    }

    pub fn discard_pending(&mut self) {
        self.pending.clear();
        self.streaming = false;
    }

    pub fn cancel(&mut self) {
        self.discard_pending();
        self.typing = false;
    }

    pub fn status(&self) -> String {
        format!(
            "{:?}{}",
            self.mode,
            if self.typing { " (typing)" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn registry() -> Registry {
        let path =
            std::env::temp_dir().join(format!("ears-commands-{}.toml", uuid::Uuid::new_v4()));
        fs::write(&path, "[[command]]\nphrase = 'open notes'\nrisk = 'safe'\naction = { type = 'literal_text', text = 'NOTES' }\n").unwrap();
        Registry::new(path)
    }
    #[test]
    fn routing_and_prefix() {
        let reg = registry();
        let mut r = Router::new(VoiceMode::Dictation);
        assert_eq!(r.word("hello"), Some("hello".into()));
        assert_eq!(r.word("world"), Some("world".into()));
        assert_eq!(r.finish("hello world", &reg), Route::Streamed);
        assert_eq!(r.word("eaRS"), None);
        assert_eq!(r.word("command-aware"), None);
        assert_eq!(r.word("mode"), None);
        assert_eq!(
            r.finish("eaRS command-aware mode", &reg),
            Route::Control("ears command-aware mode".into())
        );
        assert_eq!(r.mode, VoiceMode::CommandAware);
        assert_eq!(r.word("open"), None);
        assert_eq!(r.word("notes"), None);
        assert_eq!(r.finish("open notes", &reg), Route::Action("NOTES".into()));
        assert_eq!(r.word("unknown"), None);
        assert_eq!(r.finish("unknown", &reg), Route::Type("unknown".into()));
        assert_eq!(r.word("eaRS"), None);
        assert_eq!(r.word("command-only"), None);
        assert_eq!(r.word("mode"), None);
        r.finish("eaRS command-only mode", &reg);
        assert_eq!(r.word("unknown"), None);
        assert_eq!(r.finish("unknown", &reg), Route::Discard);
        for word in ["eaRS", "start", "typing"] {
            assert_eq!(r.word(word), None);
        }
        r.finish("eaRS start typing", &reg);
        assert_eq!(r.word("hello"), Some("hello".into()));
        r.finish("hello", &reg);
        for word in ["eaRS", "stop", "typing"] {
            assert_eq!(r.word(word), None);
        }
        r.finish("eaRS stop typing", &reg);
        assert_eq!(r.word("hello"), None);
        assert_eq!(r.finish("hello", &reg), Route::Discard);
        for word in ["eaRS", "dictation", "mode"] {
            r.word(word);
        }
        r.finish("eaRS dictation mode", &reg);
        assert_eq!(r.mode, VoiceMode::Dictation);
    }
    #[test]
    fn offline_labeled_replay_checks_deterministic_baseline() {
        let reg = registry();
        for line in include_str!("../../docs/voice-command-replay.jsonl").lines() {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            let mode = match row["mode"].as_str().unwrap() {
                "dictation" => VoiceMode::Dictation,
                "command_aware" => VoiceMode::CommandAware,
                "command_only" => VoiceMode::CommandOnly,
                other => panic!("unknown fixture mode: {other}"),
            };
            let mut router = Router::new(mode);
            for word in row["words"].as_array().unwrap() {
                router.word(word.as_str().unwrap());
            }
            let outcome = match router.finish(row["normalized"].as_str().unwrap(), &reg) {
                Route::Type(_) => "typed",
                Route::Action(_) => "literal_text",
                Route::Control(_) => "control",
                Route::Discard => "discarded",
                Route::Streamed => "streamed",
            };
            assert_eq!(outcome, row["expected"].as_str().unwrap(), "{line}");
            assert!(matches!(
                row["label"].as_str().unwrap(),
                "literal_dictation" | "registered_command_candidate" | "uncertain"
            ));
        }
    }

    #[test]
    fn invalid_registry_fails_closed() {
        let mut reg = registry();
        fs::write(&reg.path, "[[command]]\nphrase='open notes'\nrisk='safe'\naction={type='literal_text',text='x'}\n[[command]]\nphrase='OPEN  NOTES'\nrisk='safe'\naction={type='literal_text',text='y'}").unwrap();
        reg.reload();
        assert!(reg.match_phrase("open notes").is_none());
        let mut r = Router::new(VoiceMode::Dictation);
        assert_eq!(r.word("eaRS"), None);
        assert_eq!(r.word("not"), Some("eaRS not".into()));
        assert_eq!(r.finish("eaRS not", &reg), Route::Streamed);
    }

    #[test]
    fn reserved_controls_are_whole_utterances_in_every_mode() {
        let reg = registry();
        for mode in [
            VoiceMode::Dictation,
            VoiceMode::CommandAware,
            VoiceMode::CommandOnly,
        ] {
            for control in CONTROLS {
                let mut router = Router::new(mode);
                for word in control.split_whitespace() {
                    router.word(word);
                }
                assert_eq!(router.finish(control, &reg), Route::Control(control.into()));
                let mut router = Router::new(mode);
                for word in format!("{control} now").split_whitespace() {
                    router.word(word);
                }
                assert!(!matches!(
                    router.finish(&format!("{control} now"), &reg),
                    Route::Control(_)
                ));
            }
        }
    }

    #[test]
    fn registry_hot_reload_replaces_and_removes_actions() {
        let mut reg = registry();
        assert!(reg.match_phrase("open notes").is_some());
        fs::write(&reg.path, "[[command]]\nphrase='close notes'\nrisk='safe'\naction={type='literal_text',text='CLOSED'}\n").unwrap();
        reg.reload_if_changed();
        assert!(reg.match_phrase("open notes").is_none());
        assert!(reg.match_phrase("close notes").is_some());
        fs::remove_file(&reg.path).unwrap();
        reg.reload_if_changed();
        assert!(reg.match_phrase("close notes").is_none());
    }

    #[test]
    fn risk_policy_and_cancel_fail_closed() {
        let mut reg = registry();
        fs::write(&reg.path, "[[command]]\nphrase='open notes'\nrisk='destructive'\naction={type='literal_text',text='NO'}\n").unwrap();
        reg.reload();
        assert!(reg.match_phrase("open notes").is_none());
        fs::write(&reg.path, "[[command]]\nphrase='open notes'\nrisk='safe'\naction={type='literal_text',text=\"echo hi\\n\"}\n").unwrap();
        reg.reload();
        assert!(reg.match_phrase("open notes").is_none());
        let mut router = Router::new(VoiceMode::CommandOnly);
        router.word("open");
        router.word("notes");
        router.cancel(); // Escape/capture release must not dispatch pending words.
        assert_eq!(router.finish("", &reg), Route::Discard);
        for word in ["eaRS", "what", "mode"] {
            router.word(word);
        }
        assert_eq!(
            router.finish("eaRS what mode", &reg),
            Route::Control("ears what mode".into())
        );
    }

    #[test]
    fn normalization_precedes_exact_command_match() {
        let mut reg = registry();
        fs::write(
            &reg.path,
            "[[command]]\nphrase='trx issue'\nrisk='safe'\naction={type='literal_text',text='OK'}\n",
        )
        .unwrap();
        reg.reload();
        let dir = std::env::temp_dir().join(format!("ears-voice-dict-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let mut dictionary = super::super::typing::tests::test_reloader(&dir);
        let mut r = Router::new(VoiceMode::CommandOnly);
        assert_eq!(r.word("tricks"), None);
        assert_eq!(r.word("issue"), None);
        assert_eq!(r.finish("tricks issue", &reg), Route::Discard);
        for word in ["tricks", "issue"] {
            r.word(word);
        }
        assert_eq!(
            r.finish(&dictionary.replace("tricks issue"), &reg),
            Route::Action("OK".into())
        );
        let _ = fs::remove_dir_all(dir);
    }
}
