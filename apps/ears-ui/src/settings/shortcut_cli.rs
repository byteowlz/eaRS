//! Headless equivalent of shortcut settings. Never starts a listener or engine.
use super::shortcuts;
use anyhow::{Result, ensure};
use ears::config::HotkeyConfig;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{io::Read, path::Path};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Bindings {
    toggle: String,
    push_to_talk: Option<String>,
    profile_cycle: Option<String>,
}
impl From<&HotkeyConfig> for Bindings {
    fn from(config: &HotkeyConfig) -> Self {
        Self {
            toggle: config.toggle.clone(),
            push_to_talk: config.push_to_talk.clone(),
            profile_cycle: config.profile_cycle.clone(),
        }
    }
}
impl Bindings {
    fn apply(self, mut config: HotkeyConfig) -> HotkeyConfig {
        config.toggle = self.toggle;
        config.push_to_talk = self.push_to_talk;
        config.profile_cycle = self.profile_cycle;
        config
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    expected: Bindings,
    bindings: Bindings,
}
fn apply(path: &Path, input: &str) -> Result<()> {
    let value: serde_json::Value = serde_json::from_str(input)?;
    for section in ["expected", "bindings"] {
        for key in ["toggle", "push_to_talk", "profile_cycle"] {
            ensure!(
                value.get(section).and_then(|v| v.get(key)).is_some(),
                "{section}.{key} must be explicit (null disables optional bindings)"
            );
        }
    }
    let request: Request = serde_json::from_value(value)?;
    let disk = shortcuts::current(&std::fs::read_to_string(path)?)?;
    let before = request.expected.apply(disk.clone());
    let candidate = request.bindings.apply(disk);
    shortcuts::persist(path, &before, &candidate)
}
pub(crate) fn handle(args: &[String]) -> bool {
    let Some(option) = args.iter().find(|a| {
        matches!(
            a.as_str(),
            "--shortcuts" | "--set-shortcuts" | "--shortcuts-schema"
        )
    }) else {
        return false;
    };
    if args.len() != 1 {
        eprintln!("Shortcut CLI operations cannot be combined with other options");
        std::process::exit(2);
    }
    let result = (|| -> Result<serde_json::Value> {
        if option == "--shortcuts-schema" {
            let bindings = json!({"type":"object","additionalProperties":false,"required":["toggle","push_to_talk","profile_cycle"],"properties":{"toggle":{"type":"string","minLength":1},"push_to_talk":{"type":["string","null"]},"profile_cycle":{"type":["string","null"]}}});
            return Ok(
                json!({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false,"required":["expected","bindings"],"properties":{"expected":bindings,"bindings":bindings}}),
            );
        }
        let path = ears::config::AppConfig::path()?;
        if option == "--shortcuts" {
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
                Err(e) => return Err(e.into()),
            };
            return Ok(
                json!({"bindings":Bindings::from(&shortcuts::current(&text)?),"source":"saved_config","live_state_queried":false}),
            );
        }
        let mut input = String::new();
        std::io::stdin().take(65537).read_to_string(&mut input)?;
        ensure!(input.len() <= 65536, "Shortcut request exceeds 64 KiB");
        apply(&path, &input)?;
        Ok(json!({"saved":true,"live_applied":false,"applies":"next_engine_start"}))
    })();
    match result {
        Ok(value) => println!("{}", value),
        Err(error) => {
            eprintln!("ears-ui: {error:#}");
            std::process::exit(1);
        }
    }
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn headless_binding_write_uses_same_validation_and_stale_guard() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "# keep\n[hotkeys]\ntoggle='ctrl+shift+v'\n").unwrap();
        let expected = json!({"toggle":"ctrl+shift+v","push_to_talk":null,"profile_cycle":null});
        let request=json!({"expected":expected,"bindings":{"toggle":"f6","push_to_talk":null,"profile_cycle":null}}).to_string();
        apply(&path, &request).unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.contains("# keep"));
        assert_eq!(shortcuts::current(&saved).unwrap().toggle, "f6");
        assert!(apply(&path, &request).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), saved);
    }
    #[test]
    fn rejects_implicit_clears_and_shell_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "").unwrap();
        assert!(
            apply(
                &path,
                r#"{"expected":{"toggle":"f6"},"bindings":{"toggle":"f7"}}"#
            )
            .is_err()
        );
        let b = json!({"toggle":"ctrl+shift+v","push_to_talk":null,"profile_cycle":null});
        assert!(
            apply(
                &path,
                &json!({"expected":b,"bindings":b,"command":"not allowed"}).to_string()
            )
            .is_err()
        );
        assert_eq!(std::fs::read_to_string(path).unwrap(), "");
    }
}
