//! Shortcut drafts use narrow compare-and-save edits; runtime overrides are not disk settings.
use anyhow::{Result, ensure};
use ears::{config::HotkeyConfig, dictation::hotkeys::validate_hotkeys};
use std::path::Path;
use toml_edit::{DocumentMut, value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Binding {
    Toggle,
    PushToTalk,
    Profile,
}
impl Binding {
    pub const ALL: [Self; 3] = [Self::Toggle, Self::PushToTalk, Self::Profile];
    pub fn id(self) -> &'static str {
        match self {
            Self::Toggle => "toggle",
            Self::PushToTalk => "push_to_talk",
            Self::Profile => "profile_cycle",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Toggle => "Dictation",
            Self::PushToTalk => "Hold to talk",
            Self::Profile => "Next profile",
        }
    }
    pub fn get(self, config: &HotkeyConfig) -> Option<&str> {
        match self {
            Self::Toggle => Some(&config.toggle),
            Self::PushToTalk => config.push_to_talk.as_deref(),
            Self::Profile => config.profile_cycle.as_deref(),
        }
    }
    pub fn set(self, config: &mut HotkeyConfig, text: String) {
        let optional = (!text.is_empty()).then_some(text.clone());
        match self {
            Self::Toggle => config.toggle = text,
            Self::PushToTalk => config.push_to_talk = optional,
            Self::Profile => config.profile_cycle = optional,
        }
    }
}
pub(super) fn current(original: &str) -> Result<HotkeyConfig> {
    let parsed: toml::Value = toml::from_str(original)?;
    let mut merged = toml::Value::try_from(HotkeyConfig::default())?;
    if let Some(hotkeys) = parsed.get("hotkeys") {
        let table = hotkeys
            .as_table()
            .ok_or_else(|| anyhow::anyhow!("hotkeys must be a table"))?;
        for (key, value) in table {
            merged
                .as_table_mut()
                .unwrap()
                .insert(key.clone(), value.clone());
        }
    }
    Ok(merged.try_into()?)
}
fn updated(original: &str, before: &HotkeyConfig, candidate: &HotkeyConfig) -> Result<String> {
    validate_hotkeys(candidate)?;
    let disk = current(original)?;
    // Refuse overwriting another editor's bindings, but preserve unrelated fresh keys.
    for binding in Binding::ALL {
        ensure!(
            binding.get(&disk) == binding.get(before),
            "Shortcuts changed outside this window; cancel and reopen the editor"
        );
    }
    let mut document: DocumentMut = original.parse()?;
    ensure!(
        document
            .get("hotkeys")
            .is_none_or(|v| v.as_table_like().is_some()),
        "hotkeys must be a table"
    );
    for binding in Binding::ALL {
        if binding.get(before) == binding.get(candidate) {
            continue;
        }
        if let Some(text) = binding.get(candidate) {
            let item = &mut document["hotkeys"][binding.id()];
            let decor = item.as_value().map(|v| v.decor().clone());
            *item = value(text);
            if let Some(decor) = decor {
                *item.as_value_mut().unwrap().decor_mut() = decor;
            }
        } else if let Some(table) = document
            .get_mut("hotkeys")
            .and_then(|v| v.as_table_like_mut())
        {
            table.remove(binding.id());
        }
    }
    Ok(document.to_string())
}
pub(super) fn persist(path: &Path, before: &HotkeyConfig, candidate: &HotkeyConfig) -> Result<()> {
    ensure!(
        path.exists(),
        "Configuration disappeared; reopen Settings before saving"
    );
    super::auto_pause::persist_edit(path, |text| updated(text, before, candidate))
}

/// An effect is reachable only after the narrow atomic save succeeds. A returned
/// runtime error does not roll back a successfully persisted preference.
pub(super) fn persist_then<T>(
    path: &Path,
    before: &HotkeyConfig,
    candidate: &HotkeyConfig,
    apply: impl FnOnce() -> T,
) -> Result<T> {
    persist(path, before, candidate)?;
    Ok(apply())
}

/// Window-local capture fallback when no engine is running. Modifier sides are
/// never guessed; physical side-only keys use the running engine's recorder.
pub(super) fn window_combo(key: &gpui::Keystroke) -> Option<String> {
    let mut parts = Vec::new();
    if key.modifiers.control {
        parts.push("ctrl".to_string());
    }
    if key.modifiers.shift {
        parts.push("shift".to_string());
    }
    if key.modifiers.alt {
        parts.push("option".to_string());
    }
    if key.modifiers.platform {
        parts.push("cmd".to_string());
    }
    parts.push(key.key.to_ascii_lowercase());
    let text = parts.join("+");
    ears::dictation::hotkeys::parse_combo_checked(&text)
        .ok()
        .map(|_| text)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edits_only_bindings_and_preserves_comments_and_runtime_policy() {
        let text = "# keep\n[hotkeys]\ntoggle = 'ctrl+shift+v' # key\nenable_internal = false\nmode = 'push_to_talk'\n[dictation]\ninput_device = 'Example mic'\n";
        let before = current(text).unwrap();
        let mut draft = before.clone();
        draft.toggle = "right_option".into();
        draft.push_to_talk = Some("right_cmd".into());
        draft.enable_internal = true;
        let output = updated(text, &before, &draft).unwrap();
        assert!(output.contains("# keep") && output.contains("# key"));
        assert!(
            output.contains("enable_internal = false") && output.contains("mode = 'push_to_talk'")
        );
        assert!(output.contains("input_device = 'Example mic'"));
        assert_eq!(current(&output).unwrap().toggle, "right_option");
    }
    #[test]
    fn conflict_or_stale_binding_leaves_file_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let text = "[hotkeys]\ntoggle='ctrl+shift+v'\n";
        std::fs::write(&path, text).unwrap();
        let before = current(text).unwrap();
        let mut draft = before.clone();
        draft.push_to_talk = Some("control+shift+v".into());
        assert!(persist(&path, &before, &draft).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        draft.push_to_talk = Some("right_cmd".into());
        let changed = text.replace("ctrl+shift+v", "f6");
        std::fs::write(&path, &changed).unwrap();
        assert!(persist(&path, &before, &draft).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), changed);
    }
    #[test]
    fn live_effect_is_unreachable_until_atomic_save_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let before = HotkeyConfig::default();
        let mut candidate = before.clone();
        candidate.toggle = "f6".into();
        let called = std::cell::Cell::new(false);
        assert!(persist_then(&path, &before, &candidate, || called.set(true)).is_err());
        assert!(!called.get());
        std::fs::write(&path, "# keep\n").unwrap();
        persist_then(&path, &before, &candidate, || {
            assert_eq!(
                current(&std::fs::read_to_string(&path).unwrap())
                    .unwrap()
                    .toggle,
                "f6"
            );
            called.set(true);
        })
        .unwrap();
        assert!(called.get());
    }

    #[test]
    fn clearing_optional_binding_keeps_required_toggle() {
        let text = "[hotkeys]\npush_to_talk='right_option'\n";
        let before = current(text).unwrap();
        let mut draft = before.clone();
        Binding::PushToTalk.set(&mut draft, String::new());
        let output = updated(text, &before, &draft).unwrap();
        assert!(!output.contains("push_to_talk"));
        Binding::Toggle.set(&mut draft, String::new());
        assert!(updated(text, &before, &draft).is_err());
    }
}
