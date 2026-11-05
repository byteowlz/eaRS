//! App-owned scene state, display text and typed-action validation. Geometry: device_layout.
pub(super) use super::device_layout::initial;
use device_scene::{Snapshot, Value};
use serde_json::json;

fn state<'a>(s: &'a mut Snapshot, id: &str) -> &'a mut device_scene::ControlState {
    s.state
        .controls
        .get_mut(id)
        .unwrap_or_else(|| panic!("app control {id}"))
}
pub(super) fn text(s: &mut Snapshot, id: &str, value: impl Into<String>) {
    state(s, id).value = Value(json!(value.into()));
}
pub(super) fn switch(s: &mut Snapshot, id: &str, on: bool) {
    state(s, id).value = Value(json!(on));
}
/// Replace a click-through display's choices and select one.
pub(super) fn choices(s: &mut Snapshot, id: &str, labels: &[String], current: usize) {
    if let Some(control) = s.scene.devices[0].controls.iter_mut().find(|c| c.id == id) {
        control.options = Some(
            labels
                .iter()
                .enumerate()
                .map(|(i, label)| device_scene::OptionItem {
                    id: format!("choice-{i}"),
                    label: label.clone(),
                })
                .collect(),
        );
    }
    text(s, id, format!("choice-{current}"));
}
/// Show or hide a list key; hidden keys are not drawn, focused or clickable.
pub(super) fn show(s: &mut Snapshot, id: &str, visible: bool) {
    let control = state(s, id);
    control.visible = visible;
    control.enabled = visible;
}
/// Keep a key in its grid place but make it unavailable (dimmed, unfocusable).
pub(super) fn enable(s: &mut Snapshot, id: &str, enabled: bool) {
    let control = state(s, id);
    control.visible = true;
    control.enabled = enabled;
}
pub(super) fn label(s: &mut Snapshot, id: &str, label: &str) {
    if let Some(control) = s.scene.devices[0].controls.iter_mut().find(|c| c.id == id) {
        label.clone_into(&mut control.label);
        label.clone_into(&mut control.accessible_name);
    }
}
pub(super) fn capacity(s: &Snapshot, id: &str) -> (usize, usize) {
    s.control(id)
        .and_then(|(_, c, _)| c.matrix.as_ref())
        .map_or((40, 1), |m| (m.cols as usize, m.rows as usize))
}

/// Human shortcut text on the matrix: `ctrl+shift+v` -> `⌃⇧V`, `right_option` -> `Right ⌥`.
pub(super) fn shortcut_text(combo: Option<&str>) -> String {
    let Some(combo) = combo.filter(|c| !c.trim().is_empty()) else {
        return "Not set".into();
    };
    let parts: Vec<_> = combo.split('+').map(str::trim).collect();
    let symbol = |part: &str| -> Option<(Option<&'static str>, &'static str)> {
        let (side, name) = match part.split_once('_') {
            Some(("left", name)) => (Some("Left"), name),
            Some(("right", name)) => (Some("Right"), name),
            _ => (None, part),
        };
        let mark = match name {
            "ctrl" | "control" => "⌃",
            "shift" => "⇧",
            "option" | "alt" | "opt" => "⌥",
            "cmd" | "command" | "meta" | "super" => "⌘",
            "fn" => "fn",
            _ => return None,
        };
        Some((side, mark))
    };
    if let [only] = parts.as_slice()
        && let Some((side, mark)) = symbol(only)
    {
        return side.map_or(mark.to_owned(), |side| format!("{side} {mark}"));
    }
    let mut out = String::new();
    for part in parts {
        match symbol(part) {
            Some((side, mark)) => {
                if let Some(side) = side {
                    out.push_str(&side[..1]);
                }
                out.push_str(mark);
            }
            None => out.push_str(&match part {
                "space" => "Space".to_owned(),
                "tab" => "Tab".to_owned(),
                "escape" | "esc" => "Esc".to_owned(),
                "return" | "enter" => "Return".to_owned(),
                key if key.len() == 1 => key.to_uppercase(),
                key => {
                    let mut chars = key.chars();
                    chars
                        .next()
                        .map(|c| c.to_uppercase().chain(chars).collect())
                        .unwrap_or_default()
                }
            }),
        }
    }
    out
}

fn lines(text: &str, cols: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let mut word = word.to_owned();
        while word.chars().count() > cols {
            if !line.is_empty() {
                lines.push(std::mem::take(&mut line));
            }
            let head: String = word.chars().take(cols).collect();
            word = word.chars().skip(cols).collect();
            lines.push(head);
        }
        let len = line.chars().count();
        if len > 0 && len + 1 + word.chars().count() > cols {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(&word);
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}

/// Centre a message horizontally and vertically for a fixed-pitch display.
/// Long help keeps its start; a streamed transcript keeps its newest words.
pub(super) fn display_text(text: &str, cols: usize, rows: usize, newest: bool) -> String {
    let mut all = lines(text, cols);
    if all.len() > rows {
        if newest {
            all.drain(..all.len() - rows);
            all[0] = format!("…{}", all[0].chars().skip(1).collect::<String>());
        } else {
            all.truncate(rows);
            let last = &mut all[rows - 1];
            let keep: String = last.chars().take(cols - 1).collect();
            *last = format!("{keep}…");
        }
    }
    let blank = (rows - all.len()) / 2;
    let mut out: Vec<String> = vec![String::new(); blank];
    for line in all {
        let pad = (cols - line.chars().count().min(cols)) / 2;
        out.push(format!("{}{line}", " ".repeat(pad)));
    }
    out.join("\n")
}

/// Phase semantics belong to the app adapter, not just schema validation.
pub(super) fn accepts(action: &device_scene::Action, snapshot: &Snapshot) -> bool {
    if action.validate(snapshot).is_err() {
        return false;
    }
    match action.phase {
        device_scene::Phase::Commit | device_scene::Phase::Cancel => true,
        device_scene::Phase::Preview => {
            snapshot
                .control(&action.control_id)
                .is_some_and(|(_, control, _)| {
                    matches!(
                        control.kind,
                        device_scene::Kind::Shortcut | device_scene::Kind::Select
                    )
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shortcuts_read_like_keycaps() {
        assert_eq!(shortcut_text(Some("ctrl+shift+v")), "⌃⇧V");
        assert_eq!(shortcut_text(Some("right_option")), "Right ⌥");
        assert_eq!(shortcut_text(Some("left_cmd+e")), "L⌘E");
        assert_eq!(shortcut_text(Some("f6")), "F6");
        assert_eq!(shortcut_text(Some("fn")), "fn");
        assert_eq!(shortcut_text(None), "Not set");
    }
    #[test]
    fn display_text_centres_and_streams_newest_words() {
        let centred = display_text("Ready", 11, 3, false);
        assert_eq!(centred, "\n   Ready");
        let help = display_text("one two three four five six seven", 9, 2, false);
        assert_eq!(help.lines().count(), 2);
        assert!(help.ends_with('…'));
        let stream = display_text("alpha beta gamma delta epsilon zeta", 11, 2, true);
        assert!(stream.contains("zeta") && !stream.contains("alpha"));
        assert!(stream.starts_with('…') || stream.trim_start().starts_with('…'));
        for line in display_text("a very long sentence that wraps", 10, 3, false).lines() {
            assert!(line.chars().count() <= 10);
        }
    }
    #[test]
    fn stale_or_preview_buttons_cannot_commit_host_effects() {
        let snapshot = initial();
        let mut action = device_scene::Action {
            schema_version: 1,
            scene_id: snapshot.scene.id.clone(),
            scene_revision: snapshot.scene.revision,
            state_revision: snapshot.state.revision,
            control_id: "language.next".into(),
            interaction_id: "test.i1".into(),
            sequence: 0,
            phase: device_scene::Phase::Preview,
            value: Value(serde_json::Value::Null),
            source: device_scene::Source::Api,
        };
        assert!(!accepts(&action, &snapshot));
        action.phase = device_scene::Phase::Commit;
        assert!(accepts(&action, &snapshot));
        action.state_revision += 1;
        assert!(!accepts(&action, &snapshot));
    }
}
