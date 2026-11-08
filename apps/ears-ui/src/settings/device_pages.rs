//! Overlay, Dictionaries and Profiles on the receiver. Same rules as Dictation:
//! changes apply immediately, the top display carries help and results, and
//! text is typed into native fields set into the recessed displays.
use super::*;
use crate::config::{
    ContentMotion, IndicatorStyle, OverlayAnchor, OverlayAppearance, OverlayFont, PreviewPlacement,
    ShapeMotion, TranscriptStyle,
};
use crate::ui::text_input::{TextInput, TextInputEvent};
use ears::profiles::{ContextProfile, ProfileMatch};
use ears::replacement::{ReplacementDictionary, ReplacementEngine, dictionary_paths};
use std::collections::BTreeMap;

pub(super) const TEXT_FIELDS: [(&str, &str); 8] = [
    ("dict.phrase", "Type a phrase…"),
    ("dict.write", "Canonical text"),
    ("dict.heard", "phrase, phrase"),
    ("prof.name", "Unique name"),
    ("prof.bundles", "com.apple.Terminal, …"),
    ("prof.apps", "Code, firefox, …"),
    ("prof.title", "Regex (optional)"),
    ("prof.dicts", "/path/to.toml, …"),
];
const INDICATORS: [(IndicatorStyle, &str); 4] = [
    (IndicatorStyle::Dot, "Dot"),
    (IndicatorStyle::Ears, "Wave"),
    (IndicatorStyle::ParametricEars, "Bars"),
    (IndicatorStyle::Waveform, "Waveform"),
];
const SHAPES: [(ShapeMotion, &str); 2] = [
    (ShapeMotion::Smooth, "Smooth"),
    (ShapeMotion::Spring, "Spring"),
];
const CONTENTS: [(ContentMotion, &str); 3] = [
    (ContentMotion::Grow, "Grow"),
    (ContentMotion::Fade, "Fade"),
    (ContentMotion::GrowFade, "Both"),
];
const PILLS: [(OverlayAppearance, &str); 2] = [
    (OverlayAppearance::Classic, "Classic"),
    (OverlayAppearance::Black, "Black"),
];
const TOPS: [(TranscriptStyle, &str); 4] = [
    (TranscriptStyle::Current, "Phrase"),
    (TranscriptStyle::Word, "Word"),
    (TranscriptStyle::Meter, "Ears only"),
    (TranscriptStyle::Hidden, "Hidden"),
];
const PREVIEWS: [(PreviewPlacement, &str); 3] = [
    (PreviewPlacement::Inline, "Inline"),
    (PreviewPlacement::Subtitle, "Subtitles"),
    (PreviewPlacement::Cursor, "Pointer"),
];
const FONTS: [(OverlayFont, &str); 3] = [
    (OverlayFont::System, "System"),
    (OverlayFont::DepartureMono, "Departure"),
    (OverlayFont::JetbrainsMono, "JetBrains"),
];
const SCHEMES: [(&str, &str); 3] = [
    ("auto", "Auto"),
    ("oqto-dark", "Dark"),
    ("oqto-light", "Light"),
];
const PROFILE_INSERT: [(Option<InsertionMode>, &str); 4] = [
    (None, "Inherit"),
    (Some(InsertionMode::InsertAtCursor), "At cursor"),
    (Some(InsertionMode::Clipboard), "Clipboard"),
    (Some(InsertionMode::SendAsPrompt), "Prompt"),
];

fn step<T: Copy + PartialEq>(items: &[(T, &'static str)], current: T, delta: isize) -> T {
    let i = items.iter().position(|(v, _)| *v == current).unwrap_or(0) as isize;
    items[(i + delta).rem_euclid(items.len() as isize) as usize].0
}
fn name<T: PartialEq>(items: &[(T, &'static str)], current: &T) -> &'static str {
    items
        .iter()
        .find(|(v, _)| v == current)
        .map_or("", |(_, n)| n)
}
fn split_csv(text: &str) -> Vec<String> {
    text.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// State of the profile editor on the right of the Profiles page.
#[derive(Default)]
pub(super) struct ProfileEdit {
    /// `None`: nothing chosen; `Some(None)`: new profile; `Some(Some(name))`: editing.
    pub editing: Option<Option<String>>,
    pub priority: i32,
    pub language: Option<String>,
    pub insertion: Option<InsertionMode>,
}

impl NativeSettings {
    pub(super) fn create_text_fields(
        cx: &mut Context<Self>,
    ) -> BTreeMap<&'static str, Entity<TextInput>> {
        let mut fields = BTreeMap::new();
        for (id, placeholder) in TEXT_FIELDS {
            let field = cx.new(|cx| TextInput::new(cx, placeholder).bare());
            cx.subscribe(&field, move |this, _, event: &TextInputEvent, cx| {
                match (id, event) {
                    ("dict.phrase", TextInputEvent::Change) => this.sync(cx),
                    ("dict.write" | "dict.heard", TextInputEvent::Submit) => this.add_entry(cx),
                    (id, TextInputEvent::Submit) if id.starts_with("prof.") => {
                        this.save_profile(cx)
                    }
                    _ => {}
                }
            })
            .detach();
            fields.insert(id, field);
        }
        fields
    }
    fn field(&self, id: &str, cx: &gpui::App) -> String {
        self.text
            .get(id)
            .map(|f| f.read(cx).text().trim().to_owned())
            .unwrap_or_default()
    }
    fn set_field(&self, id: &str, value: &str, cx: &mut Context<Self>) {
        if let Some(f) = self.text.get(id) {
            f.update(cx, |f, cx| f.set_text(value, cx));
        }
    }

    // ---- page actions -------------------------------------------------------

    /// Returns true when the action belonged to one of these pages.
    pub(super) fn page_action(&mut self, id: &str, cx: &mut Context<Self>) -> bool {
        let delta = |base: &str| -> Option<isize> {
            id.strip_prefix(base).and_then(|rest| match rest {
                ".prev" => Some(-1),
                ".next" => Some(1),
                _ => None,
            })
        };
        if let Some(rest) = id.strip_prefix("ov.") {
            self.overlay_action(rest, delta, cx);
            return true;
        }
        if id.starts_with("dict.") {
            if let Some(d) = delta("dict.page") {
                self.dict_page = (self.dict_page as isize + d).max(0) as usize;
            } else if let Some(i) = id
                .strip_prefix("dict.entry.")
                .and_then(|r| r.strip_suffix(".remove"))
            {
                if let Ok(i) = i.parse::<usize>() {
                    self.remove_entry(self.dict_page * DICT_ROWS + i, cx);
                }
            } else if id == "dict.add" {
                self.add_entry(cx);
            }
            self.sync(cx);
            return true;
        }
        if id.starts_with("prof.") {
            if let Some(d) = delta("prof.page") {
                self.prof_page = (self.prof_page as isize + d).max(0) as usize;
            } else if let Some(i) = id
                .strip_prefix("prof.item.")
                .and_then(|r| r.parse::<usize>().ok())
            {
                let names = self.model.read(cx).profile_names();
                if let Some(name) = names.get(self.prof_page * PROFILE_ROWS + i).cloned() {
                    self.edit_profile(Some(name), cx);
                }
            } else if let Some(d) = delta("prof.priority") {
                self.profile.priority += d as i32;
            } else if let Some(d) = delta("prof.lang") {
                let mut codes: Vec<Option<String>> = std::iter::once(None)
                    .chain(LANGUAGES.iter().skip(1).map(|(c, _)| Some(c.to_string())))
                    .collect();
                if !codes.contains(&self.profile.language) {
                    codes.push(self.profile.language.clone());
                }
                let i = codes
                    .iter()
                    .position(|c| *c == self.profile.language)
                    .unwrap_or(0) as isize;
                self.profile.language =
                    codes[(i + d).rem_euclid(codes.len() as isize) as usize].clone();
            } else if let Some(d) = delta("prof.insert") {
                self.profile.insertion = step(&PROFILE_INSERT, self.profile.insertion, d);
            } else {
                match id {
                    "prof.new" => self.edit_profile(None, cx),
                    "prof.save" => self.save_profile(cx),
                    "prof.delete" => self.delete_profile(cx),
                    "prof.revert" => {
                        let original = self.profile.editing.clone().flatten();
                        self.edit_profile(original, cx);
                    }
                    _ => {}
                }
            }
            self.sync(cx);
            return true;
        }
        false
    }

    fn overlay_action(
        &mut self,
        id: &str,
        delta: impl Fn(&str) -> Option<isize>,
        cx: &mut Context<Self>,
    ) {
        let o = self.model.read(cx).ui.overlay.clone();
        let theme = self.model.read(cx).ui.theme.clone();
        let switch = |f: fn(&mut crate::config::OverlayConfig) -> &mut bool| {
            move |ui: &mut crate::config::UiConfig| {
                let v = f(&mut ui.overlay);
                *v = !*v;
            }
        };
        let full = |rest: &str| format!("ov.{rest}");
        let mut refresh_theme = false;
        let edit: Option<Box<dyn FnOnce(&mut crate::config::UiConfig)>> = match id {
            "show" => Some(Box::new(switch(|o| &mut o.enabled))),
            "dot" => Some(Box::new(switch(|o| &mut o.dormant_dot))),
            "hug" => Some(Box::new(switch(|o| &mut o.island_mode))),
            "rec" => Some(Box::new(switch(|o| &mut o.recording_notch))),
            "through" => Some(Box::new(switch(|o| &mut o.click_through))),
            "details" => Some(Box::new(switch(|o| &mut o.show_details))),
            "animate" => Some(Box::new(switch(|o| &mut o.animation))),
            "always" => Some(Box::new(switch(|o| &mut o.animate_with_reduced_motion))),
            "wordfade" => Some(Box::new(switch(|o| &mut o.word_fade))),
            _ => {
                let d = |base: &str| delta(&full(base));
                if let Some(d) = d("anchor") {
                    let all: Vec<(OverlayAnchor, &str)> =
                        OverlayAnchor::ALL.iter().map(|a| (*a, a.label())).collect();
                    let v = step(&all, o.anchor, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.anchor = v
                    }))
                } else if let Some(d) = d("indicator") {
                    let v = step(&INDICATORS, o.indicator, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.indicator = v
                    }))
                } else if let Some(d) = d("shape") {
                    let v = step(&SHAPES, o.shape_motion, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.shape_motion = v
                    }))
                } else if let Some(d) = d("content") {
                    let v = step(&CONTENTS, o.content_motion, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.content_motion = v
                    }))
                } else if let Some(d) = d("pill") {
                    let v = step(&PILLS, o.appearance, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.appearance = v
                    }))
                } else if let Some(d) = d("top") {
                    let v = step(&TOPS, o.transcript, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.transcript = v
                    }))
                } else if let Some(d) = d("preview") {
                    let v = step(&PREVIEWS, o.preview, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.preview = v
                    }))
                } else if let Some(d) = d("font") {
                    let v = step(&FONTS, o.font, d);
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.overlay.font = v
                    }))
                } else if let Some(d) = d("scheme") {
                    let current = SCHEMES
                        .iter()
                        .find(|(k, _)| *k == theme.scheme)
                        .map_or("auto", |(k, _)| *k);
                    let v = step(&SCHEMES, current, d).to_owned();
                    refresh_theme = true;
                    Some(Box::new(move |ui: &mut crate::config::UiConfig| {
                        ui.theme.scheme = v
                    }))
                } else {
                    None
                }
            }
        };
        if let Some(edit) = edit {
            self.model.update(cx, |m, cx| m.update_ui(cx, edit));
            if refresh_theme {
                crate::runtime::apply_theme(cx, None);
            }
            self.sync(cx);
        }
    }

    // ---- dictionaries -------------------------------------------------------

    pub(super) fn reload_entries(&mut self, cx: &mut Context<Self>) {
        let path = dictionary_paths(&self.model.read(cx).core.replacement)
            .first()
            .cloned();
        self.dict_file = path.as_ref().map(|p| p.display().to_string());
        self.dict_entries = path
            .and_then(|p| ReplacementDictionary::load(&p).ok())
            .map(|d| {
                d.entries
                    .into_iter()
                    .map(|e| (e.replace, e.phrases))
                    .collect()
            })
            .unwrap_or_default();
    }
    fn add_entry(&mut self, cx: &mut Context<Self>) {
        let write = self.field("dict.write", cx);
        let heard = split_csv(&self.field("dict.heard", cx));
        if write.is_empty() || heard.is_empty() {
            return self.say("Fill in Write and Heard as first.", cx);
        }
        if self.model.read(cx).configuration_is_preview {
            return self.say("Preview: no dictionary was written.", cx);
        }
        let result = (|| -> anyhow::Result<()> {
            let path = dictionary_paths(&self.model.read(cx).core.replacement)
                .first()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no dictionary configured"))?;
            let mut dictionary = ReplacementDictionary::load_or_create(&path)?;
            dictionary.add_entry(write.clone(), heard);
            dictionary.save(&path)
        })();
        match result {
            Ok(()) => {
                self.set_field("dict.write", "", cx);
                self.set_field("dict.heard", "", cx);
                self.model.update(cx, |m, cx| m.reload(cx));
                self.reload_entries(cx);
                self.say(format!("Added {write}."), cx);
            }
            Err(error) => self.say(format!("Entry not saved: {error:#}"), cx),
        }
    }
    fn remove_entry(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some((write, _)) = self.dict_entries.get(index).cloned() else {
            return;
        };
        if self.model.read(cx).configuration_is_preview {
            return self.say("Preview: no dictionary was written.", cx);
        }
        let result = (|| -> anyhow::Result<()> {
            let path = dictionary_paths(&self.model.read(cx).core.replacement)
                .first()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no dictionary configured"))?;
            let mut dictionary = ReplacementDictionary::load_or_create(&path)?;
            dictionary.remove_replacement(&write);
            dictionary.save(&path)
        })();
        match result {
            Ok(()) => {
                self.model.update(cx, |m, cx| m.reload(cx));
                self.reload_entries(cx);
                self.say(format!("Removed {write}."), cx);
            }
            Err(error) => self.say(format!("Entry not removed: {error:#}"), cx),
        }
    }

    // ---- profiles -----------------------------------------------------------

    fn edit_profile(&mut self, name: Option<String>, cx: &mut Context<Self>) {
        let existing = name
            .as_ref()
            .and_then(|n| self.model.read(cx).profiles.get(n).cloned());
        let p = existing.clone().unwrap_or(ContextProfile {
            name: String::new(),
            priority: 0,
            matcher: ProfileMatch::default(),
            dictionaries: Vec::new(),
            language: None,
            insertion_mode: None,
        });
        self.set_field("prof.name", &p.name, cx);
        self.set_field("prof.bundles", &p.matcher.bundle_id.join(", "), cx);
        self.set_field("prof.apps", &p.matcher.binary.join(", "), cx);
        self.set_field(
            "prof.title",
            p.matcher.window_title.as_deref().unwrap_or(""),
            cx,
        );
        self.set_field("prof.dicts", &p.dictionaries.join(", "), cx);
        self.profile = ProfileEdit {
            editing: Some(existing.map(|p| p.name)),
            priority: p.priority,
            language: p.language,
            insertion: p.insertion_mode,
        };
        self.sync(cx);
    }
    fn save_profile(&mut self, cx: &mut Context<Self>) {
        let Some(original) = self.profile.editing.clone() else {
            return self.say("Choose a profile or press New profile.", cx);
        };
        let name = self.field("prof.name", cx);
        if name.is_empty() {
            return self.say("A profile needs a name.", cx);
        }
        let title = self.field("prof.title", cx);
        let profile = ContextProfile {
            name: name.clone(),
            priority: self.profile.priority,
            matcher: ProfileMatch {
                bundle_id: split_csv(&self.field("prof.bundles", cx)),
                binary: split_csv(&self.field("prof.apps", cx)),
                window_title: (!title.is_empty()).then_some(title),
            },
            dictionaries: split_csv(&self.field("prof.dicts", cx)),
            language: self.profile.language.clone(),
            insertion_mode: self.profile.insertion,
        };
        if self.model.read(cx).configuration_is_preview {
            return self.say("Preview: profile not written.", cx);
        }
        let rename_from = original.filter(|o| *o != name);
        let result = self.model.update(cx, |m, cx| {
            if let Some(old) = &rename_from {
                m.profiles.remove(old);
            }
            m.profiles.upsert(profile);
            let saved = m.profiles.save();
            m.reload(cx);
            saved
        });
        match result {
            Ok(()) => {
                self.profile.editing = Some(Some(name.clone()));
                self.say(format!("Saved {name}."), cx);
            }
            Err(error) => self.say(format!("Profile not saved: {error:#}"), cx),
        }
    }
    fn delete_profile(&mut self, cx: &mut Context<Self>) {
        let Some(Some(name)) = self.profile.editing.clone() else {
            return self.say("Choose a saved profile to delete.", cx);
        };
        if self.model.read(cx).configuration_is_preview {
            return self.say("Preview: profile not deleted.", cx);
        }
        let result = self.model.update(cx, |m, cx| {
            m.profiles.remove(&name);
            let saved = m.profiles.save();
            if m.profile.as_deref() == Some(name.as_str()) {
                m.set_profile(None, cx);
            }
            m.reload(cx);
            saved
        });
        match result {
            Ok(()) => {
                self.edit_profile(None, cx);
                self.profile.editing = None;
                self.say(format!("Deleted {name}."), cx);
            }
            Err(error) => self.say(format!("Profile not deleted: {error:#}"), cx),
        }
    }

    // ---- scene ----------------------------------------------------------------

    /// Everything the extra pages show, gathered before the scene edit.
    pub(super) fn pages_view(&mut self, cx: &gpui::App) -> PagesView {
        let model = self.model.read(cx);
        let phrase = self.field("dict.phrase", cx);
        let result = if phrase.is_empty() {
            String::new()
        } else {
            match ReplacementEngine::from_config(&model.core.replacement) {
                Ok(engine) => engine.replace(&phrase),
                Err(error) => format!("Error: {error}"),
            }
        };
        let dict_pages = self.dict_entries.len().div_ceil(DICT_ROWS).max(1);
        self.dict_page = self.dict_page.min(dict_pages - 1);
        let names = model.profile_names();
        let prof_pages = names.len().div_ceil(PROFILE_ROWS).max(1);
        self.prof_page = self.prof_page.min(prof_pages - 1);
        PagesView {
            overlay: model.ui.overlay.clone(),
            radius: model.ui.theme.radius,
            scheme: model.ui.theme.scheme.clone(),
            result,
            file: self.dict_file.clone(),
            entries: self.dict_entries.clone(),
            dict_page: self.dict_page,
            names,
            prof_page: self.prof_page,
            editing: self.profile.editing.clone(),
            priority: self.profile.priority,
            language: self.profile.language.clone(),
            insertion: self.profile.insertion,
        }
    }
}

pub(super) struct PagesView {
    overlay: crate::config::OverlayConfig,
    radius: f32,
    scheme: String,
    result: String,
    file: Option<String>,
    entries: Vec<(String, Vec<String>)>,
    dict_page: usize,
    names: Vec<String>,
    prof_page: usize,
    editing: Option<Option<String>>,
    priority: i32,
    language: Option<String>,
    insertion: Option<InsertionMode>,
}
impl PagesView {
    pub(super) fn write(&self, s: &mut Snapshot) {
        let o = &self.overlay;
        for (id, on) in [
            ("ov.show", o.enabled),
            ("ov.dot", o.dormant_dot),
            ("ov.hug", o.island_mode),
            ("ov.rec", o.recording_notch),
            ("ov.through", o.click_through),
            ("ov.details", o.show_details),
            ("ov.animate", o.animation),
            ("ov.always", o.animate_with_reduced_motion),
        ] {
            scene::switch(s, id, on);
        }
        let word = o.transcript == TranscriptStyle::Word;
        scene::switch(s, "ov.wordfade", o.word_fade);
        scene::label(s, "ov.wordfade", if o.word_fade { "On" } else { "Off" });
        scene::enable(s, "ov.wordfade", word);
        scene::text(s, "ov.anchor", o.anchor.label());
        scene::text(s, "ov.indicator", name(&INDICATORS, &o.indicator));
        scene::text(s, "ov.shape", name(&SHAPES, &o.shape_motion));
        scene::text(s, "ov.content", name(&CONTENTS, &o.content_motion));
        scene::text(s, "ov.wordwidth", format!("{:.0} px", o.word_width));
        scene::text(s, "ov.wordsize", format!("{:.0} pt", o.word_size));
        scene::text(s, "ov.pill", name(&PILLS, &o.appearance));
        scene::text(s, "ov.top", name(&TOPS, &o.transcript));
        scene::text(s, "ov.preview", name(&PREVIEWS, &o.preview));
        scene::text(s, "ov.font", name(&FONTS, &o.font));
        let scheme = SCHEMES
            .iter()
            .find(|(k, _)| *k == self.scheme)
            .map_or("Auto", |(_, n)| n);
        scene::text(s, "ov.scheme", scheme);
        scene::text(s, "ov.radius", format!("{:.0} px", self.radius));
        scene::text(s, "ov.opacity", format!("{:.2}", o.opacity));
        scene::text(s, "ov.margin", format!("{:.0} px", o.margin));
        scene::text(s, "ov.linger", format!("{} ms", o.linger_ms));

        // Dictionaries.
        scene::text(s, "dict.result", self.result.clone());
        scene::text(
            s,
            "dict.file",
            self.file
                .clone()
                .unwrap_or_else(|| "No dictionary configured".into()),
        );
        let pages = self.entries.len().div_ceil(DICT_ROWS).max(1);
        for i in 0..DICT_ROWS {
            let id = format!("dict.entry.{i}");
            match self.entries.get(self.dict_page * DICT_ROWS + i) {
                Some((write, heard)) => {
                    scene::text(s, &format!("{id}.write"), write.clone());
                    scene::text(s, &id, heard.join(", "));
                    scene::enable(s, &format!("{id}.remove"), true);
                }
                None => {
                    let empty = i == 0 && self.entries.is_empty();
                    scene::text(s, &format!("{id}.write"), "");
                    scene::text(s, &id, if empty { "No entries yet" } else { "" });
                    scene::enable(s, &format!("{id}.remove"), false);
                }
            }
        }
        scene::text(s, "dict.page", format!("{} / {pages}", self.dict_page + 1));
        scene::enable(s, "dict.page.prev", self.dict_page > 0);
        scene::enable(s, "dict.page.next", self.dict_page + 1 < pages);

        // Profiles.
        let names = &self.names;
        let pages = names.len().div_ceil(PROFILE_ROWS).max(1);
        let editing = self.editing.clone().flatten();
        for i in 0..PROFILE_ROWS {
            let id = format!("prof.item.{i}");
            match names.get(self.prof_page * PROFILE_ROWS + i) {
                Some(n) => {
                    scene::show(s, &id, true);
                    scene::label(s, &id, n);
                    scene::switch(s, &id, editing.as_deref() == Some(n.as_str()));
                }
                None => {
                    scene::label(
                        s,
                        &id,
                        if i == 0 && names.is_empty() {
                            "No profiles yet"
                        } else {
                            ""
                        },
                    );
                    scene::switch(s, &id, false);
                    // Unused rows are not drawn; an empty list keeps one quiet key.
                    scene::show(s, &id, i == 0 && names.is_empty());
                    scene::enable(s, &id, false);
                }
            }
        }
        scene::text(s, "prof.page", format!("{} / {pages}", self.prof_page + 1));
        scene::enable(s, "prof.page.prev", self.prof_page > 0);
        scene::enable(s, "prof.page.next", self.prof_page + 1 < pages);
        let open = self.editing.is_some();
        scene::text(s, "prof.priority", self.priority.to_string());
        let lang = self
            .language
            .as_deref()
            .map_or("Inherit".to_owned(), |code| {
                LANGUAGES
                    .iter()
                    .find(|(c, _)| *c == code)
                    .map_or(code.to_owned(), |(_, n)| (*n).to_owned())
            });
        scene::text(s, "prof.lang", lang);
        scene::text(s, "prof.insert", name(&PROFILE_INSERT, &self.insertion));
        for id in [
            "prof.priority.prev",
            "prof.priority.next",
            "prof.lang.prev",
            "prof.lang.next",
            "prof.insert.prev",
            "prof.insert.next",
            "prof.save",
            "prof.revert",
        ] {
            scene::enable(s, id, open);
        }
        scene::enable(s, "prof.delete", editing.is_some());
        let heading = match &self.editing {
            None => "PROFILE".to_owned(),
            Some(None) => "NEW PROFILE".to_owned(),
            Some(Some(n)) => format!("PROFILE  {}", n.to_uppercase()),
        };
        scene::text(s, "prof.head.edit", heading);
    }
}

impl NativeSettings {
    /// Native text fields set into the displays of the visible page.
    pub(super) fn text_overlays(&self, cx: &gpui::App) -> Vec<gpui::AnyElement> {
        let device = self.device.read(cx);
        let (Some(p), s) = (device.projection(), device.snapshot()) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (id, field) in &self.text {
            let Some((d, c, state)) = s.control(id) else {
                continue;
            };
            if !s.control_visible(d, c) || !state.visible {
                continue;
            }
            let o = d.origin;
            let r = &c.rect;
            let pad = c.matrix.as_ref().map_or(0.4, |m| m.padding_x);
            let corner =
                |x: f64, y: f64| p.project(glam::DVec3::new(x + o[0], y + o[1], r.z + o[2] + 0.03));
            let (Some(a), Some(b)) = (
                corner(r.x - r.width / 2.0 + pad, r.y + r.height / 2.0),
                corner(r.x + r.width / 2.0 - pad, r.y - r.height / 2.0),
            ) else {
                continue;
            };
            let height = (b[1] - a[1]) as f32;
            out.push(
                div()
                    .absolute()
                    .left(gpui::px(a[0] as f32))
                    .top(gpui::px(a[1] as f32))
                    .w(gpui::px((b[0] - a[0]) as f32))
                    .h(gpui::px(height))
                    .flex()
                    .items_center()
                    .font_family("Departure Mono")
                    .text_size(gpui::px((height * 0.56).clamp(9.0, 22.0)))
                    .line_height(gpui::px(height))
                    .text_color(gpui::rgb(0xff9e52))
                    .child(field.clone())
                    .into_any_element(),
            );
        }
        out
    }
}
