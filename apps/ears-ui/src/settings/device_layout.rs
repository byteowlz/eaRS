//! Receiver geometry for Settings: one navigation level, then labelled sections.
//!
//! One baseline grid: both columns are stacks of 2.4-unit slots (a heading takes
//! a slot too), so every row lines up across the device. Physical keys align to
//! the control column edges; every display aligns to the inner value column.
//! Spacing scale 0.4/0.8/1.2; one 0.16 dot pitch. Geometry never depends on values.
use super::shortcuts::Binding;
use device_scene::{Control, Rect, Snapshot};
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;

/// Committed layout tuned with the layout editor; empty means the generated grid.
const TUNED: &str = include_str!("../../assets/settings-layout.json");
pub(super) const SNAP: f64 = 0.2;
pub(super) const MIN_SIZE: f64 = 0.8;

/// Geometry overrides written by the layout editor. Structure (ids, kinds, actions)
/// always comes from this module; the file only moves and resizes elements.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(super) struct LayoutFile {
    #[serde(default)]
    pub schema: String,
    #[serde(default)]
    pub controls: BTreeMap<String, Placement>,
    #[serde(default)]
    pub panels: BTreeMap<String, Placement>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(super) struct Placement {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
}
fn round(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
impl Placement {
    fn of(r: &Rect, module: Option<String>) -> Self {
        Self {
            x: round(r.x),
            y: round(r.y),
            width: round(r.width),
            height: round(r.height),
            module,
        }
    }
}
impl LayoutFile {
    pub fn capture(s: &Snapshot) -> Self {
        let d = &s.scene.devices[0];
        Self {
            schema: "ears.settings-layout/v1".into(),
            controls: d
                .controls
                .iter()
                .map(|c| {
                    (
                        c.id.clone(),
                        Placement::of(&c.rect, Some(c.module_id.clone())),
                    )
                })
                .collect(),
            panels: d
                .panels
                .iter()
                .map(|p| (p.id.clone(), Placement::of(&p.rect, None)))
                .collect(),
        }
    }
}
/// Recompute a display's fixed-pitch grid for its current size: the pitch never changes.
pub(super) fn refit(control: &mut Control) {
    let (width, height) = (control.rect.width, control.rect.height);
    if let Some(m) = control.matrix.as_mut() {
        m.cols = (((width - 0.8 + PITCH) / (6.0 * PITCH)).floor() as u32).max(1);
        m.rows = rows_for(height);
        m.padding_x = ((width - (m.cols * 6 - 1) as f64 * PITCH) / 2.0).max(0.0);
        m.padding_y = ((height - ((m.rows - 1) * 8 + 7) as f64 * PITCH) / 2.0).max(0.0);
    }
}
/// Apply overrides; unknown ids are ignored so an older file never breaks Settings.
pub(super) fn apply(s: &mut Snapshot, layout: &LayoutFile) {
    let d = &mut s.scene.devices[0];
    for p in &mut d.panels {
        if let Some(at) = layout.panels.get(&p.id) {
            (p.rect.x, p.rect.y, p.rect.width, p.rect.height) = (at.x, at.y, at.width, at.height);
        }
    }
    let panels: Vec<String> = d.panels.iter().map(|p| p.id.clone()).collect();
    for c in &mut d.controls {
        if let Some(at) = layout.controls.get(&c.id) {
            (c.rect.x, c.rect.y, c.rect.width, c.rect.height) = (at.x, at.y, at.width, at.height);
            if let Some(module) = at.module.as_ref().filter(|m| panels.contains(m)) {
                module.clone_into(&mut c.module_id);
            }
            refit(c);
        }
    }
}
/// The generated grid, without the committed tuned layout.
#[cfg(test)]
pub(super) fn generated() -> Snapshot {
    build()
}
pub(super) fn initial() -> Snapshot {
    let mut s = build();
    if let Ok(tuned) = serde_json::from_str::<LayoutFile>(TUNED) {
        let mut candidate = s.clone();
        apply(&mut candidate, &tuned);
        // A tuned layout that no longer validates falls back to the generated grid.
        if device_scene::ValidatedSnapshot::new(candidate.clone()).is_ok() {
            s = candidate;
        }
    }
    s
}

const BASE: &str =
    include_str!("../../../../vendor/device/spec/fixtures/device-scene-v1/ears-input.json");
pub(super) const PAGE: &str = "dictation";
pub(super) const PITCH: f64 = 0.16;
const INNER: f64 = 0.4;
const GAP: f64 = 0.8;
const PAD: f64 = 1.2;
const ROW: f64 = 1.6;
const SLOT: f64 = 2.4;
const HEADING: f64 = 1.2;
const LABEL_WIDTH: f64 = 10.4;
const SIDE_KEY: f64 = 2.4;
const SWITCH_KEY: f64 = 6.0;
const EDGE: f64 = 30.4;
const HEIGHT: f64 = 46.0;
const SLOTS: usize = 11;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Span {
    pub left: f64,
    pub right: f64,
}
impl Span {
    fn width(self) -> f64 {
        self.right - self.left
    }
    fn inset(self, amount: f64) -> Span {
        Span {
            left: self.left + amount,
            right: self.right - amount,
        }
    }
    fn first(self, width: f64) -> Span {
        Span {
            left: self.left,
            right: self.left + width,
        }
    }
    fn last(self, width: f64) -> Span {
        Span {
            left: self.right - width,
            right: self.right,
        }
    }
    fn tracks(self, count: usize, gap: f64) -> Vec<Span> {
        let width = (self.width() - gap * (count - 1) as f64) / count as f64;
        (0..count)
            .map(|i| {
                let left = self.left + i as f64 * (width + gap);
                Span {
                    left,
                    right: left + width,
                }
            })
            .collect()
    }
}

fn rows_for(height: f64) -> u32 {
    (1..=8)
        .rev()
        .find(|r| ((r - 1) * 8 + 7) as f64 * PITCH <= height - 0.4)
        .unwrap_or(1)
}
struct Builder {
    controls: Vec<Json>,
    states: serde_json::Map<String, Json>,
}
impl Builder {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        id: &str,
        module: &str,
        kind: &str,
        label: &str,
        help: &str,
        span: Span,
        y: f64,
        height: f64,
        value: Json,
    ) {
        let (material, presentation) = match kind {
            "button" | "toggle" => ("key", "printed"),
            "legend" => ("face", "printed"),
            "heading" => ("caption", "printed"),
            _ => ("glass", "matrix"),
        };
        let kind_name = match kind {
            "legend" | "heading" => "readout",
            other => other,
        };
        let mut control = json!({"id":id,"module_id":module,"page":"*","kind":kind_name,
            "rect":{"x":(span.left+span.right)/2.0,"y":y,"z":0.8,"width":span.width(),"height":height,"depth":0.15},
            "material":material,"label":label,"accessible_name":label,"description":help,
            "tab_index":-1,"action":id,"presentation":presentation});
        if kind == "select" {
            control["options"] = json!([{"id":"choice-0","label":""}]);
        }
        if presentation == "matrix" {
            let cols = ((span.width() - 0.8 + PITCH) / (6.0 * PITCH)).floor() as u32;
            let rows = rows_for(height);
            control["matrix"] = json!({"cols":cols,"rows":rows,"glyph_width":5,"glyph_height":7,
                "cell_width":6,"cell_height":8,"pitch":PITCH,
                "padding_x":(span.width() - (cols * 6 - 1) as f64 * PITCH) / 2.0,
                "padding_y":(height - ((rows - 1) * 8 + 7) as f64 * PITCH) / 2.0});
        }
        let interactive = matches!(kind, "button" | "toggle" | "shortcut" | "select");
        if interactive {
            control["tab_index"] = json!(self.controls.len());
        } else {
            control["action"] = json!("none");
        }
        self.states.insert(
            id.into(),
            json!({"value":value,"enabled":interactive,"visible":true,
            "pressed":false,"focused":false,"busy":false,"error":""}),
        );
        self.controls.push(control);
    }
}

/// One column of slots inside a module.
struct Column<'a> {
    b: &'a mut Builder,
    module: &'static str,
    full: Span,
    label: Span,
    /// Physical keys align to these edges.
    control: Span,
    /// Every display aligns to these edges.
    value: Span,
    top: f64,
}
impl<'a> Column<'a> {
    fn new(b: &'a mut Builder, module: &'static str, full: Span, top: f64) -> Self {
        let label = full.first(LABEL_WIDTH);
        let control = Span {
            left: label.right + GAP,
            right: full.right,
        };
        let value = control.inset(SIDE_KEY + INNER);
        Self {
            b,
            module,
            full,
            label,
            control,
            value,
            top,
        }
    }
    fn slot(&mut self) -> f64 {
        let top = self.top;
        self.top -= SLOT;
        top
    }
    /// A heading sits low in its slot: near its own rows, far from the previous section.
    fn heading(&mut self, id: &str, text: &str) {
        let top = self.slot();
        let y = top - 0.8 - HEADING / 2.0;
        self.b.add(
            id,
            self.module,
            "heading",
            text,
            text,
            self.full,
            y,
            HEADING,
            json!(text),
        );
    }
    fn next_row(&mut self) -> f64 {
        self.slot() - ROW / 2.0
    }
    fn legend(&mut self, id: &str, text: &str, help: &str, y: f64) {
        self.b.add(
            &format!("label.{id}"),
            self.module,
            "legend",
            text,
            help,
            self.label,
            y,
            ROW,
            json!(text),
        );
    }
    fn display(&mut self, id: &str, text: &str, help: &str, y: f64) {
        self.b.add(
            id,
            self.module,
            "readout",
            text,
            help,
            self.value,
            y,
            ROW,
            json!(""),
        );
    }
    /// Label, then [◀] display [▶].
    fn stepper(&mut self, id: &str, text: &str, help: &str, keys: (&str, &str)) {
        let y = self.next_row();
        self.legend(id, text, help, y);
        self.b.add(
            &format!("{id}.prev"),
            self.module,
            "button",
            keys.0,
            help,
            self.control.first(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
        self.display(id, text, help, y);
        self.b.add(
            &format!("{id}.next"),
            self.module,
            "button",
            keys.1,
            help,
            self.control.last(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
    }
    /// Label, then a latching key with a lamp.
    fn switch(&mut self, id: &str, text: &str, help: &str) {
        let y = self.next_row();
        self.legend(id, text, help, y);
        self.b.add(
            id,
            self.module,
            "toggle",
            "Off",
            help,
            self.control.first(SWITCH_KEY),
            y,
            ROW,
            json!(false),
        );
    }
    fn shortcut(&mut self, binding: Binding, text: &str, help: &str) {
        let y = self.next_row();
        let id = format!("shortcut.{}", binding.id());
        self.legend(&id, text, help, y);
        self.b.add(
            &id,
            self.module,
            "shortcut",
            text,
            help,
            self.value,
            y,
            ROW,
            json!(""),
        );
        self.b.add(
            &format!("{id}.clear"),
            self.module,
            "button",
            "×",
            "Remove this shortcut",
            self.control.last(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
    }
    fn status_with_key(
        &mut self,
        id: &str,
        text: &str,
        help: &str,
        key_id: &str,
        key: &str,
        key_help: &str,
    ) {
        let y = self.next_row();
        self.legend(id, text, help, y);
        self.display(id, text, help, y);
        self.b.add(
            key_id,
            self.module,
            "button",
            key,
            key_help,
            self.control.last(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
    }
    fn action(&mut self, id: &str, text: &str, help: &str) {
        let y = self.next_row();
        self.b.add(
            id,
            self.module,
            "button",
            text,
            help,
            self.control,
            y,
            ROW,
            Json::Null,
        );
    }
    /// Full-width click-through display: click it, or ◀ ▶, to step through choices.
    fn wide_select(&mut self, id: &str, text: &str, help: &str) {
        let y = self.next_row();
        let (full, module) = (self.full, self.module);
        self.b.add(
            &format!("{id}.prev"),
            module,
            "button",
            "◀",
            help,
            full.first(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
        self.b.add(
            id,
            module,
            "select",
            text,
            help,
            full.inset(SIDE_KEY + INNER),
            y,
            ROW,
            json!("choice-0"),
        );
        self.b.add(
            &format!("{id}.next"),
            module,
            "button",
            "▶",
            help,
            full.last(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
    }
}

fn module(id: &str, left: f64, right: f64, bottom: f64, top: f64) -> Json {
    json!({"id":id,"rect":{"x":(left+right)/2.0,"y":(bottom+top)/2.0,"z":0.66,
        "width":right-left,"height":top-bottom,"depth":0.1},
        "surface":"module","material":"module","label":"","page":"*"})
}

fn build() -> Snapshot {
    let mut base: Json = serde_json::from_str(BASE).expect("pinned base scene");
    let scene = &mut base["scene"];
    scene["id"] = json!("ears.settings.receiver");
    scene["camera"] = json!({"projection":"perspective","fov_y_deg":25,"yaw_deg":0,
        "pitch_deg":0,"target":[0,-0.5,-2],"distance":118});
    scene["materials"]["module"] = json!({"base":"#272b30","ink":"#d4d7dc","emissive":"#000000",
        "roughness":0.62,"metalness":0.45,"finish":"brushed"});
    scene["materials"]["caption"] = json!({"base":"#272b30","ink":"#aab0b8","emissive":"#000000",
        "roughness":0.62,"metalness":0.45,"finish":"brushed"});
    let device = &mut scene["devices"][0];
    device["chassis"]["height"] = json!(HEIGHT);
    let inner = HEIGHT / 2.0 - 1.6;
    let top_bottom = inner - (PAD + 4.4 + GAP + ROW + PAD);
    let mid_top = top_bottom - GAP;
    let column_top = mid_top - (PAD - GAP / 2.0);
    let mid_bottom = column_top - (SLOTS as f64 - 1.0) * SLOT - ROW - PAD;
    device["panels"] = json!([
        {"id":"ears.front","rect":{"x":0,"y":0,"z":0,"width":64,"height":HEIGHT,"depth":0.65},
         "surface":"faceplate","material":"face","label":"","page":"*"},
        module("ears.top", -EDGE, EDGE, top_bottom, inner),
        module("ears.left", -EDGE, -GAP / 2.0, mid_bottom, mid_top),
        module("ears.right", GAP / 2.0, EDGE, mid_bottom, mid_top),
    ]);
    let mut b = Builder {
        controls: Vec::new(),
        states: Default::default(),
    };

    let left_full = Span {
        left: -EDGE + PAD,
        right: -GAP / 2.0 - PAD,
    };
    let right_full = Span {
        left: GAP / 2.0 + PAD,
        right: EDGE - PAD,
    };
    let band = Span {
        left: left_full.left,
        right: right_full.right,
    };

    // Top module: main display, Test and Close; then the only navigation row,
    // whose keys line up with the two columns below.
    let display_top = inner - PAD;
    let close = band.last(SIDE_KEY);
    let test = Span {
        left: close.left - GAP - 6.4,
        right: close.left - GAP,
    };
    let display = Span {
        left: band.left,
        right: test.left - GAP,
    };
    b.add(
        "display",
        "ears.top",
        "readout",
        "Display",
        "Help, status and test transcripts",
        display,
        display_top - 2.2,
        4.4,
        json!(""),
    );
    b.add(
        "test",
        "ears.top",
        "toggle",
        "Test",
        "Speak to see your words here. Nothing is typed or saved.",
        test,
        display_top - 2.2,
        4.4,
        json!(false),
    );
    b.add(
        "window.close",
        "ears.top",
        "button",
        "×",
        "Close settings",
        close,
        display_top - ROW / 2.0,
        ROW,
        Json::Null,
    );
    let nav_y = display_top - 4.4 - GAP - ROW / 2.0;
    let tabs = left_full
        .tracks(2, GAP)
        .into_iter()
        .chain(right_full.tracks(2, GAP));
    for (span, (id, label)) in tabs.zip([
        ("dictation", "Dictation"),
        ("overlay", "Overlay"),
        ("dictionaries", "Dictionaries"),
        ("profiles", "Profiles"),
    ]) {
        b.add(
            &format!("nav.{id}"),
            "ears.top",
            "toggle",
            label,
            &format!("{label} settings"),
            span,
            nav_y,
            ROW,
            json!(id == "dictation"),
        );
    }

    {
        let mut c = Column::new(&mut b, "ears.left", left_full, column_top);
        c.heading("head.mic", "MICROPHONE");
        c.wide_select(
            "mic",
            "Microphone",
            "Click to step through inputs; it switches when you stop.",
        );
        c.heading("head.shortcuts", "SHORTCUTS");
        c.shortcut(
            Binding::Toggle,
            "Start / stop",
            "Starts and stops dictation. Click, then press keys.",
        );
        c.shortcut(
            Binding::PushToTalk,
            "Hold to talk",
            "Dictates while held. Click, then press keys.",
        );
        c.shortcut(
            Binding::Profile,
            "Next profile",
            "Switches to the next context profile.",
        );
        c.stepper(
            "keymode",
            "Key mode",
            "Toggle: press to start and stop. Hold: talk while held. Hybrid: both.",
            ("◀", "▶"),
        );
        c.switch(
            "escape",
            "Esc cancels",
            "Escape discards pending words instead of pausing.",
        );
        c.action(
            "shortcut.defaults",
            "Restore defaults",
            "Restore the default shortcuts",
        );
        c.heading("head.wake", "WAKE WORD");
        c.stepper(
            "wake",
            "Sensitivity",
            "Higher wakes more easily but triggers falsely more often",
            ("−", "+"),
        );
    }
    {
        let mut c = Column::new(&mut b, "ears.right", right_full, column_top);
        c.heading("head.speech", "SPEECH");
        c.stepper("language", "Language", "Transcription language", ("◀", "▶"));
        c.stepper(
            "pause",
            "Auto-pause",
            "Silence before dictation pauses itself",
            ("−", "+"),
        );
        c.stepper(
            "threshold",
            "Speech threshold",
            "Higher ignores more noise but may cut quiet speech",
            ("−", "+"),
        );
        c.heading("head.text", "TEXT");
        c.stepper(
            "insert",
            "Insert as",
            "Where finished text goes",
            ("◀", "▶"),
        );
        c.switch(
            "profiles",
            "Context profiles",
            "Use per-app dictionaries and settings",
        );
        c.heading("head.server", "SERVER");
        c.stepper(
            "server",
            "Server",
            "Transcription server from config.toml",
            ("◀", "▶"),
        );
        c.status_with_key(
            "connection",
            "Status",
            "Server connection",
            "server.use",
            "Use",
            "Connect to the chosen server (restarts dictation)",
        );
    }
    b.add(
        "brand",
        "ears.front",
        "legend",
        "eaRS",
        "eaRS",
        left_full.first(8.0),
        (mid_bottom - inner) / 2.0,
        2.0,
        json!("eaRS"),
    );

    device["controls"] = json!(b.controls);
    base["state"]["controls"] = json!(b.states);
    base["state"]["active_page"] = json!(PAGE);
    Snapshot::parse(&serde_json::to_string(&base).unwrap())
        .expect("receiver layout validates")
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use device_scene::{Kind, Presentation};
    fn rect(s: &Snapshot, id: &str) -> device_scene::Rect {
        s.control(id)
            .unwrap_or_else(|| panic!("{id}"))
            .1
            .rect
            .clone()
    }
    fn left(r: &device_scene::Rect) -> f64 {
        r.x - r.width / 2.0
    }
    fn right(r: &device_scene::Rect) -> f64 {
        r.x + r.width / 2.0
    }
    fn equal(a: f64, b: f64, what: &str) {
        assert!((a - b).abs() < 1e-8, "{what}: {a} != {b}");
    }
    #[test]
    fn one_navigation_level_and_every_dictation_setting_on_one_page() {
        let s = generated();
        s.validate().unwrap();
        let ids: Vec<_> = s.scene.devices[0]
            .controls
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        for id in [
            "nav.dictation",
            "nav.overlay",
            "nav.dictionaries",
            "nav.profiles",
            "test",
            "display",
            "mic",
            "shortcut.toggle",
            "shortcut.push_to_talk",
            "shortcut.profile_cycle",
            "keymode",
            "escape",
            "language",
            "pause",
            "threshold",
            "wake",
            "insert",
            "profiles",
            "server",
            "connection",
        ] {
            assert!(ids.contains(&id), "{id}");
        }
        assert!(
            !ids.iter().any(|id| id.starts_with("dictation.")),
            "no second navigation level"
        );
        assert!(s.scene.devices[0].controls.iter().all(|c| c.page == "*"));
    }
    #[test]
    fn columns_rows_and_compound_controls_share_edges() {
        let s = generated();
        for (labels, controls) in [
            (
                vec![
                    "label.shortcut.toggle",
                    "label.keymode",
                    "label.escape",
                    "label.wake",
                    "mic.prev",
                ],
                vec![
                    "shortcut.toggle.clear",
                    "keymode.next",
                    "shortcut.defaults",
                    "wake.next",
                    "mic.next",
                ],
            ),
            (
                vec!["label.language", "label.insert", "label.server"],
                vec!["language.next", "insert.next", "server.use"],
            ),
        ] {
            let edge = left(&rect(&s, labels[0]));
            for id in &labels {
                equal(left(&rect(&s, id)), edge, id);
            }
            let end = right(&rect(&s, controls[0]));
            for id in &controls {
                equal(right(&rect(&s, id)), end, id);
            }
        }
        for (label, control) in [
            ("label.pause", "pause"),
            ("label.shortcut.toggle", "shortcut.toggle"),
            ("label.escape", "escape"),
        ] {
            equal(rect(&s, label).y, rect(&s, control).y, label);
        }
        // Every value display shares the value column's edges.
        for id in [
            "pause",
            "threshold",
            "wake",
            "insert",
            "server",
            "keymode",
            "connection",
            "shortcut.profile_cycle",
        ] {
            equal(
                left(&rect(&s, id)),
                left(&rect(
                    &s,
                    if id == "insert"
                        || id == "pause"
                        || id == "threshold"
                        || id == "server"
                        || id == "connection"
                    {
                        "language"
                    } else {
                        "keymode"
                    },
                )),
                id,
            );
            equal(rect(&s, id).width, rect(&s, "language").width, id);
        }
        // One baseline grid shared by both columns.
        equal(
            rect(&s, "head.mic").y,
            rect(&s, "head.speech").y,
            "section tops",
        );
        equal(rect(&s, "mic").y, rect(&s, "language").y, "first rows");
        equal(
            rect(&s, "shortcut.toggle").y,
            rect(&s, "threshold").y,
            "grid lines",
        );
        equal(
            rect(&s, "shortcut.defaults").y,
            rect(&s, "server").y,
            "grid lines",
        );
        // Navigation keys line up with the columns; the top band spans both.
        let tab = rect(&s, "nav.dictation").width;
        for id in ["nav.overlay", "nav.dictionaries", "nav.profiles"] {
            equal(rect(&s, id).width, tab, id);
        }
        equal(
            left(&rect(&s, "nav.dictation")),
            left(&rect(&s, "mic.prev")),
            "tab grid",
        );
        equal(
            right(&rect(&s, "nav.overlay")),
            right(&rect(&s, "mic.next")),
            "tab grid",
        );
        equal(
            right(&rect(&s, "nav.profiles")),
            right(&rect(&s, "server.use")),
            "tab grid",
        );
        equal(
            left(&rect(&s, "display")),
            left(&rect(&s, "nav.dictation")),
            "top band",
        );
        equal(
            right(&rect(&s, "window.close")),
            right(&rect(&s, "nav.profiles")),
            "top band",
        );
        equal(rect(&s, "display").y, rect(&s, "test").y, "top band");
        equal(
            rect(&s, "display").y + rect(&s, "display").height / 2.0,
            rect(&s, "window.close").y + rect(&s, "window.close").height / 2.0,
            "top edge",
        );
    }
    #[test]
    fn layout_file_round_trips_and_refits_displays_at_fixed_pitch() {
        let s = generated();
        let mut file = LayoutFile::capture(&s);
        let mic = file.controls.get_mut("mic").unwrap();
        mic.width -= 4.0;
        mic.height = 3.2;
        let mut edited = s.clone();
        apply(&mut edited, &file);
        let (_, c, _) = edited.control("mic").unwrap();
        let m = c.matrix.as_ref().unwrap();
        equal(m.pitch, PITCH, "pitch");
        assert_eq!(m.rows, 2);
        assert!(((m.cols * 6 - 1) as f64 * PITCH) <= c.rect.width - 0.8 + 1e-9);
        let json = serde_json::to_string(&LayoutFile::capture(&edited)).unwrap();
        let mut again = s.clone();
        apply(&mut again, &serde_json::from_str(&json).unwrap());
        assert_eq!(LayoutFile::capture(&again), LayoutFile::capture(&edited));
        device_scene::ValidatedSnapshot::new(again).unwrap();
        // Unknown ids from an older file are ignored, never an error.
        let mut stale = LayoutFile::default();
        stale.controls.insert(
            "gone".into(),
            Placement {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                module: None,
            },
        );
        let mut kept = s.clone();
        apply(&mut kept, &stale);
        assert_eq!(kept, s);
        initial().validate().unwrap();
    }
    #[test]
    fn displays_use_one_pitch_and_nothing_overlaps() {
        let s = generated();
        let d = &s.scene.devices[0];
        for c in &d.controls {
            if c.material == "glass" {
                assert_eq!(c.presentation, Presentation::Matrix, "{}", c.id);
                equal(c.matrix.as_ref().unwrap().pitch, PITCH, &c.id);
            }
            if c.presentation == Presentation::Printed && c.kind == Kind::Readout {
                assert!(c.action == "none");
            }
        }
        let mic = d.controls.iter().find(|c| c.id == "mic").unwrap();
        assert!(
            mic.matrix.as_ref().unwrap().cols >= 22,
            "long input names fit"
        );
        for (i, a) in d.controls.iter().enumerate() {
            for b in &d.controls[i + 1..] {
                let x = (a.rect.x - b.rect.x).abs() - (a.rect.width + b.rect.width) / 2.0;
                let y = (a.rect.y - b.rect.y).abs() - (a.rect.height + b.rect.height) / 2.0;
                assert!(
                    x >= INNER - 1e-8 || y >= INNER - 1e-8,
                    "{} overlaps {}",
                    a.id,
                    b.id
                );
            }
        }
    }
}
