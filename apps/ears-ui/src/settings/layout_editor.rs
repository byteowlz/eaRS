//! Native layout editor for the receiver (`ears-ui --edit-settings-layout`).
//!
//! Works on the real 3D device: click to select (Shift adds), drag to move, drag
//! the corner handle to resize. Everything snaps to the 0.2-unit grid; displays
//! refit their dot grid at the fixed pitch. Moving a module carries its controls;
//! a control dropped into another module joins it. Arrows nudge (Shift x4),
//! Option+arrows resize, Cmd-S saves, Cmd-Z undoes, G shows the grid, E toggles
//! between editing and using the device. Synthetic state only: no engine, no config.
use super::device_layout::{self, LayoutFile, MIN_SIZE, SNAP};
use device_scene::{Projection, Rect, Snapshot, ValidatedSnapshot};
use glam::DVec3;
use gpui::{AnyElement, KeyDownEvent, div, prelude::*, px, rgb, rgba};
use std::path::PathBuf;

pub(super) const FLAG: &str = "--edit-settings-layout";
const HANDLE: f32 = 10.0;
const UNDO_LIMIT: usize = 200;

pub(super) fn requested() -> bool {
    std::env::args().any(|a| a == FLAG)
}
pub(super) fn path() -> PathBuf {
    std::env::var_os("EARS_SETTINGS_LAYOUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            ears::paths::config_dir()
                .join("ears")
                .join("settings-layout.json")
        })
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Move,
    Resize,
}
struct Drag {
    mode: Mode,
    start: (f64, f64),
    z: f64,
    originals: Vec<(String, Rect)>,
    moved: bool,
}
pub(super) struct LayoutEditor {
    pub active: bool,
    path: PathBuf,
    selected: Vec<String>,
    drag: Option<Drag>,
    undo: Vec<Snapshot>,
    saved: Option<LayoutFile>,
    grid: bool,
    status: String,
}

fn snap(v: f64) -> f64 {
    (v / SNAP).round() * SNAP
}
fn panel_ids(s: &Snapshot) -> Vec<String> {
    s.scene.devices[0]
        .panels
        .iter()
        .filter(|p| p.surface != device_scene::Surface::Faceplate)
        .map(|p| p.id.clone())
        .collect()
}
fn rect_of(s: &Snapshot, id: &str) -> Option<Rect> {
    let d = &s.scene.devices[0];
    d.controls
        .iter()
        .find(|c| c.id == id)
        .map(|c| c.rect.clone())
        .or_else(|| d.panels.iter().find(|p| p.id == id).map(|p| p.rect.clone()))
}
fn set_rect(s: &mut Snapshot, id: &str, r: &Rect) {
    let d = &mut s.scene.devices[0];
    if let Some(c) = d.controls.iter_mut().find(|c| c.id == id) {
        c.rect = r.clone();
        device_layout::refit(c);
    } else if let Some(p) = d.panels.iter_mut().find(|p| p.id == id) {
        p.rect = r.clone();
    }
}
/// A control belongs to the smallest module containing its centre.
fn rehome(s: &mut Snapshot, id: &str) {
    let d = &mut s.scene.devices[0];
    let Some(c) = d.controls.iter().find(|c| c.id == id) else {
        return;
    };
    let (x, y) = (c.rect.x, c.rect.y);
    let home = d
        .panels
        .iter()
        .filter(|p| {
            (p.rect.x - x).abs() <= p.rect.width / 2.0
                && (p.rect.y - y).abs() <= p.rect.height / 2.0
        })
        .min_by(|a, b| (a.rect.width * a.rect.height).total_cmp(&(b.rect.width * b.rect.height)))
        .map(|p| p.id.clone());
    if let (Some(home), Some(c)) = (home, d.controls.iter_mut().find(|c| c.id == id)) {
        c.module_id = home;
    }
}
fn visible(s: &Snapshot, id: &str) -> bool {
    s.state.controls.get(id).is_none_or(|st| st.visible)
}

/// Screen rectangle (left, top, right, bottom) of a device-local rect's front face.
fn screen(p: &Projection, s: &Snapshot, r: &Rect) -> Option<[f32; 4]> {
    let o = s.scene.devices[0].origin;
    let z = r.z + o[2];
    let a = p.project(DVec3::new(
        r.x - r.width / 2.0 + o[0],
        r.y + r.height / 2.0 + o[1],
        z,
    ))?;
    let b = p.project(DVec3::new(
        r.x + r.width / 2.0 + o[0],
        r.y - r.height / 2.0 + o[1],
        z,
    ))?;
    Some([a[0] as f32, a[1] as f32, b[0] as f32, b[1] as f32])
}
/// The device-local point under the pointer on the plane z.
fn world(p: &Projection, s: &Snapshot, x: f32, y: f32, z: f64) -> Option<(f64, f64)> {
    let o = s.scene.devices[0].origin;
    let ray = p.ray(x as f64, y as f64)?;
    let plane = z + o[2];
    if ray.direction.z.abs() < 1e-9 {
        return None;
    }
    let t = (plane - ray.origin.z) / ray.direction.z;
    let hit = ray.origin + ray.direction * t;
    Some((hit.x - o[0], hit.y - o[1]))
}

impl LayoutEditor {
    pub fn new(s: &mut Snapshot) -> Self {
        let path = path();
        let mut status = format!("Editing {}. Nothing saved yet.", path.display());
        let mut saved = None;
        if let Ok(text) = std::fs::read_to_string(&path) {
            match serde_json::from_str::<LayoutFile>(&text) {
                Ok(file) => {
                    let mut candidate = s.clone();
                    device_layout::apply(&mut candidate, &file);
                    if ValidatedSnapshot::new(candidate.clone()).is_ok() {
                        *s = candidate;
                        saved = Some(LayoutFile::capture(s));
                        status = format!("Loaded {}.", path.display());
                    } else {
                        status = format!(
                            "{} does not fit the current device; starting from the default.",
                            path.display()
                        );
                    }
                }
                Err(error) => status = format!("Cannot read {}: {error}", path.display()),
            }
        }
        Self {
            active: true,
            path,
            selected: Vec::new(),
            drag: None,
            undo: Vec::new(),
            saved,
            grid: false,
            status,
        }
    }
    fn dirty(&self, s: &Snapshot) -> bool {
        self.saved.as_ref() != Some(&LayoutFile::capture(s))
    }
    /// Topmost element under the pointer: controls before modules, smallest first.
    fn pick(&self, p: &Projection, s: &Snapshot, x: f32, y: f32) -> Option<String> {
        let d = &s.scene.devices[0];
        let inside = |r: &Rect| {
            screen(p, s, r).is_some_and(|[l, t, rr, b]| x >= l && x <= rr && y >= t && y <= b)
        };
        let area = |r: &Rect| r.width * r.height;
        d.controls
            .iter()
            .filter(|c| visible(s, &c.id) && inside(&c.rect))
            .min_by(|a, b| area(&a.rect).total_cmp(&area(&b.rect)))
            .map(|c| c.id.clone())
            .or_else(|| {
                d.panels
                    .iter()
                    .filter(|p| p.surface != device_scene::Surface::Faceplate && inside(&p.rect))
                    .min_by(|a, b| area(&a.rect).total_cmp(&area(&b.rect)))
                    .map(|p| p.id.clone())
            })
    }
    /// Validate before applying: an edit the scene would reject is refused, not rendered.
    fn commit(&mut self, before: &Snapshot, candidate: Snapshot) -> Option<Snapshot> {
        match ValidatedSnapshot::new(candidate.clone()) {
            Ok(_) => Some(candidate),
            Err(error) => {
                self.status = format!("Not applied: {error}");
                let _ = before;
                None
            }
        }
    }
    fn push_undo(&mut self, s: &Snapshot) {
        self.undo.push(s.clone());
        if self.undo.len() > UNDO_LIMIT {
            self.undo.remove(0);
        }
    }
    /// Originals for the selection; a module brings its controls along.
    fn originals(&self, s: &Snapshot) -> Vec<(String, Rect)> {
        let modules = panel_ids(s);
        let mut out: Vec<(String, Rect)> = Vec::new();
        for id in &self.selected {
            if let Some(r) = rect_of(s, id) {
                out.push((id.clone(), r));
            }
            if modules.contains(id) {
                for c in s.scene.devices[0]
                    .controls
                    .iter()
                    .filter(|c| &c.module_id == id)
                {
                    if !out.iter().any(|(o, _)| o == &c.id) {
                        out.push((c.id.clone(), c.rect.clone()));
                    }
                }
            }
        }
        out
    }
    fn moved(
        &mut self,
        s: &Snapshot,
        originals: &[(String, Rect)],
        dx: f64,
        dy: f64,
    ) -> Option<Snapshot> {
        let mut candidate = s.clone();
        // Snap the lead element's edge; everything moves by the same snapped delta.
        let Some((_, lead)) = originals.first() else {
            return None;
        };
        let left = lead.x - lead.width / 2.0;
        let top = lead.y + lead.height / 2.0;
        let (sx, sy) = (snap(left + dx) - left, snap(top + dy) - top);
        for (id, r) in originals {
            set_rect(
                &mut candidate,
                id,
                &Rect {
                    x: r.x + sx,
                    y: r.y + sy,
                    ..r.clone()
                },
            );
        }
        for (id, _) in originals {
            rehome(&mut candidate, id);
        }
        self.commit(s, candidate)
    }
    fn resized(&mut self, s: &Snapshot, id: &str, r: &Rect, dx: f64, dy: f64) -> Option<Snapshot> {
        let (left, top) = (r.x - r.width / 2.0, r.y + r.height / 2.0);
        let right = snap(r.x + r.width / 2.0 + dx).max(left + MIN_SIZE);
        let bottom = snap(r.y - r.height / 2.0 + dy).min(top - MIN_SIZE);
        let next = Rect {
            x: (left + right) / 2.0,
            y: (top + bottom) / 2.0,
            width: right - left,
            height: top - bottom,
            ..r.clone()
        };
        let mut candidate = s.clone();
        set_rect(&mut candidate, id, &next);
        self.commit(s, candidate)
    }

    // ---- pointer ------------------------------------------------------------

    pub fn mouse_down(&mut self, p: &Projection, s: &Snapshot, x: f32, y: f32, shift: bool) {
        // The resize handle of a single selection wins over picking.
        if let [only] = self.selected.as_slice()
            && let Some(r) = rect_of(s, only)
            && let Some([_, _, rr, b]) = screen(p, s, &r)
            && (x - rr).abs() <= HANDLE
            && (y - b).abs() <= HANDLE
            && let Some(start) = world(p, s, x, y, r.z)
        {
            self.drag = Some(Drag {
                mode: Mode::Resize,
                start,
                z: r.z,
                originals: vec![(only.clone(), r)],
                moved: false,
            });
            return;
        }
        let picked = self.pick(p, s, x, y);
        match (&picked, shift) {
            (Some(id), true) => {
                if let Some(i) = self.selected.iter().position(|s| s == id) {
                    self.selected.remove(i);
                } else {
                    self.selected.push(id.clone());
                }
            }
            (Some(id), false) if !self.selected.contains(id) => self.selected = vec![id.clone()],
            (Some(_), false) => {}
            (None, _) => self.selected.clear(),
        }
        if let Some(id) = picked.filter(|id| self.selected.contains(id))
            && let Some(r) = rect_of(s, &id)
            && let Some(start) = world(p, s, x, y, r.z)
        {
            // Lead with the element under the pointer so its edge snaps.
            let mut originals = self.originals(s);
            if let Some(i) = originals.iter().position(|(o, _)| *o == id) {
                originals.swap(0, i);
            }
            self.drag = Some(Drag {
                mode: Mode::Move,
                start,
                z: r.z,
                originals,
                moved: false,
            });
        }
        self.status = self.describe(s);
    }
    pub fn mouse_move(&mut self, p: &Projection, s: &Snapshot, x: f32, y: f32) -> Option<Snapshot> {
        let drag = self.drag.as_ref()?;
        let (wx, wy) = world(p, s, x, y, drag.z)?;
        let (dx, dy) = (wx - drag.start.0, wy - drag.start.1);
        if !drag.moved && dx.abs() < SNAP / 2.0 && dy.abs() < SNAP / 2.0 {
            return None;
        }
        let (mode, originals) = (drag.mode, drag.originals.clone());
        if !self.drag.as_ref().is_some_and(|d| d.moved) {
            self.push_undo(s);
            if let Some(d) = self.drag.as_mut() {
                d.moved = true;
            }
        }
        // Always apply relative to the state when the drag began.
        let base = self.undo.last().cloned().unwrap_or_else(|| s.clone());
        let next = match mode {
            Mode::Move => self.moved(&base, &originals, dx, dy),
            Mode::Resize => {
                let (id, r) = &originals[0];
                self.resized(&base, id, r, dx, dy)
            }
        };
        if let Some(n) = &next {
            self.status = self.describe(n);
        }
        next
    }
    pub fn mouse_up(&mut self) {
        self.drag = None;
    }

    // ---- keyboard -----------------------------------------------------------

    /// Returns (handled, new snapshot to apply).
    pub fn key(&mut self, event: &KeyDownEvent, s: &Snapshot) -> (bool, Option<Snapshot>) {
        let k = &event.keystroke;
        let m = k.modifiers;
        let primary = if cfg!(target_os = "macos") {
            m.platform
        } else {
            m.control
        };
        match (k.key.as_str(), primary) {
            ("s", true) => {
                self.save(s);
                return (true, None);
            }
            ("z", true) => {
                return match self.undo.pop() {
                    Some(previous) => {
                        self.status = "Undone.".into();
                        (true, Some(previous))
                    }
                    None => {
                        self.status = "Nothing to undo.".into();
                        (true, None)
                    }
                };
            }
            ("e", false) if m == gpui::Modifiers::default() => {
                self.active = !self.active;
                self.status = if self.active {
                    "Editing.".into()
                } else {
                    "Using the device. Press E to edit again.".into()
                };
                return (true, None);
            }
            _ => {}
        }
        if !self.active {
            return (false, None);
        }
        match k.key.as_str() {
            "g" if m == gpui::Modifiers::default() => {
                self.grid = !self.grid;
                (true, None)
            }
            "escape" => {
                self.selected.clear();
                self.status = self.describe(s);
                (true, None)
            }
            "left" | "right" | "up" | "down" if !self.selected.is_empty() => {
                let step = if m.shift { SNAP * 4.0 } else { SNAP };
                let (dx, dy) = match k.key.as_str() {
                    "left" => (-step, 0.0),
                    "right" => (step, 0.0),
                    "up" => (0.0, step),
                    _ => (0.0, -step),
                };
                self.push_undo(s);
                let next = if m.alt {
                    // Option resizes from the top-left corner: right grows, down grows.
                    match self.selected.clone().as_slice() {
                        [only] => rect_of(s, only).and_then(|r| self.resized(s, only, &r, dx, dy)),
                        _ => {
                            self.status = "Select one element to resize.".into();
                            None
                        }
                    }
                } else {
                    let originals = self.originals(s);
                    self.moved(s, &originals, dx, dy)
                };
                if next.is_none() {
                    self.undo.pop();
                }
                if let Some(n) = &next {
                    self.status = self.describe(n);
                }
                (true, next)
            }
            _ => (true, None),
        }
    }
    fn save(&mut self, s: &Snapshot) {
        let file = LayoutFile::capture(s);
        let result = (|| -> anyhow::Result<()> {
            let text = serde_json::to_string_pretty(&file)? + "\n";
            if let Some(dir) = self.path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let tmp = self.path.with_extension("json.tmp");
            std::fs::write(&tmp, text)?;
            std::fs::rename(&tmp, &self.path)?;
            Ok(())
        })();
        self.status = match result {
            Ok(()) => {
                self.saved = Some(file);
                format!("Saved {}.", self.path.display())
            }
            Err(error) => format!("Not saved: {error:#}"),
        };
    }
    fn describe(&self, s: &Snapshot) -> String {
        match self.selected.as_slice() {
            [] => "Click an element. Shift adds. Drag moves; corner handle resizes.".into(),
            [only] => match rect_of(s, only) {
                Some(r) => {
                    let module = s.scene.devices[0]
                        .controls
                        .iter()
                        .find(|c| &c.id == only)
                        .map(|c| format!("  in {}", c.module_id))
                        .unwrap_or_default();
                    format!(
                        "{only}  x {:.1}  y {:.1}  w {:.1}  h {:.1}  left {:.1}  top {:.1}{module}",
                        r.x,
                        r.y,
                        r.width,
                        r.height,
                        r.x - r.width / 2.0,
                        r.y + r.height / 2.0
                    )
                }
                None => only.clone(),
            },
            many => format!("{} elements selected", many.len()),
        }
    }

    // ---- overlay ------------------------------------------------------------

    pub fn overlay(&self, p: Option<Projection>, s: &Snapshot) -> AnyElement {
        let mut root = div().absolute().top_0().left_0().size_full();
        if let Some(p) = p.filter(|_| self.active) {
            let d = &s.scene.devices[0];
            if self.grid {
                let c = &d.chassis;
                let mut x = -c.width / 2.0;
                while x <= c.width / 2.0 + 1e-9 {
                    let major = ((x / 0.8).round() * 0.8 - x).abs() < 1e-6;
                    if let Some([l, t, _, b]) = screen(
                        &p,
                        s,
                        &Rect {
                            x,
                            y: 0.0,
                            z: 0.8,
                            width: 0.0,
                            height: c.height,
                            depth: 0.0,
                        },
                    ) {
                        root = root.child(
                            div()
                                .absolute()
                                .left(px(l))
                                .top(px(t))
                                .w(px(1.0))
                                .h(px(b - t))
                                .bg(if major {
                                    rgba(0xffffff26)
                                } else {
                                    rgba(0xffffff0d)
                                }),
                        );
                    }
                    x += 0.8;
                }
                let mut y = -c.height / 2.0;
                while y <= c.height / 2.0 + 1e-9 {
                    if let Some([l, t, r, _]) = screen(
                        &p,
                        s,
                        &Rect {
                            x: 0.0,
                            y,
                            z: 0.8,
                            width: c.width,
                            height: 0.0,
                            depth: 0.0,
                        },
                    ) {
                        root = root.child(
                            div()
                                .absolute()
                                .left(px(l))
                                .top(px(t))
                                .w(px(r - l))
                                .h(px(1.0))
                                .bg(rgba(0xffffff1a)),
                        );
                    }
                    y += 0.8;
                }
            }
            let outline = |r: &Rect, selected: bool, panel: bool| {
                screen(&p, s, r).map(|[l, t, rr, b]| {
                    div()
                        .absolute()
                        .left(px(l))
                        .top(px(t))
                        .w(px(rr - l))
                        .h(px(b - t))
                        .border_1()
                        .border_color(if selected {
                            rgb(0xff8a3d)
                        } else if panel {
                            rgba(0x6fb7ff55)
                        } else {
                            rgba(0xffffff40)
                        })
                })
            };
            for panel in d
                .panels
                .iter()
                .filter(|p| p.surface != device_scene::Surface::Faceplate)
            {
                if let Some(o) = outline(&panel.rect, self.selected.contains(&panel.id), true) {
                    root = root.child(o);
                }
            }
            for c in d.controls.iter().filter(|c| visible(s, &c.id)) {
                if let Some(o) = outline(&c.rect, self.selected.contains(&c.id), false) {
                    root = root.child(o);
                }
            }
            if let [only] = self.selected.as_slice()
                && let Some(r) = rect_of(s, only)
                && let Some([_, _, rr, b]) = screen(&p, s, &r)
            {
                root = root.child(
                    div()
                        .absolute()
                        .left(px(rr - HANDLE / 2.0))
                        .top(px(b - HANDLE / 2.0))
                        .w(px(HANDLE))
                        .h(px(HANDLE))
                        .bg(rgb(0xff8a3d)),
                );
            }
        }
        let dirty = if self.dirty(s) { "  · unsaved" } else { "" };
        let help = if self.active {
            "Drag move · corner resize · ⇧ add · arrows nudge (⇧×4) · ⌥ arrows resize · ⌘S save · ⌘Z undo · G grid · E use"
        } else {
            "E edit"
        };
        root.child(
            div()
                .absolute()
                .left(px(12.0))
                .bottom(px(12.0))
                .max_w(px(900.0))
                .p(px(8.0))
                .bg(rgba(0x0b0d10e6))
                .text_color(rgb(0xe8eaed))
                .text_size(px(12.0))
                .font_family(crate::theme::FONT_SANS)
                .child(div().child(format!("{}{dirty}", self.status)))
                .child(div().text_color(rgb(0x9aa0a8)).child(help)),
        )
        .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edits_snap_validate_and_carry_module_children() {
        let s = device_layout::generated();
        let mut editor = LayoutEditor {
            active: true,
            path: PathBuf::from("unused"),
            selected: vec!["mic".into()],
            drag: None,
            undo: Vec::new(),
            saved: None,
            grid: false,
            status: String::new(),
        };
        let originals = editor.originals(&s);
        let moved = editor.moved(&s, &originals, 0.33, -0.07).unwrap();
        let r = rect_of(&moved, "mic").unwrap();
        let left = r.x - r.width / 2.0;
        assert!(
            ((left / SNAP).round() * SNAP - left).abs() < 1e-9,
            "edges land on the grid"
        );
        // Resizing keeps the top-left corner and the fixed dot pitch.
        let before = rect_of(&s, "language").unwrap();
        let resized = editor.resized(&s, "language", &before, -1.0, -1.5).unwrap();
        let after = rect_of(&resized, "language").unwrap();
        assert!(((before.x - before.width / 2.0) - (after.x - after.width / 2.0)).abs() < 1e-9);
        assert!(after.height > before.height);
        assert_eq!(
            resized
                .control("language")
                .unwrap()
                .1
                .matrix
                .as_ref()
                .unwrap()
                .pitch,
            device_layout::PITCH
        );
        // A control cannot leave every module; such an edit is refused.
        editor.selected = vec!["mic".into()];
        assert!(editor.moved(&s, &editor.originals(&s), 0.0, 40.0).is_none());
        // Moving a module brings its controls.
        editor.selected = vec!["ears.right".into()];
        let carried = editor.moved(&s, &editor.originals(&s), 0.0, -0.4).unwrap();
        let shift = rect_of(&carried, "language").unwrap().y - rect_of(&s, "language").unwrap().y;
        assert!((shift + 0.4).abs() < 1e-9);
    }
}
