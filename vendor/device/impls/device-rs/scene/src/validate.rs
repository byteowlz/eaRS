use crate::*;
use std::collections::BTreeSet;
use unicode_segmentation::UnicodeSegmentation;

const SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const LIMIT: f64 = 100_000.0;

#[derive(Debug, thiserror::Error)]
#[error("{path}: {message}")]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

type Result<T = ()> = std::result::Result<T, ValidationError>;
fn require(ok: bool, path: &str, message: &str) -> Result {
    if ok {
        Ok(())
    } else {
        Err(ValidationError {
            path: path.into(),
            message: message.into(),
        })
    }
}
fn bounded(v: f64, min: f64, max: f64) -> bool {
    v.is_finite() && v >= min && v <= max
}
fn positive(v: f64) -> bool {
    bounded(v, 0.0, LIMIT) && v > 0.0
}
fn text(v: &str) -> bool {
    v.chars().count() <= 4096
}
fn id(v: &str) -> bool {
    (1..=96).contains(&v.len())
        && v.as_bytes()[0].is_ascii_lowercase()
        && v.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_.-".contains(&b))
}
fn selector(v: &str) -> bool {
    text(v)
        && (v == "*"
            || (!v.is_empty() && !v.contains('*'))
            || v.strip_suffix(".*")
                .is_some_and(|p| !p.is_empty() && !p.contains('*')))
}
/// A prefix selector matches descendants, not its root (dictation.* does not match dictation).
pub fn page_matches(selector: &str, page: &str) -> bool {
    selector == "*"
        || selector == page
        || selector.strip_suffix(".*").is_some_and(|p| {
            page.strip_prefix(p)
                .is_some_and(|tail| tail.starts_with('.') && tail.len() > 1)
        })
}
fn color(v: &str) -> bool {
    v.len() == 7 && v.starts_with('#') && v[1..].bytes().all(|b| b.is_ascii_hexdigit())
}
fn rect(r: &Rect, path: &str) -> Result {
    require(
        [r.x, r.y, r.z].iter().all(|v| bounded(*v, -LIMIT, LIMIT))
            && positive(r.width)
            && positive(r.height)
            && bounded(r.depth, 0.0, LIMIT),
        path,
        "invalid rect dimensions",
    )
}
fn insert_id(ids: &mut BTreeSet<String>, value: &str) -> Result {
    require(
        id(value) && ids.insert(value.into()),
        value,
        "invalid or duplicate scene ID",
    )
}
fn material_ref(scene: &Scene, value: &str) -> Result {
    require(
        scene.materials.contains_key(value),
        value,
        "unresolved material",
    )
}
fn range(r: &Range, path: &str) -> Result {
    require(
        bounded(r.min, -LIMIT, LIMIT)
            && bounded(r.max, -LIMIT, LIMIT)
            && r.min < r.max
            && positive(r.step),
        path,
        "invalid numeric range",
    )
}
fn matrix(m: &Matrix, r: &Rect, path: &str) -> Result {
    require(
        (1..=160).contains(&m.cols)
            && (1..=32).contains(&m.rows)
            && m.glyph_width == 5
            && (7..=8).contains(&m.glyph_height)
            && m.cell_width == 6
            && (8..=16).contains(&m.cell_height)
            && m.cell_height >= m.glyph_height
            && positive(m.pitch)
            && bounded(m.padding_x, 0.0, LIMIT)
            && bounded(m.padding_y, 0.0, LIMIT),
        path,
        "invalid fixed-pitch matrix",
    )?;
    let w = ((m.cols - 1) * m.cell_width + m.glyph_width) as f64 * m.pitch + 2.0 * m.padding_x;
    let h = ((m.rows - 1) * m.cell_height + m.glyph_height) as f64 * m.pitch + 2.0 * m.padding_y;
    require(
        w <= r.width + 1e-8 && h <= r.height + 1e-8,
        path,
        "matrix does not fit control",
    )
}

impl Viewport {
    pub fn validate(&self) -> Result {
        require(
            bounded(self.width_dip, 0.0, 16384.0)
                && self.width_dip > 0.0
                && bounded(self.height_dip, 0.0, 16384.0)
                && self.height_dip > 0.0
                && bounded(self.scale_factor, 0.0, 8.0)
                && self.scale_factor > 0.0
                && (1..=16_777_216).contains(&self.pixel_cap),
            "viewport",
            "invalid extent or pixel budget",
        )
    }
}
impl Snapshot {
    pub fn validate(&self) -> Result {
        require(self.schema_version == 1, "schema_version", "expected 1")?;
        self.viewport.validate()?;
        validate_scene(&self.scene)?;
        require(
            self.state.revision <= SAFE_INTEGER
                && text(&self.state.active_page)
                && !self.state.active_page.contains('*')
                && !self.state.active_page.is_empty(),
            "state",
            "invalid revision or active page",
        )?;
        let count: usize = self.scene.devices.iter().map(|d| d.controls.len()).sum();
        require(
            count == self.state.controls.len() && count <= 512,
            "state.controls",
            "one state per control required; maximum 512",
        )?;
        let mut focus_count = 0;
        for d in &self.scene.devices {
            for c in &d.controls {
                let state = self
                    .state
                    .controls
                    .get(&c.id)
                    .ok_or_else(|| ValidationError {
                        path: c.id.clone(),
                        message: "missing state".into(),
                    })?;
                validate_value(c, &state.value)?;
                require(text(&state.error), &c.id, "error text too long")?;
                if let Some(s) = &state.selection {
                    let value = state.value.string().ok_or_else(|| ValidationError {
                        path: c.id.clone(),
                        message: "selection requires text".into(),
                    })?;
                    require(
                        matches!(c.kind, Kind::Text | Kind::Shortcut)
                            && s.start <= s.end
                            && s.end <= value.encode_utf16().count() as u64,
                        &c.id,
                        "invalid UTF-16 selection",
                    )?;
                    let mut boundaries = BTreeSet::from([0]);
                    let mut offset = 0;
                    for grapheme in value.graphemes(true) {
                        offset += grapheme.encode_utf16().count() as u64;
                        boundaries.insert(offset);
                    }
                    require(
                        boundaries.contains(&s.start) && boundaries.contains(&s.end),
                        &c.id,
                        "selection must not split a grapheme or surrogate pair",
                    )?;
                }
                let panel = d
                    .panels
                    .iter()
                    .find(|p| p.id == c.module_id)
                    .expect("validated panel");
                if state.focused {
                    focus_count += 1;
                    require(
                        state.enabled
                            && state.visible
                            && c.tab_index >= 0
                            && page_matches(&panel.page, &self.state.active_page)
                            && page_matches(&c.page, &self.state.active_page),
                        &c.id,
                        "focus must be on an enabled active control",
                    )?;
                }
            }
        }
        require(focus_count <= 1, "state", "multiple focused controls")
    }
}
fn validate_scene(s: &Scene) -> Result {
    require(
        id(&s.id)
            && s.revision <= SAFE_INTEGER
            && s.devices.len() <= 8
            && (1..=64).contains(&s.materials.len()),
        "scene",
        "invalid scene bounds",
    )?;
    let c = &s.camera;
    require(
        bounded(c.fov_y_deg, 10.0, 80.0)
            && bounded(c.yaw_deg, -60.0, 60.0)
            && bounded(c.pitch_deg, -20.0, 60.0)
            && positive(c.distance)
            && c.target.iter().all(|v| bounded(*v, -LIMIT, LIMIT)),
        "camera",
        "invalid perspective camera",
    )?;
    let l = &s.light;
    require(
        bounded(l.azimuth_deg, -180.0, 180.0)
            && bounded(l.elevation_deg, 0.0, 90.0)
            && bounded(l.intensity, 0.0, 10.0)
            && bounded(l.ambient, 0.0, 3.0),
        "light",
        "invalid lighting",
    )?;
    for (key, m) in &s.materials {
        require(
            id(key)
                && color(&m.base)
                && color(&m.ink)
                && color(&m.emissive)
                && bounded(m.roughness, 0.0, 1.0)
                && bounded(m.metalness, 0.0, 1.0),
            key,
            "invalid material",
        )?;
    }
    let mut ids = BTreeSet::new();
    for d in &s.devices {
        validate_device(d, s, &mut ids)?;
    }
    Ok(())
}
fn validate_device(d: &Device, s: &Scene, ids: &mut BTreeSet<String>) -> Result {
    insert_id(ids, &d.id)?;
    let c = &d.chassis;
    require(
        d.origin.iter().all(|v| bounded(*v, -LIMIT, LIMIT))
            && positive(c.width)
            && positive(c.height)
            && positive(c.depth)
            && positive(c.front_thickness)
            && c.front_thickness <= c.depth
            && bounded(c.feet_height, 0.0, LIMIT)
            && d.panels.len() <= 512
            && d.controls.len() <= 512,
        &d.id,
        "invalid chassis bounds",
    )?;
    material_ref(s, &c.material)?;
    for p in &d.panels {
        insert_id(ids, &p.id)?;
        rect(&p.rect, &p.id)?;
        material_ref(s, &p.material)?;
        require(
            text(&p.label) && selector(&p.page),
            &p.id,
            "invalid panel text or page selector",
        )?;
    }
    for c in &d.controls {
        validate_control(c, d, s, ids)?;
    }
    Ok(())
}
fn validate_control(c: &Control, d: &Device, s: &Scene, ids: &mut BTreeSet<String>) -> Result {
    insert_id(ids, &c.id)?;
    rect(&c.rect, &c.id)?;
    material_ref(s, &c.material)?;
    require(
        [&c.label, &c.accessible_name, &c.description]
            .iter()
            .all(|v| text(v))
            && c.unit.as_ref().is_none_or(|v| text(v))
            && selector(&c.page)
            && id(&c.action)
            && (-1..=4096).contains(&c.tab_index),
        &c.id,
        "invalid control semantics",
    )?;
    let p = d
        .panels
        .iter()
        .find(|p| p.id == c.module_id)
        .ok_or_else(|| ValidationError {
            path: c.id.clone(),
            message: "unresolved parent panel in device".into(),
        })?;
    require(
        (c.rect.x - p.rect.x).abs() + c.rect.width * 0.5 <= p.rect.width * 0.5 + 1e-8
            && (c.rect.y - p.rect.y).abs() + c.rect.height * 0.5 <= p.rect.height * 0.5 + 1e-8,
        &c.id,
        "control outside parent panel",
    )?;
    if let Some(r) = &c.range {
        range(r, &c.id)?;
    }
    if matches!(c.kind, Kind::Knob | Kind::Fader | Kind::Meter) {
        require(c.range.is_some(), &c.id, "numeric control needs range")?;
    }
    if let Some(options) = &c.options {
        let mut ids = BTreeSet::new();
        require(options.len() <= 128, &c.id, "too many options")?;
        for o in options {
            require(
                id(&o.id) && ids.insert(&o.id) && text(&o.label),
                &c.id,
                "invalid or duplicate option",
            )?;
        }
    }
    if c.kind == Kind::Select {
        require(
            c.options.as_ref().is_some_and(|o| !o.is_empty()),
            &c.id,
            "select needs options",
        )?;
    }
    if let Some(m) = &c.matrix {
        matrix(m, &c.rect, &c.id)?;
    }
    if c.presentation == Presentation::Matrix {
        require(
            c.matrix.is_some(),
            &c.id,
            "matrix presentation needs geometry",
        )?;
    }
    Ok(())
}
/// Validate a proposed value without applying quantization. Step describes input increments, not a value lattice.
pub fn validate_value(c: &Control, v: &Value) -> Result {
    let valid = match c.kind {
        Kind::Button => v.is_null() || v.string().is_some(),
        Kind::Toggle | Kind::Indicator => v.boolean().is_some(),
        Kind::Knob | Kind::Fader | Kind::Meter => v
            .number()
            .zip(c.range.as_ref())
            .is_some_and(|(v, r)| bounded(v, r.min, r.max)),
        Kind::Select => v.string().is_some_and(|v| {
            c.options
                .as_ref()
                .is_some_and(|o| o.iter().any(|o| o.id == v))
        }),
        Kind::Text | Kind::Shortcut | Kind::Readout => v.string().is_some(),
    };
    require(
        valid,
        &c.id,
        "value does not match control kind/range/options",
    )
}
impl Action {
    pub fn validate(&self, snapshot: &Snapshot) -> Result {
        snapshot.validate()?;
        require(
            self.schema_version == 1
                && self.scene_id == snapshot.scene.id
                && self.scene_revision == snapshot.scene.revision
                && self.state_revision == snapshot.state.revision
                && id(&self.interaction_id)
                && self.sequence <= SAFE_INTEGER,
            "action",
            "invalid or stale action identity",
        )?;
        let (d, c) = snapshot
            .scene
            .devices
            .iter()
            .find_map(|d| {
                d.controls
                    .iter()
                    .find(|c| c.id == self.control_id)
                    .map(|c| (d, c))
            })
            .ok_or_else(|| ValidationError {
                path: self.control_id.clone(),
                message: "unknown action control".into(),
            })?;
        if self.phase == Phase::Cancel {
            return require(self.value.is_null(), "action", "cancel value must be null");
        }
        let state = &snapshot.state.controls[&c.id];
        let panel = d
            .panels
            .iter()
            .find(|p| p.id == c.module_id)
            .expect("validated panel");
        require(
            state.enabled
                && state.visible
                && c.action != "none"
                && page_matches(&c.page, &snapshot.state.active_page)
                && page_matches(&panel.page, &snapshot.state.active_page),
            &c.id,
            "inactive control cannot emit action",
        )?;
        validate_value(c, &self.value)
    }
}
