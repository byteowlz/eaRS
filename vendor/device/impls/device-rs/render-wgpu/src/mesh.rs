use crate::RenderError;
use bytemuck::{Pod, Zeroable};
use device_scene::*;
use glam::{DVec3, Vec2, Vec3};

#[path = "matrix_mesh.rs"]
mod matrix_mesh;
#[path = "shape.rs"]
mod shape;

const MAX_VERTICES: usize = 600_000;
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub base: [f32; 3],
    pub props: [f32; 3],
    pub local: [f32; 2],
    pub emission: [f32; 3],
}
#[derive(Clone, Debug)]
pub struct HitTriangle {
    pub vertices: [DVec3; 3],
    pub control_id: Option<String>,
}
/// Printed row legends align to their column edge; keys and displays centre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextAlign {
    Start,
    Center,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextRaster {
    Native,
    Matrix,
    MatrixClipped,
    UnicodeFallback,
}
impl TextRaster {
    pub fn is_matrix(self) -> bool {
        matches!(self, Self::Matrix | Self::MatrixClipped)
    }
}
#[derive(Clone, Debug)]
pub struct Label {
    pub text: String,
    pub center: DVec3,
    pub width: f64,
    pub height: f64,
    pub color: [u8; 3],
    pub printed: bool,
    pub raster: TextRaster,
    pub align: TextAlign,
    pub control_id: Option<String>,
}
#[derive(Default)]
pub(crate) struct Mesh {
    pub vertices: Vec<Vertex>,
    pub hits: Vec<HitTriangle>,
    pub labels: Vec<Label>,
}

fn color_bytes(value: &str) -> [u8; 3] {
    std::array::from_fn(|i| {
        u8::from_str_radix(&value[1 + i * 2..3 + i * 2], 16).expect("validated color")
    })
}
fn linear_color(value: &str) -> [f32; 3] {
    color_bytes(value).map(|b| {
        let s = b as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    })
}
fn dark() -> Material {
    Material {
        base: "#070809".into(),
        ink: "#d4d7dc".into(),
        emissive: "#000000".into(),
        roughness: 0.8,
        metalness: 0.1,
        finish: Finish::Matte,
    }
}
fn trim() -> Material {
    Material {
        base: "#53585d".into(),
        metalness: 0.8,
        roughness: 0.35,
        ..dark()
    }
}
impl Mesh {
    fn triangle(
        &mut self,
        positions: [Vec3; 3],
        normal: Vec3,
        mat: &Material,
        center: Vec3,
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        if !normal.is_finite() || positions.iter().any(|p| !p.is_finite()) {
            return Err(RenderError::Budget(
                "geometry is not representable at GPU precision",
            ));
        }
        if self.vertices.len() + 3 > MAX_VERTICES {
            return Err(RenderError::Budget("mesh vertex budget exceeded"));
        }
        for position in positions {
            self.vertices.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
                base: linear_color(&mat.base),
                props: [
                    mat.roughness as f32,
                    mat.metalness as f32,
                    match mat.finish {
                        Finish::Matte => 0.0,
                        Finish::Brushed => 1.0,
                        Finish::Spun => 2.0,
                        Finish::Glass => 3.0,
                    },
                ],
                local: [(position.x - center.x), (position.y - center.y)],
                emission: linear_color(&mat.emissive),
            });
        }
        self.hits.push(HitTriangle {
            vertices: positions.map(|p| p.as_dvec3()),
            control_id: control.map(String::from),
        });
        Ok(())
    }
    fn quad(
        &mut self,
        p: [Vec3; 4],
        mat: &Material,
        center: Vec3,
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
        self.triangle([p[0], p[1], p[2]], normal, mat, center, control)?;
        self.triangle([p[0], p[2], p[3]], normal, mat, center, control)
    }
    fn cuboid(
        &mut self,
        r: &Rect,
        origin: [f64; 3],
        mat: &Material,
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        let p = r.corners(origin).map(|p| p.as_vec3());
        let center = Vec3::new(
            (r.x + origin[0]) as f32,
            (r.y + origin[1]) as f32,
            (r.z + origin[2]) as f32,
        );
        // Outward face winding; z is always front, never a centre coordinate.
        for indices in [
            [4, 5, 7, 6],
            [1, 0, 2, 3],
            [0, 4, 6, 2],
            [5, 1, 3, 7],
            [6, 7, 3, 2],
            [0, 1, 5, 4],
        ] {
            if r.depth <= 1e-8 && indices != [4, 5, 7, 6] {
                continue;
            }
            self.quad(indices.map(|i| p[i]), mat, center, control)?;
        }
        Ok(())
    }
    fn bevel(
        &mut self,
        r: &Rect,
        origin: [f64; 3],
        mat: &Material,
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        if r.depth <= 1e-8 {
            return self.cuboid(r, origin, mat, control);
        }
        let b = 0.065_f64
            .min(r.width * 0.08)
            .min(r.height * 0.08)
            .min(r.depth * 0.45);
        // Rounded plan corners and a rounded front edge, with smooth normals.
        self.rounded_slab(r, origin, mat, control, 0.22, b)
    }
    fn cylinder(
        &mut self,
        r: &Rect,
        origin: [f64; 3],
        mat: &Material,
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        let center = Vec3::new(
            (r.x + origin[0]) as f32,
            (r.y + origin[1]) as f32,
            (r.z + origin[2]) as f32,
        );
        let radius = r.width.min(r.height) as f32 * 0.5;
        let side = Material {
            finish: Finish::Matte,
            ..mat.clone()
        };
        for i in 0..96 {
            let p = |step: f32, z: f32| {
                center
                    + Vec3::new(
                        (step * std::f32::consts::TAU / 96.0).cos() * radius,
                        (step * std::f32::consts::TAU / 96.0).sin() * radius,
                        z,
                    )
            };
            let a = p(i as f32, 0.0);
            let b = p(i as f32 + 1.0, 0.0);
            self.triangle([center, a, b], Vec3::Z, mat, center, control)?;
            self.quad(
                [
                    a,
                    p(i as f32, -r.depth as f32),
                    p(i as f32 + 1.0, -r.depth as f32),
                    b,
                ],
                &side,
                center,
                control,
            )?;
        }
        Ok(())
    }
    pub fn build(validated: &ValidatedSnapshot) -> Result<Self, RenderError> {
        let s = validated.snapshot();
        let mut mesh = Self::default();
        for d in &s.scene.devices {
            mesh.chassis(d, &s.scene)?;
            for p in &d.panels {
                if !page_matches(&p.page, &s.state.active_page) {
                    continue;
                }
                mesh.panel(d, p, s)?;
            }
            for c in &d.controls {
                if s.control_visible(d, c) {
                    mesh.control(d, c, s)?;
                }
            }
        }
        Ok(mesh)
    }
    fn chassis(&mut self, d: &Device, scene: &Scene) -> Result<(), RenderError> {
        let c = &d.chassis;
        let mat = &scene.materials[&c.material];
        let body = Rect {
            x: 0.0,
            y: 0.0,
            z: -c.front_thickness,
            width: c.width - 0.12_f64.min(c.width * 0.05),
            height: c.height - 0.1_f64.min(c.height * 0.05),
            depth: c.depth - c.front_thickness,
        };
        self.rounded_slab(&body, d.origin, mat, None, CHASSIS_RADIUS, 0.3)?;
        if c.feet_height > 0.0 {
            for x in [-0.38, 0.38] {
                for z in [-0.15, -0.85] {
                    let foot = Rect {
                        x: x * c.width,
                        y: -c.height / 2.0 - c.feet_height / 2.0,
                        z: z * c.depth,
                        width: c.width * 0.05,
                        height: c.feet_height,
                        depth: c.depth * 0.10,
                    };
                    self.bevel(&foot, d.origin, &dark(), None)?;
                }
            }
        }
        if c.vents {
            // Real top-cover inset strips, not a front-face texture pretending to be vents.
            for side in [-1.0, 1.0] {
                for i in 0..23 {
                    let x = side * c.width * 0.22 + (i as f64 - 11.0) * c.width * 0.0102;
                    let y = c.height / 2.0 - 0.049;
                    let z = -c.depth * 0.58;
                    let o = Vec3::from_array(d.origin.map(|v| v as f32));
                    let p = |x: f64, z: f64| o + Vec3::new(x as f32, y as f32, z as f32);
                    self.quad(
                        [
                            p(x - 0.07, z - c.depth * 0.19),
                            p(x - 0.07, z + c.depth * 0.19),
                            p(x + 0.07, z + c.depth * 0.19),
                            p(x + 0.07, z - c.depth * 0.19),
                        ],
                        &dark(),
                        o,
                        None,
                    )?;
                }
            }
        }
        Ok(())
    }
    fn panel(&mut self, d: &Device, p: &Panel, s: &Snapshot) -> Result<(), RenderError> {
        // Plates and modules are rounded slabs; keys and displays sit on them,
        // displays inside a raised bezel, so depth reads from real geometry and shadow.
        let (radius, edge) = match p.surface {
            Surface::Faceplate => (CHASSIS_RADIUS, 0.35),
            Surface::Module => (0.6, 0.12),
            Surface::Display => (0.25, 0.05),
        };
        self.rounded_slab(
            &p.rect,
            d.origin,
            &s.scene.materials[&p.material],
            None,
            radius,
            edge,
        )?;
        if !p.label.is_empty() {
            self.labels.push(Label {
                text: p.label.clone(),
                center: DVec3::new(
                    d.origin[0] + p.rect.x,
                    d.origin[1] + p.rect.y + p.rect.height / 2.0 - 0.35,
                    d.origin[2] + p.rect.z + 0.01,
                ),
                width: p.rect.width * 0.7,
                height: 0.5,
                color: color_bytes(&s.scene.materials[&p.material].ink),
                printed: true,
                raster: TextRaster::Native,
                align: TextAlign::Center,
                control_id: None,
            });
        }
        Ok(())
    }
    /// A printed, non-interactive legend: ink on the plate, with no key, well or aperture.
    fn legend(&mut self, d: &Device, c: &Control, s: &Snapshot) {
        let state = &s.state.controls[&c.id];
        let r = &c.rect;
        self.labels.push(Label {
            text: state.value.string().unwrap_or(&c.label).to_owned(),
            center: DVec3::new(
                r.x + d.origin[0],
                r.y + d.origin[1],
                r.z + d.origin[2] + 0.01,
            ),
            width: r.width,
            height: r.height * 0.65,
            color: color_bytes(&s.scene.materials[&c.material].ink),
            printed: true,
            raster: TextRaster::Native,
            align: TextAlign::Start,
            control_id: Some(c.id.clone()),
        });
    }
    fn fader(&mut self, d: &Device, c: &Control, s: &Snapshot) -> Result<Rect, RenderError> {
        let mut r = c.rect.clone();
        let state = &s.state.controls[&c.id];
        let mat = &s.scene.materials[&c.material];
        let control = Some(c.id.as_str());
        let horizontal = r.width >= r.height;
        let slot = Rect {
            width: if horizontal { r.width } else { r.width * 0.12 },
            height: if horizontal {
                r.height * 0.12
            } else {
                r.height
            },
            depth: 0.12,
            ..r.clone()
        };
        self.cuboid(&slot, d.origin, &dark(), control)?;
        r = fader_thumb(&r, fraction(c, state));
        self.bevel(&r, d.origin, &trim(), control)?;
        let grip = Rect {
            height: if horizontal { r.height * 0.7 } else { 0.035 },
            width: if horizontal { 0.035 } else { r.width * 0.7 },
            z: r.z + 0.005,
            depth: 0.01,
            ..r.clone()
        };
        self.cuboid(
            &grip,
            d.origin,
            &Material {
                base: mat.ink.clone(),
                ..dark()
            },
            control,
        )?;
        Ok(r)
    }
    fn control(&mut self, d: &Device, c: &Control, s: &Snapshot) -> Result<(), RenderError> {
        if is_legend(c) {
            self.legend(d, c, s);
            return Ok(());
        }
        let state = &s.state.controls[&c.id];
        let mat = &s.scene.materials[&c.material];
        let mut r = c.rect.clone();
        let control = Some(c.id.as_str());
        let latched = c.kind == Kind::Toggle && state.value.boolean() == Some(true);
        let well = Rect {
            width: r.width + 0.16,
            height: r.height + 0.16,
            z: r.z - r.depth - 0.02,
            depth: 0.08,
            ..r.clone()
        };
        self.cuboid(&well, d.origin, &dark(), control)?;
        match c.kind {
            Kind::Knob => {
                self.cylinder(
                    &Rect {
                        width: r.width + 0.20,
                        height: r.height + 0.20,
                        z: r.z - r.depth,
                        depth: 0.08,
                        ..r.clone()
                    },
                    d.origin,
                    &trim(),
                    control,
                )?;
                self.cylinder(&r, d.origin, mat, control)?;
                let a = (-225.0 + fraction(c, state) * 270.0).to_radians();
                let dot = Rect {
                    x: r.x + a.cos() * r.width * 0.34,
                    y: r.y + a.sin() * r.height * 0.34,
                    z: r.z + 0.012,
                    width: 0.13,
                    height: 0.13,
                    depth: 0.01,
                };
                self.cylinder(
                    &dot,
                    d.origin,
                    &Material {
                        base: mat.ink.clone(),
                        ..dark()
                    },
                    control,
                )?;
            }
            Kind::Fader => r = self.fader(d, c, s)?,
            Kind::Button | Kind::Toggle => {
                if state.pressed {
                    r.z -= r.depth.min(0.12) * 0.65;
                } else if latched {
                    // A latched key rests part-way down, like a hi-fi source selector.
                    r.z -= r.depth.min(0.12) * 0.35;
                }
                self.bevel(&r, d.origin, mat, control)?;
                if c.kind == Kind::Toggle {
                    self.lamp(&r, d.origin, latched, control)?;
                }
            }
            Kind::Indicator => {
                let m = if state.value.boolean() == Some(true) {
                    Material {
                        base: mat.emissive.clone(),
                        ..mat.clone()
                    }
                } else {
                    dark()
                };
                self.cylinder(&r, d.origin, &m, control)?;
            }
            Kind::Readout | Kind::Select | Kind::Shortcut | Kind::Text => {
                self.rounded_slab(&r, d.origin, mat, control, 0.1, 0.03)?;
                self.bezel(&r, d.origin, control)?;
            }
            _ => {
                self.bevel(&r, d.origin, mat, control)?;
            }
        }
        if c.kind == Kind::Meter {
            let lit = (fraction(c, state) * 24.0).floor() as usize;
            for i in 0..24 {
                let segment = Rect {
                    x: r.x - r.width * 0.45 + (i as f64 + 0.5) * r.width * 0.9 / 24.0,
                    y: r.y,
                    z: r.z + 0.005,
                    width: r.width * 0.9 / 30.0,
                    height: r.height * 0.45,
                    depth: 0.001,
                };
                let m = if i < lit {
                    Material {
                        base: mat.ink.clone(),
                        emissive: mat.emissive.clone(),
                        ..dark()
                    }
                } else {
                    Material {
                        base: "#202c29".into(),
                        ..dark()
                    }
                };
                self.cuboid(&segment, d.origin, &m, control)?;
            }
        }
        if state.focused {
            let ring = Rect {
                width: r.width + 0.10,
                height: r.height + 0.10,
                z: r.z + 0.02,
                depth: 0.01,
                ..r.clone()
            };
            for bar in subtract(&ring, &r) {
                self.cuboid(
                    &bar,
                    d.origin,
                    &Material {
                        base: "#ff9e52".into(),
                        emissive: "#ff9e52".into(),
                        ..dark()
                    },
                    control,
                )?;
            }
        }
        if c.presentation != Presentation::None {
            let text = match c.kind {
                Kind::Button | Kind::Toggle => c.label.clone(),
                Kind::Select => c
                    .options
                    .as_ref()
                    .and_then(|opts| {
                        opts.iter()
                            .find(|o| Some(o.id.as_str()) == state.value.string())
                    })
                    .map(|o| o.label.clone())
                    .unwrap_or_default(),
                _ => state
                    .value
                    .string()
                    .map(str::to_owned)
                    .unwrap_or_else(|| state.value.0.to_string()),
            };
            let raster = if c.presentation == Presentation::Matrix {
                self.matrix_text(
                    &text,
                    c.matrix.as_ref().expect("validated matrix"),
                    &r,
                    d.origin,
                    mat,
                )?
            } else {
                TextRaster::Native
            };
            // Toggle legends sit right of the lamp; everything else centres on the key.
            let lamp_offset = if c.kind == Kind::Toggle {
                LAMP_INSET
            } else {
                0.0
            };
            self.labels.push(Label {
                text,
                center: DVec3::new(
                    r.x + lamp_offset / 2.0 + d.origin[0],
                    r.y + d.origin[1],
                    r.z + d.origin[2] + 0.025,
                ),
                width: r.width * 0.90 - lamp_offset,
                height: r.height * 0.65,
                // Disabled keys keep their place in the grid but read as unavailable.
                color: {
                    let ink = color_bytes(&mat.ink);
                    if state.enabled
                        || matches!(c.kind, Kind::Readout | Kind::Meter | Kind::Indicator)
                    {
                        ink
                    } else {
                        ink.map(|v| (v as f32 * 0.38) as u8)
                    }
                },
                printed: c.presentation == Presentation::Printed,
                raster,
                align: TextAlign::Center,
                control_id: Some(c.id.clone()),
            });
        }
        Ok(())
    }
}
const LAMP_INSET: f64 = 0.9;
const CHASSIS_RADIUS: f64 = 1.4;
const BEZEL: f64 = 0.14;
impl Mesh {
    /// A satin frame standing slightly proud of a display, so the glass reads as recessed.
    fn bezel(
        &mut self,
        glass: &Rect,
        origin: [f64; 3],
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        let outer = Rect {
            width: glass.width + 2.0 * BEZEL,
            height: glass.height + 2.0 * BEZEL,
            z: glass.z + 0.05,
            depth: glass.depth + 0.05,
            ..glass.clone()
        };
        let satin = Material {
            base: "#16181b".into(),
            metalness: 0.55,
            roughness: 0.42,
            ..dark()
        };
        for bar in subtract(&outer, glass) {
            self.rounded_slab(&bar, origin, &satin, control, 0.05, 0.03)?;
        }
        Ok(())
    }
    /// Indicator lamp on a latching key: lit accent when on, dark glass when off.
    fn lamp(
        &mut self,
        key: &Rect,
        origin: [f64; 3],
        lit: bool,
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        let lamp = Rect {
            x: key.x - key.width / 2.0 + LAMP_INSET / 2.0 + 0.1,
            y: key.y,
            z: key.z + 0.012,
            width: 0.28,
            height: 0.28,
            depth: 0.01,
        };
        let colour = if lit { "#ff8a3d" } else { "#2a1f19" };
        let mat = Material {
            base: colour.into(),
            emissive: colour.into(),
            finish: Finish::Glass,
            ..dark()
        };
        self.cylinder(&lamp, origin, &mat, control)
    }
}
fn is_legend(c: &Control) -> bool {
    c.kind == Kind::Readout && c.presentation == Presentation::Printed
}
fn fraction(c: &Control, s: &ControlState) -> f64 {
    c.range
        .as_ref()
        .zip(s.value.number())
        .map(|(r, v)| ((v - r.min) / (r.max - r.min)).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}
fn subtract(r: &Rect, h: &Rect) -> Vec<Rect> {
    let (l, rr, b, t) = (
        r.x - r.width / 2.0,
        r.x + r.width / 2.0,
        r.y - r.height / 2.0,
        r.y + r.height / 2.0,
    );
    let (hl, hr, hb, ht) = (
        (h.x - h.width / 2.0).max(l),
        (h.x + h.width / 2.0).min(rr),
        (h.y - h.height / 2.0).max(b),
        (h.y + h.height / 2.0).min(t),
    );
    if hl >= hr || hb >= ht {
        return vec![r.clone()];
    }
    [
        (l, rr, b, hb),
        (l, rr, ht, t),
        (l, hl, hb, ht),
        (hr, rr, hb, ht),
    ]
    .into_iter()
    .filter(|(l, rgt, b, t)| rgt - l > 1e-8 && t - b > 1e-8)
    .map(|(l, rgt, b, t)| Rect {
        x: (l + rgt) / 2.0,
        y: (b + t) / 2.0,
        width: rgt - l,
        height: t - b,
        ..r.clone()
    })
    .collect()
}
