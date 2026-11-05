use crate::*;
use glam::{DMat4, DVec3, DVec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Extent {
    pub width_px: u32,
    pub height_px: u32,
    pub effective_scale: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}
impl Viewport {
    /// Uniformly reduce the requested pixel dimensions; rounding is explicit in scale_x/y.
    pub fn extent(&self) -> Result<Extent, ValidationError> {
        self.validate()?;
        let w = (self.width_dip * self.scale_factor).ceil().max(1.0);
        let h = (self.height_dip * self.scale_factor).ceil().max(1.0);
        let mut factor = (self.pixel_cap as f64 / (w * h)).sqrt().min(1.0);
        let dimensions = |s: f64| {
            (
                (w * s).floor().max(1.0) as u32,
                (h * s).floor().max(1.0) as u32,
            )
        };
        let (mut width_px, mut height_px) = dimensions(factor);
        // Degenerate aspect ratios can force a one-pixel minimum over budget.
        if u64::from(width_px) * u64::from(height_px) > u64::from(self.pixel_cap) {
            factor = (self.pixel_cap as f64 / w.max(h)).min(factor);
            (width_px, height_px) = dimensions(factor);
        }
        Ok(Extent {
            width_px,
            height_px,
            effective_scale: self.scale_factor * factor,
            scale_x: width_px as f64 / self.width_dip,
            scale_y: height_px as f64 / self.height_dip,
        })
    }
}
impl Rect {
    pub fn corners(&self, origin: [f64; 3]) -> [DVec3; 8] {
        std::array::from_fn(|i| {
            DVec3::new(
                self.x
                    + origin[0]
                    + if i & 1 == 0 {
                        -self.width / 2.0
                    } else {
                        self.width / 2.0
                    },
                self.y
                    + origin[1]
                    + if i & 2 == 0 {
                        -self.height / 2.0
                    } else {
                        self.height / 2.0
                    },
                self.z + origin[2] - if i & 4 == 0 { self.depth } else { 0.0 },
            )
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Projection {
    /// Column-major, right-handed world to WebGPU clip coordinates (Z in 0..1).
    pub view_projection: DMat4,
    pub inverse_view_projection: DMat4,
    pub eye: DVec3,
    pub near: f64,
    pub far: f64,
    pub viewport: Viewport,
    pub extent: Extent,
}
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: DVec3,
    pub direction: DVec3,
}

impl Projection {
    pub fn new(snapshot: &ValidatedSnapshot) -> Result<Self, ValidationError> {
        let s = snapshot.snapshot();
        let c = &s.scene.camera;
        let yaw = c.yaw_deg.to_radians();
        let pitch = c.pitch_deg.to_radians();
        let target = DVec3::from_array(c.target);
        let eye = target
            + DVec3::new(
                yaw.sin() * pitch.cos(),
                pitch.sin(),
                yaw.cos() * pitch.cos(),
            ) * c.distance;
        let view = glam::dcamera::rh::view::look_at_mat4(eye, target, DVec3::Y);
        let mut nearest = f64::INFINITY;
        let mut furthest: f64 = 0.0;
        let mut clipped = false;
        let mut inspect = |point: DVec3| {
            let depth = -view.transform_point3(point).z;
            if !depth.is_finite() || depth <= 0.00001 {
                clipped = true;
            }
            if depth > 0.0 {
                nearest = nearest.min(depth);
            }
            furthest = furthest.max(depth);
        };
        for d in &s.scene.devices {
            let chassis = Rect {
                x: 0.0,
                y: -d.chassis.feet_height / 2.0,
                z: 0.0,
                width: d.chassis.width,
                height: d.chassis.height + d.chassis.feet_height,
                depth: d.chassis.depth,
            };
            for point in chassis.corners(d.origin) {
                inspect(point);
            }
            for r in d
                .panels
                .iter()
                .map(|p| &p.rect)
                .chain(d.controls.iter().map(|c| &c.rect))
            {
                for point in r.corners(d.origin) {
                    inspect(point);
                }
            }
        }
        if clipped {
            return Err(ValidationError {
                path: "camera".into(),
                message: "scene bounds intersect or lie behind the camera plane".into(),
            });
        }
        let near = if nearest.is_finite() {
            (nearest * 0.25).max(0.00001)
        } else {
            0.01
        };
        let far = (furthest * 1.25 + 1.0).max(near + 1.0);
        let extent = s.viewport.extent()?;
        let perspective = glam::dcamera::rh::proj::directx::perspective(
            c.fov_y_deg.to_radians(),
            extent.width_px as f64 / extent.height_px as f64,
            near,
            far,
        );
        let view_projection = perspective * view;
        let inverse_view_projection = view_projection.inverse();
        if !view_projection.is_finite() || !inverse_view_projection.is_finite() {
            return Err(ValidationError {
                path: "camera".into(),
                message: "camera projection is not representable".into(),
            });
        }
        Ok(Self {
            inverse_view_projection,
            view_projection,
            eye,
            near,
            far,
            viewport: s.viewport.clone(),
            extent,
        })
    }
    /// DIP coordinates, top-left origin; None means clipped or behind the camera.
    pub fn project(&self, world: DVec3) -> Option<[f64; 3]> {
        let p = self.view_projection * world.extend(1.0);
        if p.w <= 0.0 {
            return None;
        }
        let n = p.truncate() / p.w;
        if n.z < 0.0 || n.z > 1.0 {
            return None;
        }
        Some([
            (n.x + 1.0) * 0.5 * self.viewport.width_dip,
            (1.0 - n.y) * 0.5 * self.viewport.height_dip,
            n.z,
        ])
    }
    pub fn ray(&self, x_dip: f64, y_dip: f64) -> Option<Ray> {
        if !x_dip.is_finite()
            || !y_dip.is_finite()
            || x_dip < 0.0
            || y_dip < 0.0
            || x_dip > self.viewport.width_dip
            || y_dip > self.viewport.height_dip
        {
            return None;
        }
        let x = 2.0 * x_dip / self.viewport.width_dip - 1.0;
        let y = 1.0 - 2.0 * y_dip / self.viewport.height_dip;
        let p = self.inverse_view_projection * DVec4::new(x, y, 0.0, 1.0);
        let near = p.truncate() / p.w;
        Some(Ray {
            origin: near,
            direction: (near - self.eye).normalize(),
        })
    }
}
impl Ray {
    /// Double-sided Möller–Trumbore intersection, used on the actual renderer triangles.
    pub fn triangle(&self, vertices: [DVec3; 3]) -> Option<f64> {
        let [a, b, c] = vertices;
        let e1 = b - a;
        let e2 = c - a;
        let p = self.direction.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-10 {
            return None;
        }
        let t = self.origin - a;
        let u = t.dot(p) / det;
        let q = t.cross(e1);
        let v = self.direction.dot(q) / det;
        let distance = e2.dot(q) / det;
        (u >= -1e-8 && v >= -1e-8 && u + v <= 1.0 + 1e-8 && distance >= 0.0).then_some(distance)
    }
}
