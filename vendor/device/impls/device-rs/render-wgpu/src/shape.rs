//! Rounded, edge-radiused slabs with smooth normals: chassis, plates, modules and keys.
//! Corners are quarter circles in plan; the front edge is a quarter round in
//! section, so edges catch the light instead of reading as flat cut-outs.
use super::*;

const CORNER_SEGMENTS: usize = 8;
const EDGE_RINGS: usize = 4;

/// One outline point: position in the plane and its outward unit direction.
fn outline(half: Vec2, radius: f32, inset: f32) -> Vec<(Vec2, Vec2)> {
    let r = radius.max(0.0);
    let corners = [
        (Vec2::new(half.x - r, half.y - r), 0.0_f32),
        (Vec2::new(-(half.x - r), half.y - r), 0.5),
        (Vec2::new(-(half.x - r), -(half.y - r)), 1.0),
        (Vec2::new(half.x - r, -(half.y - r)), 1.5),
    ];
    let mut points = Vec::with_capacity(4 * (CORNER_SEGMENTS + 1));
    for (centre, start) in corners {
        for i in 0..=CORNER_SEGMENTS {
            let a = (start + 0.5 * i as f32 / CORNER_SEGMENTS as f32) * std::f32::consts::PI;
            let dir = Vec2::new(a.cos(), a.sin());
            points.push((centre + dir * (r - inset).max(0.0), dir));
        }
    }
    points
}

impl Mesh {
    fn smooth(
        &mut self,
        p: [Vec3; 3],
        n: [Vec3; 3],
        mat: &Material,
        centre: Vec3,
        control: Option<&str>,
    ) -> Result<(), RenderError> {
        if p.iter().chain(n.iter()).any(|v| !v.is_finite()) {
            return Err(RenderError::Budget(
                "geometry is not representable at GPU precision",
            ));
        }
        if self.vertices.len() + 3 > MAX_VERTICES {
            return Err(RenderError::Budget("mesh vertex budget exceeded"));
        }
        let start = self.vertices.len();
        self.triangle(p, n[0], mat, centre, control)?;
        for (v, normal) in self.vertices[start..].iter_mut().zip(n) {
            v.normal = normal.normalize_or_zero().to_array();
        }
        Ok(())
    }
    /// A slab whose front face is a rounded rectangle with a rounded front edge.
    pub(super) fn rounded_slab(
        &mut self,
        r: &Rect,
        origin: [f64; 3],
        mat: &Material,
        control: Option<&str>,
        radius: f64,
        edge: f64,
    ) -> Result<(), RenderError> {
        let half = Vec2::new(r.width as f32 / 2.0, r.height as f32 / 2.0);
        let radius = (radius as f32).min(half.x).min(half.y);
        let edge = (edge as f32)
            .min(radius.max(0.001))
            .min(r.depth as f32)
            .max(0.0);
        let o = Vec3::new(
            (r.x + origin[0]) as f32,
            (r.y + origin[1]) as f32,
            (r.z + origin[2]) as f32,
        );
        let front = o.z;
        let back = front - r.depth as f32;
        let at = |p: Vec2, z: f32| Vec3::new(o.x + p.x, o.y + p.y, z);
        // Rings from the side wall (phi = 0) to the front face (phi = 90 degrees).
        let rings: Vec<Vec<(Vec3, Vec3)>> = (0..=EDGE_RINGS)
            .map(|k| {
                let phi = k as f32 / EDGE_RINGS as f32 * std::f32::consts::FRAC_PI_2;
                let inset = edge * (1.0 - phi.cos());
                let z = front - edge + edge * phi.sin();
                outline(half, radius, inset)
                    .into_iter()
                    .map(|(p, d)| {
                        let n = Vec3::new(d.x * phi.cos(), d.y * phi.cos(), phi.sin());
                        (at(p, z), n)
                    })
                    .collect()
            })
            .collect();
        let count = rings[0].len();
        // Front face.
        let face = &rings[EDGE_RINGS];
        for i in 0..count {
            let (a, b) = (face[i].0, face[(i + 1) % count].0);
            self.smooth([at(Vec2::ZERO, front), a, b], [Vec3::Z; 3], mat, o, control)?;
        }
        // Rounded front edge.
        for k in 0..EDGE_RINGS {
            let (lo, hi) = (&rings[k], &rings[k + 1]);
            for i in 0..count {
                let j = (i + 1) % count;
                self.smooth(
                    [lo[i].0, lo[j].0, hi[j].0],
                    [lo[i].1, lo[j].1, hi[j].1],
                    mat,
                    o,
                    control,
                )?;
                self.smooth(
                    [lo[i].0, hi[j].0, hi[i].0],
                    [lo[i].1, hi[j].1, hi[i].1],
                    mat,
                    o,
                    control,
                )?;
            }
        }
        // Side wall down to the back.
        if back < front - edge - 1e-4 {
            let side = &rings[0];
            for i in 0..count {
                let j = (i + 1) % count;
                let (a, b) = (side[i], side[j]);
                let (ab, bb) = (Vec3::new(a.0.x, a.0.y, back), Vec3::new(b.0.x, b.0.y, back));
                self.smooth([a.0, ab, bb], [a.1, a.1, b.1], mat, o, control)?;
                self.smooth([a.0, bb, b.0], [a.1, b.1, b.1], mat, o, control)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounded_slab_has_outward_smooth_normals_and_stays_inside_its_rect() {
        let mut mesh = Mesh::default();
        let r = Rect {
            x: 1.0,
            y: 2.0,
            z: 0.5,
            width: 4.0,
            height: 2.0,
            depth: 0.3,
        };
        mesh.rounded_slab(&r, [0.0; 3], &dark(), Some("k"), 0.4, 0.1)
            .unwrap();
        assert!(!mesh.vertices.is_empty());
        for v in &mesh.vertices {
            let [x, y, z] = v.position;
            assert!((x - 1.0).abs() <= 2.0 + 1e-4 && (y - 2.0).abs() <= 1.0 + 1e-4);
            assert!((0.2 - 1e-4..=0.5 + 1e-4).contains(&z));
            let n = Vec3::from_array(v.normal);
            assert!((n.length() - 1.0).abs() < 1e-3);
            // Normals point away from the slab centre (or forward on the face).
            let outward = Vec3::new(x - 1.0, y - 2.0, 0.0);
            assert!(n.z > 0.99 || n.dot(outward) > -1e-3);
        }
        assert!(
            mesh.hits
                .iter()
                .all(|h| h.control_id.as_deref() == Some("k"))
        );
    }
}
