//! Native-vector tessellation of clipped, grouped Oqtopus frame layers.
//!
//! Curves are flattened in vector coordinates, boolean operations preserve
//! even-odd holes, and the stripe is intersected with the body. Composited
//! regions are disjoint so their opacity applies once to the whole frame.
use geo::{BooleanOps, Coord, LineString, MultiPolygon, Polygon};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use oqtopus_core::{FramePaths, PathCommand};
use std::fmt;

#[derive(Debug)]
pub struct RenderError(pub String);
impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for RenderError {}
fn error(s: &str) -> RenderError {
    RenderError(s.into())
}

#[derive(Debug, Clone)]
pub struct PreparedFrame {
    body: MultiPolygon<f64>,
    body_only: MultiPolygon<f64>,
    stripe: MultiPolygon<f64>,
}
#[derive(Debug, Clone, Copy)]
pub enum Composition {
    Single,
    Crossfade(f64),
    Over(f64),
}
/// `stripe_mix` is 0 for body color, 1 for stripe color, and intermediate
/// for the corresponding premultiplied crossfade. Vertices form triangles.
#[derive(Debug, Clone)]
pub struct Region {
    pub triangles: Vec<[f32; 2]>,
    pub stripe_mix: f32,
    pub opacity: f32,
}

fn flatten_quad(
    a: (f64, f64),
    c: (f64, f64),
    b: (f64, f64),
    points: &mut Vec<Coord<f64>>,
    depth: u8,
) {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let cross = ((c.0 - a.0) * dy - (c.1 - a.1) * dx).abs();
    let len = (dx * dx + dy * dy).sqrt();
    let projection = (c.0 - a.0) * dx + (c.1 - a.1) * dy;
    if depth >= 12
        || (len > 0.0 && cross / len <= 0.15 && projection >= 0.0 && projection <= len * len)
        || (len == 0.0 && (c.0 - a.0).hypot(c.1 - a.1) <= 0.15)
    {
        points.push(Coord { x: b.0, y: b.1 });
        return;
    }
    let ac = ((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0);
    let cb = ((c.0 + b.0) / 2.0, (c.1 + b.1) / 2.0);
    let mid = ((ac.0 + cb.0) / 2.0, (ac.1 + cb.1) / 2.0);
    flatten_quad(a, ac, mid, points, depth + 1);
    flatten_quad(mid, cb, b, points, depth + 1);
}
fn loops_to_area(loops: &[Vec<PathCommand>]) -> Result<MultiPolygon<f64>, RenderError> {
    let mut area = MultiPolygon(vec![]);
    for commands in loops {
        let mut points = Vec::new();
        let mut current = None;
        let mut closed = false;
        for command in commands {
            match *command {
                PathCommand::Move(x, y) if current.is_none() => {
                    if !x.is_finite() || !y.is_finite() {
                        return Err(error("nonfinite coordinate"));
                    }
                    points.push(Coord { x, y });
                    current = Some((x, y));
                }
                PathCommand::Line(x, y) if current.is_some() && !closed => {
                    if !x.is_finite() || !y.is_finite() {
                        return Err(error("nonfinite coordinate"));
                    }
                    points.push(Coord { x, y });
                    current = Some((x, y));
                }
                PathCommand::Quad { control, end } if current.is_some() && !closed => {
                    if ![control.0, control.1, end.0, end.1]
                        .iter()
                        .all(|v| v.is_finite())
                    {
                        return Err(error("nonfinite coordinate"));
                    }
                    flatten_quad(current.unwrap(), control, end, &mut points, 0);
                    current = Some(end);
                }
                PathCommand::Close if current.is_some() && !closed => closed = true,
                _ => return Err(error("invalid path commands")),
            }
        }
        if !closed || points.len() < 3 {
            return Err(error("unclosed or degenerate path"));
        }
        if points.last() != points.first() {
            points.push(points[0]);
        }
        let poly = MultiPolygon(vec![Polygon::new(LineString(points), vec![])]);
        area = area.xor(&poly);
    }
    Ok(area)
}
/// Prepare vector geometry once per source frame, without choosing colors.
pub fn prepare_frame(paths: &FramePaths) -> Result<PreparedFrame, RenderError> {
    let body = loops_to_area(&paths.body)?;
    let stripe = body.intersection(&loops_to_area(&paths.stripe)?);
    let body_only = body.difference(&stripe);
    Ok(PreparedFrame {
        body,
        body_only,
        stripe,
    })
}
fn triangulate(area: &MultiPolygon<f64>) -> Result<Vec<[f32; 2]>, RenderError> {
    if area.0.is_empty() {
        return Ok(vec![]);
    }
    let mut path = Path::builder();
    for poly in &area.0 {
        for ring in std::iter::once(poly.exterior()).chain(poly.interiors().iter()) {
            let coords = &ring.0;
            if coords.len() < 4 {
                continue;
            }
            path.begin(point(coords[0].x as f32, coords[0].y as f32));
            for c in &coords[1..coords.len() - 1] {
                path.line_to(point(c.x as f32, c.y as f32));
            }
            path.end(true);
        }
    }
    let path = path.build();
    let mut mesh: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    FillTessellator::new()
        .tessellate_path(
            &path,
            &FillOptions::default().with_fill_rule(FillRule::EvenOdd),
            &mut BuffersBuilder::new(&mut mesh, |v: FillVertex| [v.position().x, v.position().y]),
        )
        .map_err(|e| RenderError(format!("tessellation: {e:?}")))?;
    Ok(mesh
        .indices
        .iter()
        .map(|i| mesh.vertices[*i as usize])
        .collect())
}
fn push(
    out: &mut Vec<Region>,
    area: MultiPolygon<f64>,
    mix: f64,
    opacity: f64,
) -> Result<(), RenderError> {
    if opacity <= 0.0 || area.0.is_empty() {
        return Ok(());
    }
    let triangles = triangulate(&area)?;
    if !triangles.is_empty() {
        out.push(Region {
            triangles,
            stripe_mix: mix as f32,
            opacity: opacity as f32,
        });
    }
    Ok(())
}
/// Produce disjoint triangle regions for one frame, a true crossfade, or a
/// full-opacity base with the second frame painted over it at the given alpha.
pub fn compose(
    a: &PreparedFrame,
    b: Option<&PreparedFrame>,
    mode: Composition,
) -> Result<Vec<Region>, RenderError> {
    let (weight, alpha_a, alpha_b) = match mode {
        Composition::Single => {
            if b.is_some() {
                return Err(error("unexpected second frame"));
            }
            (0., 1., 0.)
        }
        Composition::Crossfade(w) => {
            if !w.is_finite() || !(0.0..=1.0).contains(&w) || b.is_none() {
                return Err(error("invalid crossfade"));
            }
            (w, 1. - w, w)
        }
        Composition::Over(w) => {
            if !w.is_finite() || !(0.0..=1.0).contains(&w) || b.is_none() {
                return Err(error("invalid overlay"));
            }
            (w, 1., w)
        }
    };
    let mut out = Vec::new();
    let Some(b) = b else {
        push(&mut out, a.body_only.clone(), 0., 1.)?;
        push(&mut out, a.stripe.clone(), 1., 1.)?;
        return Ok(out);
    };
    let a_parts = [(&a.body_only, 0.), (&a.stripe, 1.)];
    let b_parts = [(&b.body_only, 0.), (&b.stripe, 1.)];
    for (area, color) in a_parts {
        push(&mut out, area.difference(&b.body), color, alpha_a)?;
    }
    for (area, color) in b_parts {
        push(&mut out, area.difference(&a.body), color, alpha_b)?;
    }
    for (left, lc) in a_parts {
        for (right, rc) in b_parts {
            push(
                &mut out,
                left.intersection(right),
                lc * (1. - weight) + rc * weight,
                1.,
            )?;
        }
    }
    Ok(out)
}
