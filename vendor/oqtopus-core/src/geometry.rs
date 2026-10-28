//! Renderer-independent vector paths matching the reference Canvas2D player.
use crate::{Frame, PackError};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq)]
pub enum PathCommand {
    Move(f64, f64),
    Line(f64, f64),
    Quad {
        control: (f64, f64),
        end: (f64, f64),
    },
    Close,
}

/// Separate even-odd paths. Renderers must clip stripe to body and apply
/// opacity to the composed body-and-stripe layer, not the individual fills.
#[derive(Debug, Clone, PartialEq)]
pub struct FramePaths {
    pub body: Vec<Vec<PathCommand>>,
    pub stripe: Vec<Vec<PathCommand>>,
}

pub(crate) fn decode_points(encoded: &[i32]) -> Result<Vec<(f64, f64)>, PackError> {
    if encoded.len() < 6 || encoded.len() % 2 != 0 {
        return Err(PackError("invalid vector loop".into()));
    }
    let (mut x, mut y) = (0i64, 0i64);
    Ok(encoded
        .chunks_exact(2)
        .map(|p| {
            x += i64::from(p[0]);
            y += i64::from(p[1]);
            (x as f64 / 10.0, y as f64 / 10.0)
        })
        .collect())
}

fn paths(loops: &[Vec<i32>], poly: bool) -> Result<Vec<Vec<PathCommand>>, PackError> {
    loops
        .iter()
        .map(|encoded| {
            let points = decode_points(encoded)?;
            let mut path = Vec::with_capacity(points.len() + 2);
            if poly {
                path.push(PathCommand::Move(points[0].0, points[0].1));
                path.extend(points[1..].iter().map(|&(x, y)| PathCommand::Line(x, y)));
            } else {
                let midpoint =
                    |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
                let start = midpoint(points[points.len() - 1], points[0]);
                path.push(PathCommand::Move(start.0, start.1));
                for i in 0..points.len() {
                    path.push(PathCommand::Quad {
                        control: points[i],
                        end: midpoint(points[i], points[(i + 1) % points.len()]),
                    });
                }
            }
            path.push(PathCommand::Close);
            Ok(path)
        })
        .collect()
}

/// Append canonical loop counts and deltas to a geometry digest.
/// Metadata is intentionally excluded; callers include any relevant flags.
pub fn update_geometry_digest(hash: &mut Sha256, frame: &Frame) {
    for loops in [&frame.body, &frame.stripe] {
        hash.update((loops.len() as u32).to_le_bytes());
        for points in loops {
            hash.update((points.len() as u32).to_le_bytes());
            for point in points {
                hash.update(point.to_le_bytes());
            }
        }
    }
}

/// Decode packed delta coordinates into closed paths in vector-canvas units.
pub fn decode_frame(frame: &Frame) -> Result<FramePaths, PackError> {
    let poly = frame
        .metadata
        .get("poly")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(FramePaths {
        body: paths(&frame.body, poly)?,
        stripe: paths(&frame.stripe, poly)?,
    })
}
