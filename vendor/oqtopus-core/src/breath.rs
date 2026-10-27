//! Procedural rest-pose breathing, matching the browser player's vector warp.
use crate::geometry::decode_points;
use crate::{Frame, FramePaths, PackError, PathCommand};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct BreathParams {
    pub swell: f64,
    pub sway: f64,
    pub breath_seconds: f64,
    pub sway_seconds: f64,
}

/// Transform the exact rest outline. The renderer must still use even-odd fill
/// and clip the stripe to the transformed body.
pub fn breathing_frame(
    frame: &Frame,
    seconds: f64,
    envelope: f64,
    params: BreathParams,
) -> Result<FramePaths, PackError> {
    if !seconds.is_finite()
        || seconds < 0.0
        || !envelope.is_finite()
        || !(0.0..=1.0).contains(&envelope)
        || !params.swell.is_finite()
        || params.swell < 0.0
        || !params.sway.is_finite()
        || params.sway < 0.0
        || !params.breath_seconds.is_finite()
        || params.breath_seconds <= 0.0
        || !params.sway_seconds.is_finite()
        || params.sway_seconds <= 0.0
    {
        return Err(PackError("invalid breath parameters".into()));
    }
    // The reference stores decoded rest coordinates in Float32Array before warping.
    let points = |v: &Vec<i32>| {
        decode_points(v).map(|points| {
            points
                .into_iter()
                .map(|(x, y)| (x as f32 as f64, y as f32 as f64))
                .collect()
        })
    };
    let body: Vec<Vec<(f64, f64)>> = frame.body.iter().map(points).collect::<Result<_, _>>()?;
    let stripe: Vec<Vec<(f64, f64)>> = frame.stripe.iter().map(points).collect::<Result<_, _>>()?;
    let xs: Vec<_> = body.iter().flatten().map(|p| p.0).collect();
    let ys: Vec<_> = body.iter().flatten().map(|p| p.1).collect();
    let stripe_ys: Vec<_> = stripe.iter().flatten().map(|p| p.1).collect();
    if xs.is_empty() || stripe_ys.is_empty() {
        return Err(PackError("breath requires body and stripe geometry".into()));
    }
    let x0 = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let x1 = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let base = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tentacle_top = stripe_ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if x1 <= x0 || base <= tentacle_top {
        return Err(PackError("invalid breath geometry".into()));
    }
    let cx = (x0 + x1) / 2.0;
    let b = (2.0 * PI * seconds / params.breath_seconds).sin() * params.swell * envelope;
    let transform = |loops: Vec<Vec<(f64, f64)>>| {
        loops
            .into_iter()
            .map(|points| {
                let mut out = Vec::with_capacity(points.len() + 1);
                for (i, (x, y)) in points.into_iter().enumerate() {
                    let u = ((y - tentacle_top) / (base - tentacle_top)).clamp(0.0, 1.0);
                    let u = u * u * (3.0 - 2.0 * u);
                    let ph = 2.0 * PI * seconds / params.sway_seconds + PI * (x - cx) / (x1 - x0);
                    let a = params.sway * envelope * u * u;
                    let x = cx + (x - cx) * (1.0 - 0.5 * b) + a * ph.sin();
                    let y = base + (y - base) * (1.0 + b) + 0.35 * a * ph.cos();
                    out.push(if i == 0 {
                        PathCommand::Move(x, y)
                    } else {
                        PathCommand::Line(x, y)
                    });
                }
                out.push(PathCommand::Close);
                out
            })
            .collect()
    };
    Ok(FramePaths {
        body: transform(body),
        stripe: transform(stripe),
    })
}
