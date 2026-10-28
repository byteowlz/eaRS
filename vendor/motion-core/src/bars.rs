//! Deterministic parameterized bar-wave geometry (no platform drawing).
use crate::{Bar, BarsWave, MotionError, Wave, WaveState};
use oqtopus_core::{FramePaths, PathCommand};
use std::f32::consts::TAU;

fn corners(bar: &Bar, cos: f32, sin: f32, travel: (f32, f32), distance: f32) -> [(f32, f32); 4] {
    let cx = bar.center[0] + travel.0 * distance;
    let cy = bar.center[1] + travel.1 * distance;
    let (w, h) = (bar.width / 2.0, bar.length / 2.0);
    [(-w, -h), (w, -h), (w, h), (-w, h)]
        .map(|(x, y)| (cx + x * cos - y * sin, cy + x * sin + y * cos))
}
fn travel_cap(config: &BarsWave, bar: &Bar, cos: f32, sin: f32, travel: (f32, f32)) -> f32 {
    let mut cap = f32::INFINITY;
    for (x, y) in corners(bar, cos, sin, travel, 0.0) {
        for (v, dir, max) in [
            (x, travel.0, config.view_box[0] - config.padding),
            (y, travel.1, config.view_box[1] - config.padding),
        ] {
            if v < config.padding || v > max {
                return 0.0;
            }
            if dir.abs() > f32::EPSILON {
                cap = cap.min(((max - v) / dir.abs()).min((v - config.padding) / dir.abs()));
            }
        }
    }
    if cap.is_finite() {
        cap.max(0.0)
    } else {
        0.0
    }
}
fn wave(config: &BarsWave, state: &WaveState, time: f32, index: usize) -> f32 {
    let period =
        state.period * (1.0 + config.detune * (2.0 * (((index * 7) % 5) as f32 / 4.0) - 1.0));
    let idx = if state.mirror {
        index.abs_diff(config.bars.len() / 2)
    } else {
        index
    } as f32;
    let phase = time / period * state.direction + idx * state.stagger;
    match state.wave {
        Wave::Sine => (TAU * phase).sin(),
        Wave::EasedTriangle => {
            let triangle = 2.0 * (2.0 * phase.rem_euclid(1.0) - 1.0).abs() - 1.0;
            let x = (triangle + 1.0) / 2.0;
            let power = 1.0 + 2.0 * state.dwell.clamp(0.0, 0.95);
            let eased = if x < 0.5 {
                (2.0 * x).powf(power) / 2.0
            } else {
                1.0 - (2.0 * (1.0 - x)).powf(power) / 2.0
            };
            eased * 2.0 - 1.0
        }
    }
}
impl BarsWave {
    /// Sample a named, data-defined state at elapsed seconds. `animate=false`
    /// returns the exact rest frame, irrespective of elapsed time.
    pub fn sample(
        &self,
        state: &str,
        elapsed: f32,
        animate: bool,
    ) -> Result<FramePaths, MotionError> {
        if !elapsed.is_finite() || elapsed < 0.0 {
            return Err(MotionError("invalid elapsed time".into()));
        }
        let state = self
            .states
            .iter()
            .find(|s| s.id == state)
            .ok_or_else(|| MotionError("unknown motion state".into()))?;
        let (sin, cos) = self.rotation_degrees.to_radians().sin_cos();
        let (ty, tx) = self.travel_degrees.to_radians().sin_cos();
        let time = elapsed * state.speed;
        if !time.is_finite() {
            return Err(MotionError("elapsed time out of range".into()));
        }
        let body = self
            .bars
            .iter()
            .enumerate()
            .map(|(i, bar)| {
                let cap = travel_cap(self, bar, cos, sin, (tx, ty));
                let d = if animate {
                    (state.amplitude * bar.length * bar.gain).min(cap) * wave(self, state, time, i)
                } else {
                    0.0
                };
                let points = corners(bar, cos, sin, (tx, ty), d);
                vec![
                    PathCommand::Move(points[0].0 as f64, points[0].1 as f64),
                    PathCommand::Line(points[1].0 as f64, points[1].1 as f64),
                    PathCommand::Line(points[2].0 as f64, points[2].1 as f64),
                    PathCommand::Line(points[3].0 as f64, points[3].1 as f64),
                    PathCommand::Close,
                ]
            })
            .collect();
        Ok(FramePaths {
            body,
            stripe: vec![],
        })
    }
}
