//! Deterministic single-clip sampler; this is not the full graph player.
use crate::{Frame, Pack, PackError, VectorDocument};

/// One complete body-and-stripe group, to be composited at this opacity.
#[derive(Debug, Clone, Copy)]
pub struct Layer<'a> {
    pub frame: &'a Frame,
    pub opacity: f64,
}

#[derive(Debug, Clone)]
pub struct Sample<'a> {
    pub frame_index: usize,
    pub layers: Vec<Layer<'a>>,
}

/// Stateless looping clip sampler for deterministic clocks and renderer tests.
/// It deliberately does not implement the player's graph, idles or state machine.
pub struct ClipSampler<'a> {
    pack: &'a Pack,
    clip: &'a VectorDocument,
    fps: f64,
}

impl<'a> ClipSampler<'a> {
    pub fn new(pack: &'a Pack, clip_id: &str) -> Result<Self, PackError> {
        let clip = pack
            .documents
            .get(&format!("clip:{clip_id}"))
            .filter(|doc| !doc.frames.is_empty())
            .ok_or_else(|| PackError("unknown or empty clip".into()))?;
        let fps = pack
            .manifest
            .get("fps")
            .and_then(|value| value.as_u64())
            .filter(|&n| n > 0)
            .ok_or_else(|| PackError("invalid fps".into()))? as f64;
        Ok(Self { pack, clip, fps })
    }

    /// Sample elapsed seconds in a cyclic clip. Reduced motion returns exact rest.
    pub fn sample(&self, seconds: f64, reduced_motion: bool) -> Result<Sample<'a>, PackError> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(PackError("invalid elapsed time".into()));
        }
        if reduced_motion {
            return Ok(Sample {
                frame_index: 0,
                layers: vec![Layer {
                    frame: &self.pack.rest,
                    opacity: 1.0,
                }],
            });
        }
        let position = seconds * self.fps;
        if !position.is_finite() {
            return Err(PackError("elapsed time out of range".into()));
        }
        let index = if self.clip.frames.len() <= 1 {
            0
        } else {
            (position % (self.clip.frames.len() - 1) as f64).round() as usize
        };
        let frame = &self.clip.frames[index];
        let weight = match frame.metadata.get("m") {
            None => 0.0,
            Some(v) => v
                .as_f64()
                .filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
                .ok_or_else(|| PackError("invalid rest blend".into()))?,
        };
        let mut layers = Vec::with_capacity(2);
        if weight > 0.0 {
            layers.push(Layer {
                frame: &self.pack.rest,
                opacity: 1.0,
            });
        }
        if weight < 1.0 {
            layers.push(Layer {
                frame,
                opacity: 1.0 - weight,
            });
        }
        Ok(Sample {
            frame_index: index,
            layers,
        })
    }
}
