//! Shared native vector compositing for procedural and authored motion.
use motion_core::MotionFrame;
use oqtopus_core::{decode_frame, update_geometry_digest, Frame, PlayerSample};
pub use oqtopus_render::Region;
use oqtopus_render::{
    compose, prepare_frame, prepare_solid_union, Composition, PreparedFrame, RenderError,
};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct RenderFrame {
    pub view_box: [f32; 2],
    pub regions: Vec<Region>,
}
#[derive(Default)]
pub struct MotionRenderer {
    cache: HashMap<[u8; 32], PreparedFrame>,
    order: VecDeque<[u8; 32]>,
}
fn geometry_key(frame: &Frame) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update([u8::from(
        frame
            .metadata
            .get("poly")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    )]);
    update_geometry_digest(&mut h, frame);
    h.finalize().into()
}
impl MotionRenderer {
    fn frame(&mut self, frame: &Frame) -> Result<PreparedFrame, RenderError> {
        let key = geometry_key(frame);
        if let Some(value) = self.cache.get(&key) {
            return Ok(value.clone());
        }
        let result = prepare_frame(&decode_frame(frame).map_err(|e| RenderError(e.to_string()))?)?;
        if self.order.len() >= 96 {
            if let Some(old) = self.order.pop_front() {
                self.cache.remove(&old);
            }
        }
        self.order.push_back(key);
        self.cache.insert(key, result.clone());
        Ok(result)
    }
    /// Produce theme-free vector triangles. Colors are applied by the host.
    pub fn render(&mut self, frame: MotionFrame<'_>) -> Result<RenderFrame, RenderError> {
        let (view_box, regions) = match frame {
            MotionFrame::Paths { view_box, paths } => {
                let shape = prepare_solid_union(&paths)?;
                (view_box, compose(&shape, None, Composition::Single)?)
            }
            MotionFrame::Vector {
                view_box,
                sample,
                rest,
            } => {
                let parts = match sample {
                    PlayerSample::Rest(_) => {
                        let shape = prepare_frame(
                            &rest.ok_or_else(|| RenderError("missing rest geometry".into()))?,
                        )?;
                        compose(&shape, None, Composition::Single)?
                    }
                    PlayerSample::Frame {
                        frame, rest_weight, ..
                    } if rest_weight <= 0.0 => {
                        let shape = self.frame(frame)?;
                        compose(&shape, None, Composition::Single)?
                    }
                    PlayerSample::Frame {
                        frame, rest_weight, ..
                    } => {
                        let base = prepare_frame(
                            &rest.ok_or_else(|| RenderError("missing rest geometry".into()))?,
                        )?;
                        if rest_weight >= 1.0 {
                            compose(&base, None, Composition::Single)?
                        } else {
                            let current = self.frame(frame)?;
                            compose(&base, Some(&current), Composition::Over(1.0 - rest_weight))?
                        }
                    }
                    PlayerSample::Crossfade { from, to, weight } => {
                        let a = self.frame(from)?;
                        let b = self.frame(to)?;
                        compose(&a, Some(&b), Composition::Crossfade(weight))?
                    }
                };
                (view_box, parts)
            }
        };
        Ok(RenderFrame { view_box, regions })
    }
}
