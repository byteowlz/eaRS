//! One clock/state/sample interface for procedural and authored vector motion.
use crate::{
    BarsWave, FramePaths, MotionAction, MotionError, MotionPackage, MotionSource, StateRule,
};
use oqtopus_core::{Player, PlayerSample, RestMode};

pub enum MotionFrame<'a> {
    Paths {
        view_box: [f32; 2],
        paths: FramePaths,
    },
    Vector {
        view_box: [f32; 2],
        sample: PlayerSample<'a>,
        rest: Option<FramePaths>,
    },
}
enum Engine<'a> {
    Bars(&'a BarsWave),
    Vector {
        player: Player<'a>,
        rules: &'a [StateRule],
        view_box: [f32; 2],
    },
}
pub struct MotionPlayer<'a> {
    engine: Engine<'a>,
    state: Option<String>,
    elapsed: f64,
}
impl MotionPackage {
    pub fn player(&self, seed: u32) -> Result<MotionPlayer<'_>, MotionError> {
        let engine = match self.source() {
            MotionSource::BarsWave(b) => Engine::Bars(b),
            MotionSource::VectorPack(pack) => {
                let size = pack
                    .manifest
                    .get("vector_size")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| MotionError("missing vector size".into()))?;
                let width = size
                    .first()
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| MotionError("invalid vector size".into()))?
                    as f32;
                let height = size
                    .get(1)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| MotionError("invalid vector size".into()))?
                    as f32;
                Engine::Vector {
                    player: Player::new(pack, seed).map_err(|e| MotionError(e.to_string()))?,
                    rules: self.state_rules(),
                    view_box: [width, height],
                }
            }
        };
        Ok(MotionPlayer {
            engine,
            state: None,
            elapsed: 0.0,
        })
    }
}
impl<'a> MotionPlayer<'a> {
    pub fn set_state(&mut self, state: &str) -> Result<(), MotionError> {
        if self.state.as_deref() == Some(state) {
            return Ok(());
        }
        match &mut self.engine {
            Engine::Bars(config) => {
                if !config.states.iter().any(|s| s.id == state) {
                    return Err(MotionError(format!("unknown motion state: {state}")));
                }
            }
            Engine::Vector { player, rules, .. } => {
                let rule = rules
                    .iter()
                    .find(|r| r.id == state)
                    .ok_or_else(|| MotionError(format!("unknown motion state: {state}")))?;
                player.release();
                let result = match &rule.action {
                    MotionAction::Idle => {
                        player.stop();
                        Ok(())
                    }
                    MotionAction::Play { clip } => player.play(clip),
                    MotionAction::Hold { clip } => player.hold(clip),
                };
                if let Err(error) = result {
                    player.release();
                    if rule.optional {
                        player.stop();
                    } else {
                        return Err(MotionError(error.to_string()));
                    }
                }
            }
        }
        self.state = Some(state.into());
        self.elapsed = 0.0;
        Ok(())
    }
    pub fn set_continuous_idle(&mut self, enabled: bool) {
        if let Engine::Vector { player, .. } = &mut self.engine {
            player.set_idle(true);
            player.set_rest(if enabled {
                RestMode::None
            } else {
                RestMode::Normal
            });
        }
    }
    pub fn tick(&mut self, seconds: f64) -> Result<(), MotionError> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(MotionError("invalid elapsed time".into()));
        }
        match &mut self.engine {
            Engine::Bars(_) => {
                let next = self.elapsed + seconds;
                if !next.is_finite() || next > f32::MAX as f64 {
                    return Err(MotionError("elapsed time out of range".into()));
                }
                self.elapsed = next;
            }
            Engine::Vector { player, .. } => player
                .tick(seconds)
                .map_err(|e| MotionError(e.to_string()))?,
        }
        Ok(())
    }
    pub fn sample(&self, reduced_motion: bool) -> Result<MotionFrame<'a>, MotionError> {
        match &self.engine {
            Engine::Bars(config) => {
                let state = self
                    .state
                    .as_deref()
                    .ok_or_else(|| MotionError("motion state is not selected".into()))?;
                Ok(MotionFrame::Paths {
                    view_box: config.view_box,
                    paths: config.sample(state, self.elapsed as f32, !reduced_motion)?,
                })
            }
            Engine::Vector {
                player, view_box, ..
            } => {
                let sample = player
                    .sample(reduced_motion)
                    .map_err(|e| MotionError(e.to_string()))?;
                let needs_rest = match &sample {
                    PlayerSample::Rest(_) => true,
                    PlayerSample::Frame { rest_weight, .. } => *rest_weight > 0.0,
                    PlayerSample::Crossfade { .. } => false,
                };
                let rest = if needs_rest {
                    Some(
                        player
                            .rest_paths(reduced_motion)
                            .map_err(|e| MotionError(e.to_string()))?,
                    )
                } else {
                    None
                };
                Ok(MotionFrame::Vector {
                    view_box: *view_box,
                    sample,
                    rest,
                })
            }
        }
    }
}
