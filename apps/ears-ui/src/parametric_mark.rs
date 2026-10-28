//! eaRS state adapter for the shared data-driven bar-wave motion package.
use gpui::{AnyElement, Hsla, IntoElement};
use motion_core::MotionPackage;
use motion_render::{MotionRenderer, RenderFrame};
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Ready,
    Listening,
    Speech,
    Processing,
    Done,
    Attention,
}
impl State {
    pub fn id(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Listening => "listening",
            Self::Speech => "speech",
            Self::Processing => "processing",
            Self::Done => "done",
            Self::Attention => "attention",
        }
    }
}
fn frame(state: State, elapsed: f32, animate: bool) -> Result<RenderFrame, String> {
    static PACKAGE: OnceLock<Result<MotionPackage, String>> = OnceLock::new();
    let package = PACKAGE
        .get_or_init(|| {
            MotionPackage::from_manifest(
                include_bytes!("../../../vendor/motion-core/assets/ears-bars.json"),
                Vec::new(),
            )
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let mut player = package.player(7).map_err(|e| e.to_string())?;
    player.set_state(state.id()).map_err(|e| e.to_string())?;
    player.tick(elapsed as f64).map_err(|e| e.to_string())?;
    let sample = player.sample(!animate).map_err(|e| e.to_string())?;
    MotionRenderer::default()
        .render(sample)
        .map_err(|e| e.to_string())
}

pub fn mark(color: Hsla, size: f32, state: State, elapsed: f32, animate: bool) -> AnyElement {
    match frame(state, elapsed, animate) {
        Ok(frame) => {
            crate::motion_mark::mark(Arc::new(frame), color, color, size).into_any_element()
        }
        Err(error) => {
            eprintln!("ears-ui: procedural motion unavailable: {error}");
            crate::ears_mark::mark(color, size, 0.0, false).into_any_element()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_procedural_motion_animates_and_reduces_to_rest() {
        let animated = frame(State::Speech, 0.37, true).unwrap();
        let later = frame(State::Speech, 0.7, true).unwrap();
        assert_ne!(
            format!("{:?}", animated.regions),
            format!("{:?}", later.regions)
        );
        for state in [
            State::Ready,
            State::Listening,
            State::Speech,
            State::Processing,
            State::Done,
            State::Attention,
        ] {
            let a = frame(state, 0.0, false).unwrap();
            let b = frame(state, 4.0, false).unwrap();
            assert_eq!(format!("{:?}", a.regions), format!("{:?}", b.regions));
            assert!(!a.regions.is_empty());
        }
    }
}
