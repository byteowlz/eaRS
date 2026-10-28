//! Explicit, CPU-only Silero ONNX scoring for client-owned mono PCM.
//!
//! Loading a detector does not open a microphone, download a model, record audio
//! or authenticate a speaker. Scoring and sample-clock segmentation are separate.
//! Input contract follows snakers4/silero-vad at 066409f3 (MIT): 512 samples
//! at 16 kHz, 64 samples of context, and a per-stream recurrent state.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use ort::{session::Session, value::Tensor};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_SAMPLES: usize = 512;
const CONTEXT_SAMPLES: usize = 64;
const STATE_SAMPLES: usize = 256;

/// Injectable scoring seam. Production uses Silero; deterministic tests supply
/// explicit probabilities without loading a model or opening a microphone.
pub trait SpeechScorer: Send {
    fn score(&mut self, frame: &[f32]) -> Result<f32>;
    fn reset(&mut self);
}

/// One detector per capture stream. Never share recurrent state across sessions.
pub struct SileroVad {
    session: Session,
    context: [f32; CONTEXT_SAMPLES],
    state: Vec<f32>,
}

impl SileroVad {
    /// Load explicit local artifacts. Initialize the shared ORT runtime before
    /// session creation so a missing library is a Result, not lazy-load panic.
    /// ORT's process-global runtime must be coordinated with any other consumer.
    pub fn load(model: &Path, runtime: &Path) -> Result<Self> {
        ensure!(model.is_file(), "client Silero model file unavailable");
        ort::init_from(runtime)
            .context("failed to load client ONNX Runtime library")?
            .commit();
        let session = Session::builder()?
            .with_intra_threads(1)
            .map_err(ort::Error::<()>::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::<()>::from)?
            .commit_from_file(model)
            .context("failed to load client Silero model")?;
        Ok(Self {
            session,
            context: [0.0; CONTEXT_SAMPLES],
            state: vec![0.0; STATE_SAMPLES],
        })
    }

    pub fn reset(&mut self) {
        self.context.fill(0.0);
        self.state.fill(0.0);
    }

    /// Speech probability for exactly one 32 ms frame. Invalid inputs and model
    /// failures clear stream history; they must never reuse a stale speech score.
    pub fn score(&mut self, frame: &[f32]) -> Result<f32> {
        let result = self.score_frame(frame);
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn score_frame(&mut self, frame: &[f32]) -> Result<f32> {
        ensure!(
            frame.len() == FRAME_SAMPLES,
            "client VAD requires 512 samples at 16 kHz"
        );
        ensure!(
            frame.iter().all(|v| v.is_finite()),
            "non-finite client VAD audio"
        );
        let mut input = Vec::with_capacity(CONTEXT_SAMPLES + FRAME_SAMPLES);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(frame);
        let outputs = self.session.run(ort::inputs![
            "input" => Tensor::from_array(([1, input.len()], input))?,
            "state" => Tensor::from_array(([2, 1, 128], self.state.clone()))?,
            "sr" => Tensor::from_array((Vec::<usize>::new(), vec![i64::from(SAMPLE_RATE)]))?,
        ])?;
        let (shape, scores) = outputs
            .get("output")
            .context("client VAD output missing")?
            .try_extract_tensor::<f32>()?;
        ensure!(
            &**shape == [1, 1] && scores.len() == 1,
            "invalid client VAD score shape"
        );
        let probability = scores[0];
        ensure!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "invalid client VAD probability"
        );
        let (shape, state) = outputs
            .get("stateN")
            .context("client VAD state missing")?
            .try_extract_tensor::<f32>()?;
        ensure!(
            &**shape == [2, 1, 128] && state.len() == STATE_SAMPLES,
            "invalid client VAD state shape"
        );
        ensure!(
            state.iter().all(|v| v.is_finite()),
            "non-finite client VAD state"
        );
        self.state.copy_from_slice(state);
        self.context
            .copy_from_slice(&frame[FRAME_SAMPLES - CONTEXT_SAMPLES..]);
        Ok(probability)
    }
}

impl SpeechScorer for SileroVad {
    fn score(&mut self, frame: &[f32]) -> Result<f32> {
        SileroVad::score(self, frame)
    }
    fn reset(&mut self) {
        SileroVad::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opt-in artifact test; no network, microphone or recordings are used.
    #[test]
    #[ignore = "requires EARS_TEST_VAD_MODEL and an ONNX Runtime dynamic library"]
    fn silero_silence_reset_and_bad_frames() {
        let path = std::env::var_os("EARS_TEST_VAD_MODEL").expect("set EARS_TEST_VAD_MODEL");
        let runtime = std::env::var_os("ORT_DYLIB_PATH").expect("set ORT_DYLIB_PATH");
        let mut detector = SileroVad::load(Path::new(&path), Path::new(&runtime)).unwrap();
        let silence = [0.0; FRAME_SAMPLES];
        let first = detector.score(&silence).unwrap();
        assert!(first < 0.5);
        for _ in 0..100 {
            assert!(detector.score(&silence).unwrap() < 0.5);
        }
        detector.reset();
        assert_eq!(detector.score(&silence).unwrap(), first);
        assert!(detector.score(&[0.0; 32]).is_err());
        assert_eq!(detector.score(&silence).unwrap(), first);
        assert!(detector.score(&[f32::NAN; FRAME_SAMPLES]).is_err());
        assert_eq!(detector.score(&silence).unwrap(), first);
    }
}
