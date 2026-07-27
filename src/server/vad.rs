//! Engine-agnostic boundary VAD.
//!
//! Runs at the server audio ingress, before audio is handed to any engine, so a
//! single acoustic voice-activity detector emits `Speech { active }` boundaries
//! uniformly for every engine (kyutai, parakeet-rs, transcribe-cpp, and any
//! future engine). This is the single owner of speech-boundary events on the
//! wire; engines keep their own internal endpointing for decoding, but do not
//! emit boundaries themselves.

/// RMS energy above which a chunk is considered speech.
const DEFAULT_RMS_THRESHOLD: f32 = 0.0025;
/// Continuous speech required before opening a turn (debounces noise bursts).
const DEFAULT_MIN_SPEECH_MS: usize = 120;
/// Per-connection boundary detector. Hangover and minimum-speech are tracked in
/// samples, so the detector is robust to variable client chunk sizes.
pub(crate) struct BoundaryVad {
    enabled: bool,
    rms_threshold: f32,
    min_speech_samples: usize,
    hangover_samples: usize,
    in_speech: bool,
    speech_samples: usize,
    silence_samples: usize,
}

impl BoundaryVad {
    pub(crate) fn new(enabled: bool, sample_rate: usize, hangover_ms: usize) -> Self {
        let per_ms = sample_rate / 1000;
        Self {
            enabled,
            rms_threshold: DEFAULT_RMS_THRESHOLD,
            min_speech_samples: DEFAULT_MIN_SPEECH_MS * per_ms,
            hangover_samples: hangover_ms.max(1) * per_ms,
            in_speech: false,
            speech_samples: 0,
            silence_samples: 0,
        }
    }

    /// Enable or disable boundary detection at runtime. Disabling also clears
    /// any in-progress speech state so a later re-enable starts clean.
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.in_speech = false;
            self.speech_samples = 0;
            self.silence_samples = 0;
        }
    }

    /// Feed one decoded PCM chunk. Returns `Some(true)` on a silence→speech
    /// transition, `Some(false)` on a speech→silence transition (after the
    /// hangover), and `None` when the speech state is unchanged.
    pub(crate) fn observe(&mut self, chunk: &[f32]) -> Option<bool> {
        if !self.enabled || chunk.is_empty() {
            return None;
        }
        let loud = rms(chunk) > self.rms_threshold;
        let samples = chunk.len();

        if self.in_speech {
            if loud {
                self.silence_samples = 0;
            } else {
                self.silence_samples += samples;
                if self.silence_samples >= self.hangover_samples {
                    self.in_speech = false;
                    self.speech_samples = 0;
                    return Some(false);
                }
            }
        } else if loud {
            self.speech_samples += samples;
            if self.speech_samples >= self.min_speech_samples {
                self.in_speech = true;
                self.silence_samples = 0;
                return Some(true);
            }
        } else {
            self.speech_samples = 0;
        }
        None
    }
}

fn rms(chunk: &[f32]) -> f32 {
    let mean_sq = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32;
    mean_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::BoundaryVad;

    fn loud(n: usize) -> Vec<f32> {
        vec![0.2; n]
    }
    fn quiet(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    #[test]
    fn disabled_never_fires() {
        let mut vad = BoundaryVad::new(false, 24_000, 300);
        assert_eq!(vad.observe(&loud(24_000)), None);
    }

    #[test]
    fn opens_after_min_speech_and_closes_after_hangover() {
        let mut vad = BoundaryVad::new(true, 24_000, 300);
        // 120 ms min-speech at 24 kHz = 2880 samples; one 3000-sample loud chunk opens.
        assert_eq!(vad.observe(&loud(3_000)), Some(true));
        // Still speaking: no transition.
        assert_eq!(vad.observe(&loud(3_000)), None);
        // 300 ms hangover = 7200 samples; short silence does not close yet.
        assert_eq!(vad.observe(&quiet(3_600)), None);
        // Accumulated silence crosses the hangover: turn closes.
        assert_eq!(vad.observe(&quiet(3_600)), Some(false));
    }

    #[test]
    fn brief_noise_burst_does_not_open_a_turn() {
        let mut vad = BoundaryVad::new(true, 24_000, 300);
        // 40 ms of loud (< 120 ms min-speech) then silence: never opens.
        assert_eq!(vad.observe(&loud(960)), None);
        assert_eq!(vad.observe(&quiet(960)), None);
    }
}
