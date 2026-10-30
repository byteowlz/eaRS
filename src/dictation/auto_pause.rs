//! Capture-clock inactivity and owned, acknowledged ASR drain control.
//! A silence deadline is never itself evidence that recognition has finished.

use super::audio_history::ClientVadConfig;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

pub const DRAIN_CAPABILITY: &str = crate::DICTATION_DRAIN_CAPABILITY;

/// Observed local capture-clock state, never a prediction of future silence.
/// Only emitted while an enabled detector has scored audio in this generation.
/// Consumers must discard stale generations and values older than 500 ms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoPauseProgress {
    pub capture_generation: u64,
    pub quiet_ms: u32,
    pub idle_ms: u32,
    /// Actual physical push-to-talk hold at emission, not configured hotkey mode.
    pub held: bool,
    /// Producer snapshot time for freshness only; never advances the silence clock.
    /// Consumers must preserve this across delivery rather than timestamp receipt.
    pub scored_at: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AutoPauseConfig {
    /// Zero disables automatic pause. Counts scored capture audio, not ASR gaps.
    pub idle_ms: u32,
    pub drain_timeout_ms: u32,
    /// Minimum Silero probability counted as speech; higher rejects more noise.
    #[serde(deserialize_with = "deserialize_speech_threshold")]
    pub speech_threshold: f32,
    /// Optional independent detector configuration. Falls back to the explicit
    /// audio-history detector config, even when recording retention is disabled.
    pub vad: Option<ClientVadConfig>,
}

impl Default for AutoPauseConfig {
    fn default() -> Self {
        Self {
            idle_ms: 0,
            drain_timeout_ms: 10_000,
            speech_threshold: 0.5,
            vad: None,
        }
    }
}

impl AutoPauseConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.idle_ms == 0 || (500..=3_600_000).contains(&self.idle_ms),
            "auto-pause idle_ms must be 500..=3600000, or 0 to disable"
        );
        ensure!(
            (100..=60_000).contains(&self.drain_timeout_ms),
            "auto-pause drain_timeout_ms must be 100..=60000"
        );
        validate_speech_threshold(self.speech_threshold)
    }
}

/// Shared validation for configuration, live commands and Settings.
pub fn validate_speech_threshold(threshold: f32) -> Result<()> {
    ensure!(
        threshold.is_finite() && threshold > 0.0 && threshold < 1.0,
        "auto-pause speech_threshold must be finite and greater than 0 and less than 1"
    );
    Ok(())
}

fn deserialize_speech_threshold<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<f32, D::Error> {
    let threshold = f32::deserialize(deserializer)?;
    validate_speech_threshold(threshold).map_err(serde::de::Error::custom)?;
    Ok(threshold)
}

struct Drain {
    id: u64,
    generation: u64,
    deadline: Instant,
    final_seen: bool,
    resume: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Completion {
    Stale,
    Paused,
    Resume,
}

pub(crate) struct IdlePause {
    idle_samples: u64,
    speech_threshold: f32,
    drain_timeout: Duration,
    clock: u64,
    last_voice: u64,
    progress_clock: Option<u64>,
    generation: Option<u64>,
    next_request: u64,
    drain: Option<Drain>,
}

impl IdlePause {
    fn new(config: &AutoPauseConfig) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.idle_ms > 0,
            "disabled auto-pause has no active controller"
        );
        Ok(Self {
            idle_samples: u64::from(config.idle_ms) * 16,
            speech_threshold: config.speech_threshold,
            drain_timeout: Duration::from_millis(u64::from(config.drain_timeout_ms)),
            clock: 0,
            last_voice: 0,
            progress_clock: None,
            generation: None,
            next_request: 1,
            drain: None,
        })
    }
    pub fn set_speech_threshold(&mut self, threshold: f32) -> Result<()> {
        validate_speech_threshold(threshold)?;
        self.speech_threshold = threshold;
        Ok(())
    }
    pub fn start(&mut self, generation: u64) {
        self.clock = 0;
        self.last_voice = 0;
        self.progress_clock = None;
        self.generation = Some(generation);
        self.drain = None;
    }
    pub fn stop(&mut self) {
        self.generation = None;
        self.drain = None;
    }
    #[cfg(any(feature = "client-vad", test))]
    pub fn observe(&mut self, probability: f32) -> Result<()> {
        ensure!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "invalid auto-pause speech probability"
        );
        self.clock = self
            .clock
            .checked_add(512)
            .ok_or_else(|| anyhow::anyhow!("auto-pause sample clock overflow"))?;
        if probability >= self.speech_threshold {
            self.last_voice = self.clock;
            if let Some(drain) = &mut self.drain {
                drain.resume = true;
            }
        }
        Ok(())
    }
    /// At most one snapshot per 100 ms of newly scored 16 kHz audio.
    /// A 512-sample observation normally makes this a 128 ms cadence; no catch-up
    /// events, implicit zero or wall-clock advancement when scoring stalls.
    pub fn progress(&mut self, generation: u64, held: bool) -> Option<AutoPauseProgress> {
        if !self.active(generation)
            || self.clock == 0
            || self
                .progress_clock
                .is_some_and(|last| self.clock.saturating_sub(last) < 1600)
        {
            return None;
        }
        self.progress_clock = Some(self.clock);
        Some(AutoPauseProgress {
            capture_generation: generation,
            quiet_ms: u32::try_from(self.clock.saturating_sub(self.last_voice) / 16)
                .unwrap_or(u32::MAX),
            idle_ms: (self.idle_samples / 16) as u32,
            held,
            scored_at: Instant::now(),
        })
    }
    pub fn due(&self, generation: u64) -> bool {
        self.generation == Some(generation)
            && self.drain.is_none()
            && self.clock.saturating_sub(self.last_voice) >= self.idle_samples
    }
    pub fn begin(&mut self, generation: u64, now: Instant) -> Result<u64> {
        ensure!(
            self.due(generation),
            "auto-pause is not due for this capture generation"
        );
        let id = self.next_request;
        self.next_request = id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("auto-pause request ID exhausted"))?;
        self.drain = Some(Drain {
            id,
            generation,
            deadline: now + self.drain_timeout,
            final_seen: false,
            resume: false,
        });
        Ok(id)
    }
    pub fn active(&self, generation: u64) -> bool {
        self.generation == Some(generation)
    }
    pub fn draining(&self) -> bool {
        self.drain.is_some()
    }
    pub fn accepts(&self, generation: u64) -> bool {
        self.drain
            .as_ref()
            .is_none_or(|drain| drain.generation == generation)
    }
    pub fn final_seen(&mut self, generation: u64) {
        if let Some(drain) = &mut self.drain {
            if drain.generation == generation {
                drain.final_seen = true;
            }
        }
    }
    pub fn resume(&mut self, generation: u64) {
        if let Some(drain) = &mut self.drain {
            if drain.generation == generation {
                drain.resume = true;
            }
        }
    }
    pub fn timed_out(&self, now: Instant) -> bool {
        self.drain
            .as_ref()
            .is_some_and(|drain| now >= drain.deadline)
    }
    pub fn acknowledge(&mut self, id: u64, generation: u64) -> Result<Completion> {
        let Some(drain) = self.drain.as_ref() else {
            return Ok(Completion::Stale);
        };
        if id != drain.id || generation != drain.generation {
            return Ok(Completion::Stale);
        }
        ensure!(
            drain.final_seen,
            "server acknowledged drain without a preceding final result"
        );
        let resume = self.drain.take().unwrap().resume;
        if resume {
            self.last_voice = self.clock;
            Ok(Completion::Resume)
        } else {
            self.generation = None;
            Ok(Completion::Paused)
        }
    }
}

/// Explicit local detector, independent of recording permission/retention.
pub(crate) struct AutoPause {
    pub control: IdlePause,
    pub buffer_limit: usize,
    #[cfg(feature = "client-vad")]
    scorer: super::client_vad::CaptureScorer,
}

impl AutoPause {
    pub fn from_config(
        config: &AutoPauseConfig,
        fallback: Option<&ClientVadConfig>,
    ) -> Result<Option<Self>> {
        config.validate()?;
        if config.idle_ms == 0 {
            return Ok(None);
        }
        let control = IdlePause::new(config)?;
        #[cfg(not(feature = "client-vad"))]
        {
            let _ = (control, fallback);
            anyhow::bail!("auto-pause requires an eaRS build with client-vad");
        }
        #[cfg(feature = "client-vad")]
        {
            let vad = config.vad.as_ref().or(fallback).ok_or_else(|| {
                anyhow::anyhow!("auto-pause needs explicit local VAD model/runtime configuration")
            })?;
            Ok(Some(Self {
                control,
                buffer_limit: config.drain_timeout_ms as usize * 24 + 24_000,
                scorer: super::client_vad::CaptureScorer::new(Box::new(vad.load_detector()?)),
            }))
        }
    }
    pub fn start(&mut self, generation: u64) {
        self.control.start(generation);
        #[cfg(feature = "client-vad")]
        self.scorer.reset();
    }
    pub fn stop(&mut self) {
        self.control.stop();
        #[cfg(feature = "client-vad")]
        self.scorer.reset();
    }
    pub fn observe(&mut self, pcm: &[f32]) -> Result<()> {
        #[cfg(feature = "client-vad")]
        {
            for frame in self.scorer.push(pcm)? {
                self.control.observe(frame.probability)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "client-vad"))]
        {
            let _ = pcm;
            anyhow::bail!("auto-pause VAD unavailable")
        }
    }
    #[cfg(all(test, feature = "client-vad"))]
    pub fn fixture(
        scorer: Box<dyn super::client_vad::SpeechScorer>,
        config: AutoPauseConfig,
    ) -> Self {
        Self {
            control: IdlePause::new(&config).unwrap(),
            buffer_limit: config.drain_timeout_ms as usize * 24 + 24_000,
            scorer: super::client_vad::CaptureScorer::new(scorer),
        }
    }
}

#[cfg(test)]
#[path = "auto_pause_threshold_tests.rs"]
mod threshold_tests;

#[cfg(test)]
#[path = "auto_pause_progress_tests.rs"]
mod progress_tests;

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn gate() -> IdlePause {
        IdlePause::new(&AutoPauseConfig {
            idle_ms: 3000,
            ..Default::default()
        })
        .unwrap()
    }
    pub(super) fn silent(gate: &mut IdlePause, frames: usize) {
        for _ in 0..frames {
            gate.observe(0.0).unwrap();
        }
    }
    #[test]
    fn config_round_trips_and_disabled_control_does_not_load_artifacts() {
        let config: AutoPauseConfig =
            toml::from_str("idle_ms = 3000\ndrain_timeout_ms = 1500\n").unwrap();
        assert_eq!(config.idle_ms, 3000);
        config.validate().unwrap();
        let round_trip: AutoPauseConfig =
            toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(round_trip, config);
        assert!(toml::from_str::<AutoPauseConfig>("idle_ms = 3000\nunknown = true").is_err());
        let disabled = AutoPauseConfig {
            vad: Some(ClientVadConfig {
                model_path: "/not-an-artifact".into(),
                model_sha256: "invalid".into(),
                runtime_path: "/not-a-runtime".into(),
            }),
            ..Default::default()
        };
        assert!(AutoPause::from_config(&disabled, None).unwrap().is_none());
    }
    #[test]
    fn deadline_uses_capture_silence_not_asr_or_wall_time() {
        let mut gate = gate();
        gate.start(7);
        assert!(!gate.due(7)); // No audio, regardless of wall-clock waiting.
        silent(&mut gate, 93);
        assert!(!gate.due(7));
        gate.observe(0.8).unwrap();
        silent(&mut gate, 93);
        assert!(!gate.due(7));
        silent(&mut gate, 1);
        assert!(gate.due(7));
        assert!(!gate.due(8));
    }
    #[test]
    fn requires_owned_final_then_ack_and_never_accepts_duplicate_or_stale_ack() {
        let mut gate = gate();
        gate.start(1);
        silent(&mut gate, 94);
        let id = gate.begin(1, Instant::now()).unwrap();
        assert!(gate.acknowledge(id, 1).is_err());
        gate.final_seen(2);
        assert!(gate.acknowledge(id, 1).is_err());
        gate.final_seen(1);
        assert_eq!(gate.acknowledge(id + 1, 1).unwrap(), Completion::Stale);
        assert_eq!(gate.acknowledge(id, 2).unwrap(), Completion::Stale);
        assert_eq!(gate.acknowledge(id, 1).unwrap(), Completion::Paused);
        assert_eq!(gate.acknowledge(id, 1).unwrap(), Completion::Stale);
    }
    #[test]
    fn speech_during_drain_requests_resume_and_new_capture_never_reuses_ids() {
        let mut gate = gate();
        let now = Instant::now();
        gate.start(1);
        silent(&mut gate, 94);
        let id = gate.begin(1, now).unwrap();
        gate.observe(0.8).unwrap();
        gate.final_seen(1);
        assert_eq!(gate.acknowledge(id, 1).unwrap(), Completion::Resume);
        assert!(!gate.due(1));
        gate.start(3);
        silent(&mut gate, 94);
        let next = gate.begin(3, now).unwrap();
        assert!(next > id);
        assert!(!gate.accepts(1));
        assert!(gate.accepts(3));
        gate.stop();
        assert_eq!(gate.acknowledge(next, 3).unwrap(), Completion::Stale);
    }
    #[test]
    fn timeout_is_bounded_and_not_an_acknowledgement() {
        let mut gate = gate();
        let now = Instant::now();
        gate.start(0);
        silent(&mut gate, 94);
        gate.begin(0, now).unwrap();
        assert!(!gate.timed_out(now + Duration::from_millis(9999)));
        assert!(gate.timed_out(now + Duration::from_secs(10)));
        assert!(gate.draining());
        gate.stop();
        assert!(!gate.draining());
    }
    #[test]
    fn invalid_config_and_scores_fail_without_silence_fallback() {
        assert!(
            AutoPause::from_config(&AutoPauseConfig::default(), None)
                .unwrap()
                .is_none()
        );
        let mut gate = gate();
        gate.start(0);
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(gate.observe(value).is_err());
        }
        assert!(!gate.due(0));
        assert!(
            IdlePause::new(&AutoPauseConfig {
                idle_ms: 1,
                ..Default::default()
            })
            .is_err()
        );
    }
}
