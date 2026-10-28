//! Backend-agnostic, sample-clock VAD segmentation for retained capture clips.
//!
//! Feed each contiguous, fixed-size mono `f32` PCM frame and the probability
//! produced by a real detector for **that same frame** to `push_frame`. Scores
//! must be finite and in [0, 1]; PCM must be finite. This module does not run a
//! detector, infer speech from RMS, resample, or accept interleaved channels.
//! Any positive sample rate is supported; detector-specific restrictions belong
//! in its adapter. All timing settings are integer frame counts, not timers.
//!
//! Onset requires consecutive speech frames. Pre-roll is measured back from
//! the FIRST such frame, not the confirmation frame. A below-threshold run of
//! `silence_frames` closes speech, retaining only `post_roll_frames` after the
//! last speech frame. Shorter pauses remain inside the clip. Spans are exact,
//! half-open sample offsets in the caller's PCM tap clock, with frame precision.
//! Clips never overlap: pre-roll is clamped to session start / previous clip end.
//!
//! Memory is bounded by configuration (and a 16M-sample hard ceiling). The
//! working buffer allows max clip length plus silence-decision lookahead.
//! Overflow is an error, NEVER a successful truncated clip. Every input error
//! discards pending audio and seals the session until `reset`. `finish` flushes
//! confirmed speech; `cancel` discards it. Neither admits more input until reset.
//! Reset the detector and frame adapter too; never feed queued old-session data.
//! Buffers are logically cleared, not securely zeroized.
//!
//! Integration remains external: detector selection, frame assembly, PCM tap
//! wiring, module registration, and matching completed clips with ASR results.
//! ASR completion events and wall-clock time deliberately have no API here.

use std::collections::VecDeque;
use std::fmt;
use std::ops::Range;

const MAX_BUFFER_SAMPLES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub sample_rate_hz: u32,
    /// Only mono is supported; reject rather than silently reinterpret stereo.
    pub channels: u16,
    pub frame_samples: usize,
    /// A frame is speech when probability >= this finite threshold in (0, 1].
    pub speech_threshold: f32,
    pub pre_roll_frames: usize,
    pub min_speech_frames: usize,
    /// Consecutive non-speech frames required to close a clip; at least one.
    pub silence_frames: usize,
    /// Must not exceed silence_frames; zero is allowed.
    pub post_roll_frames: usize,
    /// Includes pre-roll, speech, internal pauses, and retained post-roll.
    pub max_segment_frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Silence,
    /// Capture ended before the full silence decision window was available.
    EndOfInput,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub span: Range<u64>,
    /// First accepted speech frame through last speech frame (includes pauses).
    pub speech_span: Range<u64>,
    /// Exactly the samples in span, with no normalization or silent clipping.
    pub pcm: Vec<f32>,
    pub end_reason: EndReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidConfig(&'static str),
    UnsupportedChannels(u16),
    AllocationFailed,
    Closed,
    InvalidFrameLength {
        expected: usize,
        actual: usize,
    },
    InvalidProbability,
    NonFinitePcm,
    NonContiguous {
        expected: u64,
        actual: u64,
    },
    SampleClockOverflow,
    /// Entire pending clip is discarded; the end is the attempted retained end.
    SegmentTooLong {
        span: Range<u64>,
        max_samples: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "capture segmentation: {self:?}")
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
struct Speech {
    onset: u64,
    last_end: u64,
    silent_frames: usize,
}

#[derive(Debug)]
pub struct CaptureSegments {
    config: Config,
    idle_samples: usize,
    max_samples: usize,
    post_samples: usize,
    buffer: VecDeque<f32>,
    next_sample: u64,
    speech_run: usize,
    speech: Option<Speech>,
    closed: bool,
}

impl CaptureSegments {
    /// Starts at sample zero. Call reset(tap_position) for another clock origin.
    pub fn new(config: Config) -> Result<Self, Error> {
        if config.channels != 1 {
            return Err(Error::UnsupportedChannels(config.channels));
        }
        if config.sample_rate_hz == 0 || config.frame_samples == 0 {
            return Err(Error::InvalidConfig(
                "sample rate and frame size must be positive",
            ));
        }
        if !config.speech_threshold.is_finite()
            || config.speech_threshold <= 0.0
            || config.speech_threshold > 1.0
        {
            return Err(Error::InvalidConfig(
                "threshold must be finite and in (0, 1]",
            ));
        }
        if config.min_speech_frames == 0 || config.silence_frames == 0 {
            return Err(Error::InvalidConfig(
                "speech and silence debounce must be positive",
            ));
        }
        if config.post_roll_frames > config.silence_frames {
            return Err(Error::InvalidConfig("post-roll exceeds silence window"));
        }
        let idle_frames = config.pre_roll_frames.checked_add(config.min_speech_frames);
        let required_frames = idle_frames.and_then(|n| n.checked_add(config.post_roll_frames));
        if required_frames.is_none_or(|n| n > config.max_segment_frames) {
            return Err(Error::InvalidConfig(
                "max segment cannot hold pre-roll, onset and post-roll",
            ));
        }
        let capacity = config
            .max_segment_frames
            .checked_add(config.silence_frames)
            .and_then(|n| n.checked_mul(config.frame_samples))
            .filter(|&n| n <= MAX_BUFFER_SAMPLES)
            .ok_or(Error::InvalidConfig("working buffer exceeds sample budget"))?;
        let mut buffer = VecDeque::new();
        buffer
            .try_reserve_exact(capacity)
            .map_err(|_| Error::AllocationFailed)?;
        Ok(Self {
            config,
            // All products below are bounded by the validated capacity.
            idle_samples: idle_frames.unwrap() * config.frame_samples,
            max_samples: config.max_segment_frames * config.frame_samples,
            post_samples: config.post_roll_frames * config.frame_samples,
            buffer,
            next_sample: 0,
            speech_run: 0,
            speech: None,
            closed: false,
        })
    }

    /// At most one clip per frame. Start offset detects missing/reordered frames.
    /// Invalid input seals the session, even if a clip was already in progress.
    pub fn push_frame(
        &mut self,
        start_sample: u64,
        pcm: &[f32],
        speech_probability: f32,
    ) -> Result<Option<Segment>, Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        if let Err(error) = self.validate_frame(start_sample, pcm, speech_probability) {
            self.cancel();
            return Err(error);
        }
        let Some(end) = self.next_sample.checked_add(pcm.len() as u64) else {
            self.cancel();
            return Err(Error::SampleClockOverflow);
        };
        self.buffer.extend(pcm.iter().copied());
        self.next_sample = end;
        let is_speech = speech_probability >= self.config.speech_threshold;
        if let Some(speech) = &mut self.speech {
            if is_speech {
                speech.last_end = end;
                speech.silent_frames = 0;
            } else {
                speech.silent_frames += 1;
            }
        } else {
            self.trim_idle();
            self.speech_run = if is_speech { self.speech_run + 1 } else { 0 };
            if self.speech_run == self.config.min_speech_frames {
                self.speech = Some(Speech {
                    onset: end - (self.speech_run * self.config.frame_samples) as u64,
                    last_end: end,
                    silent_frames: 0,
                });
                self.speech_run = 0;
            }
        }
        self.check_endpoint()
    }

    /// Flushes only confirmed speech, using available post-roll, then seals.
    /// A shorter-than-minimum onset candidate is deliberately discarded.
    pub fn finish(&mut self) -> Result<Option<Segment>, Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        let segment = if self.speech.is_some() {
            Some(self.emit(EndReason::EndOfInput))
        } else {
            None
        };
        self.cancel();
        Ok(segment)
    }

    /// Discards all pending audio and seals; does not emit a partial clip.
    pub fn cancel(&mut self) {
        self.buffer.clear();
        self.speech = None;
        self.speech_run = 0;
        self.closed = true;
    }

    /// Begins a fresh session in the caller's absolute PCM clock.
    pub fn reset(&mut self, start_sample: u64) {
        self.cancel();
        self.next_sample = start_sample;
        self.closed = false;
    }

    pub fn next_sample(&self) -> u64 {
        self.next_sample
    }

    fn validate_frame(&self, start: u64, pcm: &[f32], probability: f32) -> Result<(), Error> {
        if start != self.next_sample {
            return Err(Error::NonContiguous {
                expected: self.next_sample,
                actual: start,
            });
        }
        if pcm.len() != self.config.frame_samples {
            return Err(Error::InvalidFrameLength {
                expected: self.config.frame_samples,
                actual: pcm.len(),
            });
        }
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(Error::InvalidProbability);
        }
        if pcm.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::NonFinitePcm);
        }
        Ok(())
    }

    fn trim_idle(&mut self) {
        let excess = self.buffer.len().saturating_sub(self.idle_samples);
        self.buffer.drain(..excess);
    }

    fn retained_span(&self, speech: &Speech) -> Range<u64> {
        let available_tail = self.next_sample - speech.last_end;
        let end = speech.last_end + available_tail.min(self.post_samples as u64);
        (self.next_sample - self.buffer.len() as u64)..end
    }

    fn check_endpoint(&mut self) -> Result<Option<Segment>, Error> {
        let Some(speech) = &self.speech else {
            return Ok(None);
        };
        let span = self.retained_span(speech);
        if span.end - span.start > self.max_samples as u64 {
            let error = Error::SegmentTooLong {
                span,
                max_samples: self.max_samples,
            };
            self.cancel();
            return Err(error);
        }
        if speech.silent_frames >= self.config.silence_frames {
            return Ok(Some(self.emit(EndReason::Silence)));
        }
        Ok(None)
    }

    fn emit(&mut self, end_reason: EndReason) -> Segment {
        let speech = self.speech.take().expect("emit requires confirmed speech");
        let span = self.retained_span(&speech);
        let pcm = self
            .buffer
            .drain(..(span.end - span.start) as usize)
            .collect();
        // Keep only unused silence for the next pre-roll, never previous audio.
        self.trim_idle();
        self.speech_run = 0;
        Segment {
            span,
            speech_span: speech.onset..speech.last_end,
            pcm,
            end_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            sample_rate_hz: 24_000,
            channels: 1,
            frame_samples: 4,
            speech_threshold: 0.5,
            pre_roll_frames: 3,
            min_speech_frames: 3,
            silence_frames: 4,
            post_roll_frames: 2,
            max_segment_frames: 30,
        }
    }

    // Every sample is an identifiable synthetic marker, not an amplitude VAD.
    fn frame(core: &mut CaptureSegments, probability: f32) -> Result<Option<Segment>, Error> {
        let start = core.next_sample();
        let pcm: Vec<_> = (start..start + 4).map(|n| n as f32).collect();
        core.push_frame(start, &pcm, probability)
    }

    fn feed(core: &mut CaptureSegments, scores: &[f32]) -> Vec<Segment> {
        scores
            .iter()
            .filter_map(|&p| frame(core, p).unwrap())
            .collect()
    }

    fn assert_exact(segment: &Segment, span: Range<u64>, speech: Range<u64>) {
        assert_eq!(segment.span, span);
        assert_eq!(segment.speech_span, speech);
        assert_eq!(segment.pcm, span.map(|n| n as f32).collect::<Vec<_>>());
    }

    #[test]
    fn silero_frame_geometry_preserves_exact_sample_markers() {
        let mut cfg = config();
        cfg.sample_rate_hz = 16_000;
        cfg.frame_samples = 512;
        let mut core = CaptureSegments::new(cfg).unwrap();
        let mut segments = Vec::new();
        for index in 0..13 {
            let start = index * 512;
            let pcm: Vec<_> = (start..start + 512).map(|n| n as f32).collect();
            let score = if (5..9).contains(&index) { 1.0 } else { 0.0 };
            if let Some(segment) = core.push_frame(start, &pcm, score).unwrap() {
                segments.push(segment);
            }
        }
        assert_eq!(segments.len(), 1);
        assert_exact(&segments[0], 1024..5632, 2560..4608);
        assert_eq!(segments[0].pcm[1536], 2560.0);
        assert_eq!(segments[0].pcm[3583], 4607.0);
    }

    #[test]
    fn onset_debounce_keeps_first_marker_and_exact_postroll() {
        let mut core = CaptureSegments::new(config()).unwrap();
        assert!(feed(&mut core, &[0.0; 5]).is_empty());
        assert!(feed(&mut core, &[1.0; 2]).is_empty());
        assert!(frame(&mut core, 1.0).unwrap().is_none());
        assert!(frame(&mut core, 1.0).unwrap().is_none());
        assert!(feed(&mut core, &[0.0; 3]).is_empty());
        let segment = frame(&mut core, 0.0).unwrap().unwrap();
        assert_exact(&segment, 8..44, 20..36);
        assert_eq!(segment.pcm[12], 20.0); // first speech sample
        assert_eq!(segment.pcm[27], 35.0); // last speech sample
        assert_eq!(core.next_sample(), 52); // decision is later than retained end
        assert_eq!(segment.end_reason, EndReason::Silence);
    }

    #[test]
    fn speech_at_session_start_is_not_lost_and_short_noise_is_rejected() {
        let mut core = CaptureSegments::new(config()).unwrap();
        assert!(feed(&mut core, &[1.0; 2]).is_empty());
        assert!(core.finish().unwrap().is_none());
        core.reset(100);
        feed(&mut core, &[1.0; 3]);
        let segment = core.finish().unwrap().unwrap();
        assert_exact(&segment, 100..112, 100..112);
        assert_eq!(segment.end_reason, EndReason::EndOfInput);
    }

    #[test]
    fn short_pause_does_not_split_or_drop_audio() {
        let mut core = CaptureSegments::new(config()).unwrap();
        feed(&mut core, &[1.0; 3]);
        assert!(feed(&mut core, &[0.0; 3]).is_empty());
        assert!(frame(&mut core, 1.0).unwrap().is_none());
        let segments = feed(&mut core, &[0.0; 4]);
        assert_exact(&segments[0], 0..36, 0..28);
    }

    #[test]
    fn consecutive_segments_keep_unused_silence_without_overlap() {
        let mut core = CaptureSegments::new(config()).unwrap();
        feed(&mut core, &[1.0; 3]);
        let first = feed(&mut core, &[0.0; 4]).pop().unwrap();
        feed(&mut core, &[1.0; 3]);
        let second = feed(&mut core, &[0.0; 4]).pop().unwrap();
        assert_exact(&first, 0..20, 0..12);
        assert_exact(&second, 20..48, 28..40);
        assert!(first.span.end <= second.span.start);
    }

    #[test]
    fn variable_asr_delivery_delays_do_not_move_spans() {
        fn capture(asr_delivery_frames: &[usize]) -> Vec<Segment> {
            let mut core = CaptureSegments::new(config()).unwrap();
            let mut clips = Vec::new();
            // Simulates unrelated ASR notifications before, during, or after
            // speech. Notifications never mutate the sample-clock core.
            for i in 0..40 {
                if asr_delivery_frames.contains(&i) {
                    std::thread::yield_now();
                }
                let p = if (5..9).contains(&i) || (20..24).contains(&i) {
                    1.0
                } else {
                    0.0
                };
                if let Some(clip) = frame(&mut core, p).unwrap() {
                    clips.push(clip);
                }
            }
            clips
        }
        let early = capture(&[1, 8, 12, 22]);
        let late = capture(&[17, 29, 38, 39]);
        assert_eq!(early, late);
        assert_exact(&early[0], 8..44, 20..36);
        assert_exact(&early[1], 68..104, 80..96);
    }

    #[test]
    fn duration_overflow_is_an_error_not_a_truncated_success() {
        let mut cfg = config();
        cfg.max_segment_frames = 8;
        let mut core = CaptureSegments::new(cfg).unwrap();
        assert!(feed(&mut core, &[1.0; 8]).is_empty());
        assert_eq!(
            frame(&mut core, 1.0),
            Err(Error::SegmentTooLong {
                span: 0..36,
                max_samples: 32
            })
        );
        assert!(core.buffer.is_empty());
        assert_eq!(core.finish(), Err(Error::Closed));
        assert_eq!(frame(&mut core, 0.0), Err(Error::Closed));
    }

    #[test]
    fn max_length_clip_allows_decision_lookahead_beyond_retained_end() {
        let mut cfg = config();
        cfg.max_segment_frames = 8;
        let mut core = CaptureSegments::new(cfg).unwrap();
        feed(&mut core, &[1.0; 6]);
        let clips = feed(&mut core, &[0.0; 4]);
        assert_exact(&clips[0], 0..32, 0..24);
        assert_eq!(core.next_sample(), 40);
    }

    #[test]
    fn postroll_and_resumed_speech_can_also_overflow() {
        let mut cfg = config();
        cfg.max_segment_frames = 8;
        let mut core = CaptureSegments::new(cfg).unwrap();
        feed(&mut core, &[1.0; 8]);
        assert!(matches!(
            frame(&mut core, 0.0),
            Err(Error::SegmentTooLong { .. })
        ));
        core.reset(0);
        feed(&mut core, &[1.0; 6]);
        feed(&mut core, &[0.0; 3]);
        assert!(matches!(
            frame(&mut core, 1.0),
            Err(Error::SegmentTooLong { .. })
        ));
    }

    #[test]
    fn reset_drops_active_candidate_and_idle_audio_across_sessions() {
        let mut core = CaptureSegments::new(config()).unwrap();
        for scores in [&[0.0; 5][..], &[1.0; 2][..], &[1.0; 4][..]] {
            core.reset(0);
            feed(&mut core, scores);
            core.reset(1000);
            feed(&mut core, &[1.0; 3]);
            let segment = core.finish().unwrap().unwrap();
            assert_exact(&segment, 1000..1012, 1000..1012);
        }
    }

    #[test]
    fn cancel_discards_and_finish_is_terminal() {
        let mut core = CaptureSegments::new(config()).unwrap();
        feed(&mut core, &[1.0; 3]);
        core.cancel();
        assert!(core.buffer.is_empty());
        assert_eq!(core.finish(), Err(Error::Closed));
        core.reset(12);
        feed(&mut core, &[1.0; 3]);
        frame(&mut core, 0.0).unwrap();
        assert_exact(&core.finish().unwrap().unwrap(), 12..28, 12..24);
        assert_eq!(frame(&mut core, 1.0), Err(Error::Closed));
        assert_eq!(core.finish(), Err(Error::Closed));
    }

    #[test]
    fn invalid_probabilities_and_pcm_fail_closed() {
        let mut core = CaptureSegments::new(config()).unwrap();
        for p in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            core.reset(0);
            feed(&mut core, &[1.0; 3]);
            assert_eq!(frame(&mut core, p), Err(Error::InvalidProbability));
            assert_eq!(core.finish(), Err(Error::Closed));
            assert!(core.buffer.is_empty());
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            core.reset(0);
            feed(&mut core, &[1.0; 3]);
            assert_eq!(
                core.push_frame(12, &[0.0, bad, 0.0, 0.0], 1.0),
                Err(Error::NonFinitePcm)
            );
            assert_eq!(core.finish(), Err(Error::Closed));
            assert!(core.buffer.is_empty());
        }
    }

    #[test]
    fn wrong_frame_sizes_and_discontinuous_clock_are_rejected() {
        let mut core = CaptureSegments::new(config()).unwrap();
        for len in [0, 3, 5, 8] {
            core.reset(0);
            assert_eq!(
                core.push_frame(0, &vec![0.0; len], 0.0),
                Err(Error::InvalidFrameLength {
                    expected: 4,
                    actual: len
                })
            );
            assert_eq!(core.finish(), Err(Error::Closed));
        }
        for offset in [0, 3, 5, 8] {
            core.reset(0);
            frame(&mut core, 1.0).unwrap();
            assert_eq!(
                core.push_frame(offset, &[0.0; 4], 1.0),
                Err(Error::NonContiguous {
                    expected: 4,
                    actual: offset
                })
            );
            assert_eq!(core.finish(), Err(Error::Closed));
        }
    }

    #[test]
    fn clock_overflow_fails_closed_without_panicking() {
        let mut core = CaptureSegments::new(config()).unwrap();
        core.reset(u64::MAX - 4);
        frame(&mut core, 1.0).unwrap();
        assert_eq!(
            core.push_frame(u64::MAX, &[0.0; 4], 1.0),
            Err(Error::SampleClockOverflow)
        );
        assert_eq!(core.finish(), Err(Error::Closed));
    }

    #[test]
    fn zero_padding_and_one_frame_debounce_work_at_threshold() {
        let mut cfg = config();
        cfg.pre_roll_frames = 0;
        cfg.post_roll_frames = 0;
        cfg.min_speech_frames = 1;
        cfg.silence_frames = 1;
        let mut core = CaptureSegments::new(cfg).unwrap();
        frame(&mut core, 0.0).unwrap();
        frame(&mut core, 0.5).unwrap();
        assert_exact(&frame(&mut core, 0.499).unwrap().unwrap(), 4..8, 4..8);
    }

    #[test]
    fn long_silence_and_subminimum_pulses_have_bounded_storage() {
        let mut core = CaptureSegments::new(config()).unwrap();
        let capacity = core.buffer.capacity();
        for i in 0..10_000 {
            assert!(frame(&mut core, if i % 3 == 0 { 0.0 } else { 1.0 })
                .unwrap()
                .is_none());
            assert!(core.buffer.len() <= core.idle_samples);
            assert_eq!(core.buffer.capacity(), capacity);
        }
        assert!(core.finish().unwrap().is_none());
    }

    #[test]
    fn rejects_unsupported_format_bad_timing_and_unbounded_configuration() {
        let base = config();
        let invalid = [
            Config {
                channels: 0,
                ..base
            },
            Config {
                channels: 2,
                ..base
            },
            Config {
                sample_rate_hz: 0,
                ..base
            },
            Config {
                frame_samples: 0,
                ..base
            },
            Config {
                speech_threshold: f32::NAN,
                ..base
            },
            Config {
                speech_threshold: f32::INFINITY,
                ..base
            },
            Config {
                speech_threshold: 0.0,
                ..base
            },
            Config {
                speech_threshold: 1.1,
                ..base
            },
            Config {
                min_speech_frames: 0,
                ..base
            },
            Config {
                silence_frames: 0,
                ..base
            },
            Config {
                post_roll_frames: 5,
                ..base
            },
            Config {
                max_segment_frames: 7,
                ..base
            },
            Config {
                pre_roll_frames: usize::MAX,
                ..base
            },
            Config {
                max_segment_frames: usize::MAX,
                ..base
            },
            Config {
                frame_samples: usize::MAX,
                ..base
            },
            Config {
                frame_samples: MAX_BUFFER_SAMPLES,
                ..base
            },
        ];
        for cfg in invalid {
            assert!(CaptureSegments::new(cfg).is_err(), "accepted {cfg:?}");
        }
    }
}
