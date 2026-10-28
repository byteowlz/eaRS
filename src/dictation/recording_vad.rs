//! Client-owned acoustic boundaries, independent of ASR delivery.
use super::{
    audio_history::{AudioTap, SAMPLE_RATE},
    capture_segments::{CaptureSegments, Config, Segment},
    client_vad::{FRAME_SAMPLES, SpeechScorer},
    wake::Resampler,
};
use anyhow::{Context, Result, ensure};

pub(super) struct RecordingVad {
    detector: Box<dyn SpeechScorer>,
    spans: CaptureSegments,
    resampler: Resampler,
    pending: std::collections::VecDeque<f32>,
    input_pending: std::collections::VecDeque<f32>,
    epoch: u64,
    clock: u64,
    wait_for_silence: usize,
}

pub(super) struct Clip {
    pub pcm: Vec<f32>,
    pub start: u64,
    pub end: u64,
    pub complete: bool,
}

impl RecordingVad {
    pub fn new(detector: Box<dyn SpeechScorer>, max_secs: u32, pre_roll_ms: u32) -> Result<Self> {
        Ok(Self {
            detector,
            spans: CaptureSegments::new(Config {
                sample_rate_hz: 16_000,
                channels: 1,
                frame_samples: FRAME_SAMPLES,
                speech_threshold: 0.5,
                pre_roll_frames: (pre_roll_ms as usize).div_ceil(32),
                min_speech_frames: 4,
                silence_frames: 16,
                post_roll_frames: 16,
                max_segment_frames: (max_secs as usize * 16_000) / FRAME_SAMPLES,
            })?,
            resampler: Resampler::new(SAMPLE_RATE as usize, 16_000),
            pending: Default::default(),
            input_pending: Default::default(),
            epoch: 0,
            clock: 0,
            wait_for_silence: 0,
        })
    }

    pub fn reset(&mut self, epoch: u64, wait_for_silence: bool) {
        self.detector.reset();
        self.spans.reset(0);
        self.resampler = Resampler::new(SAMPLE_RATE as usize, 16_000);
        self.pending.clear();
        self.input_pending.clear();
        self.epoch = epoch;
        self.clock = 0;
        self.wait_for_silence = if wait_for_silence { 16 } else { 0 };
    }

    pub fn push(&mut self, pcm: &[f32], tap: &AudioTap) -> Result<Vec<Clip>> {
        ensure!(
            pcm.iter().all(|sample| sample.is_finite()),
            "non-finite recording input"
        );
        // Fixed source quanta keep the filter's fractional phase independent
        // of the microphone callback partition (768 @ 24k = 512 @ 16k).
        self.input_pending.extend(pcm.iter().copied());
        while self.input_pending.len() >= 768 {
            let input: Vec<_> = self.input_pending.drain(..768).collect();
            self.pending.extend(self.resampler.push(&input));
        }
        let mut clips = Vec::new();
        while self.pending.len() >= FRAME_SAMPLES {
            let frame: Vec<_> = self.pending.drain(..FRAME_SAMPLES).collect();
            let probability = self.detector.score(&frame)?;
            if self.wait_for_silence > 0 {
                self.wait_for_silence = if probability < 0.5 {
                    self.wait_for_silence - 1
                } else {
                    16
                };
                self.clock += FRAME_SAMPLES as u64;
                self.spans.reset(self.clock);
                continue;
            }
            if let Some(segment) = self.spans.push_frame(self.clock, &frame, probability)? {
                clips.push(source_clip(self.epoch, segment, tap, true)?);
            }
            self.clock += FRAME_SAMPLES as u64;
        }
        Ok(clips)
    }

    pub fn finish(&mut self, tap: &AudioTap) -> Result<Option<Clip>> {
        // Preserve the original PCM through capture release, but explicitly mark
        // the clip incomplete: release is not an acoustic end-of-speech proof.
        let Some(segment) = self.spans.finish()? else {
            return Ok(None);
        };
        let start = self.epoch + segment.span.start * 3 / 2;
        let end = tap.position();
        let pcm = exact_slice(tap, start, end)?;
        Ok(Some(Clip {
            pcm,
            start,
            end,
            complete: false,
        }))
    }
}

fn source_clip(epoch: u64, segment: Segment, tap: &AudioTap, complete: bool) -> Result<Clip> {
    let start = epoch
        .checked_add(segment.span.start * 3 / 2)
        .context("recording clock overflow")?;
    let end = epoch
        .checked_add(segment.span.end * 3 / 2)
        .context("recording clock overflow")?;
    Ok(Clip {
        pcm: exact_slice(tap, start, end)?,
        start,
        end,
        complete,
    })
}

fn exact_slice(tap: &AudioTap, start: u64, end: u64) -> Result<Vec<f32>> {
    let pcm = tap.slice(start, end);
    ensure!(
        pcm.len() as u64 == end - start,
        "recording audio span no longer buffered; refusing truncated clip"
    );
    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::super::capture_segments::EndReason;
    use super::*;

    #[test]
    fn source_spans_keep_24k_markers_and_reject_evicted_audio() {
        let tap = AudioTap::new(1);
        let pcm: Vec<f32> = (0..24_000).map(|i| i as f32 / 24_000.0).collect();
        tap.push(&pcm);
        let segment = || Segment {
            span: 512..4096,
            speech_span: 1024..3072,
            pcm: vec![],
            end_reason: EndReason::Silence,
        };
        let clip = source_clip(0, segment(), &tap, true).unwrap();
        assert_eq!((clip.start, clip.end), (768, 6144));
        assert_eq!(clip.pcm, pcm[768..6144]);
        tap.push(&vec![0.0; 24_000]);
        assert!(source_clip(0, segment(), &tap, true).is_err());
    }
}
