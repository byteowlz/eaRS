//! Experimental final-pass coordinator. Not enabled in production.
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub id: u64,
    pub start: usize,
    pub end: usize,
    pub complete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crop {
    pub start: usize,
    pub end: usize,
    pub emitted_at: usize,
    pub capped: bool,
}

#[derive(Clone, Copy)]
struct Pending {
    start: usize,
    end: usize,
}

pub struct Segmenter {
    hold: usize,
    padding: usize,
    cap: usize,
    pending: Option<Pending>,
    completed: HashSet<u64>,
    emitted_end: usize,
}

impl Segmenter {
    pub fn new(rate: usize) -> Self {
        Self {
            hold: rate * 800 / 1000,
            padding: rate * 200 / 1000,
            cap: rate * 30,
            pending: None,
            completed: HashSet::new(),
            emitted_end: 0,
        }
    }

    /// Consume only VAD observations available at this source-clock tick.
    /// Repeated completed lines cannot create another utterance. Native line
    /// timestamps are clamped to available PCM; later audio is never inspected.
    pub fn observe(&mut self, now: usize, regions: &[Region], eof: bool) -> Option<Crop> {
        let mut active = false;
        for region in regions {
            let end = region.end.min(now);
            if end <= self.emitted_end {
                continue;
            }
            if region.complete && !self.completed.insert(region.id) {
                continue;
            }
            active |= !region.complete;
            let start = region
                .start
                .saturating_sub(self.padding)
                .max(self.emitted_end);
            if start >= end {
                continue;
            }
            let pending = self.pending.get_or_insert(Pending { start, end });
            pending.start = pending.start.min(start);
            pending.end = pending.end.max(end);
        }
        let pending = self.pending?;
        let capped = pending.end.saturating_sub(pending.start) >= self.cap;
        if !capped && !eof && (active || now < pending.end.saturating_add(self.hold)) {
            return None;
        }
        let end = if capped {
            pending.start.saturating_add(self.cap).min(now)
        } else {
            pending.end.saturating_add(self.padding).min(now)
        };
        self.pending = None;
        self.emitted_end = end;
        Some(Crop {
            start: pending.start,
            end,
            emitted_at: now,
            capped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn region(id: u64, start: usize, end: usize, complete: bool) -> Region {
        Region {
            id,
            start,
            end,
            complete,
        }
    }

    #[test]
    fn no_positive_speech_never_creates_a_decode() {
        let mut s = Segmenter::new(1000);
        assert_eq!(s.observe(10_000, &[], false), None);
        assert_eq!(s.observe(12_000, &[], true), None);
    }

    #[test]
    fn short_pause_keeps_context_and_completed_events_are_deduplicated() {
        let mut s = Segmenter::new(1000);
        let first = region(1, 200, 1000, true);
        assert_eq!(s.observe(1000, &[first], false), None);
        assert_eq!(
            s.observe(1500, &[first, region(2, 1300, 1500, false)], false),
            None
        );
        let second = region(2, 1300, 2200, true);
        assert_eq!(s.observe(2200, &[first, second], false), None);
        let crop = s.observe(3000, &[first, second], false).unwrap();
        assert_eq!((crop.start, crop.end), (0, 2400));
        assert_eq!(s.observe(4000, &[first, second], true), None);
    }

    #[test]
    fn a_long_pause_closes_before_a_later_turn() {
        let mut s = Segmenter::new(1000);
        assert_eq!(s.observe(1000, &[region(1, 200, 1000, true)], false), None);
        let first = s.observe(1800, &[], false).unwrap();
        assert_eq!((first.start, first.end), (0, 1200));
        assert_eq!(
            s.observe(2500, &[region(2, 2100, 2500, false)], false),
            None
        );
        let second = s
            .observe(2600, &[region(2, 2100, 2600, true)], true)
            .unwrap();
        assert_eq!(second.start, 1900);
    }

    #[test]
    fn eof_and_padding_do_not_read_future_pcm() {
        let mut s = Segmenter::new(1000);
        let crop = s.observe(900, &[region(1, 50, 1200, true)], true).unwrap();
        assert_eq!((crop.start, crop.end, crop.emitted_at), (0, 900, 900));
    }

    #[test]
    fn cap_is_bounded_and_active_continuation_does_not_replay_audio() {
        let mut s = Segmenter::new(1000);
        let crop = s
            .observe(31_000, &[region(1, 0, 31_000, false)], false)
            .unwrap();
        assert!(crop.capped);
        assert_eq!((crop.start, crop.end), (0, 30_000));
        let next = s
            .observe(31_500, &[region(1, 0, 31_500, true)], true)
            .unwrap();
        assert_eq!(next.start, 30_000);
        assert_eq!(next.end, 31_500);
    }

    #[test]
    fn active_speech_never_commits_just_because_hold_elapsed() {
        let mut s = Segmenter::new(1000);
        assert_eq!(s.observe(1000, &[region(1, 200, 1000, false)], false), None);
        assert_eq!(s.observe(3000, &[region(1, 200, 3000, false)], false), None);
    }
}
