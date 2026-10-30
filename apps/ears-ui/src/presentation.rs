//! Display-only transcript policy. No capture, insertion or routing effects.
use crate::config::{OverlayConfig, OverlayFont, PreviewPlacement};
use gpui::{App, Global, Hsla, SharedString, TextRun, Window, font, px};
use std::{borrow::Cow, time::Duration};

struct FontsRegistered(bool);
impl Global for FontsRegistered {}

pub fn bundled_fonts() -> Vec<Cow<'static, [u8]>> {
    vec![
        Cow::Borrowed(include_bytes!("../assets/fonts/DepartureMono-Regular.otf")),
        Cow::Borrowed(include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf")),
    ]
}

pub fn register_fonts(cx: &mut App) -> bool {
    if cx.has_global::<FontsRegistered>() {
        return cx.global::<FontsRegistered>().0;
    }
    let result = cx.text_system().add_fonts(bundled_fonts());
    let registered = result.is_ok();
    if let Err(error) = result {
        eprintln!("ears-ui: overlay fonts unavailable: {error:#}");
    }
    cx.set_global(FontsRegistered(registered));
    registered
}

pub fn latest_word(text: &str) -> &str {
    text.split_whitespace().last().unwrap_or("")
}

/// Last words are bounded in memory/layout, not sent anywhere. This is not ASR text mutation.
pub fn preview_text(text: &str) -> String {
    let mut chars: Vec<_> = text.trim().chars().rev().take(384).collect();
    chars.reverse();
    chars.into_iter().collect()
}

pub fn expire_progress(
    progress: &mut Option<ears::dictation::AutoPauseProgress>,
    generation: Option<u64>,
    now: std::time::Instant,
) -> bool {
    let stale = progress.as_ref().is_some_and(|p| {
        Some(p.capture_generation) != generation
            || now
                .checked_duration_since(p.scored_at)
                .is_none_or(|age| age > Duration::from_millis(500))
    });
    if stale {
        *progress = None;
    }
    stale
}

pub fn word_label_opacity(phase: crate::app::Phase, quiet: f32) -> f32 {
    if phase == crate::app::Phase::Listening {
        quiet
    } else {
        1.0
    }
}

pub fn quiet_opacity(quiet_ms: u32, idle_ms: u32, held: bool, age: Duration, enabled: bool) -> f32 {
    if !enabled || held || idle_ms == 0 || age > Duration::from_millis(500) {
        return 1.0;
    }
    (1.0 - quiet_ms as f32 / idle_ms as f32).clamp(0.05, 1.0)
}

pub fn measure(window: &Window, text: &str, family: OverlayFont, size: f32, color: Hsla) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let run = TextRun {
        len: text.len(),
        font: font(family.family()),
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    f32::from(
        window
            .text_system()
            .shape_line(SharedString::from(text.to_owned()), px(size), &[run], None)
            .width,
    )
}

/// Match rendered font when keeping the newest text in the incumbent single line.
pub fn tail(
    window: &Window,
    text: &str,
    width: f32,
    family: OverlayFont,
    size: f32,
    color: Hsla,
) -> String {
    let text = text.trim();
    if measure(window, text, family, size, color) <= width {
        return text.to_owned();
    }
    let chars: Vec<_> = text.chars().collect();
    let (mut lo, mut hi) = (0, chars.len());
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        let candidate: String = std::iter::once('…')
            .chain(chars[chars.len() - mid..].iter().copied())
            .collect();
        if measure(window, &candidate, family, size, color) <= width {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    std::iter::once('…')
        .chain(chars[chars.len() - lo..].iter().copied())
        .collect()
}

/// Fixed cap; exceptionally long words shrink, never enlarge the selected font.
pub fn word_font_size(
    window: &Window,
    text: &str,
    width: f32,
    overlay: &OverlayConfig,
    color: Hsla,
) -> f32 {
    let maximum = overlay.word_size;
    let natural = measure(window, text, overlay.font, maximum, color);
    if natural <= width || natural <= 0.0 {
        return maximum;
    }
    (maximum * width / natural).clamp(10.0, maximum)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub align_bottom: bool,
}
impl PreviewRect {
    pub fn body_top(self, height: f32) -> f32 {
        if self.align_bottom {
            self.y + self.height - height
        } else {
            self.y
        }
    }
}

/// The renderer and the headless probe use the same native font shaper and cap.
pub fn preview_height(
    system: &gpui::WindowTextSystem,
    text: &str,
    family: OverlayFont,
    size: f32,
    width: f32,
) -> f32 {
    let run = TextRun {
        len: text.len(),
        font: font(family.family()),
        color: Hsla::default(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let line_height = size * 1.4;
    let height = system
        .shape_text(
            SharedString::from(text.to_owned()),
            px(size),
            &[run],
            Some(px((width - 40.0).max(1.0))),
            None,
        )
        .map(|lines| {
            lines
                .iter()
                .map(|line| f32::from(line.size(px(line_height)).height))
                .sum::<f32>()
        })
        .unwrap_or(line_height);
    height.clamp(line_height, line_height * 3.0) + 28.0
}

pub fn preview_rect(
    screen: (f32, f32),
    placement: PreviewPlacement,
    cursor: Option<(f32, f32)>,
) -> Option<PreviewRect> {
    let (sw, sh) = screen;
    if !sw.is_finite()
        || !sh.is_finite()
        || sw < 160.0
        || sh < 160.0
        || placement == PreviewPlacement::Inline
    {
        return None;
    }
    let width = (if placement == PreviewPlacement::Cursor {
        370.0_f32
    } else {
        560.0_f32
    })
    .min(sw - 32.0);
    let height = 156.0_f32.min(sh - 32.0);
    let (x, y, align_bottom) = match placement {
        PreviewPlacement::Subtitle => ((sw - width) / 2.0, (sh - height - 72.0).max(16.0), true),
        PreviewPlacement::Cursor => {
            let (cx, cy) = cursor?;
            if !cx.is_finite() || !cy.is_finite() {
                return None;
            }
            let x = if cx + 24.0 + width > sw - 16.0 {
                cx - width - 20.0
            } else {
                cx + 24.0
            };
            let above = cy + 28.0 + height > sh - 16.0;
            let y = if above { cy - height - 28.0 } else { cy + 28.0 };
            (
                x.clamp(16.0, sw - width - 16.0),
                y.clamp(16.0, sh - height - 16.0),
                above,
            )
        }
        PreviewPlacement::Inline => return None,
    };
    Some(PreviewRect {
        x,
        y,
        width,
        height,
        align_bottom,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn words_preserve_unicode_and_punctuation() {
        assert_eq!(latest_word(" Hallo\nWelt! "), "Welt!");
        assert_eq!(latest_word("stay 👨‍👩‍👧‍👦"), "👨‍👩‍👧‍👦");
        assert_eq!(latest_word("\u{2003}"), "");
        assert_eq!(
            latest_word("noch Aufmerksamkeitssteuerung"),
            "Aufmerksamkeitssteuerung"
        );
    }
    #[test]
    fn producer_age_expires_once_and_status_messages_never_fade() {
        use crate::app::Phase;
        let scored_at = std::time::Instant::now();
        let p = ears::dictation::AutoPauseProgress {
            capture_generation: 7,
            quiet_ms: 1500,
            idle_ms: 3000,
            held: false,
            scored_at,
        };
        let mut stored = Some(p);
        assert!(!expire_progress(
            &mut stored,
            Some(7),
            scored_at + Duration::from_millis(499)
        ));
        assert!(expire_progress(
            &mut stored,
            Some(7),
            scored_at + Duration::from_millis(501)
        ));
        assert!(stored.is_none());
        assert!(!expire_progress(
            &mut stored,
            Some(7),
            scored_at + Duration::from_secs(2)
        ));
        assert!(expire_progress(&mut Some(p), Some(8), scored_at));
        assert!(expire_progress(
            &mut Some(p),
            Some(7),
            scored_at - Duration::from_millis(1)
        ));
        for phase in [
            Phase::Error,
            Phase::Armed,
            Phase::Finalizing,
            Phase::Dormant,
        ] {
            assert_eq!(word_label_opacity(phase, 0.05), 1.0);
        }
        assert_eq!(word_label_opacity(Phase::Listening, 0.05), 0.05);
    }

    #[test]
    fn pointer_gap_is_to_the_painted_body_for_short_and_multiline_captions() {
        for height in [53.2, 103.6] {
            let below = preview_rect(
                (1920.0, 1080.0),
                PreviewPlacement::Cursor,
                Some((400.0, 200.0)),
            )
            .unwrap();
            assert_eq!(below.body_top(height) - 200.0, 28.0);
            let above = preview_rect(
                (1920.0, 1080.0),
                PreviewPlacement::Cursor,
                Some((400.0, 1000.0)),
            )
            .unwrap();
            assert!((1000.0 - (above.body_top(height) + height) - 28.0).abs() < 0.01);
        }
    }

    #[test]
    fn actual_quiet_progress_never_invents_silence_or_ignores_holds() {
        assert_eq!(quiet_opacity(1500, 3000, false, Duration::ZERO, true), 0.5);
        assert_eq!(quiet_opacity(3000, 3000, true, Duration::ZERO, true), 1.0);
        assert_eq!(
            quiet_opacity(3000, 3000, false, Duration::from_millis(501), true),
            1.0
        );
        assert_eq!(quiet_opacity(3000, 0, false, Duration::ZERO, true), 1.0);
        assert_eq!(quiet_opacity(3000, 3000, false, Duration::ZERO, false), 1.0);
    }
    #[test]
    fn cursor_bounds_are_clamped_without_tracking_the_pointer() {
        for p in [(0.0, 0.0), (1920.0, 1080.0), (1910.0, 10.0), (-40.0, 600.0)] {
            let r = preview_rect((1920.0, 1080.0), PreviewPlacement::Cursor, Some(p)).unwrap();
            assert!(
                r.x >= 16.0 && r.y >= 16.0 && r.x + r.width <= 1904.0 && r.y + r.height <= 1064.0
            );
        }
        assert!(preview_rect((1920.0, 1080.0), PreviewPlacement::Cursor, None).is_none());
        assert!(preview_rect((100.0, 100.0), PreviewPlacement::Subtitle, None).is_none());
        assert!(preview_rect((f32::NAN, 1080.0), PreviewPlacement::Subtitle, None).is_none());
    }
    #[test]
    fn subtitle_is_lifted_and_text_is_bounded_without_utf8_damage() {
        let r = preview_rect((1920.0, 1080.0), PreviewPlacement::Subtitle, None).unwrap();
        assert_eq!(r.x, 680.0);
        assert_eq!(r.y + r.height, 1008.0);
        assert_eq!(preview_text(&"ä".repeat(500)).chars().count(), 384);
    }
}
