//! Additive display controls; these never change capture or insertion settings.
use super::*;
use crate::config::{OverlayFont, PreviewPlacement, TranscriptStyle};

impl SettingsView {
    pub(super) fn transcription_section(&self, theme: &Theme, cx: &Context<Self>) -> Div {
        let overlay = &self.model.read(cx).ui.overlay;
        let entity = cx.entity();
        let mut section=section("Transcript presentation",theme)
            .child(overlay_choice("transcript-style","Top text","Current preserves the existing pill. Word adds a fixed-width black alternative.",vec![(TranscriptStyle::Current,"Current".into()),(TranscriptStyle::Word,"Word".into()),(TranscriptStyle::Meter,"Ears only".into()),(TranscriptStyle::Hidden,"Hidden".into())],overlay.transcript,entity.clone(),theme,|o,v|o.transcript=v))
            .child(overlay_choice("preview-placement","Full-text preview","Passive provisional text, never an insertion target. Near pointer anchors once per capture, not at the caret.",vec![(PreviewPlacement::Inline,"Inline only".into()),(PreviewPlacement::Subtitle,"Subtitles".into()),(PreviewPlacement::Cursor,"Near pointer".into())],overlay.preview,entity.clone(),theme,|o,v|o.preview=v))
            .child(overlay_choice("overlay-font","Typeface","Bundled fonts; no download. Applies only to overlay text.",vec![(OverlayFont::System,"System".into()),(OverlayFont::DepartureMono,"Departure".into()),(OverlayFont::JetbrainsMono,"JetBrains".into())],overlay.font,entity.clone(),theme,|o,v|o.font=v));
        if !crate::platform::passive_previews_supported() {
            section=section.child(hint("Passive preview windows are unavailable on this platform; inline text remains available.",theme));
        }
        if overlay.transcript == TranscriptStyle::Word {
            let mut widths = vec![260.0, 300.0, 360.0, 420.0];
            if !widths.contains(&overlay.word_width) {
                widths.push(overlay.word_width);
            }
            section=section
                .child(overlay_choice("word-width","Word pill width","Floating word mode only. Camera-notch dimensions remain hardware-safe.",widths.into_iter().map(|v|(v,format!("{v:.0} px").into())).collect(),overlay.word_width,entity.clone(),theme,|o,v|o.word_width=v))
                .child(overlay_choice("word-size","Word size","Maximum size. Exceptionally long words shrink or ellipsize instead of widening the pill.",vec![(12.0,"12".into()),(14.0,"14".into()),(18.0,"18".into()),(22.0,"22".into()),(24.0,"24".into())],overlay.word_size,entity.clone(),theme,|o,v|o.word_size=v))
                .child(overlay_choice("word-fade","Quiet fade","Uses actual local VAD silence. Holds, stale/unavailable scores and reduced motion keep text visible.",vec![(true,"On".into()),(false,"Off".into())],overlay.word_fade,entity,theme,|o,v|o.word_fade=v));
        }
        section
    }
}
