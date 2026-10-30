//! Additive display controls; these never change capture or insertion settings.
use super::*;
use crate::config::{OverlayAppearance, OverlayFont, PreviewPlacement, TranscriptStyle};

impl SettingsView {
    pub(super) fn transcription_section(&self, theme: &Theme, cx: &Context<Self>) -> Div {
        let overlay = &self.model.read(cx).ui.overlay;
        let entity = cx.entity();
        let mut section=section("Transcript presentation",theme)
            .child(overlay_choice("pill-appearance","Pill appearance","Independent of text. Compact floating pills stay Classic; hardware notches stay black.",vec![(OverlayAppearance::Classic,"Classic".into()),(OverlayAppearance::Black,"Black".into())],overlay.appearance,entity.clone(),theme,|o,v|o.appearance=v))
            .child(overlay_choice("transcript-style","Top text","Choose phrase, centered word, indicator only, or hide the top pill.",vec![(TranscriptStyle::Current,"Phrase".into()),(TranscriptStyle::Word,"Word".into()),(TranscriptStyle::Meter,"Ears only".into()),(TranscriptStyle::Hidden,"Hidden".into())],overlay.transcript,entity.clone(),theme,|o,v|o.transcript=v))
            .child(overlay_choice("preview-placement","Full-text preview","Passive provisional text, never an insertion target. Near pointer follows the mouse; it is not an insertion target or caret lookup.",vec![(PreviewPlacement::Inline,"Inline only".into()),(PreviewPlacement::Subtitle,"Subtitles".into()),(PreviewPlacement::Cursor,"Near pointer".into())],overlay.preview,entity.clone(),theme,|o,v|o.preview=v))
            .child(overlay_choice("overlay-font","Typeface","Bundled fonts; no download. Applies only to overlay text.",vec![(OverlayFont::System,"System".into()),(OverlayFont::DepartureMono,"Departure".into()),(OverlayFont::JetbrainsMono,"JetBrains".into())],overlay.font,entity.clone(),theme,|o,v|o.font=v));
        if overlay.preview != PreviewPlacement::Inline {
            section = section.child(hint(self.model.read(cx).preview_status.label(), theme));
        }
        if !crate::platform::passive_previews_supported() {
            section=section.child(hint("Passive preview windows are unavailable on this platform; inline text remains available.",theme));
        }
        if overlay.transcript == TranscriptStyle::Word {
            let mut widths = vec![220.0, 260.0, 300.0, 360.0, 420.0];
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
