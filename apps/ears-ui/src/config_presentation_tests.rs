//! Configuration/schema coverage for additive transcript display choices.
use super::*;

#[test]
fn presentation_matrix_and_schema_agree_without_changing_legacy_defaults() {
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../../../examples/ui.schema.json")).unwrap();
    for transcript in [
        TranscriptStyle::Current,
        TranscriptStyle::Word,
        TranscriptStyle::Meter,
        TranscriptStyle::Hidden,
    ] {
        for preview in [
            PreviewPlacement::Inline,
            PreviewPlacement::Subtitle,
            PreviewPlacement::Cursor,
        ] {
            for font in [
                OverlayFont::System,
                OverlayFont::DepartureMono,
                OverlayFont::JetbrainsMono,
            ] {
                let mut config = UiConfig::default();
                config.overlay.transcript = transcript;
                config.overlay.preview = preview;
                config.overlay.font = font;
                let decoded: UiConfig = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
                assert_eq!(
                    (
                        decoded.overlay.transcript,
                        decoded.overlay.preview,
                        decoded.overlay.font
                    ),
                    (transcript, preview, font)
                );
                let json = serde_json::to_value(&config.overlay).unwrap();
                for key in ["transcript", "preview", "font"] {
                    assert!(
                        schema["properties"]["overlay"]["properties"][key]["enum"]
                            .as_array()
                            .unwrap()
                            .contains(&json[key])
                    );
                }
            }
        }
    }
    let mut invalid = UiConfig::default();
    invalid.overlay.word_width = f32::NAN;
    invalid.overlay.word_size = f32::INFINITY;
    invalid.sanitize();
    assert_eq!(invalid.overlay.word_width, 300.0);
    assert_eq!(invalid.overlay.word_size, 18.0);
    invalid.overlay.word_width = 10000.0;
    invalid.overlay.word_size = 0.0;
    invalid.sanitize();
    assert_eq!(invalid.overlay.word_width, 440.0);
    assert_eq!(invalid.overlay.word_size, 12.0);
    assert!(toml::from_str::<UiConfig>("[overlay]\ntranscript='unknown'").is_err());
}

#[test]
fn new_transcript_options_survive_configuration_roundtrip() {
    let config: UiConfig = toml::from_str("[overlay]\ntranscript = 'word'\npreview = 'subtitle'\nfont = 'departure_mono'\nword_width = 320.0\nword_size = 18.0\n").unwrap();
    let value = toml::Value::try_from(&config).unwrap();
    assert_eq!(value["overlay"]["transcript"].as_str(), Some("word"));
    assert_eq!(value["overlay"]["preview"].as_str(), Some("subtitle"));
    assert_eq!(value["overlay"]["font"].as_str(), Some("departure_mono"));
    assert_eq!(value["overlay"]["word_width"].as_float(), Some(320.0));
}
