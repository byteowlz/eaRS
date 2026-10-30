//! Non-engine CLI paths. Preview modifiers never fall through to microphone startup.
use crate::{
    config::{OverlayFont, PreviewPlacement, TranscriptStyle, UiConfig},
    presentation,
};
use gpui::Application;
const OPTIONS: &[&str] = &[
    "--settings",
    "--no-overlay",
    "--preview-settings",
    "--preview-synthetic-expanded",
    "--preview-oqtopus-reduced",
    "--preview-oqtopus",
    "--preview-parametric-override",
    "--preview-parametric",
    "--preview-parametric-reduced",
    "--preview-floating",
    "--preview-click",
    "--preview-hit-test",
    "--preview-placement",
    "--preview-synthetic",
    "--preview-word",
    "--preview-subtitle",
    "--preview-cursor",
    "--preview-departure",
    "--preview-jetbrains",
    "--overlay-config",
    "--overlay-schema",
    "--overlay-font-check",
];

fn unknown(args: &[String]) -> Option<&str> {
    args.iter()
        .map(String::as_str)
        .find(|arg| !OPTIONS.contains(arg))
}
pub fn handle(args: &[String]) -> bool {
    if let Some(arg) = unknown(args) {
        eprintln!("ears-ui: unknown option {arg}; use --help");
        std::process::exit(2);
    }
    if args.iter().any(|a| a == "--overlay-config") {
        match UiConfig::load().and_then(|ui| Ok(serde_json::to_string_pretty(&ui.overlay)?)) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("ears-ui: {error:#}");
                std::process::exit(1);
            }
        }
        return true;
    }
    if args.iter().any(|a| a == "--overlay-schema") {
        println!("{}", include_str!("../../../examples/ui.schema.json"));
        return true;
    }
    if args.iter().any(|a| a == "--overlay-font-check") {
        check_fonts();
        return true;
    }
    false
}
fn check_fonts() {
    // No native event loop/window: this path must also terminate over SSH.
    let app = Application::headless();
    let system = app.text_system();
    let registered = system.add_fonts(presentation::bundled_fonts()).is_ok();
    let names = system.all_font_names();
    let departure = names.iter().any(|name| name == "Departure Mono");
    let jetbrains = names.iter().any(|name| name == "JetBrains Mono");
    println!(
        "{}",
        serde_json::json!({"registered":registered,"departure_mono":departure,"jetbrains_mono":jetbrains})
    );
    if !registered || !departure || !jetbrains {
        std::process::exit(1);
    }
}
fn selected<T: Copy>(args: &[String], choices: &[(&str, T)], default: T) -> T {
    choices
        .iter()
        .find(|(arg, _)| args.iter().any(|a| a == arg))
        .map_or(default, |(_, value)| *value)
}
pub struct PresentationArgs {
    pub transcript: TranscriptStyle,
    pub preview: PreviewPlacement,
    pub font: OverlayFont,
    pub animated: bool,
}
impl PresentationArgs {
    pub fn parse(args: &[String]) -> Self {
        let transcript = selected(
            args,
            &[("--preview-word", TranscriptStyle::Word)],
            TranscriptStyle::Current,
        );
        let preview = selected(
            args,
            &[
                ("--preview-cursor", PreviewPlacement::Cursor),
                ("--preview-subtitle", PreviewPlacement::Subtitle),
            ],
            PreviewPlacement::Inline,
        );
        Self {
            transcript,
            preview,
            font: selected(
                args,
                &[
                    ("--preview-departure", OverlayFont::DepartureMono),
                    ("--preview-jetbrains", OverlayFont::JetbrainsMono),
                ],
                OverlayFont::System,
            ),
            animated: transcript == TranscriptStyle::Word || preview != PreviewPlacement::Inline,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preview_modifiers_are_non_engine_options_and_unknown_flags_fail_closed() {
        for flag in [
            "--preview-departure",
            "--preview-word",
            "--preview-cursor",
            "--overlay-config",
        ] {
            assert!(unknown(&[flag.into()]).is_none());
        }
        assert_eq!(
            unknown(&["--preview-curser".into()]),
            Some("--preview-curser")
        );
        let p = PresentationArgs::parse(&["--preview-word".into(), "--preview-departure".into()]);
        assert_eq!(p.transcript, TranscriptStyle::Word);
        assert_eq!(p.preview, PreviewPlacement::Inline);
        assert_eq!(p.font, OverlayFont::DepartureMono);
    }
}
