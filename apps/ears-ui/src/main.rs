//! eaRS companion UI: floating dictation island, tray icon and settings.
//!
//! Everything about dictation happens in `ears::dictation` (the core crate);
//! this binary only renders state and forwards commands.

mod app;
mod config;
mod ears_mark;
mod indicator_theme;
mod motion_mark;
mod oqtopus;
mod overlay;
mod parametric_mark;
mod platform;
mod presentation;
mod runtime;
mod settings;
mod theme;
mod transcript_preview;
mod tray;
mod ui;
mod ui_cli;

use app::AppHandle;
use app::AppModel;
use config::UiConfig;
use ears::config::AppConfig;
use gpui::AppContext;
use gpui::Application;
use runtime::Runtime;

fn print_help() {
    println!(
        "ears-ui {}\n\nUsage: ears-ui [--settings] [--no-overlay] [--oqtopus-check] [--preview-synthetic|--preview-synthetic-expanded|--preview-placement|--preview-click|--preview-settings|--preview-floating|--preview-oqtopus|--preview-oqtopus-reduced|--preview-parametric|--preview-parametric-reduced|--preview-parametric-override] [--help]\n\n  --settings           open the settings window at launch\n  --no-overlay         start with the floating island disabled\n  --preview-synthetic  display a synthetic notch fixture without starting dictation\n  --preview-synthetic-expanded  show the expanded synthetic fixture\n  --preview-placement  show a synthetic fixture at the actual screen anchor\n  --preview-click     show a clickable placement fixture without dictation\n  --preview-hit-test  show a centered clickable synthetic notch for native input tests\n  --preview-settings   show settings with synthetic state, without dictation\n  --preview-floating   show a synthetic bottom-anchored floating pill\n  --preview-word       synthetic fixed-width latest-word fixture (no dictation)\n  --preview-subtitle   add synthetic passive subtitles\n  --preview-cursor     add a synthetic pointer-anchored bubble\n  --preview-departure / --preview-jetbrains  synthetic overlay font choice\n  --overlay-config     print current overlay settings as JSON, no engine\n  --overlay-schema     print the ui.toml JSON schema, no engine\n  --oqtopus-check     validate the configured local vector pack without dictation\n  --preview-oqtopus   show the local Oqtopus pack without dictation\n  --preview-oqtopus-reduced   show its static reduced-motion rest frame\n  --preview-parametric-override  animate a synthetic icon despite reduced motion\n  --preview-parametric  show animated parametric bars in a synthetic notch\n  --preview-parametric-reduced  show the same fixture with reduced motion\n  --version            print the version\n\nConfig: {}",
        env!("CARGO_PKG_VERSION"),
        UiConfig::path().display()
    );
}

struct PreviewOptions {
    expanded: bool,
    parametric: bool,
    reduced: bool,
    placement: bool,
    floating: bool,
    motion_override: bool,
    oqtopus: bool,
    clickable: bool,
    transcript: config::TranscriptStyle,
    preview: config::PreviewPlacement,
    font: config::OverlayFont,
}

fn preview_ui(options: &PreviewOptions) -> UiConfig {
    let mut ui = UiConfig::default();
    ui.overlay.indicator = if options.oqtopus {
        config::IndicatorStyle::Oqtopus
    } else if options.parametric {
        config::IndicatorStyle::ParametricEars
    } else {
        config::IndicatorStyle::Ears
    };
    ui.overlay.transcript = options.transcript;
    ui.overlay.preview = options.preview;
    ui.overlay.font = options.font;
    ui.overlay.show_details = options.expanded;
    ui.overlay.animation = options.parametric || options.oqtopus;
    ui.overlay.click_through = !options.clickable;
    ui.overlay.island_mode = options.placement && !options.floating;
    ui.overlay.animate_with_reduced_motion = options.motion_override;
    if options.floating {
        ui.overlay.anchor = config::OverlayAnchor::Bottom;
    }
    ui
}

/// Synthetic UI fixture: no engine, microphone, hotkeys, tray, or settings writes.
fn preview_synthetic(options: PreviewOptions) {
    use ears::dictation::{ConnectionState, DictationState};
    let ui = preview_ui(&options);
    Application::new().run(move |cx| {
        platform::set_accessory_policy();
        let mut fixture = AppModel::new(AppConfig::default(), ui);
        fixture.configuration_is_preview = true;
        fixture.state = DictationState::Listening;
        fixture.connection = ConnectionState::Connected;
        fixture.speech_active = true;
        fixture.partial = "Synthetic speech".into();
        fixture.reduce_motion = options.reduced;
        let model = cx.new(|_| fixture);
        cx.set_global(AppHandle(model.clone()));
        cx.set_global(Runtime::new(model.clone()));
        runtime::apply_theme(cx, None);
        cx.on_window_closed(|cx| cx.quit()).detach();
        if options.placement || options.floating {
            let overlay_config = model.read(cx).ui.overlay.clone();
            overlay::open(cx, &overlay_config, model.clone());
        } else {
            overlay::open_preview(cx, model.clone());
        }
        if options.transcript == config::TranscriptStyle::Word
            || options.preview != config::PreviewPlacement::Inline
        {
            animate_preview(model, cx);
        }
    });
}

/// Display-only sample clock for the explicitly synthetic native fixture.
fn preview_step(model: &mut AppModel, step: u32) {
    let words = [
        "Quietly", "keep", "each", "word", "clear", "and", "leave", "room", "to", "breathe.",
    ];
    let i = (step / 3) as usize;
    if i < words.len() {
        model.partial = words[..=i].join(" ");
    }
    let quiet_ms = step
        .saturating_sub((words.len() * 3) as u32)
        .saturating_mul(128)
        .min(3000);
    model.idle_progress = Some(ears::dictation::AutoPauseProgress {
        capture_generation: 0,
        quiet_ms,
        idle_ms: 3000,
        held: false,
        scored_at: std::time::Instant::now(),
    });
}
fn animate_preview(model: gpui::Entity<AppModel>, cx: &mut gpui::App) {
    cx.spawn(async move |cx| {
        let mut step = 0_u32;
        loop {
            if cx
                .update(|cx| {
                    model.update(cx, |m, cx| {
                        preview_step(m, step);
                        cx.notify();
                    });
                    transcript_preview::sync(cx);
                })
                .is_err()
            {
                break;
            }
            step = (step + 1) % 55;
            gpui::Timer::after(std::time::Duration::from_millis(128)).await;
        }
    })
    .detach();
}

/// Render settings with synthetic state and without starting dictation.
fn preview_settings() {
    Application::new().run(|cx| {
        platform::set_accessory_policy();
        let model = cx.new(|_| {
            let mut model = AppModel::new(AppConfig::default(), UiConfig::default());
            model.configuration_is_preview = true;
            model
        });
        cx.set_global(AppHandle(model.clone()));
        cx.set_global(Runtime::new(model));
        runtime::apply_theme(cx, None);
        cx.on_window_closed(|cx| cx.quit()).detach();
        runtime::open_settings(cx);
    });
}

fn maybe_preview(args: &[String]) -> bool {
    if args.iter().any(|a| a == "--preview-settings") {
        preview_settings();
        return true;
    }
    let expanded = args.iter().any(|a| a == "--preview-synthetic-expanded");
    let oqtopus_reduced = args.iter().any(|a| a == "--preview-oqtopus-reduced");
    let oqtopus = oqtopus_reduced || args.iter().any(|a| a == "--preview-oqtopus");
    let motion_override = args.iter().any(|a| a == "--preview-parametric-override");
    let presentation = ui_cli::PresentationArgs::parse(args);
    let animated = motion_override
        || presentation.animated
        || args.iter().any(|a| a == "--preview-parametric");
    let reduced = motion_override || args.iter().any(|a| a == "--preview-parametric-reduced");
    let floating = args.iter().any(|a| a == "--preview-floating");
    let anchored_click = args.iter().any(|a| a == "--preview-click");
    let hit_test = args.iter().any(|a| a == "--preview-hit-test");
    let clickable = anchored_click || hit_test;
    let placement = anchored_click || args.iter().any(|a| a == "--preview-placement");
    if args.iter().any(|a| a.starts_with("--preview-")) {
        preview_synthetic(PreviewOptions {
            expanded,
            parametric: animated || reduced,
            reduced: oqtopus_reduced || (!oqtopus && (reduced || !animated)),
            placement,
            floating,
            motion_override,
            oqtopus,
            clickable,
            transcript: presentation.transcript,
            preview: presentation.preview,
            font: presentation.font,
        });
        return true;
    }
    false
}

fn check_oqtopus() {
    let result = oqtopus::pack_path()
        .ok_or_else(|| "no data directory available".to_string())
        .and_then(|path| oqtopus::check(&path));
    match result {
        Ok(clips) => println!("{}", serde_json::json!({"ok": true, "clips": clips})),
        Err(error) => {
            eprintln!("{}", serde_json::json!({"ok": false, "error": error}));
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("ears-ui {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.iter().any(|a| a == "--oqtopus-check") {
        check_oqtopus();
        return;
    }
    if ui_cli::handle(&args) {
        return;
    }
    if maybe_preview(&args) {
        return;
    }
    let open_settings_at_launch = args.iter().any(|a| a == "--settings");
    let no_overlay = args.iter().any(|a| a == "--no-overlay");

    let core = match AppConfig::load() {
        Ok(c) => c,
        Err(err) => {
            eprintln!("ears-ui: cannot load config.toml: {err:#}; using defaults");
            AppConfig::default()
        }
    };
    let mut ui = match UiConfig::load() {
        Ok(c) => c,
        Err(err) => {
            eprintln!("ears-ui: cannot load ui.toml: {err:#}; using defaults");
            UiConfig::default()
        }
    };
    if no_overlay {
        ui.overlay.enabled = false;
    }

    Application::new().run(move |cx| {
        eprintln!("ears-ui: app launched");
        platform::set_accessory_policy();

        let mut model = AppModel::new(core, ui);
        model.reduce_motion = platform::reduce_motion();
        let model = cx.new(|_| model);
        cx.set_global(AppHandle(model.clone()));
        cx.set_global(Runtime::new(model.clone()));
        runtime::apply_theme(cx, None);

        cx.on_window_closed(|cx| runtime::on_window_closed(cx))
            .detach();

        tray::install(cx);
        eprintln!("ears-ui: tray installed");
        runtime::start_engine(cx);
        eprintln!("ears-ui: engine started");
        runtime::spawn_tick_loop(cx);

        if open_settings_at_launch {
            runtime::open_settings(cx);
        }
    });
}
