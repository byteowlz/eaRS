//! eaRS companion UI: floating dictation island, tray icon and settings.
//!
//! Everything about dictation happens in `ears::dictation` (the core crate);
//! this binary only renders state and forwards commands.

mod app;
mod bootstrap;
mod config;
mod ears_mark;
mod indicator_theme;
mod motion_mark;
mod oqtopus;
mod overlay;
#[cfg_attr(
    not(feature = "experimental-overlays"),
    path = "accepted_parametric_mark.rs"
)]
mod parametric_mark;
mod platform;
mod presentation;
mod preview_proof;
mod recording_notch;
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
        "ears-ui {}\n\nUsage: ears-ui [--settings] [--no-overlay] [--oqtopus-check] [--preview-synthetic|--preview-synthetic-expanded|--preview-placement|--preview-click|--preview-settings|--preview-floating|--preview-oqtopus|--preview-oqtopus-reduced|--preview-parametric|--preview-parametric-reduced|--preview-parametric-override] [--help]\n\n  --settings           open the settings window at launch\n  --setup              configure without starting the engine until Apply\n  --build-info         read-only compiled capabilities JSON\n  --input-devices      input-device metadata JSON; no recording\n  --no-overlay         start with the floating island disabled\n  --preview-synthetic  display a synthetic notch fixture without starting dictation\n  --preview-synthetic-expanded  show the expanded synthetic fixture\n  --preview-placement  show a synthetic fixture at the actual screen anchor\n  --preview-click     show a clickable placement fixture without dictation\n  --preview-hit-test  show a centered clickable synthetic notch for native input tests\n  --preview-settings   show settings with synthetic state, without dictation\n  --preview-floating   show a synthetic bottom-anchored floating pill\n  --preview-word       synthetic fixed-width latest-word fixture (no dictation)\n  --preview-black / --preview-compact  synthetic skin/compact state\n  --preview-layout-report  synthetic numeric layout evidence, never text/pixels\n  --preview-settings-minimum / --preview-settings-light / --preview-settings-wide  synthetic layout variants\n  --preview-settings-appearance  synthetic Overlay tab scrolled to Appearance\n  --preview-settings-voice  synthetic Dictation tab scrolled to auto-pause and wake sliders\n  --preview-settings-mic-empty / --preview-settings-mic-missing  synthetic microphone states\n  --preview-subtitle   add synthetic passive subtitles\n  --preview-cursor     add a synthetic pointer-anchored bubble\n  --preview-recording-notch / --preview-recording-notch-active  passive recording silhouette fixtures; no engine\n  --preview-departure / --preview-jetbrains  synthetic overlay font choice\n  --overlay-config     print current overlay settings as JSON, no engine\n  --overlay-schema     print the ui.toml JSON schema, no engine\n  --oqtopus-check     validate the configured local vector pack without dictation\n  --preview-oqtopus   show the local Oqtopus pack without dictation\n  --preview-oqtopus-reduced   show its static reduced-motion rest frame\n  --preview-parametric-override  animate a synthetic icon despite reduced motion\n  --preview-parametric  show animated parametric bars in a synthetic notch\n  --preview-parametric-reduced  show the same fixture with reduced motion\n  --version            print the version\n\nConfig: {}",
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
    appearance: config::OverlayAppearance,
    compact: bool,
    recording_notch: bool,
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
    ui.overlay.appearance = options.appearance;
    ui.overlay.show_details = options.expanded;
    ui.overlay.recording_notch = options.recording_notch;
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
    if options.recording_notch && !platform::passive_previews_supported() {
        eprintln!("ears-ui: recording notch fixtures require macOS");
        std::process::exit(2);
    }
    let ui = preview_ui(&options);
    Application::new().run(move |cx| {
        platform::set_accessory_policy();
        let mut fixture = AppModel::new(AppConfig::default(), ui);
        fixture.configuration_is_preview = true;
        fixture.state = if options.compact {
            DictationState::Suspended
        } else {
            DictationState::Listening
        };
        fixture.connection = ConnectionState::Connected;
        fixture.speech_active = true;
        fixture.partial = "Synthetic speech".into();
        fixture.reduce_motion = options.reduced;
        let model = cx.new(|_| fixture);
        cx.set_global(AppHandle(model.clone()));
        cx.set_global(Runtime::new(model.clone()));
        runtime::apply_theme(cx, None);
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        open_synthetic_overlay(&options, model.clone(), cx);
        if options.transcript == config::TranscriptStyle::Word
            || options.preview != config::PreviewPlacement::Inline
        {
            animate_preview(model, cx);
        }
    });
}

fn open_synthetic_overlay(
    options: &PreviewOptions,
    model: gpui::Entity<AppModel>,
    cx: &mut gpui::App,
) {
    let config = model.read(cx).ui.overlay.clone();
    if options.recording_notch {
        recording_notch::sync(&config, cx);
        if !options.compact {
            overlay::open(cx, &config, model);
        }
    } else if options.placement || options.floating {
        overlay::open(cx, &config, model);
    } else {
        overlay::open_preview(cx, model);
    }
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
fn preview_frame(model: &gpui::Entity<AppModel>, frame: u32, cx: &mut gpui::App) {
    if frame % 4 == 0 {
        model.update(cx, |m, cx| {
            preview_step(m, (frame / 4) % 55);
            cx.notify();
        });
    }
    transcript_preview::sync(cx);
}

fn animate_preview(model: gpui::Entity<AppModel>, cx: &mut gpui::App) {
    cx.spawn(async move |cx| {
        let mut frame = 0_u32;
        loop {
            if cx
                .update(|cx| {
                    preview_frame(&model, frame, cx);
                })
                .is_err()
            {
                break;
            }
            frame = frame.wrapping_add(1);
            gpui::Timer::after(std::time::Duration::from_millis(33)).await;
        }
    })
    .detach();
}

fn settings_fixture(light: bool) -> AppModel {
    let mut ui = UiConfig::default();
    if light {
        ui.theme.scheme = "oqto-light".into();
    }
    let mut core = AppConfig::default();
    for alias in [
        "Fixture A",
        "Fixture B",
        "A deliberately long synthetic server alias",
    ] {
        core.dictation.servers.insert(
            alias.into(),
            ears::config::DictationServerConfig {
                host: "127.0.0.1".into(),
                port: 1,
                description: None,
            },
        );
    }
    let mut model = AppModel::new(core, ui);
    model.configuration_is_preview = true;
    model
}

fn settings_fixture_size(args: &[String]) -> Option<gpui::Size<gpui::Pixels>> {
    let dimensions = if args.iter().any(|arg| arg == "--preview-settings-minimum") {
        (640.0, 420.0)
    } else if args.iter().any(|arg| arg == "--preview-settings-wide") {
        (1120.0, 780.0)
    } else {
        return None;
    };
    Some(gpui::size(gpui::px(dimensions.0), gpui::px(dimensions.1)))
}

/// Render settings with synthetic state and without starting dictation.
/// Synthetic Settings state for one preview variant; never starts the engine.
fn preview_settings_fixture(args: &[String]) -> AppModel {
    let has = |flag: &str| args.iter().any(|arg| arg == flag);
    let mut fixture = settings_fixture(has("--preview-settings-light"));
    if has("--preview-settings-mic-missing") {
        fixture.core.dictation.input_device = Some("Disconnected microphone".into());
    }
    if has("--preview-settings-appearance") {
        fixture.settings_tab = settings::Tab::Overlay;
    }
    if has("--preview-settings-voice") {
        // The preview never starts the engine, so this command never runs.
        let wake = &mut fixture.core.dictation.wake;
        wake.enabled = true;
        wake.command = [
            "waqy",
            "listen",
            "--manifest",
            "synthetic",
            "--report-above",
            "0.3",
        ]
        .map(String::from)
        .into();
        wake.threshold = Some(0.42);
        fixture.core.dictation.auto_pause.idle_ms = 3000;
    }
    fixture
}

fn preview_settings(args: &[String]) {
    let viewport = settings_fixture_size(args);
    let scroll_to_end = ["--preview-settings-appearance", "--preview-settings-voice"]
        .iter()
        .any(|flag| args.iter().any(|arg| arg == flag));
    let fixture = preview_settings_fixture(args);
    Application::new().run(move |cx| {
        platform::set_accessory_policy();
        let model = cx.new(|_| fixture);
        cx.set_global(AppHandle(model.clone()));
        cx.set_global(Runtime::new(model));
        runtime::apply_theme(cx, None);
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        runtime::open_settings(cx);
        if let (Some(viewport), Some(handle)) = (viewport, cx.global::<Runtime>().settings) {
            let _ = handle.update(cx, |_, window, _| window.resize(viewport));
        }
        if let (true, Some(handle)) = (scroll_to_end, cx.global::<Runtime>().settings) {
            let _ = handle.update(cx, |view, _, _| view.preview_scroll_to_bottom());
        }
    });
}

fn maybe_preview(args: &[String]) -> bool {
    if args.iter().any(|a| {
        matches!(
            a.as_str(),
            "--preview-settings"
                | "--preview-settings-mic-empty"
                | "--preview-settings-mic-missing"
                | "--preview-settings-appearance"
                | "--preview-settings-voice"
                | "--preview-settings-minimum"
                | "--preview-settings-light"
                | "--preview-settings-wide"
        )
    }) {
        preview_settings(args);
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
            appearance: presentation.appearance,
            compact: args.iter().any(|arg| arg == "--preview-compact")
                || (args.iter().any(|arg| arg == "--preview-recording-notch")
                    && !args
                        .iter()
                        .any(|arg| arg == "--preview-recording-notch-active")),
            recording_notch: args.iter().any(|arg| {
                matches!(
                    arg.as_str(),
                    "--preview-recording-notch" | "--preview-recording-notch-active"
                )
            }),
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
    let setup_mode = match AppConfig::path()
        .and_then(|path| bootstrap::setup_required(&path, args.iter().any(|a| a == "--setup")))
    {
        Ok(required) => required,
        Err(error) => {
            eprintln!("ears-ui: cannot preserve setup state: {error:#}");
            std::process::exit(1);
        }
    };
    let open_settings_at_launch = setup_mode || args.iter().any(|a| a == "--settings");
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

    // The Linux tray backend needs GTK initialized on the main thread first.
    let tray_ready = tray::prepare();
    Application::new().run(move |cx| {
        eprintln!("ears-ui: app launched");
        platform::set_accessory_policy();

        let mut model = AppModel::new(core, ui);
        model.reduce_motion = platform::reduce_motion();
        model.setup_mode = setup_mode;
        let model = cx.new(|_| model);
        cx.set_global(AppHandle(model.clone()));
        cx.set_global(Runtime::new(model.clone()));
        runtime::apply_theme(cx, None);

        cx.on_window_closed(|cx| runtime::on_window_closed(cx))
            .detach();

        if tray_ready {
            tray::install(cx);
        }
        eprintln!(
            "ears-ui: tray {}",
            if cx.has_global::<tray::TrayState>() {
                "installed"
            } else {
                "unavailable"
            }
        );
        if !setup_mode {
            runtime::start_engine(cx);
            eprintln!("ears-ui: engine started");
        } else {
            eprintln!("ears-ui: setup mode; engine not started");
        }
        runtime::spawn_tick_loop(cx);

        if open_settings_at_launch {
            runtime::open_settings(cx);
        }
    });
}
