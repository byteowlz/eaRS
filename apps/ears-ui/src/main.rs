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
mod runtime;
mod settings;
mod theme;
mod tray;
mod ui;

use app::AppHandle;
use app::AppModel;
use config::UiConfig;
use ears::config::AppConfig;
use gpui::AppContext;
use gpui::Application;
use runtime::Runtime;

fn print_help() {
    println!(
        "ears-ui {}\n\nUsage: ears-ui [--settings] [--no-overlay] [--oqtopus-check] [--preview-synthetic|--preview-synthetic-expanded|--preview-placement|--preview-click|--preview-settings|--preview-floating|--preview-oqtopus|--preview-oqtopus-reduced|--preview-parametric|--preview-parametric-reduced|--preview-parametric-override] [--help]\n\n  --settings           open the settings window at launch\n  --no-overlay         start with the floating island disabled\n  --preview-synthetic  display a synthetic notch fixture without starting dictation\n  --preview-synthetic-expanded  show the expanded synthetic fixture\n  --preview-placement  show a synthetic fixture at the actual screen anchor\n  --preview-click     show a clickable placement fixture without dictation\n  --preview-hit-test  show a centered clickable synthetic notch for native input tests\n  --preview-settings   show settings with synthetic state, without dictation\n  --preview-floating   show a synthetic bottom-anchored floating pill\n  --oqtopus-check     validate the configured local vector pack without dictation\n  --preview-oqtopus   show the local Oqtopus pack without dictation\n  --preview-oqtopus-reduced   show its static reduced-motion rest frame\n  --preview-parametric-override  animate a synthetic icon despite reduced motion\n  --preview-parametric  show animated parametric bars in a synthetic notch\n  --preview-parametric-reduced  show the same fixture with reduced motion\n  --version            print the version\n\nConfig: {}",
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
}

/// Synthetic UI fixture: no engine, microphone, hotkeys, tray, or settings writes.
fn preview_synthetic(options: PreviewOptions) {
    let PreviewOptions {
        expanded,
        parametric,
        reduced,
        placement,
        floating,
        motion_override,
        oqtopus,
        clickable,
    } = options;
    use config::IndicatorStyle;
    use ears::dictation::{ConnectionState, DictationState};
    let mut ui = UiConfig::default();
    ui.overlay.indicator = if oqtopus {
        IndicatorStyle::Oqtopus
    } else if parametric {
        IndicatorStyle::ParametricEars
    } else {
        IndicatorStyle::Ears
    };
    ui.overlay.show_details = expanded;
    ui.overlay.animation = parametric || oqtopus;
    ui.overlay.click_through = !clickable;
    ui.overlay.island_mode = placement && !floating;
    ui.overlay.animate_with_reduced_motion = motion_override;
    if floating {
        ui.overlay.anchor = config::OverlayAnchor::Bottom;
    }
    Application::new().run(move |cx| {
        platform::set_accessory_policy();
        let mut fixture = AppModel::new(AppConfig::default(), ui);
        fixture.configuration_is_preview = true;
        fixture.state = DictationState::Listening;
        fixture.connection = ConnectionState::Connected;
        fixture.speech_active = true;
        fixture.partial = "Synthetic speech".into();
        fixture.reduce_motion = reduced;
        let model = cx.new(|_| fixture);
        cx.set_global(AppHandle(model.clone()));
        cx.set_global(Runtime::new(model.clone()));
        runtime::apply_theme(cx, None);
        cx.on_window_closed(|cx| cx.quit()).detach();
        if placement || floating {
            let overlay_config = model.read(cx).ui.overlay.clone();
            overlay::open(cx, &overlay_config, model);
        } else {
            overlay::open_preview(cx, model);
        }
    });
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
    let animated = motion_override || args.iter().any(|a| a == "--preview-parametric");
    let reduced = motion_override || args.iter().any(|a| a == "--preview-parametric-reduced");
    let floating = args.iter().any(|a| a == "--preview-floating");
    let anchored_click = args.iter().any(|a| a == "--preview-click");
    let hit_test = args.iter().any(|a| a == "--preview-hit-test");
    let clickable = anchored_click || hit_test;
    let placement = anchored_click || args.iter().any(|a| a == "--preview-placement");
    if oqtopus
        || floating
        || placement
        || expanded
        || hit_test
        || animated
        || reduced
        || args.iter().any(|a| a == "--preview-synthetic")
    {
        preview_synthetic(PreviewOptions {
            expanded,
            parametric: animated || reduced,
            reduced: oqtopus_reduced || (!oqtopus && (reduced || !animated)),
            placement,
            floating,
            motion_override,
            oqtopus,
            clickable,
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
