//! eaRS companion UI: floating dictation island, tray icon and settings.
//!
//! Everything about dictation happens in `ears::dictation` (the core crate);
//! this binary only renders state and forwards commands.

mod app;
mod config;
mod ears_mark;
mod indicator_theme;
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
        "ears-ui {}\n\nUsage: ears-ui [--settings] [--no-overlay] [--preview-synthetic|--preview-synthetic-expanded|--preview-placement|--preview-parametric|--preview-parametric-reduced] [--help]\n\n  --settings           open the settings window at launch\n  --no-overlay         start with the floating island disabled\n  --preview-synthetic  display a synthetic notch fixture without starting dictation\n  --preview-synthetic-expanded  show the expanded synthetic fixture\n  --preview-placement  show a synthetic fixture at the actual screen anchor\n  --preview-parametric  show animated parametric bars in a synthetic notch\n  --preview-parametric-reduced  show the same fixture with reduced motion\n  --version            print the version\n\nConfig: {}",
        env!("CARGO_PKG_VERSION"),
        UiConfig::path().display()
    );
}

/// Synthetic UI fixture: no engine, microphone, hotkeys, tray, or settings writes.
fn preview_synthetic(expanded: bool, parametric: bool, reduced: bool, placement: bool) {
    use config::IndicatorStyle;
    use ears::dictation::{ConnectionState, DictationState};
    let mut ui = UiConfig::default();
    ui.overlay.indicator = if parametric {
        IndicatorStyle::ParametricEars
    } else {
        IndicatorStyle::Ears
    };
    ui.overlay.show_details = expanded;
    ui.overlay.animation = parametric;
    ui.overlay.click_through = true;
    ui.overlay.island_mode = placement;
    Application::new().run(move |cx| {
        platform::set_accessory_policy();
        let mut fixture = AppModel::new(AppConfig::default(), ui);
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
        if placement {
            let overlay_config = model.read(cx).ui.overlay.clone();
            overlay::open(cx, &overlay_config, model);
        } else {
            overlay::open_preview(cx, model);
        }
    });
}

fn maybe_preview(args: &[String]) -> bool {
    let expanded = args.iter().any(|a| a == "--preview-synthetic-expanded");
    let animated = args.iter().any(|a| a == "--preview-parametric");
    let reduced = args.iter().any(|a| a == "--preview-parametric-reduced");
    let placement = args.iter().any(|a| a == "--preview-placement");
    if placement
        || expanded
        || animated
        || reduced
        || args.iter().any(|a| a == "--preview-synthetic")
    {
        preview_synthetic(expanded, animated || reduced, !animated, placement);
        return true;
    }
    false
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
