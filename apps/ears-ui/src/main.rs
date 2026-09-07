//! eaRS companion UI: floating dictation island, tray icon and settings.
//!
//! Everything about dictation happens in `ears::dictation` (the core crate);
//! this binary only renders state and forwards commands.

mod app;
mod config;
mod overlay;
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
        "ears-ui {}\n\nUsage: ears-ui [--settings] [--no-overlay] [--help]\n\n  --settings     open the settings window at launch\n  --no-overlay   start with the floating island disabled\n  --version      print the version\n\nConfig: {}",
        env!("CARGO_PKG_VERSION"),
        UiConfig::path().display()
    );
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
