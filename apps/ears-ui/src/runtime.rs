//! Process-level plumbing: hosts the headless dictation engine, owns the
//! settings window, resolves the theme and runs the UI tick loop.

use crate::app::AppHandle;
use crate::app::AppModel;
use crate::overlay;
use crate::platform;
use crate::settings::SettingsView;
use crate::theme::Theme;
use crate::tray;
use anyhow::Result;
use crossbeam_channel::Receiver;
use ears::dictation::DictationEvent;
use ears::dictation::DictationOptions;
use ears::dictation::DictationSession;
use gpui::App;
use gpui::AppContext;
use gpui::BorrowAppContext;
use gpui::Bounds;
use gpui::Entity;
use gpui::Global;
use gpui::TitlebarOptions;
use gpui::WindowAppearance;
use gpui::WindowBounds;
use gpui::WindowHandle;
use gpui::WindowKind;
use gpui::WindowOptions;
use gpui::px;
use gpui::size;
use std::thread::JoinHandle;
use std::time::Duration;

pub const SETTINGS_SIZE: (f32, f32) = (780.0, 560.0);

pub struct Runtime {
    #[allow(dead_code)]
    pub model: Entity<AppModel>,
    pub settings: Option<WindowHandle<SettingsView>>,
    events: Option<Receiver<DictationEvent>>,
    engine_thread: Option<JoinHandle<Result<()>>>,
    appearance: WindowAppearance,
    quitting: bool,
}

impl Global for Runtime {}

impl Runtime {
    pub fn new(model: Entity<AppModel>) -> Self {
        Self {
            model,
            settings: None,
            events: None,
            engine_thread: None,
            appearance: WindowAppearance::Dark,
            quitting: false,
        }
    }
}

pub fn model(cx: &App) -> Entity<AppModel> {
    cx.global::<AppHandle>().0.clone()
}

/// Build engine options from the core + UI config.
fn engine_options(model: &AppModel) -> DictationOptions {
    let ui = &model.ui.dictation;
    DictationOptions {
        server: if ui.server.trim().is_empty() {
            None
        } else {
            Some(ui.server.trim().to_string())
        },
        insertion_mode: ui.insertion_mode,
        escape_cancels: Some(ui.escape_cancels),
        profiles: ui.profiles,
        verbose: std::env::var_os("EARS_UI_VERBOSE").is_some(),
        write_pid_file: true,
        ..DictationOptions::default()
    }
}

/// Start (or restart) the dictation engine on its own thread.
pub fn start_engine(cx: &mut App) {
    let model = model(cx);
    let (core, options, hotkey_mode) = model.read_with(cx, |m, _| {
        let mut core = m.core.clone();
        core.hotkeys.mode = m.hotkey_mode;
        (core, engine_options(m), m.hotkey_mode)
    });

    // A CLI `ears-dictation` instance would type everything twice.
    match ears::dictation::stop_running_instance() {
        Ok(true) => eprintln!("ears-ui: stopped the running ears-dictation instance"),
        Ok(false) => {}
        Err(err) => eprintln!("ears-ui: could not stop running dictation: {err:#}"),
    }

    match ears::dictation::spawn(core, options) {
        Ok(DictationSession {
            handle,
            events,
            thread,
        }) => {
            handle.set_hotkey_mode(hotkey_mode);
            model.update(cx, |m, cx| {
                m.engine = Some(handle);
                m.engine_stopped = false;
                m.error = None;
                cx.notify();
            });
            cx.update_global::<Runtime, _>(|rt, _| {
                rt.events = Some(events);
                rt.engine_thread = Some(thread);
            });
        }
        Err(err) => {
            eprintln!("ears-ui: engine failed to start: {err:#}");
            model.update(cx, |m, cx| {
                m.engine = None;
                m.engine_stopped = true;
                m.apply_event(
                    DictationEvent::Error {
                        message: format!("Engine failed to start: {err:#}"),
                    },
                    cx,
                );
            });
        }
    }
}

pub fn restart_engine(cx: &mut App) {
    let model = model(cx);
    let old = model.update(cx, |m, cx| {
        let handle = m.engine.take();
        if let Some(h) = &handle {
            h.shutdown();
        }
        m.engine_stopped = true;
        cx.notify();
        handle
    });
    let thread = cx.update_global::<Runtime, _>(|rt, _| {
        rt.events = None;
        rt.engine_thread.take()
    });
    drop(old);
    if let Some(thread) = thread {
        // The engine exits promptly on Shutdown; join on a helper thread so
        // the UI never blocks on a slow socket close.
        std::thread::spawn(move || {
            let _ = thread.join();
        });
    }
    start_engine(cx);
}

/// If the engine thread exited, surface its error (e.g. no Accessibility
/// permission for text insertion) as a UI error.
fn reap_engine(cx: &mut App) {
    let finished =
        cx.update_global::<Runtime, _>(|rt, _| rt.engine_thread.as_ref().map(|t| t.is_finished()));
    if finished != Some(true) {
        return;
    }
    let thread = cx.update_global::<Runtime, _>(|rt, _| rt.engine_thread.take());
    let Some(thread) = thread else { return };
    let outcome = std::thread::spawn(move || thread.join()).join();
    let message = match outcome {
        Ok(Ok(Ok(()))) | Ok(Err(_)) => None, // clean stop or join error: ignore
        Ok(Ok(Err(err))) => Some(format!("Dictation stopped: {err:#}")),
        Err(_) => Some("Dictation thread panicked".to_string()),
    };
    if let Some(message) = message {
        eprintln!("ears-ui: {message}");
        let model = model(cx);
        model.update(cx, |m, cx| {
            m.engine = None;
            m.engine_stopped = true;
            m.apply_event(DictationEvent::Error { message }, cx);
        });
    }
}

/// Drain engine events into the model.
pub fn pump_events(cx: &mut App) {
    reap_engine(cx);
    let events: Vec<DictationEvent> = {
        let rt = cx.global::<Runtime>();
        match &rt.events {
            Some(rx) => rx.try_iter().collect(),
            None => Vec::new(),
        }
    };
    if events.is_empty() {
        return;
    }
    let model = model(cx);
    model.update(cx, |m, cx| {
        for event in events {
            m.apply_event(event, cx);
        }
    });
}

/// Resolve the theme for the current appearance and install it globally.
pub fn apply_theme(cx: &mut App, appearance: Option<WindowAppearance>) {
    let appearance = match appearance {
        Some(a) => {
            cx.update_global::<Runtime, _>(|rt, _| rt.appearance = a);
            a
        }
        None => cx.global::<Runtime>().appearance,
    };
    let (scheme, radius) =
        model(cx).read_with(cx, |m, _| (m.ui.theme.scheme.clone(), m.ui.theme.radius));
    let theme = Theme::resolve(&scheme, radius, appearance);
    cx.set_global(theme);
    cx.refresh_windows();
}

pub fn open_settings(cx: &mut App) {
    let existing = cx.global::<Runtime>().settings;
    if let Some(handle) = existing {
        let ok = handle
            .update(cx, |_, window, _| {
                window.activate_window();
            })
            .is_ok();
        if ok {
            platform::activate_app();
            return;
        }
        cx.update_global::<Runtime, _>(|rt, _| rt.settings = None);
    }
    let model = model(cx);
    let bounds = Bounds::centered(None, size(px(SETTINGS_SIZE.0), px(SETTINGS_SIZE.1)), cx);
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        titlebar: Some(TitlebarOptions {
            title: Some("eaRS Settings".into()),
            appears_transparent: false,
            traffic_light_position: None,
        }),
        focus: true,
        show: true,
        kind: WindowKind::Normal,
        is_movable: true,
        is_resizable: true,
        is_minimizable: true,
        window_min_size: Some(size(px(640.0), px(420.0))),
        app_id: Some("dev.byteowlz.ears-ui".into()),
        ..Default::default()
    };
    match cx.open_window(options, |window, cx| {
        let appearance = window.appearance();
        apply_theme(cx, Some(appearance));
        cx.new(|cx| SettingsView::new(model.clone(), window, cx))
    }) {
        Ok(handle) => {
            cx.update_global::<Runtime, _>(|rt, _| rt.settings = Some(handle));
            platform::activate_app();
        }
        Err(err) => eprintln!("ears-ui: cannot open settings: {err:#}"),
    }
}

#[allow(dead_code)]
pub fn close_settings(cx: &mut App) {
    if let Some(handle) = cx.global::<Runtime>().settings {
        let _ = handle.update(cx, |_, window, _| window.remove_window());
    }
    cx.update_global::<Runtime, _>(|rt, _| rt.settings = None);
}

/// Called when any window closes: forget the settings handle if it was it.
pub fn on_window_closed(cx: &mut App) {
    let settings = cx.global::<Runtime>().settings;
    if let Some(handle) = settings {
        let alive = cx
            .windows()
            .iter()
            .any(|w| w.window_id() == gpui::AnyWindowHandle::from(handle).window_id());
        if !alive {
            cx.update_global::<Runtime, _>(|rt, _| rt.settings = None);
        }
    }
}

pub fn quit(cx: &mut App) {
    let already = cx.update_global::<Runtime, _>(|rt, _| std::mem::replace(&mut rt.quitting, true));
    if already {
        return;
    }
    let model = model(cx);
    model.update(cx, |m, _| {
        if let Some(engine) = m.engine.take() {
            engine.shutdown();
        }
    });
    ears::dictation::remove_pid_file();
    cx.quit();
}

/// One UI tick: pump engine events, expire timers, sync the overlay and
/// tray, process tray clicks.
pub fn tick(cx: &mut App) {
    pump_events(cx);
    let model = model(cx);
    model.update(cx, |m, cx| {
        m.tick(cx);
    });
    tray::pump(cx);
    overlay::sync(cx);
}

/// Spawn the tick loop. Runs fast while the island is active, slow otherwise.
pub fn spawn_tick_loop(cx: &mut App) {
    cx.spawn(async move |cx| {
        loop {
            let active = cx
                .update(|cx| {
                    tick(cx);
                    overlay::is_active(cx)
                })
                .unwrap_or(false);
            let delay = if active { 33 } else { 120 };
            cx.background_executor()
                .timer(Duration::from_millis(delay))
                .await;
        }
    })
    .detach();
}
