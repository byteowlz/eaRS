//! Menu-bar (tray) presence: persistent icon whose look follows the engine
//! state, with a menu for the controls people actually want without opening
//! the settings window. Works with the overlay disabled.
//!
//! The icon is a small RGBA mic glyph rasterised in code so we ship no assets;
//! its colour follows the same role mapping the island uses (idle = muted,
//! capturing = primary, error = danger, disconnected = warning).

use crate::config::TrayConfig;
use crate::runtime;
use ears::dictation::ConnectionState;
use ears::dictation::DictationState;
use ears::dictation::InsertionMode;
use gpui::App;
use gpui::Global;
use std::sync::Arc;
use tray_icon::Icon;
use tray_icon::TrayIcon;
use tray_icon::TrayIconBuilder;
use tray_icon::menu::CheckMenuItem;
use tray_icon::menu::Menu;
use tray_icon::menu::MenuEvent;
use tray_icon::menu::MenuId;
use tray_icon::menu::PredefinedMenuItem;

/// On Linux the tray backend is libertray/libappindicator over GTK: GTK must
/// be initialized once, on the main thread, before any menu widget exists.
/// Without it the app used to abort at startup (`GTK has not been initialized`).
/// When GTK is unavailable (headless session, no display) the tray is skipped
/// and eaRS keeps running with Settings and hotkeys.
#[cfg(target_os = "linux")]
mod gtk_boot {
    use std::sync::OnceLock;

    static READY: OnceLock<bool> = OnceLock::new();

    pub fn prepare() -> bool {
        *READY.get_or_init(|| {
            // `init` uses gtk_init_check: a headless/tty session returns Err
            // instead of aborting, and a repeated call on the main thread is a
            // no-op.
            gtk::init().is_ok() && gtk::is_initialized_main_thread()
        })
    }

    /// Deliver tray/indicator events to the GTK side from our own loop.
    pub fn pump_events() {
        if !gtk::is_initialized_main_thread() {
            return;
        }
        while gtk::events_pending() {
            gtk::main_iteration_do(false);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod gtk_boot {
    pub fn prepare() -> bool {
        true
    }
    pub fn pump_events() {}
}

/// Prepare the platform tray backend. Call once, early, before GPUI starts:
/// the Linux backend needs GTK on the main thread first.
pub fn prepare() -> bool {
    gtk_boot::prepare()
}

const ID_TOGGLE: &str = "ears-toggle";
const ID_SETTINGS: &str = "ears-settings";
const ID_MODE_INSERT: &str = "ears-mode-insert";
const ID_MODE_COPY: &str = "ears-mode-copy";
const ID_MODE_PROMPT: &str = "ears-mode-prompt";
const ID_LAUNCH: &str = "ears-launch";
const ID_QUIT: &str = "ears-quit";

pub struct TrayState {
    _tray: TrayIcon,
    toggle_item: Arc<CheckMenuItem>,
    mode_insert: Arc<CheckMenuItem>,
    mode_copy: Arc<CheckMenuItem>,
    mode_prompt: Arc<CheckMenuItem>,
    launch_item: Arc<CheckMenuItem>,
    /// Signature of the state the icon + check marks currently reflect.
    applied: Option<String>,
}

impl Global for TrayState {}

/// Menu ids stay constant; event dispatch runs on the GPUI main thread from
/// the tick loop.
pub fn install(cx: &mut App) {
    if cx.has_global::<TrayState>() {
        return;
    }
    if !gtk_boot::prepare() {
        eprintln!("ears-ui: tray disabled: the GTK tray backend is unavailable in this session");
        return;
    }

    let toggle_item = Arc::new(CheckMenuItem::with_id(
        ID_TOGGLE,
        "Pause dictation",
        true,
        false,
        None,
    ));
    let settings_item = tray_icon::menu::MenuItem::with_id(ID_SETTINGS, "Settings…", true, None);
    let mode_insert = Arc::new(CheckMenuItem::with_id(
        ID_MODE_INSERT,
        "Insert at cursor",
        true,
        true,
        None,
    ));
    let mode_copy = Arc::new(CheckMenuItem::with_id(
        ID_MODE_COPY,
        "Copy to clipboard",
        true,
        false,
        None,
    ));
    let mode_prompt = Arc::new(CheckMenuItem::with_id(
        ID_MODE_PROMPT,
        "Send as prompt",
        true,
        false,
        None,
    ));
    let launch_item = Arc::new(CheckMenuItem::with_id(
        ID_LAUNCH,
        "Launch at login",
        true,
        launch_at_login_enabled(),
        None,
    ));
    let quit_item = tray_icon::menu::MenuItem::with_id(ID_QUIT, "Quit eaRS", true, None);

    let menu = Menu::new();
    if let Err(err) = menu.append_items(&[
        &*toggle_item,
        &settings_item,
        &PredefinedMenuItem::separator(),
        &*mode_insert,
        &*mode_copy,
        &*mode_prompt,
        &PredefinedMenuItem::separator(),
        &*launch_item,
        &PredefinedMenuItem::separator(),
        &quit_item,
    ]) {
        eprintln!("ears-ui: tray menu error: {err}");
        return;
    }

    let icon = icon_for(false, true, true);
    let tray = match TrayIconBuilder::new()
        .with_id("ears-ui")
        .with_menu(Box::new(menu))
        .with_icon(icon)
        .with_tooltip("eaRS dictation")
        .with_menu_on_left_click(true)
        .build()
    {
        Ok(tray) => tray,
        Err(err) => {
            eprintln!("ears-ui: cannot create tray icon: {err}");
            return;
        }
    };

    cx.set_global(TrayState {
        _tray: tray,
        toggle_item,
        mode_insert,
        mode_copy,
        mode_prompt,
        launch_item,
        applied: None,
    });
}

/// Drain tray menu events and re-sync the icon/menu with the model.
pub fn pump(cx: &mut App) {
    gtk_boot::pump_events();
    while let Ok(event) = MenuEvent::receiver().try_recv() {
        dispatch(&event.id, cx);
    }
    sync(cx);
}

fn dispatch(id: &MenuId, cx: &mut App) {
    let model = runtime::model(cx);
    match id.as_ref() {
        ID_TOGGLE => model.update(cx, |m, _| {
            if m.state == DictationState::Listening {
                m.set_capturing(false);
            } else {
                m.set_capturing(true);
            }
        }),
        ID_SETTINGS => runtime::open_settings(cx),
        ID_MODE_INSERT | ID_MODE_COPY | ID_MODE_PROMPT => {
            let mode = match id.as_ref() {
                ID_MODE_COPY => InsertionMode::Clipboard,
                ID_MODE_PROMPT => InsertionMode::SendAsPrompt,
                _ => InsertionMode::InsertAtCursor,
            };
            model.update(cx, |m, cx| m.set_insertion_mode(mode, cx));
        }
        ID_LAUNCH => {
            let enable = !launch_at_login_enabled();
            match set_launch_at_login(enable) {
                Ok(()) => {
                    model.update(cx, |m, cx| {
                        m.update_ui(cx, |ui| {
                            ui.tray = TrayConfig {
                                launch_at_login: enable,
                            }
                        });
                    });
                }
                Err(err) => eprintln!("ears-ui: launch-at-login failed: {err:#}"),
            }
        }
        ID_QUIT => runtime::quit(cx),
        _ => {}
    }
}

/// Reflect engine state into the icon colour and menu check marks. Cheap: it
/// only rebuilds the icon when the state signature changes.
fn sync(cx: &mut App) {
    if !cx.has_global::<TrayState>() {
        return;
    }
    let model = runtime::model(cx);
    let (capturing, listening, connected, mode, launch_cfg) = model.read_with(cx, |m, _| {
        (
            m.state == DictationState::Listening,
            matches!(
                m.phase(),
                crate::app::Phase::Listening | crate::app::Phase::Armed
            ),
            m.connection != ConnectionState::Disconnected,
            m.insertion_mode,
            m.ui.tray.launch_at_login,
        )
    });
    let healthy = connected || capturing;
    let signature = format!("{capturing}|{listening}|{healthy}|{mode:?}|{launch_cfg}");
    {
        let state = cx.global_mut::<TrayState>();
        if state.applied.as_deref() == Some(signature.as_str()) {
            return;
        }
        state.toggle_item.set_text(if capturing {
            "Pause dictation"
        } else {
            "Resume dictation"
        });
        state.toggle_item.set_checked(capturing);
        state
            .mode_insert
            .set_checked(mode == InsertionMode::InsertAtCursor);
        state
            .mode_copy
            .set_checked(mode == InsertionMode::Clipboard);
        state
            .mode_prompt
            .set_checked(mode == InsertionMode::SendAsPrompt);
        state.launch_item.set_checked(launch_at_login_enabled());
    }
    let healthy = listening || capturing || healthy;
    let icon = icon_for(listening || capturing, healthy, capturing);
    let state = cx.global_mut::<TrayState>();
    if let Err(err) = state._tray.set_icon(Some(icon)) {
        eprintln!("ears-ui: cannot update tray icon: {err}");
    }
    state.applied = Some(signature);
}

// ---------------------------------------------------------------------------
// Icon
// ---------------------------------------------------------------------------

/// The real eaRS mark (rendered from `logo/ears_logo_white.svg` by
/// `scripts/export-icons.sh`), tinted with the island's role colours: muted
/// when off or unhealthy, primary while capturing, muted foreground when idle.
/// Recolouring one artwork covers both light and dark bars.
fn icon_for(active: bool, healthy: bool, enabled: bool) -> Icon {
    let (r, g, b) = if !enabled || !healthy {
        (0.62, 0.65, 0.63) // muted
    } else if active {
        (0.231, 0.655, 0.486) // primary (oqto base0B)
    } else {
        (0.698, 0.725, 0.71) // foreground-muted
    };
    let (width, height, mut pixels) = mark_rgba();
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[0] = (r * 255.0) as u8;
        pixel[1] = (g * 255.0) as u8;
        pixel[2] = (b * 255.0) as u8;
        // Alpha carries the mark's shape and stays as rendered.
    }
    Icon::from_rgba(pixels, width, height).expect("bundled tray mark")
}

/// Decoded once from the committed PNG.
fn mark_rgba() -> (u32, u32, Vec<u8>) {
    static MARK: std::sync::OnceLock<(u32, u32, Vec<u8>)> = std::sync::OnceLock::new();
    MARK.get_or_init(|| {
        let bytes = include_bytes!("../assets/tray/ears-mark.png");
        let image = image::load_from_memory(bytes)
            .expect("bundled tray mark is a valid PNG")
            .to_rgba8();
        (image.width(), image.height(), image.into_raw())
    })
    .clone()
}

// ---------------------------------------------------------------------------
// Launch at login
// ---------------------------------------------------------------------------

fn auto_launcher() -> anyhow::Result<auto_launch::AutoLaunch> {
    let exe = std::env::current_exe()?;
    let mut builder = auto_launch::AutoLaunchBuilder::new();
    builder
        .set_app_name("ears-ui")
        .set_app_path(&exe.display().to_string());
    #[cfg(target_os = "macos")]
    builder.set_use_launch_agent(true);
    builder.build().map_err(|err| anyhow::anyhow!("{err}"))
}

pub fn launch_at_login_enabled() -> bool {
    auto_launcher()
        .and_then(|launcher| {
            launcher
                .is_enabled()
                .map_err(|err| anyhow::anyhow!("{err}"))
        })
        .unwrap_or(false)
}

pub fn set_launch_at_login(enable: bool) -> anyhow::Result<()> {
    let launcher = auto_launcher()?;
    let result = if enable {
        launcher.enable()
    } else {
        launcher.disable()
    };
    result.map_err(|err| anyhow::anyhow!("{err}"))
}
