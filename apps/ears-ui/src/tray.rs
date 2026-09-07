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

/// Rasterise a 16×16 microphone glyph. `active` picks the accent colour,
/// `healthy` picks between it and the warning grey, `enabled` between grey
/// and the active colour. Template-style monochrome would fight schemes, so
/// this follows the island's role mapping instead.
fn icon_for(active: bool, healthy: bool, enabled: bool) -> Icon {
    const S: usize = 32; // draw at 2x for retina crispness
    let (r, g, b) = if !enabled || !healthy {
        (0.62, 0.65, 0.63) // muted
    } else if active {
        (0.231, 0.655, 0.486) // primary (oqto base0B)
    } else {
        (0.698, 0.725, 0.71) // foreground-muted
    };

    let mut pixels = vec![0u8; S * S * 4];
    let centre = (S - 1) as f32 / 2.0;
    for y in 0..S {
        for x in 0..S {
            let (fx, fy) = (x as f32, y as f32);
            // Capsule: the mic body.
            let body_top = 4.0;
            let body_bottom = 15.0;
            let body_r = 4.6;
            let cx = centre;
            let body_mid = (body_top + body_bottom) / 2.0;
            let body_half = (body_bottom - body_top) / 2.0;
            let in_body = (fx - cx).powi(2) / (body_r * body_r)
                + (fy - body_mid).powi(2) / (body_half * body_half)
                <= 1.0;
            // Arc: the holder.
            let arc_c = (cx, 13.5);
            let arc_r = 8.4;
            let arc_t = 1.7;
            let dist = ((fx - arc_c.0).powi(2) + (fy - arc_c.1).powi(2)).sqrt();
            let in_arc = dist <= arc_r && dist >= arc_r - arc_t && fy >= arc_c.1 && fy <= 21.0;
            // Stem + base.
            let in_stem = (fx - cx).abs() <= 1.6 && fy > 15.0 && fy <= 21.0;
            let in_base = (fx - cx).abs() <= 6.0 && fy >= 22.5 && fy <= 25.0;

            if in_body || in_arc || in_stem || in_base {
                let index = (y * S + x) * 4;
                pixels[index] = (r * 255.0) as u8;
                pixels[index + 1] = (g * 255.0) as u8;
                pixels[index + 2] = (b * 255.0) as u8;
                pixels[index + 3] = 255;
            }
        }
    }
    Icon::from_rgba(pixels, S as u32, S as u32).expect("static icon rgba")
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
