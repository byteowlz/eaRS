//! Opt-in, recordable camera silhouette. No recorder, capture state or input hooks.
//! Passive replicas exist on every attached display, independent of the live HUD.
use crate::{
    config::{OverlayAnchor, OverlayConfig},
    overlay,
    platform::{self, Notch},
};
use gpui::{
    App, AppContext, BorrowAppContext, Bounds, Context, Global, IntoElement, ParentElement, Pixels,
    Render, Styled, Window, WindowBackgroundAppearance, WindowBounds, WindowHandle, WindowKind,
    WindowOptions, div, point, px, size,
};
use std::{borrow::Cow, collections::BTreeMap};

pub fn enabled(config: &OverlayConfig, supported: bool) -> bool {
    supported && config.enabled && config.recording_notch
}

/// An effective presentation, never a write to saved anchor/island/drag choices.
pub fn effective(config: &OverlayConfig, supported: bool) -> Cow<'_, OverlayConfig> {
    if !enabled(config, supported) {
        return Cow::Borrowed(config);
    }
    let mut effective = config.clone();
    effective.anchor = OverlayAnchor::Top;
    effective.margin = 0.0;
    effective.island_mode = true;
    effective.offset_x = 0.0;
    effective.offset_y = 0.0;
    Cow::Owned(effective)
}

/// Match the real cutout; otherwise reuse the accepted synthetic Mac notch.
pub fn geometry(hardware: Option<Notch>, screen_width: f32) -> Option<Notch> {
    if !screen_width.is_finite() || screen_width < overlay::WIN_W {
        return None;
    }
    if let Some(notch) = hardware.filter(|n| {
        n.width.is_finite()
            && n.height.is_finite()
            && n.width > 0.0
            && n.width <= overlay::WIN_W
            && n.width <= screen_width
            && n.height > 0.0
            && n.height <= 100.0
    }) {
        return Some(notch);
    }
    Some(Notch {
        width: 185.0,
        height: 32.0,
    })
}

#[derive(Clone, PartialEq)]
struct Placement {
    bounds: Bounds<Pixels>,
    notch: Notch,
}
struct Replica {
    placement: Placement,
    handle: WindowHandle<ReplicaView>,
}
#[derive(Default)]
struct Replicas(BTreeMap<u32, Replica>);
impl Global for Replicas {}

fn placements(
    allowed: bool,
    screens: impl IntoIterator<Item = (u32, Bounds<Pixels>, Option<Notch>)>,
) -> BTreeMap<u32, Placement> {
    if !allowed {
        return BTreeMap::new();
    }
    screens
        .into_iter()
        .filter_map(|(id, bounds, hardware)| {
            geometry(hardware, f32::from(bounds.size.width))
                .map(|notch| (id, Placement { bounds, notch }))
        })
        .collect()
}

pub fn sync(config: &OverlayConfig, cx: &mut App) {
    let enabled = enabled(config, platform::passive_previews_supported());
    if !cx.has_global::<Replicas>() {
        if !enabled {
            return;
        }
        cx.set_global(Replicas::default());
    }
    let displays = cx.displays();
    let desired = placements(
        enabled,
        displays.iter().map(|display| {
            let id = u32::from(display.id());
            (id, display.bounds(), platform::notch(id))
        }),
    );
    cx.update_global::<Replicas, _>(|replicas, cx| {
        let stale: Vec<_> = replicas
            .0
            .keys()
            .filter(|id| !desired.contains_key(id))
            .copied()
            .collect();
        for id in stale {
            remove(replicas.0.remove(&id), cx);
        }
        for display in displays {
            let id = u32::from(display.id());
            let Some(placement) = desired.get(&id) else {
                continue;
            };
            if replicas
                .0
                .get(&id)
                .is_some_and(|replica| replica.placement == *placement)
            {
                continue;
            }
            remove(replicas.0.remove(&id), cx);
            if let Some(handle) = open(display.id(), placement, cx) {
                replicas.0.insert(
                    id,
                    Replica {
                        placement: placement.clone(),
                        handle,
                    },
                );
            }
        }
    });
}
fn open(
    display: gpui::DisplayId,
    placement: &Placement,
    cx: &mut App,
) -> Option<WindowHandle<ReplicaView>> {
    let x = (f32::from(placement.bounds.size.width) - overlay::WIN_W) / 2.0;
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(Bounds::new(
            point(px(x), px(0.0)),
            size(px(overlay::WIN_W), px(placement.notch.height)),
        ))),
        display_id: Some(display),
        titlebar: None,
        focus: false,
        show: true,
        kind: WindowKind::PopUp,
        is_movable: false,
        is_resizable: false,
        is_minimizable: false,
        window_background: WindowBackgroundAppearance::Transparent,
        app_id: Some("dev.byteowlz.ears-ui.recording-notch".into()),
        ..Default::default()
    };
    let handle = match cx.open_window(options, |window, cx| {
        platform::prepare_overlay_panel(window);
        platform::set_click_through(window, true);
        cx.new(|_| ReplicaView(placement.notch))
    }) {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!("ears-ui: recording notch unavailable: {error:#}");
            return None;
        }
    };
    let positioned = handle
        .update(cx, |_, window, _| {
            platform::position_preview(window, u32::from(display), x, 0.0)
        })
        .unwrap_or(false);
    if !positioned {
        let _ = handle.update(cx, |_, window, _| window.remove_window());
        return None;
    }
    Some(handle)
}

fn remove(replica: Option<Replica>, cx: &mut App) {
    if let Some(replica) = replica {
        let _ = replica
            .handle
            .update(cx, |_, window, _| window.remove_window());
    }
}

struct ReplicaView(Notch);
impl Render for ReplicaView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let notch = self.0;
        div()
            .size_full()
            .relative()
            .child(
                div()
                    .absolute()
                    .top_0()
                    .left(px((overlay::WIN_W - notch.width) / 2.0))
                    .w(px(notch.width))
                    .h(px(notch.height))
                    .rounded_bl(px(notch.height * 0.35))
                    .rounded_br(px(notch.height * 0.35))
                    .bg(gpui::black()),
            )
            .child(overlay::notch_throats(notch.width, notch.height))
    }
}

#[cfg(test)]
#[path = "recording_notch_tests.rs"]
mod tests;
