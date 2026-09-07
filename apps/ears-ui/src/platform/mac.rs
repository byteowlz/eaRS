//! macOS shim: everything goes through public AppKit API (objc2), no private
//! symbols. See docs/DYNAMIC_ISLAND_SPIKE.md for the reasoning.

use super::Notch;
use gpui::Window;
use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::NSApplication;
use objc2_app_kit::NSApplicationActivationPolicy;
use objc2_app_kit::NSEvent;
use objc2_app_kit::NSScreen;
use objc2_app_kit::NSView;
use objc2_app_kit::NSWindow;
use objc2_app_kit::NSWorkspace;
use objc2_foundation::NSNumber;
use objc2_foundation::NSPoint;
use objc2_foundation::NSString;
use raw_window_handle::HasWindowHandle;
use raw_window_handle::RawWindowHandle;

fn mtm() -> MainThreadMarker {
    MainThreadMarker::new().expect("ears-ui platform calls run on the main thread")
}

/// Hide the Dock icon and app switcher entry; the app lives in the menu bar.
pub fn set_accessory_policy() {
    let app = NSApplication::sharedApplication(mtm());
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
}

/// Bring the app forward when a regular window (settings) opens.
pub fn activate_app() {
    let app = NSApplication::sharedApplication(mtm());
    #[allow(deprecated)]
    app.activateIgnoringOtherApps(true);
}

pub fn reduce_motion() -> bool {
    NSWorkspace::sharedWorkspace().accessibilityDisplayShouldReduceMotion()
}

fn screen_number(screen: &NSScreen) -> Option<u32> {
    let description = screen.deviceDescription();
    let key = NSString::from_str("NSScreenNumber");
    let value = description.objectForKey(&key)?;
    let number = value.downcast_ref::<NSNumber>()?;
    Some(number.as_u32())
}

fn screen_for(display: u32) -> Option<Retained<NSScreen>> {
    NSScreen::screens(mtm())
        .iter()
        .find(|s| screen_number(s) == Some(display))
}

/// CGDirectDisplayID of the screen under the mouse cursor.
pub fn cursor_display() -> Option<u32> {
    let point = NSEvent::mouseLocation();
    NSScreen::screens(mtm()).iter().find_map(|screen| {
        let frame = screen.frame();
        let inside = point.x >= frame.origin.x
            && point.x < frame.origin.x + frame.size.width
            && point.y >= frame.origin.y
            && point.y < frame.origin.y + frame.size.height;
        if inside { screen_number(&screen) } else { None }
    })
}

/// Height of the menu bar on the given display (0 when hidden).
pub fn menu_bar_height(display: u32) -> f32 {
    let Some(screen) = screen_for(display) else {
        return 0.0;
    };
    let frame = screen.frame();
    let visible = screen.visibleFrame();
    let top = frame.origin.y + frame.size.height;
    let visible_top = visible.origin.y + visible.size.height;
    (top - visible_top).max(0.0) as f32
}

/// Camera notch geometry, if this display has one. Uses
/// `safeAreaInsets` (height) and the auxiliary top areas (width).
pub fn notch(display: u32) -> Option<Notch> {
    let screen = screen_for(display)?;
    let insets = screen.safeAreaInsets();
    if insets.top <= 0.0 {
        return None;
    }
    let frame = screen.frame();
    let left = screen.auxiliaryTopLeftArea();
    let right = screen.auxiliaryTopRightArea();
    let width = (frame.size.width - left.size.width - right.size.width).max(0.0);
    if width <= 0.0 {
        return None;
    }
    Some(Notch {
        width: width as f32,
        height: insets.top as f32,
    })
}

/// The NSWindow behind a GPUI window.
pub fn ns_window(window: &Window) -> Option<Retained<NSWindow>> {
    let handle = HasWindowHandle::window_handle(window).ok()?;
    match handle.as_raw() {
        RawWindowHandle::AppKit(h) => {
            let view = h.ns_view.as_ptr() as *mut NSView;
            // SAFETY: GPUI hands us a live NSView pointer for this window.
            let view = unsafe { &*view };
            view.window()
        }
        _ => None,
    }
}

pub fn set_click_through(window: &Window, on: bool) {
    if let Some(ns) = ns_window(window) {
        ns.setIgnoresMouseEvents(on);
    }
}

#[allow(dead_code)]
pub fn set_window_alpha(window: &Window, alpha: f32) {
    if let Some(ns) = ns_window(window) {
        ns.setAlphaValue(alpha as f64);
    }
}

/// Mouse position in AppKit screen coordinates (origin bottom-left).
pub fn mouse_screen_position() -> Option<(f32, f32)> {
    let p = NSEvent::mouseLocation();
    Some((p.x as f32, p.y as f32))
}

/// Window frame origin in AppKit screen coordinates.
pub fn window_origin(window: &Window) -> Option<(f32, f32)> {
    let ns = ns_window(window)?;
    let frame = ns.frame();
    Some((frame.origin.x as f32, frame.origin.y as f32))
}

pub fn set_window_origin(window: &Window, x: f32, y: f32) {
    if let Some(ns) = ns_window(window) {
        ns.setFrameOrigin(NSPoint::new(x as f64, y as f64));
    }
}

/// GPUI's `start_window_move` is a no-op on macOS; the overlay drags the
/// window itself. Returns false so callers fall back to manual dragging.
pub fn start_native_drag(_window: &Window) -> bool {
    false
}

/// Disable the system shadow: the pill draws its own.
pub fn set_shadow(window: &Window, on: bool) {
    if let Some(ns) = ns_window(window) {
        ns.setHasShadow(on);
    }
}
