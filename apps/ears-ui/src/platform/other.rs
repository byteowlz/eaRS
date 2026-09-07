//! Linux / Windows shim. Most island niceties degrade gracefully here:
//! no notch, no click-through toggle at runtime, dragging goes through the
//! compositor via GPUI's `start_window_move`.

use super::Notch;
use gpui::Window;

pub fn set_accessory_policy() {}

pub fn activate_app() {}

pub fn reduce_motion() -> bool {
    // GTK: gtk-enable-animations; Windows: SPI_GETCLIENTAREAANIMATION.
    // Neither is worth a dependency yet; the ui.toml switch covers it.
    false
}

pub fn cursor_display() -> Option<u32> {
    None
}

pub fn menu_bar_height(_display: u32) -> f32 {
    0.0
}

pub fn notch(_display: u32) -> Option<Notch> {
    None
}

pub fn set_click_through(_window: &Window, _on: bool) {}

pub fn set_window_alpha(_window: &Window, _alpha: f32) {}

pub fn mouse_screen_position() -> Option<(f32, f32)> {
    None
}

pub fn window_origin(_window: &Window) -> Option<(f32, f32)> {
    None
}

pub fn set_window_origin(_window: &Window, _x: f32, _y: f32) {}

pub fn start_native_drag(window: &Window) -> bool {
    window.start_window_move();
    true
}

pub fn set_shadow(_window: &Window, _on: bool) {}
