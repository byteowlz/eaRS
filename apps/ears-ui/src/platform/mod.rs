//! Thin per-platform shims for the few things GPUI does not expose:
//! click-through windows, moving a window from Rust, the cursor's display,
//! notch geometry, reduce-motion, activation policy.

#[cfg(target_os = "macos")]
mod mac;
#[cfg(not(target_os = "macos"))]
mod other;

#[cfg(target_os = "macos")]
pub use mac::*;
#[cfg(not(target_os = "macos"))]
pub use other::*;

/// Geometry of a camera notch in logical pixels (macOS laptops).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Notch {
    pub width: f32,
    pub height: f32,
}

/// Open a folder or file with the platform file manager.
#[allow(dead_code)]
pub fn reveal(path: &std::path::Path) {
    #[cfg(target_os = "macos")]
    let cmd = ("open", vec![path.as_os_str().to_owned()]);
    #[cfg(target_os = "windows")]
    let cmd = ("explorer", vec![path.as_os_str().to_owned()]);
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = ("xdg-open", vec![path.as_os_str().to_owned()]);
    let _ = std::process::Command::new(cmd.0).args(cmd.1).spawn();
}
