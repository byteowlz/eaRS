//! Frontmost application detection.
//!
//! Context profiles (see [`crate::profiles`]) pick a dictionary, language and
//! insertion mode based on the application that currently has keyboard focus.
//! This module answers "what is in front right now?" on each platform:
//!
//! - macOS: `NSWorkspace.frontmostApplication` for bundle id / name / binary,
//!   plus the Accessibility API for the focused window title (requires the
//!   Accessibility permission, which dictation already needs for typing).
//! - Linux: `hyprctl`, `swaymsg` or `xdotool`, whichever compositor is running.
//! - Windows: `GetForegroundWindow` + `QueryFullProcessImageNameW`.
//!
//! Every path is best-effort: a failure yields `None` rather than an error so
//! the dictation engine can fall back to the default profile.

use serde::Deserialize;
use serde::Serialize;

/// A snapshot of the frontmost application.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontmostApp {
    /// macOS bundle identifier (e.g. `com.apple.Safari`). `None` elsewhere.
    pub bundle_id: Option<String>,
    /// Human readable application name.
    pub name: Option<String>,
    /// Executable file name without directory (e.g. `code`, `Safari`).
    pub binary: Option<String>,
    /// Title of the focused window, when the platform exposes it.
    pub window_title: Option<String>,
    /// Process id of the frontmost application.
    pub pid: Option<u32>,
}

impl FrontmostApp {
    /// Short human readable label for logs and UI: name, binary or bundle id.
    pub fn label(&self) -> String {
        self.name
            .clone()
            .or_else(|| self.binary.clone())
            .or_else(|| self.bundle_id.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }
}

/// Query the frontmost application. Returns `None` when detection is not
/// possible on this platform or the query failed.
pub fn frontmost_app() -> Option<FrontmostApp> {
    platform::frontmost_app()
}

fn file_stem_of(path: &str) -> Option<String> {
    std::path::Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
}

#[cfg(target_os = "macos")]
mod platform {
    use super::FrontmostApp;
    use super::file_stem_of;
    use objc2_app_kit::NSWorkspace;
    use objc2_core_foundation::CFRetained;
    use objc2_core_foundation::CFString;
    use objc2_core_foundation::CFType;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    #[repr(C)]
    struct AXUIElement(c_void);

    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        fn AXUIElementCreateApplication(pid: libc::pid_t) -> *mut AXUIElement;
        fn AXUIElementCopyAttributeValue(
            element: *mut AXUIElement,
            attribute: *const CFString,
            value: *mut *const CFType,
        ) -> i32;
        fn CFRelease(cf: *const c_void);
    }

    pub fn frontmost_app() -> Option<FrontmostApp> {
        let workspace = NSWorkspace::sharedWorkspace();
        let app = workspace.frontmostApplication()?;
        let bundle_id = app.bundleIdentifier().map(|s| s.to_string());
        let name = app.localizedName().map(|s| s.to_string());
        let binary = app
            .executableURL()
            .and_then(|url| url.path())
            .and_then(|path| file_stem_of(&path.to_string()));
        let pid = app.processIdentifier();
        let window_title = if pid > 0 {
            focused_window_title(pid)
        } else {
            None
        };
        Some(FrontmostApp {
            bundle_id,
            name,
            binary,
            window_title,
            pid: u32::try_from(pid).ok(),
        })
    }

    fn copy_attribute(element: *mut AXUIElement, name: &str) -> Option<CFRetained<CFType>> {
        let attribute = CFString::from_str(name);
        let mut value: *const CFType = std::ptr::null();
        let status = unsafe {
            AXUIElementCopyAttributeValue(element, &*attribute, &mut value as *mut *const CFType)
        };
        if status != 0 || value.is_null() {
            return None;
        }
        let ptr = NonNull::new(value as *mut CFType)?;
        // AXUIElementCopyAttributeValue follows the Create/Copy rule: we own
        // one reference, which CFRetained releases on drop.
        Some(unsafe { CFRetained::from_raw(ptr) })
    }

    fn focused_window_title(pid: libc::pid_t) -> Option<String> {
        let app_element = unsafe { AXUIElementCreateApplication(pid) };
        if app_element.is_null() {
            return None;
        }
        let title = copy_attribute(app_element, "AXFocusedWindow").and_then(|window| {
            let window_ptr = (&*window) as *const CFType as *mut AXUIElement;
            copy_attribute(window_ptr, "AXTitle")
                .and_then(|title| title.downcast_ref::<CFString>().map(|s| s.to_string()))
        });
        unsafe { CFRelease(app_element as *const c_void) };
        title.filter(|t| !t.is_empty())
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::FrontmostApp;
    use std::process::Command;

    pub fn frontmost_app() -> Option<FrontmostApp> {
        if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() {
            if let Some(app) = hyprland() {
                return Some(app);
            }
        }
        if std::env::var_os("SWAYSOCK").is_some() {
            if let Some(app) = sway() {
                return Some(app);
            }
        }
        if std::env::var_os("DISPLAY").is_some() {
            if let Some(app) = xdotool() {
                return Some(app);
            }
        }
        None
    }

    fn run(cmd: &str, args: &[&str]) -> Option<String> {
        let output = Command::new(cmd).args(args).output().ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn comm_for_pid(pid: u32) -> Option<String> {
        std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn hyprland() -> Option<FrontmostApp> {
        let raw = run("hyprctl", &["activewindow", "-j"])?;
        let json: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let class = json
            .get("class")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let title = json
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let pid = json
            .get("pid")
            .and_then(|v| v.as_i64())
            .and_then(|p| u32::try_from(p).ok());
        Some(FrontmostApp {
            bundle_id: None,
            name: class.clone(),
            binary: pid.and_then(comm_for_pid).or(class),
            window_title: title,
            pid,
        })
    }

    fn sway() -> Option<FrontmostApp> {
        let raw = run("swaymsg", &["-t", "get_tree"])?;
        let tree: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let node = find_focused(&tree)?;
        let app_id = node
            .get("app_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                node.get("window_properties")
                    .and_then(|p| p.get("class"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });
        let title = node
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let pid = node
            .get("pid")
            .and_then(|v| v.as_i64())
            .and_then(|p| u32::try_from(p).ok());
        Some(FrontmostApp {
            bundle_id: None,
            name: app_id.clone(),
            binary: pid.and_then(comm_for_pid).or(app_id),
            window_title: title,
            pid,
        })
    }

    fn find_focused(node: &serde_json::Value) -> Option<&serde_json::Value> {
        if node.get("focused").and_then(|v| v.as_bool()) == Some(true) {
            return Some(node);
        }
        for key in ["nodes", "floating_nodes"] {
            if let Some(children) = node.get(key).and_then(|v| v.as_array()) {
                for child in children {
                    if let Some(found) = find_focused(child) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }

    fn xdotool() -> Option<FrontmostApp> {
        let window = run("xdotool", &["getactivewindow"])?;
        let title = run("xdotool", &["getwindowname", &window]);
        let pid = run("xdotool", &["getwindowpid", &window]).and_then(|p| p.parse::<u32>().ok());
        let class = run("xprop", &["-id", &window, "WM_CLASS"]).and_then(|line| {
            // WM_CLASS(STRING) = "code", "Code"
            line.split('"')
                .filter(|s| !s.trim().is_empty() && !s.contains('='))
                .nth(1)
                .map(|s| s.to_string())
        });
        Some(FrontmostApp {
            bundle_id: None,
            name: class.clone(),
            binary: pid.and_then(comm_for_pid).or(class),
            window_title: title,
            pid,
        })
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::FrontmostApp;
    use super::file_stem_of;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::OpenProcess;
    use windows_sys::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION;
    use windows_sys::Win32::System::Threading::QueryFullProcessImageNameW;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetForegroundWindow;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetWindowTextW;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;

    pub fn frontmost_app() -> Option<FrontmostApp> {
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.is_null() {
                return None;
            }
            let mut title_buf = [0u16; 512];
            let len = GetWindowTextW(hwnd, title_buf.as_mut_ptr(), title_buf.len() as i32);
            let window_title = if len > 0 {
                Some(String::from_utf16_lossy(&title_buf[..len as usize]))
            } else {
                None
            };
            let mut pid: u32 = 0;
            GetWindowThreadProcessId(hwnd, &mut pid);
            let mut binary = None;
            if pid != 0 {
                let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
                if !handle.is_null() {
                    let mut path_buf = [0u16; 1024];
                    let mut size = path_buf.len() as u32;
                    if QueryFullProcessImageNameW(handle, 0, path_buf.as_mut_ptr(), &mut size) != 0
                    {
                        let path = String::from_utf16_lossy(&path_buf[..size as usize]);
                        binary = file_stem_of(&path);
                    }
                    CloseHandle(handle);
                }
            }
            Some(FrontmostApp {
                bundle_id: None,
                name: binary.clone(),
                binary,
                window_title,
                pid: (pid != 0).then_some(pid),
            })
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod platform {
    use super::FrontmostApp;

    pub fn frontmost_app() -> Option<FrontmostApp> {
        None
    }
}
