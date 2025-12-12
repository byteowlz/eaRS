# Wayland Virtual Keyboard Implementation Research

## Executive Summary

This document provides comprehensive research on implementing a virtual keyboard for the eaRS dictation project on Wayland. The goal is to programmatically inject keystrokes into focused applications for speech-to-text dictation.

**Current Implementation**: eaRS uses `enigo` v0.6 for keyboard simulation in `src/bin/ears-dictation.rs`.

**Key Finding**: Wayland's security model makes virtual keyboard implementation more complex than X11, but several viable approaches exist.

---

## 1. Wayland Virtual Keyboard Protocol Overview

### 1.1 Protocol: `zwp_virtual_keyboard_v1`

Wayland provides the `zwp_virtual_keyboard_v1` protocol (part of `wayland-protocols`) for virtual keyboard functionality:

- **Protocol File**: `virtual-keyboard-unstable-v1.xml`
- **Purpose**: Allows clients to create virtual keyboard devices
- **Security**: Requires compositor support and often specific permissions
- **Standard**: Unstable protocol (zwp = "Z Wayland Protocol")

### 1.2 How It Works

1. Client connects to Wayland compositor
2. Requests `zwp_virtual_keyboard_manager_v1` interface
3. Creates a virtual keyboard device
4. Sends key events using Linux keycodes (from `linux/input-event-codes.h`)
5. Compositor injects events into the focused application

### 1.3 Security Considerations

**Wayland's Security Model**:
- **Isolation**: Applications cannot see or control other applications
- **No global input injection**: Unlike X11's `XTest` extension
- **Compositor permission**: Virtual keyboard requires compositor support
- **DBus portal alternative**: Some compositors use XDG Desktop Portal

**Security Challenges**:
1. Not all compositors support `zwp_virtual_keyboard_v1`
2. May require running with elevated permissions
3. Some compositors block virtual keyboards entirely
4. User must explicitly grant permission (in some setups)

**Supported Compositors** (as of 2024):
- ✅ Sway
- ✅ Hyprland
- ✅ KWin (KDE Plasma 5.27+)
- ✅ Wlroots-based compositors
- ⚠️ GNOME Mutter (limited, uses alternative methods)

---

## 2. Available Rust Crates and Libraries

### 2.1 Current State: `enigo` v0.6

**Status**: Limited Wayland support in v0.6

```rust
// Current code in ears-dictation.rs (lines 247-248, 391-392)
let mut enigo = Enigo::new(&Settings::default())?;
enigo.text(word)?;
enigo.key(Key::Space, Direction::Click)?;
```

**Issues**:
- v0.6 primarily uses X11 backend on Linux
- Wayland support is experimental/incomplete
- Falls back to X11 through XWayland

**Note**: Latest version is v0.6.1 (current version in use)

### 2.2 Better Alternatives

#### Option A: `enigo` v2.0+ (Future)
- **Status**: In development, improved Wayland support planned
- **Approach**: Will likely use virtual keyboard protocol or DBus portal
- **Timeline**: Unknown release date
- **Risk**: Not yet available

#### Option B: `wayland-client` + `wayland-protocols`

**Crates**:
```toml
wayland-client = "0.31"
wayland-protocols = "0.32"
wayland-protocols-wlr = "0.3"  # For wlr-specific protocols
```

**Pros**:
- Direct protocol implementation
- Full control over virtual keyboard
- Most up-to-date with Wayland standards
- Active maintenance

**Cons**:
- Requires manual protocol handling
- More boilerplate code
- Need to handle keymapping manually

**Implementation Complexity**: Medium-High

#### Option C: `umacro-lib`

**Crate**: `umacro-lib = "0.1.0"`

**Description**: "Simulate Keyboard & Mouse events in Linux (X11 & Wayland) with uinput"

**Pros**:
- Supports both X11 and Wayland
- Uses `uinput` kernel interface
- Simpler API than raw Wayland protocols

**Cons**:
- Requires `/dev/uinput` access (usually root or `input` group)
- Less well-known/maintained
- May require udev rules configuration

**Implementation Complexity**: Low-Medium

#### Option D: `uinput` Direct Access

**Crate**: `uinput = "0.1"`

**Approach**: Create a virtual input device at the kernel level

**Pros**:
- Works with both X11 and Wayland
- Compositor-agnostic
- Most reliable for text injection

**Cons**:
- Requires `/dev/uinput` permissions
- User must be in `input` group or run as root
- Requires udev rules for non-root access
- Lower-level API

**Implementation Complexity**: Medium

#### Option E: External Tool Wrapper - `ydotool`

**Tool**: `ydotool` (command-line utility)

**Approach**: Shell out to external binary

**Installation**:
```bash
# Arch
sudo pacman -S ydotool
# Ubuntu/Debian
sudo apt install ydotool
```

**Usage**:
```rust
use std::process::Command;

Command::new("ydotool")
    .args(["type", text])
    .output()?;
```

**Pros**:
- Works on all Wayland compositors
- Well-tested
- No Rust dependencies

**Cons**:
- External dependency
- Requires ydotool daemon (`ydotoold`) running
- Slower than native implementation
- IPC overhead

**Implementation Complexity**: Very Low

---

## 3. Code Examples and Implementation Approaches

### 3.1 Current Implementation (enigo v0.6)

**Location**: `src/bin/ears-dictation.rs:382-409`

```rust
fn handle_message(json: &Value, enigo: &mut Enigo, capturing: &Arc<Mutex<bool>>) -> Result<()> {
    let is_capturing = *capturing.lock().unwrap();
    
    if let Some(event_type) = json.get("type").and_then(|v| v.as_str()) {
        match event_type {
            "word" if is_capturing => {
                if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
                    if !word.is_empty() {
                        eprintln!("[TYPING WORD] {}", word);
                        enigo.text(word)?;
                        enigo.key(Key::Space, Direction::Click)?;
                    }
                }
            }
            "final" if is_capturing => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        eprintln!("[TYPING FINAL] {}", text);
                        enigo.text(text)?;
                        enigo.key(Key::Space, Direction::Click)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
```

### 3.2 Recommended Approach: `wayland-client` + Virtual Keyboard Protocol

**Dependencies**:
```toml
[dependencies]
wayland-client = "0.31"
wayland-protocols = { version = "0.32", features = ["client", "unstable"] }
wayland-protocols-wlr = "0.3"
```

**Example Implementation**:

```rust
use wayland_client::{Connection, Dispatch, QueueHandle};
use wayland_protocols::wp::virtual_keyboard::zwp_virtual_keyboard_unstable_v1::client::{
    zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
    zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
};

struct VirtualKeyboard {
    keyboard: ZwpVirtualKeyboardV1,
}

impl VirtualKeyboard {
    fn new() -> Result<Self> {
        // Connect to Wayland
        let conn = Connection::connect_to_env()?;
        let mut event_queue = conn.new_event_queue();
        let qh = event_queue.handle();
        let display = conn.display();
        
        // Get registry and bind to virtual keyboard manager
        // (implementation details omitted for brevity)
        
        Ok(Self { keyboard })
    }
    
    fn type_text(&self, text: &str) -> Result<()> {
        for ch in text.chars() {
            let keycode = char_to_linux_keycode(ch)?;
            
            // Key press
            self.keyboard.key(
                current_time_ms(),
                keycode,
                1, // pressed
            );
            
            // Key release
            self.keyboard.key(
                current_time_ms() + 10,
                keycode,
                0, // released
            );
        }
        Ok(())
    }
    
    fn send_key(&self, key: Key, pressed: bool) -> Result<()> {
        let keycode = key_to_linux_keycode(key)?;
        self.keyboard.key(
            current_time_ms(),
            keycode,
            if pressed { 1 } else { 0 },
        );
        Ok(())
    }
}

// Helper to convert characters to Linux keycodes
fn char_to_linux_keycode(ch: char) -> Result<u32> {
    // Map characters to KEY_* constants from linux/input-event-codes.h
    match ch.to_ascii_lowercase() {
        'a' => Ok(30), // KEY_A
        'b' => Ok(48), // KEY_B
        // ... etc
        ' ' => Ok(57), // KEY_SPACE
        _ => Err(anyhow!("Unsupported character")),
    }
}
```

**Challenges**:
1. **Keycode Mapping**: Must maintain char → Linux keycode mapping
2. **Shift/Modifiers**: Need to handle uppercase, symbols
3. **Keymap Handling**: Virtual keyboard needs keymap file (XKB format)
4. **Timing**: Key press/release timing matters

### 3.3 Alternative Approach: `uinput` Direct

**Dependencies**:
```toml
[dependencies]
uinput = "0.1"
```

**Example Implementation**:

```rust
use uinput::Device;
use uinput::event::keyboard::{Key, Keyboard};

struct UInputKeyboard {
    device: Device,
}

impl UInputKeyboard {
    fn new() -> Result<Self> {
        let device = uinput::open("/dev/uinput")?
            .name("eaRS Virtual Keyboard")?
            .event(Keyboard::All)?  // Enable all keyboard keys
            .create()?;
        
        Ok(Self { device })
    }
    
    fn type_text(&self, text: &str) -> Result<()> {
        for ch in text.chars() {
            let key = char_to_key(ch)?;
            
            // Handle shift for uppercase
            if ch.is_uppercase() {
                self.device.press(&Key::LeftShift)?;
            }
            
            self.device.press(&key)?;
            self.device.release(&key)?;
            
            if ch.is_uppercase() {
                self.device.release(&Key::LeftShift)?;
            }
            
            self.device.synchronize()?;
        }
        Ok(())
    }
}

fn char_to_key(ch: char) -> Result<Key> {
    match ch.to_ascii_lowercase() {
        'a' => Ok(Key::A),
        'b' => Ok(Key::B),
        // ...
        ' ' => Ok(Key::Space),
        _ => Err(anyhow!("Unsupported character")),
    }
}
```

**Setup Requirements**:

1. **User Permissions** (choose one):
   ```bash
   # Option A: Add user to input group
   sudo usermod -a -G input $USER
   newgrp input  # Or logout/login
   
   # Option B: Create udev rule
   echo 'KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"' | \
       sudo tee /etc/udev/rules.d/99-uinput.rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

2. **Load uinput module**:
   ```bash
   sudo modprobe uinput
   # Make permanent:
   echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf
   ```

### 3.4 Simplest Approach: `ydotool` Wrapper

**Installation**:
```bash
# User must install ydotool
sudo pacman -S ydotool  # Arch
sudo apt install ydotool  # Debian/Ubuntu

# Start daemon
systemctl --user enable --now ydotool.service
# Or manually:
ydotoold &
```

**Rust Implementation**:

```rust
use std::process::Command;
use anyhow::{Result, Context};

struct YdotoolKeyboard;

impl YdotoolKeyboard {
    fn new() -> Result<Self> {
        // Check if ydotool is available
        Command::new("which")
            .arg("ydotool")
            .output()
            .context("ydotool not found in PATH")?;
        
        Ok(Self)
    }
    
    fn type_text(&self, text: &str) -> Result<()> {
        Command::new("ydotool")
            .arg("type")
            .arg("--")  // Prevent text starting with - from being parsed as flag
            .arg(text)
            .output()
            .context("Failed to execute ydotool")?;
        Ok(())
    }
    
    fn send_key(&self, key: &str) -> Result<()> {
        Command::new("ydotool")
            .arg("key")
            .arg(format!("{}:1", key))  // Press
            .arg(format!("{}:0", key))  // Release
            .output()?;
        Ok(())
    }
}

// Usage in handle_message:
fn handle_message(json: &Value, keyboard: &YdotoolKeyboard) -> Result<()> {
    if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
        keyboard.type_text(word)?;
        keyboard.type_text(" ")?;  // Add space
    }
    Ok(())
}
```

---

## 4. Comparison with X11 Approaches

### 4.1 X11 Tools

**xdotool**:
- Uses X11 `XTest` extension
- Global input injection
- No permissions needed (in X11 session)
- Does NOT work on pure Wayland

**ydotool**:
- Designed as Wayland alternative to xdotool
- Uses `uinput` (works on both X11 and Wayland)
- Requires daemon and permissions
- API similar to xdotool

### 4.2 Key Differences

| Aspect | X11 | Wayland |
|--------|-----|---------|
| **Protocol** | XTest extension | zwp_virtual_keyboard_v1 or uinput |
| **Permissions** | None (in X session) | Requires group membership or root |
| **Security** | Global input injection | Isolated, compositor-mediated |
| **Tools** | xdotool, xte | ydotool, wtype |
| **Rust Support** | enigo, x11rb | wayland-client, uinput |
| **Complexity** | Low | Medium-High |

### 4.3 XWayland Fallback

**Current Behavior**: 
- `enigo` v0.6 likely uses X11 backend
- Works through XWayland compatibility layer
- Not ideal: adds latency, not "native" Wayland

**Detection**:
```rust
fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
}

fn is_x11() -> bool {
    std::env::var("DISPLAY").is_ok()
}
```

---

## 5. Security Considerations

### 5.1 Wayland Security Model

**Design Philosophy**:
- Applications cannot spy on each other
- No global screen capture without permission
- No arbitrary input injection

**Implications for eaRS**:
1. Virtual keyboard requires explicit setup
2. User must grant permissions (group membership)
3. Some compositors may block entirely
4. DBus portal may prompt user

### 5.2 Permission Requirements

**For uinput approach**:
```bash
# Check current groups
groups

# Should include 'input'
# If not:
sudo usermod -a -G input $USER
```

**For virtual keyboard protocol**:
- No special permissions usually needed
- Compositor support is the blocker

### 5.3 User Communication

**Documentation Should Include**:
1. List of supported compositors
2. Setup instructions for uinput permissions
3. Troubleshooting for unsupported compositors
4. Fallback to XWayland (if applicable)

---

## 6. Recommendations

### 6.1 Recommended Implementation Strategy

**Phase 1: Immediate (Use uinput)**

**Approach**: Implement `uinput`-based virtual keyboard

**Rationale**:
- ✅ Works on ALL Wayland compositors
- ✅ Works on X11 too (unified codebase)
- ✅ Mature, stable kernel interface
- ✅ Reasonable Rust support
- ⚠️ Requires user setup (permissions)

**Implementation**:
1. Add `uinput` crate dependency
2. Create wrapper abstraction: `VirtualKeyboard` trait
3. Implement for both `uinput` (Linux) and `enigo` (fallback/other platforms)
4. Add documentation for permission setup
5. Add runtime check and helpful error messages

**Phase 2: Future (Native Wayland Protocol)**

When more compositors support virtual keyboard protocol:
- Add `wayland-client` implementation
- Keep `uinput` as fallback
- Auto-detect and choose best method

### 6.2 Hybrid Approach (Recommended)

```rust
// Abstraction layer
trait KeyboardInjector {
    fn type_text(&mut self, text: &str) -> Result<()>;
    fn press_key(&mut self, key: Key) -> Result<()>;
    fn release_key(&mut self, key: Key) -> Result<()>;
}

// Implementations
struct UInputKeyboard { /* ... */ }
struct WaylandVirtualKeyboard { /* ... */ }
struct EnigoKeyboard { /* ... */ }

// Factory
fn create_keyboard() -> Result<Box<dyn KeyboardInjector>> {
    if is_wayland() {
        // Try native Wayland protocol first
        if let Ok(kb) = WaylandVirtualKeyboard::new() {
            return Ok(Box::new(kb));
        }
        // Fallback to uinput
        Ok(Box::new(UInputKeyboard::new()?))
    } else {
        // X11 or other
        Ok(Box::new(EnigoKeyboard::new()?))
    }
}
```

### 6.3 Quick Win: ydotool Wrapper

**For fastest implementation**:
1. Add dependency check for `ydotool` at startup
2. Print helpful error if not found
3. Wrap ydotool commands in Rust functions
4. Update README with installation instructions

**Pros**: Works immediately, no complex code
**Cons**: External dependency, users must install/configure

### 6.4 Dependencies Summary

**Option A - uinput (Recommended)**:
```toml
uinput = "0.1"
```

**Option B - ydotool wrapper**:
```toml
# No Rust dependencies, system package required
```

**Option C - Native Wayland**:
```toml
wayland-client = "0.31"
wayland-protocols = { version = "0.32", features = ["client", "unstable"] }
```

---

## 7. Implementation Roadmap

### Step 1: Add Runtime Detection
- Detect Wayland vs X11
- Check for uinput availability
- Print helpful diagnostics

### Step 2: Implement uinput Backend
- Create `UInputKeyboard` struct
- Implement text typing with modifiers
- Handle special keys (Backspace, Enter, etc.)

### Step 3: Update ears-dictation
- Replace direct `enigo` usage with abstraction
- Add configuration option for input method
- Update error messages

### Step 4: Documentation
- Update README with Wayland support info
- Add setup guide for uinput permissions
- Document supported compositors

### Step 5: Testing
- Test on Sway, Hyprland, KWin
- Verify X11 fallback still works
- Test special characters, modifiers

---

## 8. Code Locations

Current keyboard injection code:
- **Main file**: `src/bin/ears-dictation.rs`
- **Key functions**: 
  - `handle_message()` (line 382-409)
  - Enigo initialization (line 247-248)

**Files to modify**:
1. `Cargo.toml` - Add dependencies
2. `src/bin/ears-dictation.rs` - Update keyboard handling
3. `README.md` - Add Wayland setup instructions
4. `docs/` - Add this document

---

## 9. Additional Resources

### Official Documentation
- [Wayland Protocol Spec](https://wayland.freedesktop.org/docs/html/)
- [Virtual Keyboard Protocol](https://gitlab.freedesktop.org/wayland/wayland-protocols/-/blob/main/unstable/virtual-keyboard/virtual-keyboard-unstable-v1.xml)
- [Linux Input Subsystem](https://www.kernel.org/doc/html/latest/input/input.html)

### Rust Crates
- [wayland-client docs](https://docs.rs/wayland-client/)
- [uinput crate](https://docs.rs/uinput/)
- [enigo](https://docs.rs/enigo/)

### Similar Projects
- [wtype](https://github.com/atx/wtype) - Wayland text typer (C, good reference)
- [ydotool](https://github.com/ReimuNotMoe/ydotool) - Generic command-line automation tool
- [wl-clipboard](https://github.com/bugaevc/wl-clipboard) - Wayland clipboard utilities

### Community
- [r/wayland](https://reddit.com/r/wayland)
- [Wayland Matrix channel](https://matrix.to/#/#wayland:matrix.org)

---

## 10. Conclusion

**Summary**:
- Wayland virtual keyboard is more complex than X11 but achievable
- Multiple viable approaches exist
- **uinput is recommended** for broadest compatibility
- Requires user setup (permissions) but works reliably
- Can maintain X11 support through abstraction layer

**Next Steps**:
1. Decide on implementation approach (recommend uinput)
2. Create abstraction layer for keyboard injection
3. Implement uinput backend
4. Add documentation for user setup
5. Test across different compositors

**Estimated Effort**:
- Quick solution (ydotool wrapper): 2-4 hours
- Robust solution (uinput): 1-2 days
- Full solution (native Wayland + fallbacks): 3-5 days
