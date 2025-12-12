# Wayland Virtual Keyboard - Implementation Example

This document provides a practical, copy-paste ready implementation for adding Wayland virtual keyboard support to eaRS.

## Recommended Approach: uinput with Abstraction Layer

This implementation provides:
- Native uinput support for Wayland/X11
- Fallback to enigo for other platforms
- Clean abstraction for future improvements
- Proper error handling and user-friendly messages

---

## Step 1: Update Cargo.toml

```toml
[dependencies]
# Existing dependencies...
enigo = "0.6"
rdev = "0.5"

# Add for Wayland/Linux virtual keyboard
[target.'cfg(target_os = "linux")'.dependencies]
uinput = "0.1"
```

---

## Step 2: Create Virtual Keyboard Abstraction

Create a new file: `src/virtual_keyboard.rs`

```rust
//! Virtual keyboard abstraction for cross-platform keyboard input injection.
//! 
//! Supports:
//! - Linux (Wayland/X11): uinput kernel interface
//! - Other platforms: enigo fallback

use anyhow::{Context, Result, anyhow};

#[cfg(target_os = "linux")]
use uinput::{Device, event::keyboard};

#[cfg(not(target_os = "linux"))]
use enigo::{Direction, Enigo, Key as EnigoKey, Keyboard, Settings};

/// Cross-platform virtual keyboard trait
pub trait VirtualKeyboard {
    /// Type text into the focused application
    fn type_text(&mut self, text: &str) -> Result<()>;
    
    /// Press and release a special key
    fn press_key(&mut self, key: SpecialKey) -> Result<()>;
}

/// Special keys that can be pressed
#[derive(Debug, Clone, Copy)]
pub enum SpecialKey {
    Enter,
    Backspace,
    Delete,
    Tab,
    Space,
    Escape,
    Left,
    Right,
    Up,
    Down,
}

/// Create the appropriate keyboard implementation for the current platform
pub fn create_virtual_keyboard() -> Result<Box<dyn VirtualKeyboard>> {
    #[cfg(target_os = "linux")]
    {
        UInputKeyboard::new()
            .map(|kb| Box::new(kb) as Box<dyn VirtualKeyboard>)
            .or_else(|e| {
                eprintln!("Warning: Failed to create uinput keyboard: {}", e);
                eprintln!("Falling back to enigo (may not work properly on Wayland)");
                Ok(Box::new(EnigoKeyboard::new()?) as Box<dyn VirtualKeyboard>)
            })
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        Ok(Box::new(EnigoKeyboard::new()?))
    }
}

// ============================================================================
// Linux uinput Implementation
// ============================================================================

#[cfg(target_os = "linux")]
pub struct UInputKeyboard {
    device: Device,
}

#[cfg(target_os = "linux")]
impl UInputKeyboard {
    pub fn new() -> Result<Self> {
        // Try to open /dev/uinput
        let device = uinput::open("/dev/uinput")
            .context("Failed to open /dev/uinput. Please ensure:\n\
                      1. You are in the 'input' group: sudo usermod -a -G input $USER\n\
                      2. The uinput module is loaded: sudo modprobe uinput\n\
                      3. You have logged out and back in after adding to group")?
            .name("eaRS Virtual Keyboard")?
            .event(keyboard::Keyboard::All)?
            .create()
            .context("Failed to create uinput device")?;
        
        Ok(Self { device })
    }
    
    fn type_char(&mut self, ch: char) -> Result<()> {
        let needs_shift = ch.is_ascii_uppercase() || matches!(ch, 
            '!' | '@' | '#' | '$' | '%' | '^' | '&' | '*' | '(' | ')' |
            '_' | '+' | '{' | '}' | '|' | ':' | '"' | '<' | '>' | '?'
        );
        
        let key = char_to_key(ch)?;
        
        if needs_shift {
            self.device.press(&keyboard::Key::LeftShift)?;
        }
        
        self.device.click(&key)?;
        
        if needs_shift {
            self.device.release(&keyboard::Key::LeftShift)?;
        }
        
        self.device.synchronize()?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl VirtualKeyboard for UInputKeyboard {
    fn type_text(&mut self, text: &str) -> Result<()> {
        for ch in text.chars() {
            self.type_char(ch)?;
        }
        Ok(())
    }
    
    fn press_key(&mut self, key: SpecialKey) -> Result<()> {
        let uinput_key = match key {
            SpecialKey::Enter => keyboard::Key::Enter,
            SpecialKey::Backspace => keyboard::Key::BackSpace,
            SpecialKey::Delete => keyboard::Key::Delete,
            SpecialKey::Tab => keyboard::Key::Tab,
            SpecialKey::Space => keyboard::Key::Space,
            SpecialKey::Escape => keyboard::Key::Esc,
            SpecialKey::Left => keyboard::Key::Left,
            SpecialKey::Right => keyboard::Key::Right,
            SpecialKey::Up => keyboard::Key::Up,
            SpecialKey::Down => keyboard::Key::Down,
        };
        
        self.device.click(&uinput_key)?;
        self.device.synchronize()?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn char_to_key(ch: char) -> Result<keyboard::Key> {
    use keyboard::Key::*;
    
    let key = match ch.to_ascii_lowercase() {
        'a' => A, 'b' => B, 'c' => C, 'd' => D, 'e' => E,
        'f' => F, 'g' => G, 'h' => H, 'i' => I, 'j' => J,
        'k' => K, 'l' => L, 'm' => M, 'n' => N, 'o' => O,
        'p' => P, 'q' => Q, 'r' => R, 's' => S, 't' => T,
        'u' => U, 'v' => V, 'w' => W, 'x' => X, 'y' => Y,
        'z' => Z,
        
        '0' | ')' => _0, '1' | '!' => _1, '2' | '@' => _2,
        '3' | '#' => _3, '4' | '$' => _4, '5' | '%' => _5,
        '6' | '^' => _6, '7' | '&' => _7, '8' | '*' => _8,
        '9' | '(' => _9,
        
        ' ' => Space,
        '-' | '_' => Minus,
        '=' | '+' => Equal,
        '[' | '{' => LeftBrace,
        ']' | '}' => RightBrace,
        '\\' | '|' => BackSlash,
        ';' | ':' => SemiColon,
        '\'' | '"' => Apostrophe,
        ',' | '<' => Comma,
        '.' | '>' => Dot,
        '/' | '?' => Slash,
        '`' | '~' => Grave,
        
        '\n' => Enter,
        '\t' => Tab,
        
        _ => return Err(anyhow!("Unsupported character: '{}'", ch)),
    };
    
    Ok(key)
}

// ============================================================================
// Fallback: Enigo Implementation
// ============================================================================

pub struct EnigoKeyboard {
    enigo: Enigo,
}

impl EnigoKeyboard {
    pub fn new() -> Result<Self> {
        let enigo = Enigo::new(&Settings::default())
            .context("Failed to initialize enigo keyboard controller")?;
        Ok(Self { enigo })
    }
}

impl VirtualKeyboard for EnigoKeyboard {
    fn type_text(&mut self, text: &str) -> Result<()> {
        self.enigo.text(text)
            .context("Failed to type text with enigo")
    }
    
    fn press_key(&mut self, key: SpecialKey) -> Result<()> {
        use enigo::Key::*;
        
        let enigo_key = match key {
            SpecialKey::Enter => Return,
            SpecialKey::Backspace => Backspace,
            SpecialKey::Delete => Delete,
            SpecialKey::Tab => Tab,
            SpecialKey::Space => Space,
            SpecialKey::Escape => Escape,
            SpecialKey::Left => LeftArrow,
            SpecialKey::Right => RightArrow,
            SpecialKey::Up => UpArrow,
            SpecialKey::Down => DownArrow,
        };
        
        self.enigo.key(enigo_key, Direction::Click)
            .context("Failed to press key with enigo")
    }
}
```

---

## Step 3: Update src/lib.rs

Add to `src/lib.rs`:

```rust
// Add this line with other module declarations
pub mod virtual_keyboard;
```

---

## Step 4: Update ears-dictation.rs

Modify `src/bin/ears-dictation.rs`:

### Replace imports at the top:

```rust
// Replace this line:
// use enigo::{Direction, Enigo, Key, Keyboard, Settings};

// With:
use ears::virtual_keyboard::{create_virtual_keyboard, VirtualKeyboard, SpecialKey};
```

### Update main() function:

```rust
// Find this section (around line 247):
// let mut enigo = Enigo::new(&Settings::default())
//     .context("Failed to initialize keyboard controller")?;

// Replace with:
let mut keyboard = create_virtual_keyboard()
    .context("Failed to initialize virtual keyboard. \
              On Linux/Wayland, ensure you are in the 'input' group.")?;
```

### Update handle_message() function:

```rust
// Replace the entire handle_message function (around line 382-409):

fn handle_message(
    json: &Value, 
    keyboard: &mut Box<dyn VirtualKeyboard>, 
    capturing: &Arc<Mutex<bool>>
) -> Result<()> {
    let is_capturing = *capturing.lock().unwrap();

    if let Some(event_type) = json.get("type").and_then(|v| v.as_str()) {
        match event_type {
            "word" if is_capturing => {
                if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
                    if !word.is_empty() {
                        eprintln!("[TYPING WORD] {}", word);
                        keyboard.type_text(word)?;
                        keyboard.press_key(SpecialKey::Space)?;
                    }
                }
            }
            "final" if is_capturing => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        eprintln!("[TYPING FINAL] {}", text);
                        keyboard.type_text(text)?;
                        keyboard.press_key(SpecialKey::Space)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
```

### Update the WebSocket message handling loop:

```rust
// Find the line that calls handle_message (around line 321):
// handle_message(&json, &mut enigo, &capturing)?;

// Replace with:
handle_message(&json, &mut keyboard, &capturing)?;
```

---

## Step 5: User Setup Documentation

Add to `README.md` or create `docs/LINUX_SETUP.md`:

### Linux/Wayland Setup for Dictation

The eaRS dictation feature uses the Linux `uinput` interface for reliable keyboard input on both Wayland and X11.

#### Required Setup:

**1. Load the uinput kernel module:**

```bash
sudo modprobe uinput
```

**2. Make it permanent (load on boot):**

```bash
echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf
```

**3. Add your user to the `input` group:**

```bash
sudo usermod -a -G input $USER
```

**4. Log out and log back in** (or reboot) for group changes to take effect.

**5. Verify your setup:**

```bash
# Check you're in the input group
groups | grep input

# Check /dev/uinput is accessible
ls -l /dev/uinput
```

You should see something like:
```
crw-rw---- 1 root input 10, 223 Dec 12 10:00 /dev/uinput
```

#### Alternative: udev Rule (if above doesn't work)

```bash
echo 'KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"' | \
    sudo tee /etc/udev/rules.d/99-uinput.conf

sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### Supported Environments:

- ✅ Wayland (all compositors: Sway, Hyprland, KWin, GNOME, etc.)
- ✅ X11
- ✅ XWayland

#### Troubleshooting:

**Error: "Failed to open /dev/uinput"**
- Ensure uinput module is loaded: `lsmod | grep uinput`
- Check you're in input group: `groups | grep input`
- Verify you've logged out/in after adding to group

**Dictation not typing:**
- Check `ears dictation debug` output for errors
- Ensure the target application has focus
- Try `sudo dmesg | tail` to check for kernel messages

---

## Step 6: Testing

### Manual Test:

```bash
# Start the server
cargo run --release -- server start

# Start dictation in debug mode
cargo run --release -- dictation debug

# Speak and verify text appears in a text editor
```

### Test Special Characters:

Create a test that verifies:
- Uppercase letters
- Numbers
- Common punctuation: `.`, `,`, `!`, `?`
- Special chars: `@`, `#`, `$`, `%`

---

## Advanced: Command Pattern Support

To support voice commands like "new line" or "delete that", extend `handle_message`:

```rust
fn handle_message(
    json: &Value, 
    keyboard: &mut Box<dyn VirtualKeyboard>, 
    capturing: &Arc<Mutex<bool>>
) -> Result<()> {
    let is_capturing = *capturing.lock().unwrap();

    if let Some(event_type) = json.get("type").and_then(|v| v.as_str()) {
        match event_type {
            "word" if is_capturing => {
                if let Some(word) = json.get("word").and_then(|v| v.as_str()) {
                    if !word.is_empty() {
                        // Check for voice commands
                        match word.to_lowercase().as_str() {
                            "new line" | "newline" => {
                                eprintln!("[COMMAND] New line");
                                keyboard.press_key(SpecialKey::Enter)?;
                                return Ok(());
                            }
                            "delete" | "backspace" => {
                                eprintln!("[COMMAND] Delete");
                                keyboard.press_key(SpecialKey::Backspace)?;
                                return Ok(());
                            }
                            _ => {
                                eprintln!("[TYPING WORD] {}", word);
                                keyboard.type_text(word)?;
                                keyboard.press_key(SpecialKey::Space)?;
                            }
                        }
                    }
                }
            }
            "final" if is_capturing => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        eprintln!("[TYPING FINAL] {}", text);
                        keyboard.type_text(text)?;
                        keyboard.press_key(SpecialKey::Space)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
```

---

## Summary

This implementation provides:

1. **Native Wayland support** via uinput kernel interface
2. **Backward compatibility** with X11
3. **Fallback** to enigo on platforms without uinput
4. **Clean abstraction** for future enhancements
5. **Clear user documentation** for setup

**Estimated implementation time**: 2-3 hours

**Benefits over current approach**:
- ✅ Works reliably on Wayland compositors
- ✅ No XWayland dependency
- ✅ Better performance (kernel-level input)
- ✅ Support for all special keys
- ✅ Extensible for future features
