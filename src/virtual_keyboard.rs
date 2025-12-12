//! Virtual keyboard abstraction for cross-platform keyboard input injection.
//!
//! Supports:
//! - Linux (Wayland/X11): uinput kernel interface
//! - Other platforms: enigo fallback

use anyhow::{Context, Result, anyhow};

#[cfg(target_os = "linux")]
use uinput::{Device, event::keyboard};

use enigo::{Direction, Enigo, Keyboard, Settings};

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
