//! Port of the byteowlz design-system mechanism to GPUI.
//!
//! The design system is "mechanism, not look": a scheme provides 24 base24
//! colour *slots*; a closed set of *roles* maps onto those slots; a single
//! *radius dial* drives a proportional scale. Fonts, spacing and effects are
//! per-tool identity and live here as constants.
//!
//! Schemes are the same JSON files the TypeScript implementation ships
//! (`{ id, name, mode, system, slots }`), so a user can point `ui.toml` at any
//! scheme from `../design-system` and get identical colours.

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use gpui::App;
use gpui::Global;
use gpui::Hsla;
use gpui::Pixels;
use gpui::Rgba;
use gpui::SharedString;
use gpui::WindowAppearance;
use gpui::px;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const OQTO_DARK: &str = include_str!("schemes/oqto-dark.json");
pub const OQTO_LIGHT: &str = include_str!("schemes/oqto-light.json");

/// Identity: font stacks. GPUI resolves `.SystemUIFont` to the platform UI
/// font on macOS; the fallbacks cover Linux/Windows.
pub const FONT_SANS: &str = ".SystemUIFont";
pub const FONT_MONO: &str = "Menlo";
/// Identity: spacing unit (0.25rem at 16px).
pub const SPACING: f32 = 4.0;

const SLOT_NAMES: [&str; 24] = [
    "base00", "base01", "base02", "base03", "base04", "base05", "base06", "base07", "base08",
    "base09", "base0A", "base0B", "base0C", "base0D", "base0E", "base0F", "base10", "base11",
    "base12", "base13", "base14", "base15", "base16", "base17",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Dark,
    Light,
}

/// Gamma sRGB colour with alpha, 0..=1 components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    pub fn hsla(self) -> Hsla {
        Rgba {
            r: self.r,
            g: self.g,
            b: self.b,
            a: self.a,
        }
        .into()
    }

    #[allow(dead_code)]
    pub fn with_alpha(mut self, a: f32) -> Self {
        self.a = a.clamp(0.0, 1.0);
        self
    }

    /// `color-mix(in srgb, self, other amount)`.
    pub fn mix(self, other: Color, amount: f32) -> Self {
        let t = amount.clamp(0.0, 1.0);
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    pub fn darken(self, amount: f32) -> Self {
        self.mix(Color::BLACK, amount)
    }

    pub fn lighten(self, amount: f32) -> Self {
        self.mix(Color::WHITE, amount)
    }

    /// Parse `#rgb`, `#rrggbb`, `#rrggbbaa`, `rgb()`, `rgba()` and
    /// `oklch(L C H [/ A])`.
    pub fn parse(input: &str) -> Result<Self> {
        let s = input.trim();
        if let Some(hex) = s.strip_prefix('#') {
            return parse_hex(hex);
        }
        let lower = s.to_ascii_lowercase();
        if let Some(body) = lower
            .strip_prefix("oklch(")
            .and_then(|b| b.strip_suffix(')'))
        {
            return parse_oklch(body);
        }
        if let Some(body) = lower
            .strip_prefix("rgba(")
            .or_else(|| lower.strip_prefix("rgb("))
            .and_then(|b| b.strip_suffix(')'))
        {
            return parse_rgb(body);
        }
        bail!("unsupported colour syntax: {input}")
    }
}

fn parse_hex(hex: &str) -> Result<Color> {
    let digit = |c: u8| -> Result<u8> {
        (c as char)
            .to_digit(16)
            .map(|d| d as u8)
            .ok_or_else(|| anyhow!("invalid hex digit {}", c as char))
    };
    let bytes = hex.as_bytes();
    let (r, g, b, a) = match bytes.len() {
        3 | 4 => {
            let r = digit(bytes[0])? * 17;
            let g = digit(bytes[1])? * 17;
            let b = digit(bytes[2])? * 17;
            let a = if bytes.len() == 4 {
                digit(bytes[3])? * 17
            } else {
                255
            };
            (r, g, b, a)
        }
        6 | 8 => {
            let pair =
                |i: usize| -> Result<u8> { Ok(digit(bytes[i])? * 16 + digit(bytes[i + 1])?) };
            let a = if bytes.len() == 8 { pair(6)? } else { 255 };
            (pair(0)?, pair(2)?, pair(4)?, a)
        }
        _ => bail!("invalid hex colour #{hex}"),
    };
    Ok(Color {
        r: r as f32 / 255.0,
        g: g as f32 / 255.0,
        b: b as f32 / 255.0,
        a: a as f32 / 255.0,
    })
}

fn split_components(body: &str) -> (Vec<String>, Option<String>) {
    let (main, alpha) = match body.split_once('/') {
        Some((m, a)) => (m, Some(a.trim().to_string())),
        None => (body, None),
    };
    let parts = main
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect();
    (parts, alpha)
}

fn parse_number(s: &str) -> Result<f32> {
    let s = s.trim();
    if let Some(p) = s.strip_suffix('%') {
        return Ok(p.trim().parse::<f32>()? / 100.0);
    }
    Ok(s.parse::<f32>()?)
}

fn parse_alpha(alpha: Option<String>) -> Result<f32> {
    match alpha {
        Some(a) => parse_number(&a),
        None => Ok(1.0),
    }
}

fn parse_rgb(body: &str) -> Result<Color> {
    let (parts, alpha) = split_components(body);
    if parts.len() < 3 {
        bail!("rgb() needs three components");
    }
    let channel = |s: &str| -> Result<f32> {
        if s.ends_with('%') {
            parse_number(s)
        } else {
            Ok(s.parse::<f32>()? / 255.0)
        }
    };
    let a = if parts.len() >= 4 {
        parse_number(&parts[3])?
    } else {
        parse_alpha(alpha)?
    };
    Ok(Color {
        r: channel(&parts[0])?.clamp(0.0, 1.0),
        g: channel(&parts[1])?.clamp(0.0, 1.0),
        b: channel(&parts[2])?.clamp(0.0, 1.0),
        a: a.clamp(0.0, 1.0),
    })
}

fn parse_oklch(body: &str) -> Result<Color> {
    let (parts, alpha) = split_components(body);
    if parts.len() < 3 {
        bail!("oklch() needs three components");
    }
    let l = parse_number(&parts[0])?;
    let c = parse_number(&parts[1])?;
    let h = parts[2]
        .trim_end_matches("deg")
        .parse::<f32>()
        .context("oklch hue")?;
    let a = parse_alpha(alpha)?;
    Ok(oklch_to_srgb(l, c, h, a))
}

/// OKLCH -> OKLab -> linear sRGB -> gamma sRGB (clamped).
fn oklch_to_srgb(l: f32, c: f32, h_deg: f32, alpha: f32) -> Color {
    let h = h_deg.to_radians();
    let a = c * h.cos();
    let b = c * h.sin();

    let l_ = l + 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let m_ = l - 0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let s_ = l - 0.089_484_177_5 * a - 1.291_485_548_0 * b;

    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;

    let r = 4.076_741_662_1 * l3 - 3.307_711_591_3 * m3 + 0.230_969_929_2 * s3;
    let g = -1.268_438_004_6 * l3 + 2.609_757_401_1 * m3 - 0.341_319_396_5 * s3;
    let bl = -0.004_196_086_3 * l3 - 0.703_418_614_7 * m3 + 1.707_614_701_0 * s3;

    let gamma = |v: f32| -> f32 {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };
    Color {
        r: gamma(r),
        g: gamma(g),
        b: gamma(bl),
        a: alpha.clamp(0.0, 1.0),
    }
}

/// A parsed scheme: 24 slots, always base24 after derivation.
#[derive(Debug, Clone)]
pub struct Scheme {
    pub id: String,
    pub name: String,
    pub mode: Mode,
    pub slots: [Color; 24],
}

#[derive(Deserialize)]
struct SchemeFile {
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    system: Option<String>,
    slots: HashMap<String, String>,
}

impl Scheme {
    pub fn from_json(json: &str) -> Result<Self> {
        let file: SchemeFile = serde_json::from_str(json).context("parsing scheme JSON")?;
        let mut parsed: Vec<Option<Color>> = vec![None; 24];
        for (name, value) in &file.slots {
            if let Some(idx) = SLOT_NAMES.iter().position(|n| n.eq_ignore_ascii_case(name)) {
                parsed[idx] = Some(Color::parse(value).with_context(|| format!("slot {name}"))?);
            }
        }
        for (idx, slot) in parsed.iter().enumerate().take(16) {
            if slot.is_none() {
                bail!("scheme {} is missing slot {}", file.id, SLOT_NAMES[idx]);
            }
        }
        let base16 =
            file.system.as_deref() == Some("base16") || parsed[16..].iter().any(|s| s.is_none());
        if base16 {
            // base16 policy: derive the base24 extension from the core slots.
            let core: Vec<Color> = parsed[..16].iter().map(|c| c.unwrap()).collect();
            parsed[16] = Some(core[0].darken(0.18));
            parsed[17] = Some(core[0].darken(0.34));
            for (target, source) in (18..24).zip([8usize, 10, 11, 12, 13, 14]) {
                parsed[target] = Some(core[source].lighten(0.22));
            }
        }
        let mut slots = [Color::BLACK; 24];
        for (i, slot) in parsed.into_iter().enumerate() {
            slots[i] = slot.unwrap();
        }
        let mode = match file.mode.as_deref() {
            Some("light") => Mode::Light,
            Some("dark") => Mode::Dark,
            _ => {
                let bg = slots[0];
                if 0.2126 * bg.r + 0.7152 * bg.g + 0.0722 * bg.b > 0.5 {
                    Mode::Light
                } else {
                    Mode::Dark
                }
            }
        };
        Ok(Scheme {
            name: if file.name.is_empty() {
                file.id.clone()
            } else {
                file.name
            },
            id: file.id,
            mode,
            slots,
        })
    }

    pub fn builtin(id: &str) -> Option<Scheme> {
        let json = match id {
            "oqto-dark" => OQTO_DARK,
            "oqto-light" => OQTO_LIGHT,
            _ => return None,
        };
        Scheme::from_json(json).ok()
    }

    pub fn from_path(path: &Path) -> Result<Scheme> {
        let json =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Scheme::from_json(&json)
    }

    pub fn slot(&self, index: usize) -> Color {
        self.slots[index]
    }
}

/// The closed role layer. Field order mirrors `spec/roles.md`.
#[derive(Debug, Clone)]
pub struct Roles {
    pub background: Hsla,
    pub surface: Hsla,
    pub surface_sunken: Hsla,
    pub foreground: Hsla,
    pub muted_foreground: Hsla,
    pub primary: Hsla,
    pub primary_foreground: Hsla,
    pub secondary: Hsla,
    pub secondary_foreground: Hsla,
    #[allow(dead_code)]
    pub muted: Hsla,
    pub accent: Hsla,
    #[allow(dead_code)]
    pub accent_foreground: Hsla,
    pub success: Hsla,
    pub warning: Hsla,
    pub danger: Hsla,
    #[allow(dead_code)]
    pub danger_foreground: Hsla,
    #[allow(dead_code)]
    pub info: Hsla,
    pub border: Hsla,
    pub ring: Hsla,
    pub input: Hsla,
}

impl Roles {
    pub fn from_scheme(scheme: &Scheme) -> Self {
        let s = |i: usize| scheme.slot(i);
        Roles {
            background: s(0x00).hsla(),
            surface: s(0x01).hsla(),
            surface_sunken: s(0x11).hsla(),
            foreground: s(0x05).hsla(),
            muted_foreground: s(0x04).hsla(),
            primary: s(0x0B).hsla(),
            primary_foreground: s(0x00).hsla(),
            secondary: s(0x02).hsla(),
            secondary_foreground: s(0x05).hsla(),
            muted: s(0x02).hsla(),
            accent: s(0x02).hsla(),
            accent_foreground: s(0x05).hsla(),
            success: s(0x0B).hsla(),
            warning: s(0x0A).hsla(),
            danger: s(0x08).hsla(),
            danger_foreground: s(0x07).hsla(),
            info: s(0x0D).hsla(),
            border: s(0x01).hsla(),
            ring: s(0x0B).hsla(),
            input: s(0x01).hsla(),
        }
    }
}

/// Radius dial: one value, proportional scale, everything sharp at 0.
#[derive(Debug, Clone, Copy)]
pub struct Radius {
    pub dial: f32,
}

impl Radius {
    pub fn sm(&self) -> Pixels {
        px(self.dial * 0.5)
    }
    pub fn md(&self) -> Pixels {
        px(self.dial * 0.75)
    }
    pub fn lg(&self) -> Pixels {
        px(self.dial)
    }
    #[allow(dead_code)]
    pub fn xl(&self) -> Pixels {
        px(self.dial * 1.25)
    }
}

#[derive(Debug, Clone)]
pub struct Theme {
    #[allow(dead_code)]
    pub scheme_id: String,
    #[allow(dead_code)]
    pub scheme_name: String,
    pub mode: Mode,
    pub slots: [Color; 24],
    #[allow(dead_code)]
    pub roles: Roles,
    pub radius: Radius,
    pub font_sans: SharedString,
    pub font_mono: SharedString,
    pub spacing: f32,
}

impl Global for Theme {}

impl Theme {
    pub fn from_scheme(scheme: &Scheme, radius_dial: f32) -> Self {
        Theme {
            scheme_id: scheme.id.clone(),
            scheme_name: scheme.name.clone(),
            mode: scheme.mode,
            slots: scheme.slots,
            roles: Roles::from_scheme(scheme),
            radius: Radius {
                dial: radius_dial.max(0.0),
            },
            font_sans: FONT_SANS.into(),
            font_mono: FONT_MONO.into(),
            spacing: SPACING,
        }
    }

    /// Resolve `ui.toml` `[theme]` into a theme for the given OS appearance.
    /// Unknown ids and unreadable files fall back to the house scheme that
    /// matches the appearance, so the UI never renders without a theme.
    pub fn resolve(scheme: &str, radius_dial: f32, appearance: WindowAppearance) -> Self {
        let dark = matches!(
            appearance,
            WindowAppearance::Dark | WindowAppearance::VibrantDark
        );
        let fallback = || {
            Scheme::builtin(if dark { "oqto-dark" } else { "oqto-light" })
                .expect("built-in schemes parse")
        };
        let trimmed = scheme.trim();
        let scheme = if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
            fallback()
        } else if let Some(builtin) = Scheme::builtin(trimmed) {
            builtin
        } else {
            let path = ears::replacement::expand_tilde(trimmed);
            match Scheme::from_path(&path) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("ears-ui: cannot load theme {trimmed}: {err:#}; using built-in");
                    fallback()
                }
            }
        };
        Theme::from_scheme(&scheme, radius_dial)
    }

    pub fn is_dark(&self) -> bool {
        self.mode == Mode::Dark
    }

    /// Slot colour by base24 index (0x00..=0x17).
    #[allow(dead_code)]
    pub fn slot(&self, index: usize) -> Hsla {
        self.slots[index].hsla()
    }

    /// Alpha variant of a role colour (the design system derives every
    /// translucent colour from a role, never from a fresh literal).
    pub fn alpha(color: Hsla, alpha: f32) -> Hsla {
        let mut c = color;
        c.a = alpha.clamp(0.0, 1.0);
        c
    }

    /// Spacing scale helper: `space(4)` = 16px.
    pub fn space(&self, units: f32) -> Pixels {
        px(self.spacing * units)
    }
}

pub fn theme(cx: &App) -> &Theme {
    cx.global::<Theme>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_and_rgba() {
        let c = Color::parse("#3ba77c").unwrap();
        assert!((c.r - 0x3b as f32 / 255.0).abs() < 1e-6);
        let c = Color::parse("rgba(255, 255, 255, 0.07)").unwrap();
        assert!((c.a - 0.07).abs() < 1e-6 && c.r == 1.0);
        let c = Color::parse("#fff8").unwrap();
        assert!((c.a - 136.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn oklch_white_is_white() {
        let c = Color::parse("oklch(1 0 0)").unwrap();
        assert!(c.r > 0.99 && c.g > 0.99 && c.b > 0.99);
        let g = Color::parse("oklch(0.470 0.115 163)").unwrap();
        assert!(g.g > g.r && g.g > g.b);
    }

    #[test]
    fn builtin_schemes_load() {
        let dark = Scheme::builtin("oqto-dark").unwrap();
        assert_eq!(dark.mode, Mode::Dark);
        let light = Scheme::builtin("oqto-light").unwrap();
        assert_eq!(light.mode, Mode::Light);
        let theme = Theme::from_scheme(&dark, 8.0);
        assert_eq!(theme.radius.md(), px(6.0));
        assert_eq!(theme.radius.sm(), px(4.0));
    }

    #[test]
    fn base16_schemes_get_derived_extension() {
        let json = r##"{"id":"t","system":"base16","slots":{
            "base00":"#202020","base01":"#282828","base02":"#383838","base03":"#585858",
            "base04":"#b8b8b8","base05":"#d8d8d8","base06":"#e8e8e8","base07":"#f8f8f8",
            "base08":"#ab4642","base09":"#dc9656","base0A":"#f7ca88","base0B":"#a1b56c",
            "base0C":"#86c1b9","base0D":"#7cafc2","base0E":"#ba8baf","base0F":"#a16946"}}"##;
        let scheme = Scheme::from_json(json).unwrap();
        assert!(scheme.slots[0x10].r < scheme.slots[0].r);
        assert!(scheme.slots[0x12].r > scheme.slots[0x08].r);
        assert_eq!(scheme.mode, Mode::Dark);
    }
}
