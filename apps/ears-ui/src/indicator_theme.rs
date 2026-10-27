//! State colours from the active system palette, with a theme fallback.
use crate::parametric_mark::State;
use crate::platform;
use crate::theme::{Color, Theme};
use gpui::{Global, Hsla};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub struct IndicatorPalette {
    colors: [Color; 6],
}
impl Global for IndicatorPalette {}

impl IndicatorPalette {
    pub fn resolve(theme: &Theme, scheme: &str) -> Self {
        let mut colors = [
            theme.slots[0x05],
            theme.slots[0x0c],
            theme.slots[0x0d],
            theme.slots[0x0e],
            theme.slots[0x0b],
            theme.slots[0x08],
        ];
        colors = resolve_colors(
            colors,
            scheme,
            &candidates(),
            platform::system_indicator_colors,
        );
        Self { colors }
    }

    pub fn color(&self, state: State, background: Color) -> Hsla {
        let i = match state {
            State::Ready => 0,
            State::Listening => 1,
            State::Speech => 2,
            State::Processing => 3,
            State::Done => 4,
            State::Attention => 5,
        };
        visible(self.colors[i], background).hsla()
    }
}

fn resolve_colors(
    fallback: [Color; 6],
    scheme: &str,
    paths: &[PathBuf],
    system: impl FnOnce() -> Option<[Color; 6]>,
) -> [Color; 6] {
    if !scheme.trim().is_empty() && !scheme.trim().eq_ignore_ascii_case("auto") {
        return fallback;
    }
    if let Some(path) = paths.iter().find(|path| path.is_file()) {
        return std::fs::read_to_string(path)
            .ok()
            .and_then(|text| from_omarchy(&text, fallback))
            .unwrap_or(fallback);
    }
    system().unwrap_or(fallback)
}

fn candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("OMARCHY_THEME_FILE") {
        paths.push(path.into());
    }
    let home = dirs::home_dir();
    let state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(".local/state")));
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(".config")));
    if let Some(state) = state {
        paths.push(state.join("omarchy/current/theme/colors.toml"));
    }
    if let Some(config) = config {
        paths.push(config.join("omarchy/current/theme/colors.toml"));
    }
    paths
}

fn from_omarchy(text: &str, mut colors: [Color; 6]) -> Option<[Color; 6]> {
    let value: toml::Value = toml::from_str(text).ok()?;
    let table = value.as_table()?;
    let keys: [&[&str]; 6] = [
        &["foreground", "color7"],
        &["cyan", "color6", "accent"],
        &["blue", "color4", "accent"],
        &["magenta", "color5", "accent"],
        &["green", "color2", "accent"],
        &["red", "color1", "accent"],
    ];
    for (index, options) in keys.iter().enumerate() {
        if let Some(color) = options
            .iter()
            .filter_map(|key| table.get(*key).and_then(toml::Value::as_str))
            .find_map(|v| {
                Color::parse(v)
                    .ok()
                    .filter(|c| [c.r, c.g, c.b, c.a].iter().all(|v| v.is_finite()))
            })
        {
            colors[index] = Color { a: 1.0, ..color };
        }
    }
    Some(colors)
}

fn luminance(color: Color) -> f32 {
    let linear = |v: f32| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * linear(color.r) + 0.7152 * linear(color.g) + 0.0722 * linear(color.b)
}
fn contrast(a: Color, b: Color) -> f32 {
    let (a, b) = (luminance(a), luminance(b));
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}
fn visible(color: Color, background: Color) -> Color {
    if contrast(color, background) >= 3.0 {
        return color;
    }
    if contrast(Color::WHITE, background) >= contrast(Color::BLACK, background) {
        Color::WHITE
    } else {
        Color::BLACK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_ansi_and_semantic_theme_and_preserves_missing_slots() {
        let fallback = [Color::WHITE; 6];
        let actual = from_omarchy("foreground='#112233'\ncyan='#00aaaa'\ncolor4='#3344ff'\ncolor5='#bb33ff'\ncolor2='#22bb44'\ncolor1='#ff3322'", fallback).unwrap();
        assert_eq!(actual[0], Color::parse("#112233").unwrap());
        assert_eq!(actual[1], Color::parse("#00aaaa").unwrap());
        assert_eq!(actual[2], Color::parse("#3344ff").unwrap());
        assert_eq!(actual[5], Color::parse("#ff3322").unwrap());
        assert_eq!(
            from_omarchy("cyan='#123456'", fallback).unwrap()[2],
            Color::WHITE
        );
        assert!(from_omarchy("invalid = [", fallback).is_none());
    }
    #[test]
    fn theme_sources_reload_and_explicit_scheme_wins() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ears-theme-test-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("colors.toml");
        let fallback = [Color::WHITE; 6];
        let system = [Color::BLACK; 6];
        assert_eq!(
            resolve_colors(fallback, "auto", &[path.clone()], || Some(system)),
            system
        );
        std::fs::write(&path, "blue = '#112233'").unwrap();
        assert_eq!(
            resolve_colors(fallback, "auto", &[path.clone()], || Some(system))[2],
            Color::parse("#112233").unwrap()
        );
        std::fs::write(&path, "blue = '#445566'").unwrap();
        assert_eq!(
            resolve_colors(fallback, "auto", &[path.clone()], || Some(system))[2],
            Color::parse("#445566").unwrap()
        );
        assert_eq!(
            resolve_colors(fallback, "custom", &[path.clone()], || Some(system)),
            fallback
        );
        std::fs::write(&path, "broken = [").unwrap();
        assert_eq!(
            resolve_colors(fallback, "auto", &[path.clone()], || Some(system)),
            fallback
        );
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn low_contrast_theme_color_remains_visible() {
        let black = Color::BLACK;
        let dark = Color::parse("#101010").unwrap();
        assert_eq!(visible(dark, black), Color::WHITE);
        assert_eq!(visible(Color::WHITE, Color::WHITE), Color::BLACK);
    }
}
