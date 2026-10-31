//! Launch-method-independent configuration paths. Finder/shortcuts do not inherit
//! shell XDG variables; keep the normal home-based location on every desktop OS.
use std::ffi::OsStr;
use std::path::PathBuf;

fn config_base(xdg: Option<&OsStr>, home: Option<PathBuf>, legacy: Option<PathBuf>) -> PathBuf {
    if let Some(path) = xdg.filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    let preferred = home
        .map(|path| path.join(".config"))
        .unwrap_or_else(|| PathBuf::from(".config"));
    // Do not silently abandon a pre-existing native-OS installation. No files move.
    if !preferred.join("ears").exists() {
        if let Some(old) = legacy.filter(|path| path.join("ears").exists()) {
            return old;
        }
    }
    preferred
}

pub fn config_dir() -> PathBuf {
    config_base(
        std::env::var_os("XDG_CONFIG_HOME").as_deref(),
        dirs::home_dir(),
        dirs::config_dir(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fresh_desktop_uses_xdg_convention_without_shell_environment() {
        let home =
            std::env::temp_dir().join(format!("ears-paths-no-write-{}", uuid::Uuid::new_v4()));
        assert_eq!(
            config_base(None, Some(home.clone()), None),
            home.join(".config")
        );
        assert!(!home.exists());
    }
    #[test]
    fn explicit_override_wins_and_legacy_preferences_are_retained() {
        let root = std::env::temp_dir().join(format!("ears-paths-{}", std::process::id()));
        let (home, legacy) = (root.join("home"), root.join("legacy"));
        std::fs::create_dir_all(legacy.join("ears")).unwrap();
        assert_eq!(
            config_base(None, Some(home.clone()), Some(legacy.clone())),
            legacy
        );
        assert_eq!(
            config_base(
                Some(OsStr::new("/explicit")),
                Some(home.clone()),
                Some(legacy.clone())
            ),
            PathBuf::from("/explicit")
        );
        std::fs::create_dir_all(home.join(".config/ears")).unwrap();
        assert_eq!(
            config_base(None, Some(home.clone()), Some(legacy)),
            home.join(".config")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
