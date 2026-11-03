use super::*;

#[test]
fn picker_keeps_default_and_disables_ambiguous_or_unnamed_devices() {
    assert_eq!(choices(None).len(), 1);
    let catalog = InputDevices {
        default_name: None,
        devices: vec![
            InputDeviceInfo {
                index: 0,
                name: Some("USB microphone".into()),
            },
            InputDeviceInfo {
                index: 1,
                name: Some("USB microphone".into()),
            },
            InputDeviceInfo {
                index: 2,
                name: None,
            },
            InputDeviceInfo {
                index: 3,
                name: Some("Built-in microphone".into()),
            },
        ],
    };
    let rows = choices(Some(&catalog));
    assert!(rows[0].selectable && rows[0].name.is_none());
    assert!(!rows[1].selectable && !rows[2].selectable && !rows[3].selectable);
    assert!(rows[4].selectable);
}

#[test]
fn microphone_edit_preserves_other_preferences_and_default_removes_only_the_key() {
    let text = "# keep\n[dictation]\nenabled=true\ninput_device='Old microphone' # selection\n[dictation.auto_pause]\nidle_ms=3000 # keep too\n[future]\nvalue='unchanged'\n";
    let updated = updated_document(text, Some("USB microphone")).unwrap();
    for retained in ["# keep", "# selection", "idle_ms=3000 # keep too"] {
        assert!(updated.contains(retained), "lost comment: {retained}");
    }
    let mut expected: toml::Value = toml::from_str(text).unwrap();
    expected["dictation"]["input_device"] = "USB microphone".into();
    assert_eq!(toml::from_str::<toml::Value>(&updated).unwrap(), expected);
    expected["dictation"]
        .as_table_mut()
        .unwrap()
        .remove("input_device");
    assert_eq!(
        toml::from_str::<toml::Value>(&updated_document(&updated, None).unwrap()).unwrap(),
        expected
    );
    for (invalid, choice) in [
        ("dictation=3", Some("USB microphone")),
        (text, Some("")),
        ("not TOML", None),
    ] {
        assert!(updated_document(invalid, choice).is_err());
    }
}

#[test]
fn atomic_save_keeps_permissions_symlink_and_failed_edit_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    assert!(persist(&path, Some("USB microphone")).is_err());
    assert!(!path.exists());
    std::fs::write(&path, "# keep\n[dictation]\nenabled=true\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    persist(&path, Some("USB microphone")).unwrap();
    let before = std::fs::read_to_string(&path).unwrap();
    assert!(persist(&path, Some("")).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let link = dir.path().join("linked.toml");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        persist(&link, None).unwrap();
        assert!(link.is_symlink());
        assert!(
            !std::fs::read_to_string(&path)
                .unwrap()
                .contains("input_device")
        );
    }
}

#[test]
fn concurrent_config_change_is_not_overwritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(&path, "# original\n").unwrap();
    assert!(
        super::super::auto_pause::persist_edit(&path, |text| {
            std::fs::write(&path, "# concurrent edit\n")?;
            updated_document(text, Some("USB microphone"))
        })
        .is_err()
    );
    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        "# concurrent edit\n"
    );
}
