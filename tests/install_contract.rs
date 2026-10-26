//! Dependency/install guards for trx-fahg. `cargo install` normally ignores
//! Cargo.lock, so an optional pin must actually be activated by its engine.

#[test]
fn nemotron_activates_exact_dynamic_ort_pin() {
    let manifest: toml::Value = toml::from_str(include_str!("../Cargo.toml")).unwrap();
    let features = manifest["features"].as_table().unwrap();
    assert!(!features.contains_key("parakeet"));
    assert!(
        !manifest["dependencies"]
            .as_table()
            .unwrap()
            .contains_key("prkt")
    );
    let engine = features["parakeet-rs"].as_array().unwrap();
    assert!(engine.iter().any(|f| f.as_str() == Some("dep:ort")));
    let ort = &manifest["dependencies"]["ort"];
    assert_eq!(ort["version"].as_str(), Some("=2.0.0-rc.12"));
    assert_eq!(ort["default-features"].as_bool(), Some(false));
    assert!(
        ort["features"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| { f.as_str() == Some("load-dynamic") })
    );
}

#[test]
fn supported_local_installers_use_the_lockfile() {
    for (path, source) in [
        ("justfile", include_str!("../justfile")),
        (
            "install-ears.sh",
            include_str!("../scripts/install-ears.sh"),
        ),
        (
            "install-ears.ps1",
            include_str!("../scripts/install-ears.ps1"),
        ),
    ] {
        for line in source
            .lines()
            .filter(|line| line.contains("cargo install --"))
        {
            assert!(line.contains("--locked"), "{path}: {line}");
        }
    }
}
