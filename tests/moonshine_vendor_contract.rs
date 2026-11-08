//! Portable runtime selection contract; no model download or microphone.
use std::{fs, path::Path};

#[test]
fn ordinary_moonshine_build_uses_the_vendored_sys_crate() {
    let manifest = fs::read_to_string("Cargo.toml").unwrap();
    assert!(
        manifest.contains("moonshine-sys = { path = \"vendor/moonshine-sys\" }"),
        "normal builds still select the unpatched upstream sys crate"
    );
    assert!(Path::new("vendor/moonshine-sys/core-source.tar.gz").is_file());
    assert!(manifest.contains("moonshine-rs = { path = \"vendor/moonshine-rs\""));
    let wrapper = fs::read_to_string("vendor/moonshine-rs/Cargo.toml").unwrap();
    assert!(wrapper.contains("path = \"../moonshine-sys\""));
}

#[test]
fn vendored_builder_cannot_silently_download_an_unpatched_sdk() {
    let builder = fs::read_to_string("vendor/moonshine-sys/build.rs")
        .expect("the vendored builder must be present");
    assert!(!builder.contains("download_prebuilt_release"));
    assert!(!builder.contains("Falling back"));
    assert!(builder.contains("rerun-if-changed"));
}
