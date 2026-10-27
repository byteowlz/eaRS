#![cfg(feature = "moonshine")]

use std::process::Command;

#[test]
fn local_export_override_is_explicit_and_never_downloads_a_fallback() {
    let root = std::env::temp_dir().join(format!("ears-moonshine-cli-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ears"))
        .args([
            "server",
            "run",
            "--engine",
            "moonshine",
            "--moonshine-model",
        ])
        .arg(root.join("missing-model"))
        .env("HF_HOME", root.join("hf"))
        .env("HF_HUB_OFFLINE", "1")
        .env("XDG_STATE_HOME", root.join("state"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("Moonshine model directory not found"),
        "{error}"
    );
    assert!(!root.join("hf").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn linux_installer_copies_both_shared_libraries_beside_binaries() {
    let root = std::env::temp_dir().join(format!("ears-sdk-install-{}", uuid::Uuid::new_v4()));
    let target = root.join("target");
    let sdk = target.join("release/build/moonshine-sys-fixture/out/moonshine-prebuilt/moonshine-voice-linux-x86_64/lib");
    std::fs::create_dir_all(&sdk).unwrap();
    for name in ["libmoonshine.so", "libonnxruntime.so.1"] {
        std::fs::write(sdk.join(name), name).unwrap();
    }
    let bin = root.join("bin");
    let status = Command::new("bash")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/install-moonshine-runtime.sh"
        ))
        .arg(&bin)
        .arg(&target)
        .status()
        .unwrap();
    assert!(status.success());
    for name in ["libmoonshine.so", "libonnxruntime.so.1"] {
        assert_eq!(std::fs::read_to_string(bin.join(name)).unwrap(), name);
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn all_entry_points_expose_moonshine_and_server_model_selection() {
    for (binary, args) in [
        (env!("CARGO_BIN_EXE_ears"), vec!["server", "run", "--help"]),
        (env!("CARGO_BIN_EXE_ears-server"), vec!["--help"]),
        (env!("CARGO_BIN_EXE_ears-dictation"), vec!["--help"]),
    ] {
        let output = Command::new(binary).args(args).output().unwrap();
        assert!(output.status.success());
        let help = String::from_utf8_lossy(&output.stdout);
        assert!(help.contains("moonshine"), "{help}");
        if !binary.ends_with("ears-dictation") {
            assert!(help.contains("--moonshine-model"), "{help}");
        }
    }
}
