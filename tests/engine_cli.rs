//! Legacy Parakeet removal must not make the old name an alias for Nemotron.

use std::process::Command;

#[test]
fn legacy_engine_is_rejected_by_all_entry_points() {
    for (binary, args) in [
        (env!("CARGO_BIN_EXE_ears"), vec!["server", "run"]),
        (env!("CARGO_BIN_EXE_ears-server"), vec![]),
        (env!("CARGO_BIN_EXE_ears-dictation"), vec![]),
    ] {
        let output = Command::new(binary)
            .args(args)
            .args(["--engine", "parakeet"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{binary}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("invalid value 'parakeet'"), "{stderr}");
    }
}

#[cfg(feature = "parakeet-rs")]
#[test]
fn nemotron_remains_available_and_legacy_flags_are_gone() {
    for (binary, args) in [
        (env!("CARGO_BIN_EXE_ears"), vec!["server", "run"]),
        (env!("CARGO_BIN_EXE_ears-server"), vec![]),
    ] {
        let output = Command::new(binary)
            .args(&args)
            .arg("--help")
            .output()
            .unwrap();
        assert!(output.status.success());
        let help = String::from_utf8_lossy(&output.stdout);
        assert!(help.contains("--parakeet-rs-model"), "{help}");
        assert!(help.contains("--parakeet-rs-lang"), "{help}");
        for flag in [
            "--parakeet-device",
            "--parakeet-model-dir",
            "--parakeet-repo",
        ] {
            let output = Command::new(binary)
                .args(&args)
                .arg(flag)
                .arg("unused")
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2), "{binary} accepted {flag}");
        }
    }
    let output = Command::new(env!("CARGO_BIN_EXE_ears-dictation"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&output.stdout).contains("parakeet-rs"));
}
