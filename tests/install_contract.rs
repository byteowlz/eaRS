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

/// Execute the real just recipe with command stubs: no installations or network.
fn run_install_all(os: &str, arch: &str, input: &str, target: Option<&str>) -> String {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let justfile = include_str!("../justfile");
    let recipe = justfile
        .split("install-all:\n")
        .nth(1)
        .unwrap()
        .split("\n# Install the ears-ui")
        .next()
        .unwrap();
    let body = recipe
        .lines()
        .map(|line| line.strip_prefix("    ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let harness = "cargo() { echo MOCK_CARGO \"$*\"; echo MOCK_TARGET \"${CARGO_TARGET_DIR-unset}\"; }\njust() { echo MOCK_JUST \"$*\"; }\nbash() { echo MOCK_BASH \"$*\"; echo MOCK_HELPER_TARGET \"${CARGO_TARGET_DIR-unset}\"; }\nuname() { echo \"$TEST_ARCH\"; }\n";
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(format!("{harness}\n{body}"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("OSTYPE", os)
        .env("TEST_ARCH", arch)
        .env_remove("CARGO_TARGET_DIR");
    if let Some(target) = target {
        command.env("CARGO_TARGET_DIR", target);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

#[test]
fn install_all_includes_moonshine_and_runtime_on_supported_hosts() {
    for input in ["y\n", "c\n5\n"] {
        let output = run_install_all("darwin-test", "arm64", input, None);
        let install = output
            .lines()
            .find(|line| line.starts_with("MOCK_CARGO install "))
            .unwrap();
        assert!(install.contains("moonshine"), "{output}");
        assert!(
            output.contains("MOCK_BASH ./scripts/install-moonshine-runtime.sh"),
            "{output}"
        );
        assert!(
            output.find("MOCK_CARGO install").unwrap()
                < output
                    .find("MOCK_BASH ./scripts/install-moonshine-runtime.sh")
                    .unwrap()
        );
    }
    let output = run_install_all("linux-gnu", "x86_64", "y\n", None);
    assert!(
        output
            .lines()
            .any(|line| line.starts_with("MOCK_CARGO install ") && line.contains("moonshine")),
        "{output}"
    );
    assert!(
        output.contains("MOCK_BASH ./scripts/install-moonshine-runtime.sh"),
        "{output}"
    );
    let expected_target = format!("{}/target", env!("CARGO_MANIFEST_DIR"));
    assert!(
        output.contains(&format!("MOCK_TARGET {expected_target}")),
        "{output}"
    );
    assert!(
        output.contains(&format!("MOCK_HELPER_TARGET {expected_target}")),
        "{output}"
    );
    let custom = run_install_all(
        "linux-gnu",
        "x86_64",
        "y\n",
        Some("/tmp/custom-ears-target"),
    );
    assert!(custom.contains("MOCK_TARGET /tmp/custom-ears-target"));
    assert!(custom.contains("MOCK_HELPER_TARGET /tmp/custom-ears-target"));
}

#[test]
fn install_all_preserves_unsupported_and_cancelled_paths() {
    let unsupported = run_install_all("darwin-test", "x86_64", "y\n", None);
    assert!(
        !unsupported
            .lines()
            .any(|line| line.starts_with("MOCK_CARGO install ") && line.contains("moonshine"))
    );
    assert!(!unsupported.contains("MOCK_BASH ./scripts/install-moonshine-runtime.sh"));
    let cancelled = run_install_all("darwin-test", "arm64", "n\n", None);
    assert!(!cancelled.contains("MOCK_CARGO install"));
    assert!(!cancelled.contains("MOCK_BASH ./scripts/install-moonshine-runtime.sh"));
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
