//! Packaging diagnostics must exit before config, model or capture startup.
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[test]
fn every_entry_point_reports_version_and_capabilities_without_writes() {
    let home = std::env::temp_dir().join(format!(
        "ears-metadata-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&home).unwrap();
    let result = std::panic::catch_unwind(|| {
        for binary in [
            env!("CARGO_BIN_EXE_ears"),
            env!("CARGO_BIN_EXE_ears-server"),
            env!("CARGO_BIN_EXE_ears-dictation"),
        ] {
            for flag in ["--help", "--version", "--build-info"] {
                let mut command = Command::new(binary);
                command
                    .arg(flag)
                    .current_dir(&home)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                for variable in [
                    "HOME",
                    "USERPROFILE",
                    "APPDATA",
                    "LOCALAPPDATA",
                    "XDG_CONFIG_HOME",
                    "XDG_STATE_HOME",
                    "XDG_DATA_HOME",
                    "XDG_CACHE_HOME",
                    "HF_HOME",
                ] {
                    command.env(variable, &home);
                }
                let mut child = command.spawn().unwrap();
                let start = Instant::now();
                while child.try_wait().unwrap().is_none() {
                    if start.elapsed() > Duration::from_secs(15) {
                        let _ = child.kill();
                        let _ = child.wait();
                        panic!("metadata command timed out: {binary} {flag}");
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                let output = child.wait_with_output().unwrap();
                assert!(output.status.success(), "{binary} {flag}: {output:?}");
                assert!(
                    output.stderr.is_empty(),
                    "{binary} {flag}: {:?}",
                    output.stderr
                );
                let text = String::from_utf8(output.stdout).unwrap();
                if flag == "--build-info" {
                    let info: serde_json::Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(info["schema"], "ears.build-info/v1");
                    assert_eq!(info["version"], env!("CARGO_PKG_VERSION"));
                    assert_eq!(info["client_vad"], cfg!(feature = "client-vad"));
                    assert_eq!(info["models_loaded"], false);
                    assert_eq!(info["microphone_started"], false);
                } else if flag == "--version" {
                    assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
                }
                assert_eq!(
                    std::fs::read_dir(&home).unwrap().count(),
                    0,
                    "{binary} {flag} wrote user state"
                );
            }
        }
    });
    std::fs::remove_dir_all(&home).unwrap();
    if let Err(error) = result {
        std::panic::resume_unwind(error);
    }
}
