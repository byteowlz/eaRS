use super::*;

fn capabilities(compiled: bool) -> Vec<u8> {
    serde_json::to_vec_pretty(&serde_json::json!({
        "protocol": "voqx-capabilities/1", "version": "0.1.0",
        "recognition": {"compiled": compiled, "identify_json": compiled,
            "preprocessing": ["uniform", "valid-v1"]},
        "model_state": "not_checked", "registry_state": "not_checked"
    }))
    .unwrap()
}

#[test]
fn capability_compatibility_never_implies_model_or_identity_readiness() {
    let initial = || Discovery::new(DiscoveryStatus::ProbeFailed, None);
    let ready = apply_capabilities(initial(), &capabilities(true));
    assert_eq!(ready.status, DiscoveryStatus::CompatibleExecutable);
    assert_eq!(ready.model_state, "not_checked");
    assert_eq!(ready.registry_state, "not_checked");
    assert_eq!(ready.preprocessing, ["uniform", "valid-v1"]);
    assert_eq!(
        apply_capabilities(initial(), &capabilities(false)).status,
        DiscoveryStatus::Unsupported
    );
    for bad in [
        b"garbage".as_slice(),
        b"[]",
        b"{}",
        b"{\"protocol\":\"voqx-capabilities/2\"}",
    ] {
        assert_eq!(
            apply_capabilities(initial(), bad).status,
            DiscoveryStatus::Unsupported
        );
    }
}

#[test]
fn explicit_path_never_falls_back_and_automatic_search_includes_user_prefixes() {
    let override_path = Path::new("/chosen/missing/voqx");
    let path = vec![PathBuf::from("/shell/bin")];
    let home = Path::new("/test-home");
    assert_eq!(
        candidates(
            Some(override_path),
            Some(Path::new("/bundle")),
            &path,
            Some(home)
        ),
        vec![override_path.to_path_buf()]
    );
    let found = candidates(None, Some(Path::new("/bundle")), &path, Some(home));
    assert_eq!(found[0], Path::new("/bundle").join(filename()));
    assert_eq!(found[1], Path::new("/shell/bin").join(filename()));
    assert!(found.contains(&home.join(".cargo/bin").join(filename())));
    assert!(found.contains(&home.join(".local/bin").join(filename())));
    assert!(
        candidates(Some(Path::new("missing-command")), None, &[], Some(home)).is_empty(),
        "bare overrides absent from PATH must not accidentally resolve from cwd"
    );
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(body: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir =
                std::env::temp_dir().join(format!("ears-provider-{}-{nonce}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let file = dir.join("provider with spaces");
            std::fs::write(&file, format!("#!/usr/bin/env bash\n{body}\n")).unwrap();
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o700)).unwrap();
            Self(file)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
        }
    }

    #[tokio::test]
    async fn executable_probe_uses_only_capabilities_and_json_arguments() {
        let json = String::from_utf8(capabilities(true)).unwrap();
        let fixture = Fixture::new(&format!("[ \"$#\" = 2 ] && [ \"$1\" = capabilities ] && [ \"$2\" = --json ] || exit 3\nprintf '%s\\n' '{json}'"));
        let result = discover(Some(&fixture.0)).await;
        assert_eq!(result.status, DiscoveryStatus::CompatibleExecutable);
        assert_eq!(result.executable, Some(fixture.0.canonicalize().unwrap()));
        assert_eq!(result.version.as_deref(), Some("0.1.0"));
        assert_eq!(
            std::fs::read_dir(fixture.0.parent().unwrap())
                .unwrap()
                .count(),
            1
        );
        std::fs::set_permissions(&fixture.0, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(
            discover(Some(&fixture.0)).await.status,
            DiscoveryStatus::NotExecutable
        );
        assert_eq!(
            discover(Some(&fixture.0.with_extension("missing")))
                .await
                .status,
            DiscoveryStatus::NotFound
        );
    }

    #[tokio::test]
    async fn stale_failed_and_hanging_providers_do_not_look_ready() {
        // These cases check response classification, not a one-second process-start benchmark.
        // The deliberately slow case below separately checks the deadline.
        let old = Fixture::new("exit 2");
        let failed = probe(&old.0, Duration::from_secs(5)).await;
        assert_eq!(failed.status, DiscoveryStatus::ProbeFailed, "{failed:?}");
        let malformed = Fixture::new("echo not-json");
        let invalid = probe(&malformed.0, Duration::from_secs(5)).await;
        assert_eq!(invalid.status, DiscoveryStatus::Unsupported, "{invalid:?}");
        let slow = Fixture::new("exec sleep 3");
        let start = Instant::now();
        let result = probe(&slow.0, Duration::from_millis(80)).await;
        assert_eq!(result.status, DiscoveryStatus::ProbeFailed);
        assert!(result
            .detail
            .as_deref()
            .unwrap_or_default()
            .contains("timed out"));
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
