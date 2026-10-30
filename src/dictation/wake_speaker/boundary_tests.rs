use super::*;

fn named_config() -> SpeakerConfig {
    SpeakerConfig {
        identity: "vox-owner".into(),
        threshold: 0.6,
        ..Default::default()
    }
}

#[test]
fn named_target_requires_confident_matching_identity_not_just_a_score() {
    let config = named_config();
    for value in [
        serde_json::json!({"score": 0.99}),
        serde_json::json!({"voice_id": "vox-owner", "score": 0.99}),
        serde_json::json!({"voice_id": null, "score": 0.99, "status": "confident"}),
        serde_json::json!({"voice_id": "", "score": 0.99, "status": "confident"}),
        serde_json::json!({"voice_id": "vox-owner", "score": 0.99, "status": "tentative"}),
        serde_json::json!({"voice_id": "vox-owner", "score": 0.2, "status": "confident"}),
        serde_json::json!({"voice_id": "vox-owner", "score": 2, "status": "confident"}),
        serde_json::json!({"voice_id": "vox-owner", "score": null, "status": "insufficient_audio"}),
    ] {
        let result = parse_response(&config, &serde_json::to_vec_pretty(&value).unwrap());
        assert!(
            matches!(result, SpeakerVerdict::Unknown { .. }),
            "{value}: {result:?}"
        );
        assert!(!result.filters(SpeakerMode::Filter));
    }
    let other = br#"{"voice_id":"vox-other","score":0.95,"status":"confident"}"#;
    assert_eq!(
        parse_response(&config, other),
        SpeakerVerdict::Other { score: 0.95 }
    );
    let own = br#"{"voice_id":"vox-owner","score_sample":0.88,"status":"confident"}"#;
    assert_eq!(
        parse_response(&config, own),
        SpeakerVerdict::Owner { score: 0.88 }
    );
    assert!(matches!(
        parse_response(&SpeakerConfig::default(), own),
        SpeakerVerdict::Unknown { .. }
    ));
}

#[test]
fn stdout_must_be_one_complete_object() {
    for bytes in [
        b"".as_slice(),
        b"null",
        b"[]",
        b"[0.9,null,\"vox-owner\",\"confident\",null]",
        b"noise\n{\"score\":0.9}",
        b"{\"score\":0.9}\n{\"score\":0.8}",
        b"{\"score\":1e999}",
        b"\xff",
    ] {
        assert!(matches!(
            parse_response(&named_config(), bytes),
            SpeakerVerdict::Unknown { .. }
        ));
    }
}

#[cfg(unix)]
#[test]
fn pipes_are_drained_concurrently_and_output_is_bounded() {
    use super::tests::{fake, tmpdir};
    use std::time::Instant;
    let dir = tmpdir("bounded-io");
    let wav = write_wav(&dir, &[0.1; 16_000], 16_000).unwrap();
    let prefix = "awk 'BEGIN {for (i=0;i<32000;i++) printf \"x\"}' >&2";
    let config = fake(&dir, &format!("{prefix}; echo '{{\"voice_id\":\"vox-tommy\",\"score\":0.9,\"status\":\"confident\"}}'"));
    assert_eq!(
        check(&config, &wav, 2000),
        SpeakerVerdict::Owner { score: 0.9 }
    );
    for redirect in ["", ">&2"] {
        let config = fake(
            &dir,
            &format!("awk 'BEGIN {{for (i=0;i<100000;i++) printf \"x\"}}' {redirect}"),
        );
        let start = Instant::now();
        let result = check(&config, &wav, 2000);
        assert!(
            matches!(result, SpeakerVerdict::Unknown { ref reason } if reason.contains("limit")),
            "{result:?}"
        );
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}

#[cfg(unix)]
#[test]
fn deadline_covers_process_and_pipe_eof_without_blocking_reader_threads() {
    use super::tests::{fake, tmpdir};
    use std::time::Instant;
    let dir = tmpdir("deadline");
    let wav = write_wav(&dir, &[0.1; 16_000], 16_000).unwrap();
    for script in ["exec sleep 2", "sleep 1 & exit 0"] {
        let mut config = fake(&dir, script);
        config.timeout_ms = 80;
        let start = Instant::now();
        let result = check(&config, &wav, 2000);
        assert_eq!(result, unknown("speaker command timed out"));
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}

#[cfg(unix)]
#[test]
fn nonzero_exit_cannot_smuggle_valid_json_and_stderr_is_not_evidence() {
    use super::tests::{fake, tmpdir};
    let dir = tmpdir("bad-exit");
    let wav = write_wav(&dir, &[0.1; 16_000], 16_000).unwrap();
    let config = fake(&dir, "echo '{\"voice_id\":\"vox-tommy\",\"score\":0.99,\"status\":\"confident\"}'; echo private-fixture-text >&2; exit 7");
    let result = check(&config, &wav, 2000);
    assert!(
        matches!(result, SpeakerVerdict::Unknown { ref reason } if !reason.contains("private-fixture-text"))
    );
}

#[tokio::test]
async fn async_callers_must_use_a_blocking_worker_not_a_nested_runtime() {
    let result = process::run(
        &["unused".into()],
        Path::new("fixture.wav"),
        Duration::from_secs(1),
    );
    assert_eq!(
        result,
        Err("speaker check requires a blocking worker".into())
    );
}
