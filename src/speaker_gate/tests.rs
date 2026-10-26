use super::*;
use std::sync::mpsc;
use voqx_core::recognition::Thresholds;

fn scope(mode: Mode) -> Scope {
    Scope {
        consumer: "synthetic-consumer".into(),
        target: "mock-target".into(),
        generation: 1,
        policy: Policy {
            mode,
            allowed_voice_ids: vec!["vox-a".into(), "vox-b".into()],
            revision: 1,
            deadline_ms: 1000,
        },
        recognition: Revision {
            engine: "synthetic".into(),
            model: "test-checkpoint".into(),
            profiles: "synthetic-r1".into(),
        },
    }
}
fn evidence(voice: &str, status: RecognitionStatus, samples: usize) -> Evidence {
    Evidence {
        recognition: Recognition {
            voice_id: Some(voice.into()),
            display_name: Some("PRIVATE-FIXTURE-NAME".into()),
            engine: "synthetic".into(),
            score: Some(0.99),
            status,
        },
        revision: scope(Mode::Allowlist).recognition,
        covered_samples: samples,
        speaker_consistency: SpeakerConsistency::Consistent,
    }
}
struct Fake {
    answer: Option<Result<Evidence, Reason>>,
    pcm: Option<Arc<std::sync::Mutex<Vec<f32>>>>,
}
impl EvidenceProvider for Fake {
    fn revision(&self) -> Revision {
        scope(Mode::Allowlist).recognition
    }
    fn instruction_capable(&self) -> bool {
        true
    }
    fn evaluate(&mut self, pcm: &[f32], _: &AtomicBool) -> Result<Evidence, Reason> {
        if let Some(recorded) = &self.pcm {
            *recorded.lock().unwrap() = pcm.to_vec();
        }
        self.answer.take().unwrap()
    }
}
fn gate(answer: Result<Evidence, Reason>) -> Gate {
    Gate::new(
        scope(Mode::Allowlist),
        Some(Box::new(Fake {
            answer: Some(answer),
            pcm: None,
        })),
    )
}
fn input(gate: &mut Gate) -> (UtteranceId, AsrInput) {
    let id = gate.begin().unwrap();
    // Generated PCM, never a private recording. Split capture/VAD delivery.
    gate.push_pcm(&id, &[0.1, 0.2]).unwrap();
    gate.push_pcm(&id, &[0.3, 0.4]).unwrap();
    let audio = gate.seal_audio(&id).unwrap();
    (id, audio)
}
fn submit(
    gate: &mut Gate,
    input: &AsrInput,
    revision: u64,
    text: &str,
    now: Instant,
) -> Result<(), Reason> {
    let final_text = input
        .transcribe(revision, |_| Ok::<_, ()>(text.into()))
        .unwrap();
    gate.submit_final(final_text, now)
}
fn settle(gate: &mut Gate, id: &UtteranceId, now: Instant) -> Outcome {
    for _ in 0..1000 {
        gate.poll(now);
        let state = gate.outcome(id).unwrap();
        if state != Outcome::Pending {
            return state;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("test worker failed to settle")
}

#[test]
fn two_allowed_people_join_exact_pcm_final_and_dispatch_once() {
    for voice in ["vox-a", "vox-b"] {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut gate = Gate::new(
            scope(Mode::Allowlist),
            Some(Box::new(Fake {
                answer: Some(Ok(evidence(voice, RecognitionStatus::Confident, 4))),
                pcm: Some(seen.clone()),
            })),
        );
        let now = Instant::now();
        let (id, audio) = input(&mut gate);
        assert_eq!(gate.outcome(&id).unwrap(), Outcome::Pending); // No partial dispatch.
        // Same PCM token on duplicate acoustic boundary; no new operation.
        assert_eq!(gate.seal_audio(&id).unwrap().id(), &id);
        let final_text = audio
            .transcribe(1, |pcm| {
                assert_eq!(pcm, &[0.1, 0.2, 0.3, 0.4]);
                Ok::<_, ()>("open notes".into())
            })
            .unwrap();
        gate.submit_final(final_text, now).unwrap();
        assert_eq!(settle(&mut gate, &id, now), Outcome::Eligible);
        assert_eq!(*seen.lock().unwrap(), vec![0.1, 0.2, 0.3, 0.4]);
        let mut effects = 0;
        gate.dispatch(
            &id,
            "open notes",
            &scope(Mode::Allowlist),
            now,
            || true,
            |text| {
                assert_eq!(text, "open notes");
                effects += 1;
            },
        )
        .unwrap();
        assert_eq!(
            gate.dispatch(
                &id,
                "open notes",
                &scope(Mode::Allowlist),
                now,
                || true,
                |_| effects += 1
            ),
            Err(Reason::AlreadyConsumed)
        );
        assert_eq!(effects, 1);
        assert_eq!(
            submit(&mut gate, &audio, 2, "changed", now),
            Err(Reason::Stale)
        );
    }
}

#[test]
fn disabled_needs_no_provider_but_annotation_cannot_dispatch() {
    let now = Instant::now();
    for mode in [Mode::Disabled, Mode::Annotate] {
        let provider = (mode == Mode::Annotate).then(|| {
            Box::new(Fake {
                answer: Some(Ok(evidence("vox-a", RecognitionStatus::Confident, 4))),
                pcm: None,
            }) as Box<dyn EvidenceProvider>
        });
        let mut gate = Gate::new(scope(mode), provider);
        let (id, audio) = input(&mut gate);
        submit(&mut gate, &audio, 1, "ordinary text", now).unwrap();
        assert_eq!(
            settle(&mut gate, &id, now),
            if mode == Mode::Disabled {
                Outcome::Eligible
            } else {
                Outcome::Observed(None)
            }
        );
        let mut effects = 0;
        let result = gate.dispatch(
            &id,
            "ordinary text",
            &scope(mode),
            if mode == Mode::Disabled {
                now + Duration::from_secs(60)
            } else {
                now
            },
            || true,
            |_| effects += 1,
        );
        assert_eq!(result.is_ok(), mode == Mode::Disabled);
        assert_eq!(effects, usize::from(mode == Mode::Disabled));
    }
}

#[test]
fn every_nonconfident_or_ambiguous_result_has_zero_effects() {
    let cases = [
        (RecognitionStatus::Unknown, Reason::Unknown),
        (RecognitionStatus::Tentative, Reason::Tentative),
        (
            RecognitionStatus::InsufficientAudio,
            Reason::InsufficientAudio,
        ),
        (
            RecognitionStatus::IncompatibleModel,
            Reason::IncompatibleModel,
        ),
    ];
    for (status, expected) in cases {
        denied(Ok(evidence("vox-a", status, 4)), expected);
    }
    denied(
        Ok(evidence("vox-other", RecognitionStatus::Confident, 4)),
        Reason::Disallowed,
    );
    for status in [
        SpeakerConsistency::Uncertain,
        SpeakerConsistency::Overlap,
        SpeakerConsistency::SpeakerChange,
    ] {
        let mut overlap = evidence("vox-a", RecognitionStatus::Confident, 4);
        overlap.speaker_consistency = status;
        denied(Ok(overlap), Reason::AmbiguousSpeaker);
    }
    let mut unsupported = evidence("vox-a", RecognitionStatus::Confident, 4);
    unsupported.speaker_consistency = SpeakerConsistency::Unsupported;
    denied(Ok(unsupported), Reason::SpeakerCheckUnavailable);
    denied(
        Ok(evidence("vox-a", RecognitionStatus::Confident, 2)),
        Reason::UnsupportedCoverage,
    );
    let mut incompatible = evidence("vox-a", RecognitionStatus::Confident, 4);
    incompatible.revision.model = "different-checkpoint".into();
    denied(Ok(incompatible), Reason::IncompatibleModel);
    let mut wrong_engine = evidence("vox-a", RecognitionStatus::Confident, 4);
    wrong_engine.recognition.engine = "different-engine".into();
    denied(Ok(wrong_engine), Reason::IncompatibleModel);
    let mut missing_voice = evidence("vox-a", RecognitionStatus::Confident, 4);
    missing_voice.recognition.voice_id = None;
    denied(Ok(missing_voice), Reason::Unknown);
    for score in [None, Some(f32::NAN), Some(f32::INFINITY), Some(1.1)] {
        let mut invalid_score = evidence("vox-a", RecognitionStatus::Confident, 4);
        invalid_score.recognition.score = score;
        denied(Ok(invalid_score), Reason::Unknown);
    }
    for reason in [
        Reason::ProviderError,
        Reason::ProviderUnavailable,
        Reason::Timeout,
    ] {
        denied(Err(reason), reason);
    }
}
fn denied(answer: Result<Evidence, Reason>, expected: Reason) {
    let mut gate = gate(answer);
    let now = Instant::now();
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "open notes", now).unwrap();
    assert_eq!(settle(&mut gate, &id, now), Outcome::Denied(expected));
    let mut effects = 0;
    assert_eq!(
        gate.dispatch(
            &id,
            "open notes",
            &scope(Mode::Allowlist),
            now,
            || true,
            |_| effects += 1
        ),
        Err(expected)
    );
    assert_eq!(effects, 0);
}

struct Controlled {
    requests: mpsc::Sender<()>,
    answers: mpsc::Receiver<Result<Evidence, Reason>>,
}
impl EvidenceProvider for Controlled {
    fn revision(&self) -> Revision {
        scope(Mode::Allowlist).recognition
    }
    fn instruction_capable(&self) -> bool {
        true
    }
    fn evaluate(&mut self, _: &[f32], _: &AtomicBool) -> Result<Evidence, Reason> {
        self.requests.send(()).unwrap();
        self.answers.recv().unwrap()
    }
}
fn controlled() -> (
    Gate,
    mpsc::Receiver<()>,
    mpsc::Sender<Result<Evidence, Reason>>,
) {
    let (request_tx, request_rx) = mpsc::channel();
    let (answer_tx, answer_rx) = mpsc::channel();
    (
        Gate::new(
            scope(Mode::Allowlist),
            Some(Box::new(Controlled {
                requests: request_tx,
                answers: answer_rx,
            })),
        ),
        request_rx,
        answer_tx,
    )
}
fn started(requests: &mpsc::Receiver<()>) {
    requests.recv_timeout(Duration::from_secs(2)).unwrap();
}

#[test]
fn correction_out_of_order_old_approval_cannot_authorize_new_payload() {
    let (mut gate, requests, answers) = controlled();
    let now = Instant::now();
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "open notes", now).unwrap();
    started(&requests);
    submit(&mut gate, &audio, 2, "do not open notes", now).unwrap();
    answers
        .send(Ok(evidence("vox-a", RecognitionStatus::Confident, 4)))
        .unwrap();
    started(&requests);
    gate.poll(now);
    assert_eq!(gate.outcome(&id).unwrap(), Outcome::Pending);
    assert_eq!(
        submit(&mut gate, &audio, 1, "open notes", now),
        Err(Reason::Stale)
    );
    answers
        .send(Ok(evidence("vox-other", RecognitionStatus::Confident, 4)))
        .unwrap();
    assert_eq!(
        settle(&mut gate, &id, now),
        Outcome::Denied(Reason::Disallowed)
    );
    let mut effects = 0;
    assert!(
        gate.dispatch(
            &id,
            "do not open notes",
            &scope(Mode::Allowlist),
            now,
            || true,
            |_| effects += 1
        )
        .is_err()
    );
    assert_eq!(effects, 0);
}

#[test]
fn cancel_reconnect_target_generation_policy_and_revocation_fence_late_verdicts() {
    for change in 0..6 {
        let (mut gate, requests, answers) = controlled();
        let now = Instant::now();
        let (old_id, old_audio) = input(&mut gate);
        submit(&mut gate, &old_audio, 1, "open notes", now).unwrap();
        started(&requests);
        let mut current = scope(Mode::Allowlist);
        match change {
            0 => {}                       // cancel/restart even without caller changing generation
            1 => current.generation += 1, // reconnect / ASR restart
            2 => current.target = "different-target".into(),
            3 => current.policy.revision += 1,
            4 => current.policy.allowed_voice_ids.clear(), // revoke without revision bump
            _ => current.recognition.profiles = "revoked-profile".into(),
        }
        gate.reconfigure(current.clone());
        assert_eq!(
            submit(&mut gate, &old_audio, 2, "late final", now),
            Err(Reason::Stale)
        );
        let (id, audio) = input(&mut gate);
        submit(&mut gate, &audio, 1, "new instruction", now).unwrap();
        answers
            .send(Ok(evidence("vox-a", RecognitionStatus::Confident, 4)))
            .unwrap();
        if change < 5 {
            started(&requests);
            gate.poll(now);
            assert_eq!(gate.outcome(&id).unwrap(), Outcome::Pending);
            answers
                .send(Ok(evidence("vox-other", RecognitionStatus::Confident, 4)))
                .unwrap();
            assert_eq!(
                settle(&mut gate, &id, now),
                Outcome::Denied(Reason::Disallowed)
            );
        } else {
            assert_eq!(
                settle(&mut gate, &id, now),
                Outcome::Denied(Reason::IncompatibleModel)
            );
        }
        let mut effects = 0;
        assert!(
            gate.dispatch(
                &old_id,
                "open notes",
                &current,
                now,
                || true,
                |_| effects += 1
            )
            .is_err()
        );
        assert!(
            gate.dispatch(
                &id,
                "new instruction",
                &current,
                now,
                || true,
                |_| effects += 1
            )
            .is_err()
        );
        assert_eq!(effects, 0);
    }
}

#[test]
fn current_scope_payload_and_independent_permissions_are_rechecked_at_effect() {
    for change in 0..5 {
        let mut gate = gate(Ok(evidence("vox-a", RecognitionStatus::Confident, 4)));
        let now = Instant::now();
        let (id, audio) = input(&mut gate);
        submit(&mut gate, &audio, 1, "open notes", now).unwrap();
        assert_eq!(settle(&mut gate, &id, now), Outcome::Eligible);
        let mut current = scope(Mode::Allowlist);
        match change {
            0 => current.target = "other".into(),
            1 => current.policy.allowed_voice_ids.clear(),
            2 => current.recognition.profiles = "revoked".into(),
            _ => {}
        }
        let mut effects = 0;
        assert!(
            gate.dispatch(
                &id,
                if change == 3 { "changed" } else { "open notes" },
                &current,
                now,
                || change != 4,
                |_| effects += 1
            )
            .is_err()
        );
        assert_eq!(effects, 0);
    }
}

#[test]
fn timeout_and_queue_overload_are_bounded_and_late_approval_never_dispatches() {
    let (mut gate, requests, answers) = controlled();
    let now = Instant::now();
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "first", now).unwrap();
    started(&requests);
    let (_, second) = input(&mut gate);
    submit(&mut gate, &second, 1, "queued", now).unwrap();
    let (third_id, third) = input(&mut gate);
    assert_eq!(
        submit(&mut gate, &third, 1, "overloaded", now),
        Err(Reason::Overloaded)
    );
    assert_eq!(
        gate.outcome(&third_id).unwrap(),
        Outcome::Denied(Reason::Overloaded)
    );
    gate.begin().unwrap();
    assert_eq!(gate.begin(), Err(Reason::Overloaded));
    let late = now + Duration::from_secs(2);
    gate.poll(late);
    assert_eq!(gate.outcome(&id).unwrap(), Outcome::Denied(Reason::Timeout));
    answers
        .send(Ok(evidence("vox-a", RecognitionStatus::Confident, 4)))
        .unwrap();
    let mut effects = 0;
    assert_eq!(
        gate.dispatch(
            &id,
            "first",
            &scope(Mode::Allowlist),
            late,
            || true,
            |_| effects += 1
        ),
        Err(Reason::Timeout)
    );
    assert_eq!(effects, 0);
}

#[test]
fn missing_provider_payload_reuse_bad_audio_and_resource_limits_fail_closed() {
    let now = Instant::now();
    let mut unavailable = Gate::new(scope(Mode::Allowlist), None);
    let (id, audio) = input(&mut unavailable);
    assert_eq!(
        submit(&mut unavailable, &audio, 1, "open notes", now),
        Err(Reason::ProviderUnavailable)
    );
    assert_eq!(
        unavailable.outcome(&id).unwrap(),
        Outcome::Denied(Reason::ProviderUnavailable)
    );
    assert_eq!(unavailable.status().readiness, Readiness::Unavailable);
    let mut gate = Gate::new(scope(Mode::Disabled), None);
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "one", now).unwrap();
    submit(&mut gate, &audio, 1, "one", now).unwrap(); // identical repeat is idempotent
    assert_eq!(
        submit(&mut gate, &audio, 1, "two", now),
        Err(Reason::PayloadConflict)
    );
    assert_eq!(
        gate.outcome(&id).unwrap(),
        Outcome::Denied(Reason::PayloadConflict)
    );
    for pcm in [vec![f32::NAN], vec![0.0; MAX_SAMPLES + 1]] {
        let id = gate.begin().unwrap();
        assert_eq!(gate.push_pcm(&id, &pcm), Err(Reason::InvalidAudio));
        assert!(matches!(gate.seal_audio(&id), Err(Reason::InvalidAudio)));
        gate.discard(&id);
    }
    let id = gate.begin().unwrap();
    assert!(matches!(
        gate.seal_audio(&id),
        Err(Reason::InsufficientAudio)
    ));
}

#[test]
fn provider_panic_is_a_denial_not_an_ungated_fallback() {
    struct Crash;
    impl EvidenceProvider for Crash {
        fn revision(&self) -> Revision {
            scope(Mode::Allowlist).recognition
        }
        fn instruction_capable(&self) -> bool {
            true
        }
        fn evaluate(&mut self, _: &[f32], _: &AtomicBool) -> Result<Evidence, Reason> {
            panic!("synthetic provider crash")
        }
    }
    let mut gate = Gate::new(scope(Mode::Allowlist), Some(Box::new(Crash)));
    let now = Instant::now();
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "open notes", now).unwrap();
    assert_eq!(
        settle(&mut gate, &id, now),
        Outcome::Denied(Reason::ProviderError)
    );
}

#[test]
fn status_is_private_safe_and_long_prefix_never_counts_as_full_coverage() {
    let mut gate = gate(Ok(evidence("vox-a", RecognitionStatus::Confident, 4)));
    let now = Instant::now();
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "PRIVATE-FIXTURE-TEXT", now).unwrap();
    settle(&mut gate, &id, now);
    let json = serde_json::to_string(&(gate.status(), gate.outcome(&id).unwrap())).unwrap();
    for private in [
        "PRIVATE-FIXTURE-TEXT",
        "PRIVATE-FIXTURE-NAME",
        "vox-a",
        "synthetic-r1",
    ] {
        assert!(!json.contains(private));
    }
    assert_eq!(
        voqx::supported_window(239_999),
        Err(Reason::InsufficientAudio)
    );
    assert_eq!(voqx::supported_window(240_000), Ok(()));
    assert_eq!(
        voqx::supported_window(240_001),
        Err(Reason::UnsupportedCoverage)
    );
    assert_eq!(
        voqx::supported_window(480_000),
        Err(Reason::UnsupportedCoverage)
    );
}

#[test]
fn canonical_registry_reuses_voice_ids_and_only_checkpoint_recognition_profiles() {
    use voqx_core::identity::{RecognitionProfile, VoiceIdentity};
    let mut registry = voqx_core::registry::VoiceRegistry::default();
    for (id, vector) in [("vox-a", vec![1.0, 0.0]), ("vox-b", vec![0.0, 1.0])] {
        let mut identity = VoiceIdentity::new(id, "SYNTHETIC");
        identity.recognition_profiles.push(RecognitionProfile::new(
            "synthetic",
            "checkpoint-1",
            vec![vector],
            Vec::new(),
        ));
        registry.identities.push(identity);
    }
    let thresholds = Thresholds {
        confident: 0.9,
        tentative: 0.7,
    }; // fixture geometry, NOT calibrated security thresholds
    let prepared = voqx::PreparedRegistry::new(
        registry.clone(),
        "synthetic",
        "checkpoint-1",
        thresholds,
        "fixture-only",
    )
    .unwrap();
    assert_eq!(
        prepared
            .identify_embedding(&[1.0, 0.0], "synthetic")
            .unwrap()
            .voice_id
            .as_deref(),
        Some("vox-a")
    );
    assert_eq!(
        prepared
            .identify_embedding(&[0.0, 1.0], "synthetic")
            .unwrap()
            .voice_id
            .as_deref(),
        Some("vox-b")
    );
    assert!(matches!(
        voqx::PreparedRegistry::new(
            registry.clone(),
            "synthetic",
            "wrong-checkpoint",
            thresholds,
            "fixture-only"
        ),
        Err(Reason::IncompatibleModel)
    ));
    assert!(matches!(
        prepared.identify_embedding(&[1.0], "synthetic"),
        Err(Reason::IncompatibleModel)
    ));
    assert!(matches!(
        prepared.identify_embedding(&[f32::NAN, 0.0], "synthetic"),
        Err(Reason::ProviderError)
    ));
    assert!(matches!(
        prepared.identify_embedding(&[1.0, 0.0], "other-engine"),
        Err(Reason::IncompatibleModel)
    ));
    let mut malformed = registry.clone();
    malformed.identities[0].recognition_profiles[0]
        .centroid
        .clear();
    assert!(matches!(
        voqx::PreparedRegistry::new(
            malformed,
            "synthetic",
            "checkpoint-1",
            thresholds,
            "fixture-only"
        ),
        Err(Reason::IncompatibleModel)
    ));
    registry.identities[0].recognition_profiles.clear(); // synthesis/name alone cannot match
    let revoked = voqx::PreparedRegistry::new(
        registry,
        "synthetic",
        "checkpoint-1",
        thresholds,
        "fixture-only",
    )
    .unwrap();
    assert_ne!(prepared.revision(), revoked.revision());
}

#[test]
fn completed_audio_dictionary_first_exact_command_integration_is_effect_free_until_settled() {
    use crate::dictation::voice_commands::{Registry, Route, Router, VoiceMode};
    use crate::replacement::{
        ReplacementConfig, ReplacementDictionary, ReplacementEngine, ReplacementEntry,
    };
    let dir =
        std::env::temp_dir().join(format!("ears-speaker-integration-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let dictionary = dir.join("dictionary.toml");
    ReplacementDictionary {
        version: 1,
        entries: vec![ReplacementEntry {
            replace: "trx".into(),
            phrases: vec!["tricks".into()],
        }],
    }
    .save(&dictionary)
    .unwrap();
    let replacements = ReplacementEngine::from_config(&ReplacementConfig {
        enabled: true,
        dictionary_paths: vec![dictionary.to_string_lossy().into_owned()],
        case_sensitive: false,
    })
    .unwrap();
    let commands = dir.join("commands.toml");
    std::fs::write(&commands, "[[command]]\nphrase='trx issue'\nrisk='safe'\naction={type='literal_text',text='MOCK-INTENT'}\n").unwrap();
    let registry = Registry::new(commands);
    for voice in ["vox-a", "vox-b", "vox-other"] {
        let mut gate = gate(Ok(evidence(voice, RecognitionStatus::Confident, 4)));
        let now = Instant::now();
        let (id, audio) = input(&mut gate);
        let mut intents = Vec::new();
        // The actual completed-audio ASR seam receives the exact sealed PCM.
        // This fixture substitutes ASR + speaker inference, NOT the gate/matcher.
        let final_text = audio
            .transcribe(1, |pcm| {
                assert_eq!(pcm, &[0.1, 0.2, 0.3, 0.4]);
                Ok::<_, ()>(replacements.replace("tricks issue"))
            })
            .unwrap();
        assert!(intents.is_empty());
        gate.submit_final(final_text, now).unwrap();
        settle(&mut gate, &id, now);
        let result = gate.dispatch(
            &id,
            "trx issue",
            &scope(Mode::Allowlist),
            now,
            || true,
            |text| {
                let mut router = Router::new(VoiceMode::CommandOnly);
                assert!(router.word(text).is_none());
                if let Route::Action(text) = router.finish(text, &registry) {
                    intents.push(text);
                }
            },
        );
        if voice == "vox-other" {
            assert_eq!(result, Err(Reason::Disallowed));
            assert!(intents.is_empty());
        } else {
            result.unwrap();
            assert_eq!(intents, vec!["MOCK-INTENT"]);
        }
        assert!(
            gate.dispatch(
                &id,
                "trx issue",
                &scope(Mode::Allowlist),
                now,
                || true,
                |_| panic!("duplicate effect")
            )
            .is_err()
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn appended_suffix_and_expired_settled_approval_cannot_dispatch_a_prefix() {
    let now = Instant::now();
    let mut gate = Gate::new(scope(Mode::Disabled), None);
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "prefix", now).unwrap();
    assert_eq!(gate.push_pcm(&id, &[0.5]), Err(Reason::PayloadConflict));
    assert_eq!(
        gate.dispatch(
            &id,
            "prefix",
            &scope(Mode::Disabled),
            now,
            || true,
            |_| panic!("prefix effect")
        ),
        Err(Reason::PayloadConflict)
    );
    let mut gate = Gate::new(
        scope(Mode::Allowlist),
        Some(Box::new(Fake {
            answer: Some(Ok(evidence("vox-a", RecognitionStatus::Confident, 4))),
            pcm: None,
        })),
    );
    let (id, audio) = input(&mut gate);
    submit(&mut gate, &audio, 1, "expired", now).unwrap();
    assert_eq!(settle(&mut gate, &id, now), Outcome::Eligible);
    let late = now + Duration::from_secs(2);
    gate.poll(late);
    assert_eq!(gate.outcome(&id).unwrap(), Outcome::Denied(Reason::Timeout));
    assert_eq!(
        gate.dispatch(
            &id,
            "expired",
            &scope(Mode::Allowlist),
            late,
            || true,
            |_| panic!("expired effect")
        ),
        Err(Reason::Timeout)
    );
}

#[test]
fn evidence_only_annotation_is_reference_only_not_ready_or_dispatchable() {
    struct EvidenceOnly;
    impl EvidenceProvider for EvidenceOnly {
        fn revision(&self) -> Revision {
            scope(Mode::Allowlist).recognition
        }
        fn instruction_capable(&self) -> bool {
            false
        }
        fn evaluate(&mut self, _: &[f32], _: &AtomicBool) -> Result<Evidence, Reason> {
            let mut result = evidence("vox-a", RecognitionStatus::Confident, 4);
            result.speaker_consistency = SpeakerConsistency::Unsupported;
            Ok(result)
        }
    }
    for mode in [Mode::Annotate, Mode::Allowlist] {
        let mut gate = Gate::new(scope(mode), Some(Box::new(EvidenceOnly)));
        assert_eq!(gate.status().readiness, Readiness::EvidenceOnly);
        let now = Instant::now();
        let (id, audio) = input(&mut gate);
        submit(&mut gate, &audio, 1, "reference", now).unwrap();
        assert_eq!(
            settle(&mut gate, &id, now),
            if mode == Mode::Annotate {
                Outcome::Observed(Some(Reason::SpeakerCheckUnavailable))
            } else {
                Outcome::Denied(Reason::SpeakerCheckUnavailable)
            }
        );
        assert!(
            gate.dispatch(
                &id,
                "reference",
                &scope(mode),
                now,
                || true,
                |_| panic!("reference became instruction")
            )
            .is_err()
        );
    }
}

#[cfg(feature = "speaker-voqx")]
#[test]
#[ignore = "requires separately authorized, explicit local model inference; never run in fixture CI"]
fn opt_in_local_voqx_adapter_remains_evidence_only() {
    use voqx_core::identity::{RecognitionProfile, VoiceIdentity};
    assert_eq!(
        std::env::var("EARS_SPEAKER_INFERENCE_CONSENT").as_deref(),
        Ok("explicit-local-generated-pcm-only")
    );
    let model_dir = std::path::PathBuf::from(
        std::env::var_os("EARS_SPEAKER_MODEL_DIR").expect("explicit model directory required"),
    );
    assert!(model_dir.is_dir());
    let embedder = voqx_recognition::Embedder::new("cpu", Some(&model_dir)).unwrap();
    // Synthetic profiles only: no enrollment or real person's registry is read.
    let mut registry = voqx_core::registry::VoiceRegistry::default();
    let mut identity = VoiceIdentity::new("vox-synthetic", "SYNTHETIC");
    identity.recognition_profiles.push(RecognitionProfile::new(
        voqx_recognition::EMBEDDING_MODEL_ID,
        "fixture-checkpoint-unattested",
        vec![vec![1.0; 256]],
        Vec::new(),
    ));
    registry.identities.push(identity);
    let profiles = voqx::PreparedRegistry::new(
        registry,
        voqx_recognition::EMBEDDING_MODEL_ID,
        "fixture-checkpoint-unattested",
        Thresholds {
            confident: 0.99,
            tentative: 0.98,
        },
        "uncalibrated-fixture-only",
    )
    .unwrap();
    let mut provider = voqx::VoqxProvider::from_prepared(embedder, profiles).unwrap();
    assert!(!provider.instruction_capable());
    let cancelled = AtomicBool::new(false);
    assert!(matches!(
        provider.evaluate(&[0.1; 4], &cancelled),
        Err(Reason::InsufficientAudio)
    ));
    let pcm: Vec<f32> = (0..240_000)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();
    let result = provider.evaluate(&pcm, &cancelled).unwrap();
    assert_eq!(result.covered_samples, 240_000);
    assert_eq!(result.speaker_consistency, SpeakerConsistency::Unsupported);
}
