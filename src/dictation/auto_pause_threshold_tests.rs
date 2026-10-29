//! Threshold behavior is separate from drain-protocol lifecycle fixtures.
use super::tests::{gate, silent};
use super::*;

#[test]
fn threshold_controls_silence_and_resumed_speech_without_resetting_drain() {
    let mut gate = gate();
    gate.start(7);
    for _ in 0..94 {
        gate.observe(0.6).unwrap();
    }
    assert!(!gate.due(7));
    gate.set_speech_threshold(0.75).unwrap();
    for _ in 0..94 {
        gate.observe(0.6).unwrap();
    }
    assert!(gate.due(7));
    let id = gate.begin(7, Instant::now()).unwrap();
    gate.observe(0.74).unwrap();
    gate.final_seen(7);
    assert_eq!(gate.acknowledge(id, 7).unwrap(), Completion::Paused);
    gate.start(8);
    silent(&mut gate, 94);
    let id = gate.begin(8, Instant::now()).unwrap();
    gate.observe(0.75).unwrap(); // Inclusive speech boundary, even while draining.
    gate.set_speech_threshold(0.9).unwrap();
    gate.final_seen(8);
    assert_eq!(gate.acknowledge(id, 8).unwrap(), Completion::Resume);
}

#[test]
fn threshold_validation_and_config_roundtrip() {
    for threshold in [f32::NAN, f32::INFINITY, -0.1, 0.0, 1.0, 1.1] {
        let config = AutoPauseConfig {
            speech_threshold: threshold,
            ..Default::default()
        };
        assert!(AutoPause::from_config(&config, None).is_err());
        assert!(gate().set_speech_threshold(threshold).is_err());
    }
    let config: AutoPauseConfig = toml::from_str("idle_ms=3000\nspeech_threshold=0.75").unwrap();
    config.validate().unwrap();
    assert_eq!(config.speech_threshold, 0.75);
    let restored: AutoPauseConfig = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
    assert_eq!(restored, config);
    assert_eq!(AutoPauseConfig::default().speech_threshold, 0.5);
    for value in ["0", "1", "-0.1", "1.1", "nan", "inf", "-inf"] {
        assert!(toml::from_str::<AutoPauseConfig>(&format!("speech_threshold={value}")).is_err());
    }
}
