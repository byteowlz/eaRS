//! Opt-in real voqx CLI contract check. Generated signal, isolated registry;
//! this is not a speaker-accuracy, live microphone or text-insertion benchmark.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use ears::dictation::wake_speaker::{check, write_wav, SpeakerConfig, SpeakerMode, SpeakerVerdict};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires explicit VOQX_BINARY and cached voqx model weights"]
fn documented_command_works_with_real_multiline_cli_evidence() -> Result<()> {
    let binary =
        std::env::var_os("VOQX_BINARY").context("set VOQX_BINARY to the built voqx binary")?;
    let binary = Path::new(&binary).canonicalize()?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir =
        Scratch(std::env::temp_dir().join(format!("ears-voqx-{}-{nonce}", std::process::id())));
    std::fs::create_dir(&dir.0)?;
    let rate = 24_000;
    let samples: Vec<f32> = (0..rate * 2)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.2 * (t * (900.0 + t * 30.0)).sin() + 0.1 * (t * 1700.0).sin()
        })
        .collect();
    let wav = write_wav(&dir.0, &samples, rate as u32)?;
    let registry = dir.0.join("registry");
    let environment = [
        format!("XDG_CONFIG_HOME={}", dir.0.join("config").display()),
        format!("XDG_DATA_HOME={}", dir.0.join("data").display()),
        format!("XDG_STATE_HOME={}", dir.0.join("state").display()),
        "HF_HUB_OFFLINE=1".to_string(),
    ];
    let enrollment = Command::new("/usr/bin/env")
        .args(&environment)
        .arg(&binary)
        .args(["enroll", "Fixture", "--audio"])
        .arg(&wav)
        .args(["--recognition", "--preprocessing", "valid-v1", "--registry"])
        .arg(&registry)
        .output()?;
    anyhow::ensure!(
        enrollment.status.success(),
        "isolated fixture enrollment failed: {}",
        enrollment.status
    );
    let registry_before = std::fs::read(registry.join("registry.json"))?;
    let mut command = vec!["/usr/bin/env".into()];
    command.extend(environment);
    command.extend([
        binary.display().to_string(),
        "identify".into(),
        "--json".into(),
        "--mode".into(),
        "cpu".into(),
        "--preprocessing".into(),
        "valid-v1".into(),
        "--registry".into(),
        registry.display().to_string(),
    ]);
    let config = SpeakerConfig {
        command,
        identity: "vox-fixture".into(),
        threshold: 0.6,
        mode: SpeakerMode::Annotate,
        min_ms: 1000,
        timeout_ms: 10_000,
    };
    let verdict = check(&config, &wav, 2000);
    assert!(
        matches!(verdict, SpeakerVerdict::Owner { score } if score > 0.99),
        "{verdict:?}"
    );
    assert!(!verdict.filters(SpeakerMode::Annotate));
    let other = check(
        &SpeakerConfig {
            identity: "vox-another".into(),
            ..config.clone()
        },
        &wav,
        2000,
    );
    assert!(
        matches!(other, SpeakerVerdict::Other { score } if score > 0.99),
        "{other:?}"
    );
    assert!(!other.filters(SpeakerMode::Annotate));
    let short = check(&config, &wav, 500);
    assert!(matches!(short, SpeakerVerdict::Unknown { .. }));
    let mut incompatible = config;
    let index = incompatible
        .command
        .iter()
        .position(|arg| arg == "valid-v1")
        .context("missing preprocessing")?;
    incompatible.command[index] = "uniform".into();
    assert!(matches!(
        check(&incompatible, &wav, 2000),
        SpeakerVerdict::Unknown { .. }
    ));
    assert_eq!(
        std::fs::read(registry.join("registry.json"))?,
        registry_before,
        "identification must not mutate the registry"
    );
    Ok(())
}
