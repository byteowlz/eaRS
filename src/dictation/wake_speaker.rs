//! Annotation-only speaker evidence after a wake. No capture or insertion decision is changed.
//!
//! The command receives only the captured WAV path after its configured arguments and must return
//! one JSON object (compact or multiline). The expected identity is checked locally, not passed as
//! an extra positional argument. Missing evidence, uncertainty and process failures are Unknown.
//! Scores are not calibrated probabilities or authentication. Filtering is not implemented.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod boundary_tests;
mod process;
pub(crate) use process::execute as execute_command;

/// What the gate is allowed to do with its verdict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerMode {
    /// Record the verdict, change nothing. The default.
    #[default]
    Annotate,
    /// Reserved for a future calibrated policy. The current caller still annotates only.
    Filter,
}

/// `[dictation.wake.speaker]`. Off unless `command` is set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeakerConfig {
    /// Speaker-identification command; the WAV path is appended as the last argument.
    /// e.g. `["voqx", "identify", "--json", "--mode", "cpu"]`.
    pub command: Vec<String>,
    /// Expected voice_id, compared locally with the response; never appended to the command.
    pub identity: String,
    /// Score at or above which the voice counts as the enrolled speaker.
    pub threshold: f32,
    pub mode: SpeakerMode,
    /// Audio shorter than this cannot be judged and resolves to `Unknown`.
    pub min_ms: u32,
    /// Give up after this long and treat the verdict as `Unknown`.
    pub timeout_ms: u32,
}

impl SpeakerConfig {
    pub fn active(&self) -> bool {
        !self.command.is_empty()
    }

    pub fn threshold(&self) -> f32 {
        if self.threshold > 0.0 {
            self.threshold
        } else {
            0.5
        }
    }

    pub fn min_ms(&self) -> u32 {
        if self.min_ms > 0 {
            self.min_ms
        } else {
            3000
        }
    }

    pub fn timeout(&self) -> Duration {
        Duration::from_millis(u64::from(if self.timeout_ms > 0 {
            self.timeout_ms
        } else {
            4000
        }))
    }
}

/// The verdict for one wake. `Unknown` never filters: a broken or missing check must not silently
/// change what the user's microphone does.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakerVerdict {
    Owner { score: f32 },
    Other { score: f32 },
    Unknown { reason: String },
}

impl SpeakerVerdict {
    pub fn score(&self) -> Option<f32> {
        match self {
            Self::Owner { score } | Self::Other { score } => Some(*score),
            Self::Unknown { .. } => None,
        }
    }

    /// True only when the gate is in `filter` mode and the verdict is a confident `Other`.
    pub fn filters(&self, mode: SpeakerMode) -> bool {
        mode == SpeakerMode::Filter && matches!(self, Self::Other { .. })
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Owner { .. } => "owner",
            Self::Other { .. } => "other",
            Self::Unknown { .. } => "unknown",
        }
    }
}

/// The voqx JSON contract. With a configured target, both voice_id and confident status are
/// required. Legacy anonymous score-only tools are supported only without a configured identity.
#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    score: Option<f32>,
    #[serde(default)]
    score_sample: Option<f32>,
    #[serde(default)]
    voice_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    label: Option<String>,
}

impl Response {
    fn number(&self) -> Option<f32> {
        self.score.or(self.score_sample)
    }
}

/// Run on a blocking annotation worker, never the capture/insertion thread.
/// Process and parsing failures return Unknown; this function never changes dictation state.
pub fn check(config: &SpeakerConfig, wav: &Path, pcm_ms: u32) -> SpeakerVerdict {
    if !config.active() {
        return unknown("no speaker command configured");
    }
    if !config.threshold.is_finite() || !(0.0..=1.0).contains(&config.threshold) {
        return unknown("invalid speaker threshold");
    }
    if wav.extension().and_then(|e| e.to_str()) != Some("wav") {
        return unknown("only wav is handed to the speaker command");
    }
    if pcm_ms < config.min_ms() {
        return unknown("audio is below the configured duration floor");
    }
    match process::run(&config.command, wav, config.timeout()) {
        Ok(stdout) => parse_response(config, &stdout),
        Err(reason) => unknown(reason),
    }
}

fn unknown(reason: impl Into<String>) -> SpeakerVerdict {
    SpeakerVerdict::Unknown {
        reason: reason.into(),
    }
}

fn parse_response(config: &SpeakerConfig, stdout: &[u8]) -> SpeakerVerdict {
    if stdout
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        != Some(b'{')
    {
        return unknown("speaker command must return exactly one JSON object");
    }
    let parsed: Response = match serde_json::from_slice(stdout) {
        Ok(parsed) => parsed,
        Err(_) => return unknown("speaker command must return exactly one JSON object"),
    };
    if parsed
        .status
        .as_deref()
        .is_some_and(|status| status != "confident")
    {
        return unknown("speaker status is not confident");
    }
    if parsed.label.is_some() {
        return unknown("label-only or mixed-label evidence is unsupported");
    }
    let score = match parsed.number() {
        Some(score) if score.is_finite() && (-1.0..=1.0).contains(&score) => score,
        _ => return unknown("speaker score is missing or invalid"),
    };
    if !config.identity.is_empty() {
        if parsed.status.as_deref() != Some("confident") {
            return unknown("named speaker evidence requires confident status");
        }
        let Some(id) = parsed.voice_id.as_deref().filter(|id| !id.is_empty()) else {
            return unknown("named speaker evidence requires voice_id");
        };
        if score < config.threshold() {
            return unknown("speaker score is below the consumer threshold");
        }
        return if id == config.identity {
            SpeakerVerdict::Owner { score }
        } else {
            SpeakerVerdict::Other { score }
        };
    }
    if parsed.voice_id.is_some() {
        return unknown("no expected identity configured for named evidence");
    }
    // Compatibility with custom score-only verifiers that define their own target.
    if score >= config.threshold() {
        SpeakerVerdict::Owner { score }
    } else {
        SpeakerVerdict::Other { score }
    }
}

/// Write mono PCM at the supplied capture rate for the speaker command. Returns the path.
///
/// The file is the caller's to remove: it lives beside the wake log so an operator can inspect what
/// a verdict was based on when the gate is in `annotate` mode.
pub fn write_wav(dir: &Path, samples: &[f32], rate: u32) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let name = format!(
        "wake-{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    );
    let path = dir.join(name);
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let data_len = (samples.len() * 2) as u32;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_len).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?; // PCM
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&rate.to_le_bytes())?;
    file.write_all(&(rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        file.write_all(&((clamped * 32767.0) as i16).to_le_bytes())?;
    }
    file.flush()?;
    Ok(path)
}

/// Collects the audio a speaker verdict is based on: the pre-roll before the detection plus what
/// follows it, up to `min_ms`. Pure so the decision of *when* to judge is testable without audio.
pub struct SpeakerTap {
    min_samples: usize,
    cap_samples: usize,
    buf: Vec<f32>,
    armed: bool,
    fired: bool,
}

impl SpeakerTap {
    pub fn new(min_ms: u32, rate: usize, cap_ms: u32) -> Self {
        Self {
            min_samples: (min_ms as usize * rate) / 1000,
            cap_samples: (cap_ms as usize * rate) / 1000,
            buf: Vec::new(),
            armed: false,
            fired: false,
        }
    }

    /// A wake started: seed with the pre-roll audio that precedes the phrase.
    pub fn arm(&mut self, preroll: &[f32]) {
        self.buf.clear();
        self.armed = true;
        self.fired = false;
        let take = preroll.len().min(self.cap_samples);
        self.buf.extend_from_slice(&preroll[preroll.len() - take..]);
    }

    pub fn disarm(&mut self) {
        self.armed = false;
        self.fired = false;
        self.buf.clear();
    }

    pub fn armed(&self) -> bool {
        self.armed
    }

    pub fn buffered_ms(&self, rate: usize) -> u32 {
        ((self.buf.len() * 1000) / rate.max(1)) as u32
    }

    /// Append captured audio; returns the audio to judge exactly once, when enough has arrived.
    pub fn push(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        if !self.armed || self.fired {
            return None;
        }
        let room = self.cap_samples.saturating_sub(self.buf.len());
        self.buf.extend_from_slice(&chunk[..chunk.len().min(room)]);
        if self.buf.len() < self.min_samples {
            return None;
        }
        self.fired = true;
        Some(std::mem::take(&mut self.buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn fake(dir: &Path, body: &str) -> SpeakerConfig {
        let path = dir.join("speaker.sh");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        SpeakerConfig {
            command: vec![path.display().to_string()],
            identity: "vox-tommy".into(),
            threshold: 0.6,
            mode: SpeakerMode::Annotate,
            min_ms: 1000,
            timeout_ms: 4000,
        }
    }

    pub(super) fn tmpdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ears-speaker-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn generic(dir: &Path, body: &str) -> SpeakerConfig {
        let mut config = fake(dir, body);
        config.identity.clear();
        config
    }

    #[test]
    fn anonymous_score_only_contract_retains_threshold_behavior() {
        let dir = tmpdir("verdict");
        let wav = write_wav(&dir, &[0.0; 16_000], 16_000).unwrap();
        assert_eq!(
            check(&generic(&dir, "echo '{\"score\": 0.91}'"), &wav, 2000),
            SpeakerVerdict::Owner { score: 0.91 }
        );
        assert_eq!(
            check(&generic(&dir, "echo '{\"score\": 0.12}'"), &wav, 2000),
            SpeakerVerdict::Other { score: 0.12 }
        );
    }

    #[test]
    fn short_audio_and_missing_commands_are_unknown_never_other() {
        let dir = tmpdir("short");
        let wav = write_wav(&dir, &[0.0; 4000], 16_000).unwrap();
        let verdict = check(&fake(&dir, "echo '{\"score\": 0.01}'"), &wav, 250);
        assert!(
            matches!(verdict, SpeakerVerdict::Unknown { .. }),
            "short audio must not be judged: {verdict:?}"
        );
        assert!(
            !verdict.filters(SpeakerMode::Filter),
            "unknown must never filter"
        );
        let none = SpeakerConfig::default();
        assert!(matches!(
            check(&none, &wav, 5000),
            SpeakerVerdict::Unknown { .. }
        ));
    }

    #[test]
    fn failures_degrade_to_unknown_and_never_filter() {
        let dir = tmpdir("fail");
        let wav = write_wav(&dir, &[0.0; 16_000], 16_000).unwrap();
        for body in [
            "exit 3",
            "echo 'not json'",
            "echo '{\"label\": \"insufficient_audio\"}'",
            "echo '{\"score\": null}'",
            "sleep 10; echo '{\"score\": 0.99}'",
        ] {
            let verdict = check(&fake(&dir, body), &wav, 2000);
            assert!(
                matches!(verdict, SpeakerVerdict::Unknown { .. }),
                "body {body:?} gave {verdict:?}"
            );
            assert!(!verdict.filters(SpeakerMode::Filter));
        }
    }

    #[test]
    fn only_a_confident_other_filters_and_only_in_filter_mode() {
        let dir = tmpdir("filter");
        let wav = write_wav(&dir, &[0.0; 16_000], 16_000).unwrap();
        let other = check(&generic(&dir, "echo '{\"score\": 0.2}'"), &wav, 2000);
        assert!(other.filters(SpeakerMode::Filter));
        assert!(
            !other.filters(SpeakerMode::Annotate),
            "annotate must never drop a wake"
        );
        let owner = check(&generic(&dir, "echo '{\"score\": 0.9}'"), &wav, 2000);
        assert!(!owner.filters(SpeakerMode::Filter));
    }

    #[test]
    fn only_audio_is_appended_identity_stays_in_consumer_policy() {
        let dir = tmpdir("args");
        let wav = write_wav(&dir, &[0.0; 16_000], 16_000).unwrap();
        let recorder = dir.join("args.txt");
        let config = fake(
            &dir,
            &format!(
                "printf '%s\\n' \"$#\" \"$1\" > '{}'; echo '{{\"score\": 0.7}}'",
                recorder.display()
            ),
        );
        let _ = check(&config, &wav, 2000);
        let recorded = std::fs::read_to_string(&recorder).unwrap();
        assert_eq!(recorded, format!("1\n{}\n", wav.display()));
    }

    #[test]
    fn real_pretty_json_shape_is_understood() {
        let dir = tmpdir("pretty-json");
        let wav = write_wav(&dir, &[0.1; 16_000], 16_000).unwrap();
        let response = serde_json::to_string_pretty(&serde_json::json!({
            "voice_id": "vox-tommy", "score": 0.88, "score_sample": 0.88,
            "score_centroid": 0.6, "status": "confident",
            "audio": {"observed_duration_seconds": 1.0, "calibration_band": null}
        }))
        .unwrap();
        assert_eq!(
            check(
                &fake(&dir, &format!("printf '%s\\n' '{response}'")),
                &wav,
                2000
            ),
            SpeakerVerdict::Owner { score: 0.88 }
        );
    }

    #[test]
    fn voqx_json_shape_is_understood() {
        let dir = tmpdir("voqx");
        let wav = write_wav(&dir, &[0.0; 16_000], 16_000).unwrap();
        // the shape `voqx identify --json` actually prints
        let confident = r#"{"voice_id":"vox-tommy","display_name":"Tommy","engine":"speakrs-wespeaker","score":0.88,"score_sample":0.88,"score_centroid":0.63,"status":"confident"}"#;
        let other_person = r#"{"voice_id":"vox-roman","display_name":"Roman","engine":"speakrs-wespeaker","score":0.91,"score_sample":0.91,"status":"confident"}"#;
        let tentative = r#"{"voice_id":"vox-tommy","score":0.77,"status":"tentative"}"#;
        let insufficient = r#"{"voice_id":null,"score":null,"status":"insufficient_audio"}"#;
        let config = fake(&dir, &format!("echo '{confident}'"));
        assert_eq!(
            check(&config, &wav, 5000),
            SpeakerVerdict::Owner { score: 0.88 }
        );
        // another enrolled person scoring high must not count as the owner
        let config = fake(&dir, &format!("echo '{other_person}'"));
        assert_eq!(
            check(&config, &wav, 5000),
            SpeakerVerdict::Other { score: 0.91 }
        );
        // uncertainty never becomes a confident "other" that could filter
        let config = fake(&dir, &format!("echo '{tentative}'"));
        assert!(matches!(
            check(&config, &wav, 5000),
            SpeakerVerdict::Unknown { .. }
        ));
        let config = fake(&dir, &format!("echo '{insufficient}'"));
        let verdict = check(&config, &wav, 5000);
        assert!(
            matches!(verdict, SpeakerVerdict::Unknown { .. }),
            "got {verdict:?}"
        );
        assert!(!verdict.filters(SpeakerMode::Filter));
    }

    #[test]
    fn tap_waits_for_the_minimum_then_fires_once() {
        let rate = 16_000;
        let mut tap = SpeakerTap::new(1000, rate, 12_000);
        tap.arm(&[0.5; 8000]); // 500 ms pre-roll
        assert_eq!(tap.buffered_ms(rate), 500);
        assert!(
            tap.push(&[0.1; 4000]).is_none(),
            "1 s of audio must not trigger a 3 s floor"
        );
        let fired = tap.push(&[0.1; 64_000]).expect("3.5 s should fire");
        assert!(fired.len() >= 3 * rate);
        assert!(
            tap.push(&[0.2; 16_000]).is_none(),
            "the tap fires once per wake"
        );
    }

    #[test]
    fn tap_disarm_clears_and_a_second_wake_rearms() {
        let rate = 16_000;
        let mut tap = SpeakerTap::new(1000, rate, 12_000);
        tap.arm(&[0.0; 0]);
        assert!(tap.push(&[0.3; 32_000]).is_some());
        tap.disarm();
        assert!(
            tap.push(&[0.3; 32_000]).is_none(),
            "disarmed tap must ignore audio"
        );
        tap.arm(&[0.0; 0]);
        assert!(
            tap.push(&[0.3; 32_000]).is_some(),
            "a new wake re-arms the tap"
        );
    }

    #[test]
    fn tap_never_exceeds_its_cap() {
        let rate = 16_000;
        let mut tap = SpeakerTap::new(60_000, rate, 4000); // floor longer than the cap
        tap.arm(&[0.0; 0]);
        tap.push(&[0.4; 16_000 * 10]);
        assert!(
            tap.buffered_ms(rate) <= 4000,
            "cap must bound the buffer: {} ms",
            tap.buffered_ms(rate)
        );
    }

    #[test]
    fn wav_written_by_the_gate_is_a_valid_16k_mono_file() {
        let dir = tmpdir("wav");
        let samples = vec![0.25f32; 32_000];
        let path = write_wav(&dir, &samples, 16_000).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1, "mono");
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            16_000
        );
        assert_eq!(bytes.len(), 44 + samples.len() * 2);
    }
}
