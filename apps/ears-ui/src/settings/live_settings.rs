//! Engine settings that apply immediately: saved with a narrow, comment-preserving
//! config edit first, then sent to the running engine. A failed save applies nothing.
use super::auto_pause::persist_edit;
use anyhow::{Context, Result, ensure};
use ears::config::AppConfig;
use ears::dictation::DictationCommand;
use std::path::Path;
use toml_edit::{DocumentMut, value};

/// One live setting: where it lives in `config.toml` and how the engine hears it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LiveSetting {
    SpeechThreshold,
    PauseAfterSilence,
    WakeSensitivity,
}

impl LiveSetting {
    fn location(self) -> (&'static str, &'static str) {
        match self {
            Self::SpeechThreshold => ("auto_pause", "speech_threshold"),
            Self::PauseAfterSilence => ("auto_pause", "idle_ms"),
            Self::WakeSensitivity => ("wake", "threshold"),
        }
    }

    /// Slider value (seconds for the pause, probability otherwise).
    pub fn current(self, core: &AppConfig) -> f32 {
        match self {
            Self::SpeechThreshold => core.dictation.auto_pause.speech_threshold,
            Self::PauseAfterSilence => core.dictation.auto_pause.idle_ms as f32 / 1000.0,
            Self::WakeSensitivity => core.dictation.wake.threshold.unwrap_or(0.5),
        }
    }

    fn validate(self, slider: f32) -> Result<()> {
        match self {
            Self::SpeechThreshold => ears::dictation::auto_pause::validate_speech_threshold(slider),
            Self::PauseAfterSilence => {
                ears::dictation::auto_pause::validate_idle_ms(idle_ms(slider))
            }
            Self::WakeSensitivity => {
                ensure!(
                    slider.is_finite() && (0.0..=1.0).contains(&slider),
                    "wake sensitivity must be between 0 and 1"
                );
                Ok(())
            }
        }
    }

    pub fn apply(self, core: &mut AppConfig, slider: f32) -> DictationCommand {
        match self {
            Self::SpeechThreshold => {
                core.dictation.auto_pause.speech_threshold = slider;
                DictationCommand::SetAutoPauseSpeechThreshold(slider)
            }
            Self::PauseAfterSilence => {
                core.dictation.auto_pause.idle_ms = idle_ms(slider);
                DictationCommand::SetAutoPauseIdle(idle_ms(slider))
            }
            Self::WakeSensitivity => {
                core.dictation.wake.threshold = Some(slider);
                DictationCommand::SetWakeThreshold(Some(slider))
            }
        }
    }
}

fn idle_ms(seconds: f32) -> u32 {
    (seconds * 1000.0).round().clamp(0.0, u32::MAX as f32) as u32
}

fn updated_document(original: &str, setting: LiveSetting, slider: f32) -> Result<String> {
    setting.validate(slider)?;
    let mut document: DocumentMut = original.parse().context("Cannot edit config.toml")?;
    let (table, key) = setting.location();
    ensure!(
        document
            .get("dictation")
            .is_none_or(|item| item.as_table_like().is_some()),
        "dictation must be a TOML table"
    );
    ensure!(
        document
            .get("dictation")
            .and_then(|item| item.get(table))
            .is_none_or(|item| item.as_table_like().is_some()),
        "dictation.{table} must be a TOML table"
    );
    let item = &mut document["dictation"][table][key];
    let decor = item.as_value().map(|v| v.decor().clone());
    *item = match setting {
        LiveSetting::PauseAfterSilence => value(i64::from(idle_ms(slider))),
        // Two decimals, as the slider shows; avoids f32 noise like 0.4199999.
        _ => value((f64::from(slider) * 100.0).round() / 100.0),
    };
    if let Some(decor) = decor {
        *item.as_value_mut().unwrap().decor_mut() = decor;
    }
    Ok(document.to_string())
}

pub fn persist(path: &Path, setting: LiveSetting, slider: f32) -> Result<()> {
    persist_edit(path, |original| updated_document(original, setting, slider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_setting_edits_only_its_key_and_keeps_comments() {
        let source = "# keep\n[dictation.auto_pause]\nidle_ms = 3000 # silence\nspeech_threshold = 0.7\n[dictation.wake]\nenabled = true # on\n";
        let paused = updated_document(source, LiveSetting::PauseAfterSilence, 1.75).unwrap();
        assert!(paused.contains("idle_ms = 1750 # silence"));
        assert!(
            paused.contains("speech_threshold = 0.7") && paused.contains("enabled = true # on")
        );
        let wake = updated_document(source, LiveSetting::WakeSensitivity, 0.42).unwrap();
        let parsed: toml::Value = toml::from_str(&wake).unwrap();
        assert_eq!(
            parsed["dictation"]["wake"]["threshold"].as_float(),
            Some(0.42)
        );
        assert!(wake.contains("# keep") && wake.contains("idle_ms = 3000 # silence"));
        let speech = updated_document(source, LiveSetting::SpeechThreshold, 0.55).unwrap();
        let parsed: toml::Value = toml::from_str(&speech).unwrap();
        assert_eq!(
            parsed["dictation"]["auto_pause"]["speech_threshold"].as_float(),
            Some(0.55)
        );
        // An empty config gains just the edited key.
        let fresh = updated_document("", LiveSetting::WakeSensitivity, 0.4).unwrap();
        assert_eq!(
            toml::from_str::<toml::Value>(&fresh).unwrap()["dictation"]["wake"]["threshold"]
                .as_float(),
            Some(0.4)
        );
    }

    #[test]
    fn invalid_values_and_malformed_tables_write_nothing() {
        assert!(updated_document("", LiveSetting::PauseAfterSilence, 0.2).is_err());
        assert!(updated_document("", LiveSetting::WakeSensitivity, f32::NAN).is_err());
        assert!(updated_document("", LiveSetting::SpeechThreshold, 1.0).is_err());
        assert!(
            updated_document("[dictation]\nwake = 3\n", LiveSetting::WakeSensitivity, 0.4).is_err()
        );
        assert!(updated_document("not toml", LiveSetting::PauseAfterSilence, 2.0).is_err());
    }

    #[test]
    fn applying_updates_the_model_and_names_the_engine_command() {
        let mut core = AppConfig::default();
        assert_eq!(LiveSetting::WakeSensitivity.current(&core), 0.5);
        assert_eq!(
            LiveSetting::PauseAfterSilence.apply(&mut core, 2.25),
            DictationCommand::SetAutoPauseIdle(2250)
        );
        assert_eq!(core.dictation.auto_pause.idle_ms, 2250);
        assert_eq!(LiveSetting::PauseAfterSilence.current(&core), 2.25);
        assert_eq!(
            LiveSetting::WakeSensitivity.apply(&mut core, 0.4),
            DictationCommand::SetWakeThreshold(Some(0.4))
        );
        assert_eq!(core.dictation.wake.threshold, Some(0.4));
    }
}
