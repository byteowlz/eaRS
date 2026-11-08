//! Bounded receiver values: one range contract for scene geometry, previews,
//! keyboard edits, dot-matrix readouts and the existing save/apply boundaries.
use super::super::live_settings::LiveSetting;
use super::{NativeSettings, Request};
use crate::{config::UiConfig, ui::slider::SliderSpec};
use device_scene::{Action, Phase, Range, Snapshot, Value};
use ears::config::AppConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::settings) enum Numeric {
    Live(LiveSetting),
    Radius,
    Opacity,
    Margin,
    Linger,
    WordWidth,
    WordSize,
}
pub(in crate::settings) const ALL: [Numeric; 9] = [
    Numeric::Live(LiveSetting::SpeechThreshold),
    Numeric::Live(LiveSetting::PauseAfterSilence),
    Numeric::Live(LiveSetting::WakeSensitivity),
    Numeric::Radius,
    Numeric::Opacity,
    Numeric::Margin,
    Numeric::Linger,
    Numeric::WordWidth,
    Numeric::WordSize,
];
impl Numeric {
    pub(in crate::settings) fn id(self) -> &'static str {
        match self {
            Self::Live(LiveSetting::SpeechThreshold) => "threshold",
            Self::Live(LiveSetting::PauseAfterSilence) => "pause",
            Self::Live(LiveSetting::WakeSensitivity) => "wake",
            Self::Radius => "ov.radius",
            Self::Opacity => "ov.opacity",
            Self::Margin => "ov.margin",
            Self::Linger => "ov.linger",
            Self::WordWidth => "ov.wordwidth",
            Self::WordSize => "ov.wordsize",
        }
    }
    pub(in crate::settings) fn find(id: &str) -> Option<Self> {
        ALL.into_iter().find(|s| s.id() == id)
    }
    pub(in crate::settings) fn spec(self, core: &AppConfig) -> SliderSpec {
        use super::super::{LINGER, MARGIN, OPACITY, PAUSE, RADIUS, WAKE};
        match self {
            Self::Live(LiveSetting::SpeechThreshold) => SliderSpec::PROBABILITY,
            Self::Live(LiveSetting::PauseAfterSilence) => SliderSpec { max: 60.0, ..PAUSE },
            Self::Live(LiveSetting::WakeSensitivity) => SliderSpec {
                min: ears::dictation::wake::report_floor(&core.dictation.wake.command)
                    .unwrap_or(0.0)
                    .clamp(0.0, 0.99),
                max: 1.0,
                ..WAKE
            },
            Self::Radius => RADIUS,
            Self::Opacity => OPACITY,
            Self::Margin => MARGIN,
            Self::Linger => LINGER,
            Self::WordWidth => SliderSpec {
                id: "word-width",
                min: 220.0,
                max: 420.0,
                step: 1.0,
                format: |v| format!("{v:.0} px"),
            },
            Self::WordSize => SliderSpec {
                id: "word-size",
                min: 12.0,
                max: 24.0,
                step: 1.0,
                format: |v| format!("{v:.0} pt"),
            },
        }
    }
    fn current(self, core: &AppConfig, ui: &UiConfig) -> f32 {
        match self {
            Self::Live(setting) => setting.current(core),
            Self::Radius => ui.theme.radius,
            Self::Opacity => ui.overlay.opacity,
            Self::Margin => ui.overlay.margin,
            Self::Linger => ui.overlay.linger_ms as f32,
            Self::WordWidth => ui.overlay.word_width,
            Self::WordSize => ui.overlay.word_size,
        }
    }
    fn enabled(self, core: &AppConfig, ui: &UiConfig) -> bool {
        match self {
            Self::Live(LiveSetting::PauseAfterSilence) => core.dictation.auto_pause.idle_ms > 0,
            Self::Live(LiveSetting::WakeSensitivity) => {
                core.dictation.wake.enabled && !core.dictation.wake.command.is_empty()
            }
            Self::WordWidth | Self::WordSize => {
                ui.overlay.transcript == crate::config::TranscriptStyle::Word
            }
            _ => true,
        }
    }
    fn apply_ui(self, ui: &mut UiConfig, value: f32) {
        match self {
            Self::Radius => ui.theme.radius = value,
            Self::Opacity => ui.overlay.opacity = value,
            Self::Margin => ui.overlay.margin = value,
            Self::Linger => ui.overlay.linger_ms = value.round() as u64,
            Self::WordWidth => ui.overlay.word_width = value,
            Self::WordSize => ui.overlay.word_size = value,
            Self::Live(_) => unreachable!("live settings use the engine save/apply seam"),
        }
    }
}

pub(super) fn write(
    snapshot: &mut Snapshot,
    core: &AppConfig,
    ui: &UiConfig,
    preview: Option<(Numeric, f32)>,
) {
    for setting in ALL {
        let id = setting.id();
        let slider = format!("{id}.slider");
        let spec = setting.spec(core);
        let value = preview
            .filter(|(s, _)| *s == setting)
            .map_or_else(|| setting.current(core, ui), |(_, v)| v);
        let enabled = setting.enabled(core, ui);
        if let Some(c) = snapshot.scene.devices[0]
            .controls
            .iter_mut()
            .find(|c| c.id == slider)
        {
            c.range = Some(Range {
                min: spec.min as f64,
                max: spec.max as f64,
                step: spec.step as f64,
            });
        }
        if let Some(state) = snapshot.state.controls.get_mut(&slider) {
            state.value = Value(serde_json::json!(value.clamp(spec.min, spec.max)));
            state.enabled = enabled;
        }
        let off = !enabled && matches!(setting, Numeric::Live(_));
        super::scene::text(
            snapshot,
            id,
            if off {
                "Off".into()
            } else {
                (spec.format)(value)
            },
        );
    }
}

impl NativeSettings {
    pub(super) fn range_action(&mut self, action: &Action, cx: &mut gpui::Context<Self>) -> bool {
        let Some(setting) = action
            .control_id
            .strip_suffix(".slider")
            .and_then(Numeric::find)
        else {
            return false;
        };
        match action.phase {
            Phase::Preview => {
                self.range_preview = action.value.number().map(|v| (setting, v as f32));
                self.sync(cx);
            }
            Phase::Cancel => {
                self.range_preview = None;
                self.sync(cx);
            }
            Phase::Commit => {
                self.range_preview = None;
                if let Some(value) = action.value.number() {
                    self.commit_range(setting, value as f32, cx);
                }
            }
        }
        true
    }
    fn commit_range(&mut self, setting: Numeric, value: f32, cx: &mut gpui::Context<Self>) {
        let model = self.model.read(cx);
        if !setting.enabled(&model.core, &model.ui) || !value.is_finite() {
            return;
        }
        let value = setting.spec(&model.core).snap(value);
        if (setting.current(&model.core, &model.ui) - value).abs() <= f32::EPSILON {
            self.sync(cx);
            return;
        }
        if let Numeric::Live(live) = setting {
            self.pending_live = Some(match live {
                LiveSetting::SpeechThreshold => "Speech threshold",
                LiveSetting::PauseAfterSilence => "Auto-pause",
                LiveSetting::WakeSensitivity => "Wake sensitivity",
            });
            cx.emit(Request::Live(live, value));
        } else {
            self.model.update(cx, |m, cx| {
                m.update_ui(cx, |ui| setting.apply_ui(ui, value))
            });
            if setting == Numeric::Radius {
                crate::runtime::apply_theme(cx, None)
            }
            self.sync(cx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::AppModel;
    #[test]
    fn ranges_validate_and_scene_values_never_rewrite_saved_outliers() {
        let mut model = AppModel::new(AppConfig::default(), UiConfig::default());
        model.configuration_is_preview = true;
        model.ui.overlay.opacity = 0.1;
        let mut scene = crate::settings::device_layout::generated();
        write(&mut scene, &model.core, &model.ui, None);
        assert_eq!(model.ui.overlay.opacity, 0.1);
        let (_, slider, _) = scene.control("ov.opacity.slider").unwrap();
        assert_eq!(slider.kind, device_scene::Kind::Fader);
        assert!(
            !scene.state.controls["ov.opacity"]
                .value
                .string()
                .unwrap()
                .is_empty()
        );
        for setting in ALL {
            let spec = setting.spec(&model.core);
            assert!(spec.min < spec.max && spec.step > 0.0);
            assert_eq!(spec.snap(spec.min - 10.0), spec.min);
            assert_eq!(spec.snap(spec.max + 10.0), spec.max);
        }
        scene.validate().unwrap();
        write(
            &mut scene,
            &model.core,
            &model.ui,
            Some((Numeric::Opacity, 0.65)),
        );
        assert_eq!(
            scene.state.controls["ov.opacity"].value.string(),
            Some("0.65")
        );
        assert_eq!(
            model.ui.overlay.opacity, 0.1,
            "preview is not a config edit"
        );
    }
}
