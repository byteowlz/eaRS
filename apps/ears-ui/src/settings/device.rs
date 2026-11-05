//! eaRS adapter for the shared native device. The renderer never owns app effects.
use super::{
    Tab, device_scene as scene, microphone,
    shortcuts::{self, Binding},
};
use crate::app::AppModel;
use device_gpui::DeviceView;
use device_scene::{Action, OptionItem, Source};
use ears::{
    audio::InputDevices,
    config::HotkeyConfig,
    dictation::hotkeys::{HotkeyRecording, validate_hotkeys},
};
use gpui::{Context, Entity, EventEmitter, KeyDownEvent, Window, div, prelude::*};
use std::{borrow::Cow, time::Duration};

pub(super) fn attach(
    model: Entity<AppModel>,
    window: &mut Window,
    cx: &mut Context<super::SettingsView>,
) -> Option<Entity<NativeSettings>> {
    let native_enabled = !model.read(cx).configuration_is_preview
        || std::env::args().any(|a| {
            matches!(
                a.as_str(),
                "--preview-settings-device" | "--preview-settings-shortcuts"
            )
        });
    let native =
        native_enabled.then(|| cx.new(|cx| NativeSettings::new(model.clone(), window, cx)));
    if let Some(native) = &native {
        cx.subscribe_in(native, window, |this, _, event: &Navigate, window, cx| {
            this.native_active = false;
            this.navigation_focus.focus(window);
            this.tab = event.0;
            this.model.update(cx, |m, _| m.settings_tab = event.0);
            cx.notify();
        })
        .detach();
    }
    native
}

#[derive(Clone, Copy)]
pub(super) struct Navigate(pub Tab);
struct Draft {
    before: HotkeyConfig,
    config: HotkeyConfig,
    selected: Binding,
}
pub(super) struct NativeSettings {
    model: Entity<AppModel>,
    device: Entity<DeviceView>,
    drag: super::window::Drag,
    draft: Option<Draft>,
    recording: Option<HotkeyRecording>,
    listening: bool,
    local_ready: bool,
    capture_generation: u64,
    catalog: Option<InputDevices>,
    loading: bool,
    input_draft: Option<Option<String>>,
    waiting_input: bool,
    status: String,
}
impl EventEmitter<Navigate> for NativeSettings {}
impl NativeSettings {
    pub(super) fn new(
        model: Entity<AppModel>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        // Host typography, never font paths supplied by a scene.
        if let Err(error) = cx.text_system().add_fonts(vec![
            Cow::Borrowed(include_bytes!("../../assets/fonts/Michroma-Regular.ttf")),
            Cow::Borrowed(include_bytes!("../../assets/fonts/Geo-Regular.ttf")),
        ]) {
            eprintln!("ears-ui: device fonts: {error:#}");
        }
        let mut snapshot = scene::initial();
        if std::env::args().any(|a| a == "--preview-settings-shortcuts") {
            snapshot.state.active_page = scene::SHORTCUTS.into();
        }
        let device = cx.new(|cx| {
            let mut view = DeviceView::new(
                snapshot,
                "Michroma".into(),
                crate::theme::FONT_SANS.into(),
                window,
                cx,
            );
            view.set_control_font("brand", "Geo".into());
            view
        });
        cx.subscribe_in(&device, window, |this, _, action: &Action, window, cx| {
            this.action(action, window, cx)
        })
        .detach();
        cx.observe(&model, |this, _, cx| this.sync(cx)).detach();
        cx.observe_window_activation(window, |this, window, cx| {
            if !window.is_window_active() {
                this.deactivate(cx);
            }
        })
        .detach();
        let catalog = model
            .read(cx)
            .configuration_is_preview
            .then(microphone::fixture);
        let mut this = Self {
            model,
            device,
            drag: Default::default(),
            draft: None,
            recording: None,
            listening: false,
            local_ready: true,
            capture_generation: 0,
            catalog,
            loading: false,
            input_draft: None,
            waiting_input: false,
            status: String::new(),
        };
        let page = this.device.read(cx).snapshot().state.active_page.clone();
        this.page(&page, cx);
        this
    }
    fn deactivate(&mut self, cx: &mut Context<Self>) {
        self.drag.end();
        if self.draft.is_some() {
            self.cancel(cx);
        }
    }
    pub(super) fn focus(&self, window: &mut Window, cx: &gpui::App) {
        self.device.read(cx).focus(window);
    }
    pub(super) fn cancel(&mut self, cx: &mut Context<Self>) {
        self.capture_generation += 1;
        self.listening = false;
        self.recording = None;
        self.draft = None;
        self.status = "Edit cancelled; saved shortcuts unchanged.".into();
        self.sync(cx);
    }
    fn sync(&mut self, cx: &mut Context<Self>) {
        let model = self.model.read(cx);
        let config = self
            .draft
            .as_ref()
            .map(|d| &d.config)
            .unwrap_or(&model.core.hotkeys)
            .clone();
        let selected = self.draft.as_ref().map(|d| d.selected);
        let has_draft = self.draft.is_some();
        let saved = model.core.dictation.input_device.clone();
        let requested = self.input_draft.as_ref().unwrap_or(&saved).clone();
        let pending = model.input_pending.is_some();
        if self.waiting_input && !pending {
            self.waiting_input = false;
            self.status = if model.input_error.is_none() {
                "Input confirmed and saved."
            } else {
                "Input switch failed; previous saved input retained."
            }
            .into();
        }
        let status = if let Some(error) = &model.input_error {
            format!("Input error: {error}")
        } else if pending {
            "Switching microphone; waiting for engine confirmation.".into()
        } else {
            self.status.clone()
        };
        let mut choices = microphone::choices(self.catalog.as_ref())
            .into_iter()
            .filter(|c| c.selectable)
            .collect::<Vec<_>>();
        let missing = !choices.iter().any(|c| c.name == requested);
        if missing {
            choices.push(microphone::Choice {
                name: requested.clone(),
                label: format!(
                    "Unavailable: {}",
                    requested.as_deref().unwrap_or("System default")
                ),
                selectable: false,
            });
        }
        let options = choices
            .iter()
            .enumerate()
            .map(|(i, c)| OptionItem {
                id: format!("input-{i}"),
                label: c.label.clone(),
            })
            .collect::<Vec<_>>();
        let current = choices
            .iter()
            .position(|c| c.name == requested)
            .unwrap_or(0);
        let can_apply = self.input_draft.is_some() && !pending && !missing;
        let loading = self.loading;
        let listening = self.listening;
        self.device.update(cx, |view, cx| {
            view.set_keyboard_enabled(!listening);
            view.edit_snapshot(
                |snapshot| {
                    scene::bindings(snapshot, &config, selected, has_draft);
                    let control = snapshot.scene.devices[0]
                        .controls
                        .iter_mut()
                        .find(|c| c.id == "input.device")
                        .unwrap();
                    control.options = Some(options);
                    scene::text(snapshot, "input.device", format!("input-{current}"));
                    scene::text(
                        snapshot,
                        "input.saved",
                        format!("Saved: {}", saved.as_deref().unwrap_or("System default")),
                    );
                    scene::text(snapshot, "status", status);
                    for id in ["input.device", "input.refresh"] {
                        snapshot.state.controls.get_mut(id).unwrap().enabled = !pending && !loading;
                    }
                    snapshot
                        .state
                        .controls
                        .get_mut("input.apply")
                        .unwrap()
                        .enabled = can_apply;
                    snapshot
                        .state
                        .controls
                        .get_mut("input.cancel")
                        .unwrap()
                        .enabled = self.input_draft.is_some() && !pending;
                },
                cx,
            );
        });
        cx.notify();
    }
    fn refresh(&mut self, cx: &mut Context<Self>) {
        if self.loading {
            return;
        }
        if self.model.read(cx).configuration_is_preview {
            self.catalog = Some(microphone::fixture());
            self.sync(cx);
            return;
        }
        self.loading = true;
        self.status = "Listing microphones; no capture started.".into();
        self.sync(cx);
        let task = cx
            .background_executor()
            .spawn(async { ears::audio::input_devices() });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |this, cx| {
                this.loading = false;
                match result {
                    Ok(catalog) => {
                        this.catalog = Some(catalog);
                        this.status = "Click the microphone to select; Apply confirms.".into();
                    }
                    Err(error) => this.status = format!("Cannot list microphones: {error}"),
                }
                this.sync(cx);
            });
        })
        .detach();
    }
    fn action(&mut self, action: &Action, window: &mut Window, cx: &mut Context<Self>) {
        if !scene::accepts(action, self.device.read(cx).snapshot()) {
            return;
        }
        if action.phase == device_scene::Phase::Cancel {
            self.cancel(cx);
            return;
        }
        let id = action.control_id.as_str();
        if let Some(binding) = Binding::ALL
            .into_iter()
            .find(|b| id == format!("shortcut.{}", b.id()))
        {
            self.begin(binding, action.source, cx);
            return;
        }
        match id {
            "window.close" => {
                self.cancel(cx);
                window.remove_window();
            }
            "nav.dictation" | "nav.overlay" | "nav.dictionaries" | "nav.profiles"
            | "dictation.general" => self.navigate(id, cx),
            scene::INPUT | scene::SHORTCUTS => {
                self.cancel(cx);
                self.page(id, cx);
            }
            "input.refresh" => self.refresh(cx),
            "input.device" => self.select_input(action, cx),
            "input.cancel" => {
                self.input_draft = None;
                self.status = "Input selection cancelled.".into();
                self.sync(cx);
            }
            "input.apply" => self.apply_input(cx),
            "shortcut.cancel" => self.cancel(cx),
            "shortcut.save" => self.save(cx),
            "shortcut.defaults" => self.defaults(cx),
            "shortcut.modifier" => {
                if let Some(key) = action.value.string().filter(|key| *key != "choose") {
                    self.device.update(cx, |view, cx| {
                        view.edit_snapshot(|s| scene::text(s, "shortcut.modifier", key), cx)
                    });
                    self.candidate(key.into(), cx);
                }
            }
            "shortcut.clear" => self.clear_binding(cx),
            _ => {}
        }
    }
    fn navigate(&mut self, id: &str, cx: &mut Context<Self>) {
        let tab = match id {
            "nav.overlay" => Tab::Overlay,
            "nav.dictionaries" => Tab::Dictionaries,
            "nav.profiles" => Tab::Profiles,
            _ => Tab::Dictation,
        };
        self.cancel(cx);
        if id == "nav.dictation" {
            self.page(scene::INPUT, cx);
        } else {
            cx.emit(Navigate(tab));
        }
    }
    fn select_input(&mut self, action: &Action, cx: &mut Context<Self>) {
        let index = action
            .value
            .string()
            .and_then(|s| s.strip_prefix("input-"))
            .and_then(|s| s.parse::<usize>().ok());
        let choices = microphone::choices(self.catalog.as_ref())
            .into_iter()
            .filter(|c| c.selectable)
            .collect::<Vec<_>>();
        if let Some(choice) = index.and_then(|i| choices.get(i)) {
            self.input_draft = Some(choice.name.clone());
            self.status = "Input selected; Apply to switch, Cancel to keep current input.".into();
            self.sync(cx);
        }
    }
    fn apply_input(&mut self, cx: &mut Context<Self>) {
        let Some(name) = self.input_draft.clone() else {
            return;
        };
        let running = self.model.read(cx).engine_running();
        let preview = self.model.read(cx).configuration_is_preview;
        match self
            .model
            .update(cx, |model, cx| microphone::apply(model, name, cx))
        {
            Ok(()) => {
                self.input_draft = None;
                self.waiting_input = running;
                self.status = if preview {
                    "Preview only; input selection simulated."
                } else if running {
                    "Switch requested; waiting for confirmation."
                } else {
                    "Input preference saved for the next engine start."
                }
                .into();
            }
            Err(error) => self.status = format!("Input not changed: {error:#}"),
        }
        self.sync(cx);
    }
    fn defaults(&mut self, cx: &mut Context<Self>) {
        self.stop_listening();
        let before = self.model.read(cx).core.hotkeys.clone();
        let mut config = before.clone();
        let defaults = HotkeyConfig::default();
        for b in Binding::ALL {
            b.set(&mut config, b.get(&defaults).unwrap_or("").into());
        }
        self.draft = Some(Draft {
            before,
            config,
            selected: Binding::Toggle,
        });
        self.status = "Defaults staged; Save applies them, Cancel keeps existing shortcuts.".into();
        self.sync(cx);
    }
    fn clear_binding(&mut self, cx: &mut Context<Self>) {
        self.stop_listening();
        if let Some(d) = &mut self.draft
            && d.selected != Binding::Toggle
        {
            d.selected.set(&mut d.config, String::new());
        }
        self.status = "Binding cleared in draft; Save to apply.".into();
        self.sync(cx);
    }
    fn page(&mut self, page: &str, cx: &mut Context<Self>) {
        self.status = if page == scene::SHORTCUTS {
            "Select a shortcut to record. Esc cancels. Save applies."
        } else {
            "Click microphone to select; Apply switches. Refresh lists inputs."
        }
        .into();
        self.device.update(cx, |view, cx| {
            view.edit_snapshot(|s| s.state.active_page = page.into(), cx)
        });
        self.sync(cx);
    }
    fn stop_listening(&mut self) {
        self.listening = false;
        self.capture_generation += 1;
    }
    fn begin(&mut self, binding: Binding, source: Source, cx: &mut Context<Self>) {
        self.stop_listening();
        self.recording = None;
        let model = self.model.read(cx);
        let control = model.engine.as_ref().and_then(|e| e.hotkey_controller());
        if model.engine_running() && control.is_none() {
            self.status = "Recorder unavailable; check the engine hotkey listener.".into();
            self.sync(cx);
            return;
        }
        if let Some(control) = control.filter(|_| model.engine_running()) {
            match control.record() {
                Ok(recording) => self.recording = Some(recording),
                Err(error) => {
                    self.status = error.to_string();
                    self.sync(cx);
                    return;
                }
            }
        }
        if let Some(draft) = &mut self.draft {
            draft.selected = binding;
        } else {
            let config = model.core.hotkeys.clone();
            self.draft = Some(Draft {
                before: config.clone(),
                config,
                selected: binding,
            });
        }
        self.device.update(cx, |view, cx| {
            view.edit_snapshot(|s| scene::text(s, "shortcut.modifier", "choose"), cx)
        });
        self.listening = true;
        self.local_ready = source != Source::Keyboard;
        self.status = if self.recording.is_some() {
            "Press a shortcut (modifier sides supported); release it, then Save."
        } else {
            "Press a chord, or choose a modifier key preset. Esc cancels."
        }
        .into();
        self.sync(cx);
        if self.recording.is_some() {
            self.poll_recording(cx);
        }
    }
    fn poll_recording(&mut self, cx: &mut Context<Self>) {
        let generation = self.capture_generation;
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(25))
                    .await;
                let active = this
                    .update(cx, |this, cx| {
                        if !this.listening || this.capture_generation != generation {
                            return false;
                        }
                        if let Some(candidate) = this.recording.as_ref().and_then(|r| r.candidate())
                        {
                            this.candidate(candidate, cx);
                            return false;
                        }
                        true
                    })
                    .unwrap_or(false);
                if !active {
                    break;
                }
            }
        })
        .detach();
    }
    fn candidate(&mut self, text: String, cx: &mut Context<Self>) {
        if let Some(draft) = &mut self.draft {
            draft.selected.set(&mut draft.config, text);
            let mut effective = draft.config.clone();
            if let Some(control) = self
                .model
                .read(cx)
                .engine
                .as_ref()
                .and_then(|e| e.hotkey_controller())
            {
                effective.escape_cancels = control.config().escape_cancels;
            }
            self.status = match validate_hotkeys(&effective) {
                Ok(()) => {
                    "Shortcut captured. Save applies it; Cancel keeps the old binding.".into()
                }
                Err(error) => format!("Not valid: {error}. Choose another shortcut or Cancel."),
            };
        }
        self.stop_listening();
        self.sync(cx);
    }
    fn save(&mut self, cx: &mut Context<Self>) {
        let Some(draft) = &self.draft else {
            return;
        };
        let candidate = draft.config.clone();
        let model = self.model.read(cx);
        let preview = model.configuration_is_preview;
        let control = model
            .engine
            .as_ref()
            .and_then(|e| e.hotkey_controller())
            .filter(|_| model.engine_running());
        let mut effective = candidate.clone();
        if let Some(control) = &control {
            effective.escape_cancels = control.config().escape_cancels;
        }
        let result = validate_hotkeys(&effective).and_then(|_| {
            if preview {
                Ok(None)
            } else {
                shortcuts::persist_then(
                    &ears::config::AppConfig::path()?,
                    &draft.before,
                    &candidate,
                    || control.as_ref().map(|c| c.reconfigure(&candidate)),
                )
            }
        });
        let applied = match result {
            Ok(applied) => applied,
            Err(error) => {
                self.status = format!("Not saved: {error:#}");
                self.sync(cx);
                return;
            }
        };
        self.model.update(cx, |m, cx| {
            for b in Binding::ALL {
                b.set(&mut m.core.hotkeys, b.get(&candidate).unwrap_or("").into());
            }
            cx.notify();
        });
        self.stop_listening();
        self.recording = None;
        self.draft = None;
        self.status = if preview {
            "Preview only; no configuration written.".into()
        } else {
            match applied {
                Some(Ok(())) => {
                    "Saved and applied; shortcuts rearm after held keys are released.".into()
                }
                Some(Err(e)) => format!("Saved, not active: {e}. Restart the engine to apply."),
                None => "Saved; applies when the engine starts.".into(),
            }
        };
        self.sync(cx);
    }
    fn key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if !self.listening && super::window::close_key(event) {
            self.cancel(cx);
            window.remove_window();
            cx.stop_propagation();
            return;
        }
        if event.keystroke.key == "escape" && self.draft.is_some() {
            self.cancel(cx);
            cx.stop_propagation();
            return;
        }
        if self.listening {
            if self.recording.is_none()
                && self.local_ready
                && !event.is_held
                && let Some(text) = shortcuts::window_combo(&event.keystroke)
            {
                self.candidate(text, cx);
            }
            cx.stop_propagation();
        }
    }
}
impl Render for NativeSettings {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let error = self.device.read(cx).error().map(str::to_owned);
        div()
            .size_full()
            .relative()
            .when_some(error, |root, error| {
                root.child(
                    div()
                        .absolute()
                        .top(gpui::px(12.0))
                        .left(gpui::px(12.0))
                        .id("native-settings-fallback")
                        .bg(gpui::rgb(0x22262a))
                        .text_color(gpui::rgb(0xffffff))
                        .p(gpui::px(12.0))
                        .cursor_pointer()
                        .on_click(cx.listener(|_, _, _, cx| cx.emit(Navigate(Tab::Dictation))))
                        .child(format!(
                            "Renderer unavailable: {error}. Open standard settings"
                        )),
                )
            })
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseDownEvent, window, cx| {
                    if this
                        .device
                        .read(cx)
                        .is_chassis_at(event.position.x.into(), event.position.y.into())
                    {
                        this.drag.begin(window);
                    }
                }),
            )
            .on_mouse_move(cx.listener(|this, event, window, _| this.drag.update(event, window)))
            .on_mouse_up(
                gpui::MouseButton::Left,
                cx.listener(|this, _, _, _| this.drag.end()),
            )
            .on_mouse_up_out(
                gpui::MouseButton::Left,
                cx.listener(|this, _, _, _| this.drag.end()),
            )
            .on_key_down(cx.listener(|this, event, window, cx| this.key(event, window, cx)))
            .on_key_up(cx.listener(|this, _, _, _| this.local_ready = true))
            .child(self.device.clone())
    }
}
