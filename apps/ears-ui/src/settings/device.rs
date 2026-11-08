//! eaRS adapter for the shared receiver: one page, immediate apply, top display.
//! The renderer never owns app effects; every change goes through AppModel here.
use super::{
    Tab, device_scene as scene,
    live_settings::LiveSetting,
    microphone,
    shortcuts::{self, Binding},
};
use crate::app::AppModel;
use device_gpui::{DeviceView, Hover};
use device_scene::{Action, Snapshot};
use ears::{
    audio::InputDevices,
    config::{DictationHotkeyMode, HotkeyConfig},
    dictation::{
        DictationCommand, InsertionMode,
        hotkeys::{HotkeyRecording, validate_hotkeys},
    },
};
use gpui::{Context, Entity, EventEmitter, KeyDownEvent, Window, div, prelude::*};
use std::{
    borrow::Cow,
    time::{Duration, Instant},
};

const LANGUAGES: [(&str, &str); 7] = [
    ("auto", "Auto"),
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("ja", "Japanese"),
];
const INSERT: [(InsertionMode, &str); 3] = [
    (InsertionMode::InsertAtCursor, "At cursor"),
    (InsertionMode::Clipboard, "Clipboard"),
    (InsertionMode::SendAsPrompt, "Prompt"),
];
const KEY_MODES: [(DictationHotkeyMode, &str); 3] = [
    (DictationHotkeyMode::Toggle, "Toggle"),
    (DictationHotkeyMode::PushToTalk, "Hold"),
    (DictationHotkeyMode::Hybrid, "Hybrid"),
];
const MESSAGE_FOR: Duration = Duration::from_secs(5);

#[path = "device_pages.rs"]
mod pages;
#[path = "device_ranges.rs"]
pub(super) mod ranges;
#[path = "device_value_entry.rs"]
mod value_entry;
use super::device_layout::{DICT_ROWS, PROFILE_ROWS};

fn page_id(tab: Tab) -> &'static str {
    match tab {
        Tab::Dictation => "dictation",
        Tab::Overlay => "overlay",
        Tab::Dictionaries => "dictionaries",
        Tab::Profiles => "profiles",
    }
}
const PREVIEW_WORDS: &str = "This is a synthetic preview of how test words stream into the display";

/// Requests the Settings host owns (navigation, live config edits, server restart).
#[derive(Clone, Debug)]
pub(super) enum Request {
    Navigate(Tab),
    Live(LiveSetting, f32),
    Server(String),
}

pub(super) fn attach(
    model: Entity<AppModel>,
    window: &mut Window,
    cx: &mut Context<super::SettingsView>,
) -> Option<Entity<NativeSettings>> {
    let enabled = !model.read(cx).configuration_is_preview
        || super::layout_editor::requested()
        || std::env::args().any(|a| a.starts_with("--preview-settings-"));
    let native = enabled.then(|| cx.new(|cx| NativeSettings::new(model.clone(), window, cx)))?;
    cx.subscribe_in(
        &native,
        window,
        |this, native, request: &Request, window, cx| match request {
            Request::Navigate(tab) => {
                // Keep the host's tab (and the remembered tab) in step with the device.
                this.select_tab(*tab, window, cx);
            }
            Request::Live(setting, value) => {
                this.commit_live(*setting, *value, cx);
                let error = this.live_error.clone();
                native.update(cx, |native, cx| native.live_result(error, cx));
            }
            Request::Server(alias) => {
                let alias = alias.clone();
                this.server
                    .update(cx, |field, cx| field.set_text(&alias, cx));
                this.apply_server(cx);
            }
        },
    )
    .detach();
    Some(native)
}

#[derive(Default)]
struct TestCapture {
    active: bool,
    started: Option<Instant>,
    capturing_seen: bool,
    seen_utterance: Option<String>,
    words: String,
    preview_index: usize,
}
struct Message {
    text: String,
    until: Option<Instant>,
}
pub(super) struct NativeSettings {
    model: Entity<AppModel>,
    device: Entity<DeviceView>,
    drag: super::window::Drag,
    listening: Option<Binding>,
    recording: Option<HotkeyRecording>,
    local_ready: bool,
    capture_generation: u64,
    catalog: Option<InputDevices>,
    mic_choice: Option<usize>,
    mic_generation: u64,
    waiting_input: Option<Option<String>>,
    server_choice: Option<String>,
    help: Option<String>,
    message: Option<Message>,
    test: TestCapture,
    #[cfg(target_os = "macos")]
    window_buttons: Option<crate::platform::SettingsButtons>,
    pending_live: Option<&'static str>,
    range_preview: Option<(ranges::Numeric, f32)>,
    value_entry: Option<value_entry::ValueEntry>,
    editor: Option<super::layout_editor::LayoutEditor>,
    page: Tab,
    text: std::collections::BTreeMap<&'static str, Entity<crate::ui::text_input::TextInput>>,
    dict_page: usize,
    prof_page: usize,
    dict_entries: Vec<(String, Vec<String>)>,
    dict_file: Option<String>,
    profile: pages::ProfileEdit,
}
impl EventEmitter<Request> for NativeSettings {}

impl NativeSettings {
    fn new(model: Entity<AppModel>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        // Display text fields need their key bindings on every Settings path.
        crate::ui::text_input::init(cx);
        // Host typography, never font paths supplied by a scene.
        if let Err(error) = cx.text_system().add_fonts(vec![
            Cow::Borrowed(include_bytes!("../../assets/fonts/Michroma-Regular.ttf")),
            Cow::Borrowed(include_bytes!("../../assets/fonts/Geo-Regular.ttf")),
        ]) {
            eprintln!("ears-ui: device fonts: {error:#}");
        }
        let mut snapshot = scene::initial();
        let page = model.read(cx).settings_tab;
        snapshot.state.active_page = page_id(page).into();
        let text = Self::create_text_fields(cx);
        let editor = super::layout_editor::requested()
            .then(|| super::layout_editor::LayoutEditor::new(&mut snapshot));
        let device = cx.new(|cx| {
            let mut view = DeviceView::new(
                snapshot,
                "Michroma".into(),
                crate::theme::FONT_SANS.into(),
                window,
                cx,
            );
            view.set_control_font("brand", "Geo".into());
            view.set_control_icon("test", std::rc::Rc::new(mic_icon));
            view
        });
        cx.subscribe_in(&device, window, |this, _, action: &Action, window, cx| {
            this.action(action, window, cx)
        })
        .detach();
        cx.subscribe(&device, |this, _, hover: &Hover, cx| {
            let help = hover.0.as_deref().and_then(|id| this.help_for(id, cx));
            if help != this.help {
                this.help = help;
                this.sync(cx);
            }
        })
        .detach();
        cx.observe(&model, |this, _, cx| this.sync(cx)).detach();
        #[cfg(target_os = "macos")]
        cx.observe(&device, |_, _, cx| cx.notify()).detach();
        cx.observe_window_activation(window, |this, window, cx| {
            if window.is_window_active() {
                this.refresh_inputs(cx);
            } else {
                this.drag.end();
                this.range_preview = None;
                this.stop_listening(cx);
                this.sync(cx);
            }
        })
        .detach();
        let mut this = Self {
            model,
            device,
            drag: Default::default(),
            listening: None,
            recording: None,
            local_ready: true,
            capture_generation: 0,
            catalog: None,
            mic_choice: None,
            mic_generation: 0,
            waiting_input: None,
            server_choice: None,
            help: None,
            message: None,
            test: TestCapture::default(),
            #[cfg(target_os = "macos")]
            window_buttons: crate::platform::SettingsButtons::new(window),
            pending_live: None,
            range_preview: None,
            value_entry: None,
            editor,
            page,
            text,
            dict_page: 0,
            prof_page: 0,
            dict_entries: Vec::new(),
            dict_file: None,
            profile: Default::default(),
        };
        this.reload_entries(cx);
        this.refresh_inputs(cx);
        this
    }
    pub(super) fn focus(&self, window: &mut Window, cx: &gpui::App) {
        self.device.read(cx).focus(window);
    }
    pub(super) fn renderer_available(&self, cx: &gpui::App) -> bool {
        self.device.read(cx).error().is_none()
    }
    fn help_for(&self, id: &str, cx: &gpui::App) -> Option<String> {
        if id == "display" {
            return None;
        }
        let snapshot = self.device.read(cx).snapshot();
        let (_, control, _) = snapshot.control(id)?;
        let help = control.description.trim();
        (!help.is_empty() && help != control.label).then(|| help.to_owned())
    }
    /// Show a short confirmation or error, then return to help/idle text.
    fn note(&mut self, text: impl Into<String>, cx: &mut Context<Self>) {
        self.message = Some(Message {
            text: text.into(),
            until: Some(Instant::now() + MESSAGE_FOR),
        });
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(MESSAGE_FOR).await;
            let _ = this.update(cx, |this, cx| this.sync(cx));
        })
        .detach();
    }
    fn say(&mut self, text: impl Into<String>, cx: &mut Context<Self>) {
        self.note(text, cx);
        self.sync(cx);
    }
    pub(super) fn live_result(&mut self, error: Option<String>, cx: &mut Context<Self>) {
        let label = self.pending_live.take().unwrap_or("Setting");
        match error {
            Some(error) => self.say(error, cx),
            None => self.say(format!("{label} saved."), cx),
        }
    }
    fn observe_input(&mut self, model: &ModelView, cx: &mut Context<Self>) {
        if self.waiting_input.is_some() && model.input_pending.is_none() {
            let requested = self.waiting_input.take().flatten();
            let name = requested.as_deref().unwrap_or("System default");
            match &model.input_error {
                Some(error) => self.note(format!("Microphone not changed: {error}"), cx),
                None => self.note(format!("Using {name}."), cx),
            }
        }
    }

    // ---- display -------------------------------------------------------

    fn display_message(&mut self, model: &ModelView) -> (String, bool) {
        if let Some(binding) = self.listening {
            return (
                format!(
                    "Press the new {} shortcut. Esc cancels.",
                    binding.label().to_lowercase()
                ),
                false,
            );
        }
        if self.test.active {
            let mut words = self.test.words.clone();
            let partial = model.partial.trim();
            if !partial.is_empty() {
                if !words.is_empty() {
                    words.push(' ');
                }
                words.push_str(partial);
            }
            if words.is_empty() {
                return (
                    "Listening… speak now. Press Test again to stop.".into(),
                    false,
                );
            }
            return (words, true);
        }
        if let Some(message) = &self.message {
            if message.until.is_none_or(|until| Instant::now() < until) {
                return (message.text.clone(), false);
            }
            self.message = None;
        }
        if let Some(help) = &self.help {
            return (help.clone(), false);
        }
        let start = scene::shortcut_text(Some(&model.hotkeys.toggle));
        let idle = if model.setup_mode {
            "Choose a server below and press Use to start dictation.".to_owned()
        } else {
            match model.connection {
                ears::dictation::ConnectionState::Connected => {
                    format!("Ready. Press {start} to dictate, or Test to try it here.")
                }
                ears::dictation::ConnectionState::Connecting => "Connecting to the server…".into(),
                ears::dictation::ConnectionState::Disconnected => {
                    "The server is offline. Check Server below.".into()
                }
            }
        };
        (idle, false)
    }

    // ---- sync ------------------------------------------------------------

    fn sync(&mut self, cx: &mut Context<Self>) {
        self.observe_test(cx);
        let model = self.model.read(cx).clone_view();
        self.observe_input(&model, cx);
        let (message, newest) = self.display_message(&model);
        let listening = self.listening;
        let test_on = self.test.active;
        let editing = self.editor.as_ref().is_some_and(|e| e.active);
        let pages = self.pages_view(cx);
        let page = self.page;
        let (mic_choices, mic_active) = self.mic_choices(&model);
        let mic_labels: Vec<String> = mic_choices.iter().map(|c| c.label.clone()).collect();
        let mic_shown = self
            .mic_choice
            .unwrap_or(mic_active)
            .min(mic_labels.len().saturating_sub(1));
        let server_choice = self.server_choice.clone().unwrap_or(model.server.clone());
        let (range_core, range_ui) = self
            .model
            .read_with(cx, |m, _| (m.core.clone(), m.ui.clone()));
        let range_preview = self.range_preview;
        self.device.update(cx, |view, cx| {
            view.set_keyboard_enabled(listening.is_none() && !editing);
            view.edit_snapshot(
                |s| {
                    let (cols, rows) = scene::capacity(s, "display");
                    scene::text(
                        s,
                        "display",
                        scene::display_text(&message, cols, rows, newest),
                    );
                    // The microphone glows while a test runs; the label is the accessible name.
                    scene::label(s, "test", if test_on { "Stop test" } else { "Test" });
                    pages.write(s);
                    for tab in Tab::ALL {
                        scene::switch(s, &format!("nav.{}", page_id(tab)), tab == page);
                    }
                    scene::choices(s, "mic", &mic_labels, mic_shown);
                    scene::enable(s, "mic.prev", mic_labels.len() > 1);
                    scene::enable(s, "mic.next", mic_labels.len() > 1);
                    scene::enable(s, "mic", mic_labels.len() > 1);
                    write_shortcuts(s, &model.hotkeys, listening);
                    write_settings(s, &model, &server_choice);
                    ranges::write(s, &range_core, &range_ui, range_preview);
                },
                cx,
            );
        });
        cx.notify();
    }
    /// Selectable inputs plus a saved input that is not connected; index of the active one.
    fn mic_choices(&self, model: &ModelView) -> (Vec<microphone::Choice>, usize) {
        let saved = &model.input_device;
        let active = model.input_pending.clone().unwrap_or_else(|| saved.clone());
        let mut choices: Vec<_> = microphone::choices(self.catalog.as_ref())
            .into_iter()
            .filter(|c| c.selectable)
            .collect();
        if !choices.iter().any(|c| &c.name == saved) {
            choices.push(microphone::Choice {
                label: format!(
                    "{} (not connected)",
                    saved.as_deref().unwrap_or("System default")
                ),
                name: saved.clone(),
                selectable: false,
            });
        }
        let current = choices.iter().position(|c| c.name == active).unwrap_or(0);
        (choices, current)
    }
    /// Step the shown input; the switch happens once the user stops stepping, so
    /// passing over inputs never opens them.
    fn step_mic(&mut self, target: MicStep, cx: &mut Context<Self>) {
        let model = self.model.read(cx).clone_view();
        let (choices, active) = self.mic_choices(&model);
        if choices.len() < 2 {
            return;
        }
        let shown = self.mic_choice.unwrap_or(active);
        let next = match target {
            MicStep::By(delta) => {
                (shown as isize + delta).rem_euclid(choices.len() as isize) as usize
            }
            MicStep::To(index) => index.min(choices.len() - 1),
        };
        self.mic_choice = Some(next);
        self.mic_generation += 1;
        let generation = self.mic_generation;
        self.sync(cx);
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(MIC_SETTLE).await;
            let _ = this.update(cx, |this, cx| {
                if this.mic_generation == generation {
                    this.commit_mic(cx);
                }
            });
        })
        .detach();
    }
    fn commit_mic(&mut self, cx: &mut Context<Self>) {
        let Some(index) = self.mic_choice.take() else {
            return;
        };
        let model = self.model.read(cx).clone_view();
        let (choices, active) = self.mic_choices(&model);
        let Some(choice) = choices.get(index).cloned() else {
            return self.sync(cx);
        };
        if index == active {
            return self.sync(cx);
        }
        if !choice.selectable {
            return self.say(format!("{} is not connected.", choice.label), cx);
        }
        let name = choice.name.clone();
        match self
            .model
            .update(cx, |m, cx| microphone::apply(m, name.clone(), cx))
        {
            Ok(()) if model.preview => {
                self.say(format!("Preview: input set to {}.", choice.label), cx)
            }
            Ok(()) if model.engine_running => {
                self.waiting_input = Some(name);
                self.say(format!("Switching to {}…", choice.label), cx);
            }
            Ok(()) => self.say(
                format!("{} will be used when dictation starts.", choice.label),
                cx,
            ),
            Err(error) => self.say(format!("Microphone not changed: {error:#}"), cx),
        }
    }

    // ---- actions ---------------------------------------------------------

    fn action(&mut self, action: &Action, window: &mut Window, cx: &mut Context<Self>) {
        if !scene::accepts(action, self.device.read(cx).snapshot()) {
            return;
        }
        // Any other control ends a typed value entry; its own display starts one.
        let entering = self
            .value_entry
            .as_ref()
            .is_some_and(|e| e.setting.id() == action.control_id);
        if !entering {
            self.cancel_value_entry(cx);
        }
        if action.phase == device_scene::Phase::Commit
            && self.value_entry_action(&action.control_id, window, cx)
        {
            return;
        }
        if self.range_action(action, cx) {
            return;
        }
        if action.phase == device_scene::Phase::Cancel {
            self.stop_listening(cx);
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
        if let Some(binding) = Binding::ALL
            .into_iter()
            .find(|b| id == format!("shortcut.{}.clear", b.id()))
        {
            let mut config = self.model.read(cx).core.hotkeys.clone();
            binding.set(&mut config, String::new());
            self.commit_shortcuts(
                config,
                &format!("{} shortcut removed.", binding.label()),
                cx,
            );
            return;
        }
        if self.page_action(id, cx) {
            return;
        }
        match id {
            "mic.prev" => return self.step_mic(MicStep::By(-1), cx),
            "mic.next" => return self.step_mic(MicStep::By(1), cx),
            "mic" => {
                // Clicking the display steps to the option the device proposes.
                let index = action
                    .value
                    .string()
                    .and_then(|v| v.strip_prefix("choice-"))
                    .and_then(|v| v.parse::<usize>().ok());
                return match index {
                    Some(index) => self.step_mic(MicStep::To(index), cx),
                    None => self.step_mic(MicStep::By(1), cx),
                };
            }
            _ => {}
        }
        let step = |id: &str, base: &str| -> Option<isize> {
            id.strip_prefix(base).and_then(|rest| match rest {
                ".prev" => Some(-1),
                ".next" => Some(1),
                _ => None,
            })
        };
        match id {
            "window.close" => {
                self.stop_listening(cx);
                window.remove_window();
            }
            "nav.dictation" => self.set_page(Tab::Dictation, cx),
            "nav.overlay" => self.set_page(Tab::Overlay, cx),
            "nav.dictionaries" => self.set_page(Tab::Dictionaries, cx),
            "nav.profiles" => self.set_page(Tab::Profiles, cx),
            "test" => self.toggle_test(cx),
            "shortcut.defaults" => {
                let mut config = self.model.read(cx).core.hotkeys.clone();
                let defaults = HotkeyConfig::default();
                for b in Binding::ALL {
                    b.set(&mut config, b.get(&defaults).unwrap_or("").into());
                }
                self.commit_shortcuts(config, "Default shortcuts restored.", cx);
            }
            "escape" => {
                let on = !self.model.read(cx).ui.dictation.escape_cancels;
                self.model.update(cx, |m, cx| {
                    m.update_ui(cx, |ui| ui.dictation.escape_cancels = on);
                    m.send(DictationCommand::SetEscapeCancels(on));
                });
                self.say(
                    if on {
                        "Esc now discards pending words."
                    } else {
                        "Esc now pauses dictation."
                    },
                    cx,
                );
            }
            "profiles" => {
                let on = !self.model.read(cx).ui.dictation.profiles;
                self.model
                    .update(cx, |m, cx| m.update_ui(cx, |ui| ui.dictation.profiles = on));
                self.say(
                    if on {
                        "Context profiles on."
                    } else {
                        "Context profiles off."
                    },
                    cx,
                );
            }
            "server.use" => {
                let choice = self
                    .server_choice
                    .take()
                    .unwrap_or_else(|| self.model.read(cx).ui.dictation.server.clone());
                self.say(format!("Connecting to {choice}…"), cx);
                cx.emit(Request::Server(choice));
            }
            _ => {
                if let Some(delta) = step(id, "keymode") {
                    let current = self.model.read(cx).hotkey_mode;
                    let (mode, label) = cycle(&KEY_MODES, |(m, _)| *m == current, delta);
                    self.model.update(cx, |m, cx| m.set_hotkey_mode(mode, cx));
                    self.say(format!("Key mode: {label}."), cx);
                } else if let Some(delta) = step(id, "language") {
                    let current = self
                        .model
                        .read(cx)
                        .language
                        .clone()
                        .unwrap_or_else(|| "auto".into());
                    let (code, _) = cycle(&LANGUAGES, |(c, _)| *c == current, delta);
                    self.model
                        .update(cx, |m, cx| m.set_language(code.into(), cx));
                    self.sync(cx);
                } else if let Some(delta) = step(id, "insert") {
                    let current = self.model.read(cx).insertion_mode;
                    let (mode, _) = cycle(&INSERT, |(m, _)| *m == current, delta);
                    self.model
                        .update(cx, |m, cx| m.set_insertion_mode(mode, cx));
                    self.sync(cx);
                } else if let Some(delta) = step(id, "server") {
                    let model = self.model.read(cx);
                    let mut aliases: Vec<String> =
                        model.core.dictation.servers.keys().cloned().collect();
                    aliases.sort();
                    let current = self
                        .server_choice
                        .clone()
                        .unwrap_or_else(|| model.ui.dictation.server.clone());
                    if !aliases.is_empty() {
                        let index = aliases.iter().position(|a| *a == current);
                        let next = match index {
                            Some(i) => {
                                (i as isize + delta).rem_euclid(aliases.len() as isize) as usize
                            }
                            None => 0,
                        };
                        let saved = model.ui.dictation.server.clone();
                        self.server_choice =
                            (aliases[next] != saved).then(|| aliases[next].clone());
                        self.sync(cx);
                    }
                }
            }
        }
    }
    fn refresh_inputs(&mut self, cx: &mut Context<Self>) {
        if self.model.read(cx).configuration_is_preview {
            self.catalog = Some(microphone::fixture());
            self.sync(cx);
            return;
        }
        // Metadata only: listing inputs never opens a microphone.
        let task = cx
            .background_executor()
            .spawn(async { ears::audio::input_devices() });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |this, cx| match result {
                Ok(catalog) => {
                    this.catalog = Some(catalog);
                    this.sync(cx);
                }
                Err(error) => this.say(format!("Cannot list microphones: {error}"), cx),
            });
        })
        .detach();
    }

    // ---- test capture ------------------------------------------------------

    fn toggle_test(&mut self, cx: &mut Context<Self>) {
        let model = self.model.read(cx);
        if self.test.active {
            if !model.configuration_is_preview {
                model.send(DictationCommand::SetTestCapture(false));
            }
            self.finish_test(cx);
            return;
        }
        self.stop_listening(cx);
        let model = self.model.read(cx);
        if model.configuration_is_preview {
            self.test = TestCapture {
                active: true,
                started: Some(Instant::now()),
                ..Default::default()
            };
            self.run_preview_test(cx);
            self.sync(cx);
            return;
        }
        if !model.engine_running() {
            self.say("Start dictation first: choose a server and press Use.", cx);
            return;
        }
        if model.is_capturing() {
            self.say("Stop dictation before testing.", cx);
            return;
        }
        self.test = TestCapture {
            active: true,
            started: Some(Instant::now()),
            seen_utterance: model.last_utterance.clone(),
            ..Default::default()
        };
        model.send(DictationCommand::SetTestCapture(true));
        self.sync(cx);
    }
    fn observe_test(&mut self, cx: &mut Context<Self>) {
        if !self.test.active || self.model.read(cx).configuration_is_preview {
            return;
        }
        let model = self.model.read(cx);
        let capturing = model.is_capturing();
        if model.last_utterance != self.test.seen_utterance {
            self.test.seen_utterance = model.last_utterance.clone();
            if let Some(text) = &model.last_utterance {
                if !self.test.words.is_empty() {
                    self.test.words.push(' ');
                }
                self.test.words.push_str(text.trim());
            }
        }
        if capturing {
            self.test.capturing_seen = true;
        } else if self.test.capturing_seen
            || self
                .test
                .started
                .is_some_and(|t| t.elapsed() > Duration::from_secs(3))
        {
            // Auto-pause, a hotkey or an error ended the capture.
            self.finish_test(cx);
        }
    }
    fn finish_test(&mut self, cx: &mut Context<Self>) {
        let preview = self.model.read(cx).configuration_is_preview;
        let words = std::mem::take(&mut self.test).words;
        let result = if words.trim().is_empty() {
            "No words recognised. Check the microphone and try again.".to_owned()
        } else if preview {
            format!("Preview only, no microphone used: {words}")
        } else {
            words
        };
        self.message = Some(Message {
            text: result,
            until: None,
        });
        self.sync(cx);
    }
    fn run_preview_test(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(220))
                    .await;
                let more = this
                    .update(cx, |this, cx| {
                        if !this.test.active {
                            return false;
                        }
                        let words: Vec<_> = PREVIEW_WORDS.split(' ').collect();
                        if this.test.preview_index >= words.len() {
                            this.finish_test(cx);
                            return false;
                        }
                        if !this.test.words.is_empty() {
                            this.test.words.push(' ');
                        }
                        this.test.words.push_str(words[this.test.preview_index]);
                        this.test.preview_index += 1;
                        this.sync(cx);
                        true
                    })
                    .unwrap_or(false);
                if !more {
                    break;
                }
            }
        })
        .detach();
    }

    // ---- shortcuts -------------------------------------------------------

    fn stop_listening(&mut self, cx: &mut Context<Self>) {
        if self.listening.take().is_some() {
            self.capture_generation += 1;
            self.recording = None;
            self.sync(cx);
        }
    }
    fn begin(&mut self, binding: Binding, source: device_scene::Source, cx: &mut Context<Self>) {
        self.stop_listening(cx);
        let model = self.model.read(cx);
        let control = model.engine.as_ref().and_then(|e| e.hotkey_controller());
        if let Some(control) = control.filter(|_| model.engine_running()) {
            match control.record() {
                Ok(recording) => self.recording = Some(recording),
                Err(error) => {
                    self.say(error.to_string(), cx);
                    return;
                }
            }
        }
        self.message = None;
        self.listening = Some(binding);
        self.local_ready = source != device_scene::Source::Keyboard;
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
                        if this.listening.is_none() || this.capture_generation != generation {
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
        let Some(binding) = self.listening else {
            return;
        };
        let mut config = self.model.read(cx).core.hotkeys.clone();
        binding.set(&mut config, text.clone());
        // Keep the recorder guard until the save and live apply have finished.
        let label = scene::shortcut_text(Some(&text));
        self.commit_shortcuts(config, &format!("{} is now {label}.", binding.label()), cx);
        self.listening = None;
        self.capture_generation += 1;
        self.recording = None;
        self.sync(cx);
    }
    /// Validate, save atomically, then apply live. A failure changes nothing.
    fn commit_shortcuts(&mut self, candidate: HotkeyConfig, done: &str, cx: &mut Context<Self>) {
        let model = self.model.read(cx);
        let before = model.core.hotkeys.clone();
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
                    &before,
                    &candidate,
                    || control.as_ref().map(|c| c.reconfigure(&candidate)),
                )
            }
        });
        let message = match result {
            Err(error) => format!("Not changed: {error:#}"),
            Ok(applied) => {
                self.model.update(cx, |m, cx| {
                    for b in Binding::ALL {
                        b.set(&mut m.core.hotkeys, b.get(&candidate).unwrap_or("").into());
                    }
                    cx.notify();
                });
                match applied {
                    Some(Err(e)) => format!("Saved, but not active yet: {e}"),
                    _ if preview => format!("Preview: {done}"),
                    _ => done.to_owned(),
                }
            }
        };
        self.say(message, cx);
    }
    fn set_page(&mut self, tab: Tab, cx: &mut Context<Self>) {
        self.range_preview = None;
        self.page = tab;
        self.help = None;
        if tab == Tab::Dictionaries {
            self.reload_entries(cx);
        }
        self.device.update(cx, |view, cx| {
            view.edit_snapshot(|s| s.state.active_page = page_id(tab).into(), cx)
        });
        cx.emit(Request::Navigate(tab));
        self.sync(cx);
    }
    /// Copy only geometry from an editor result, so state revisions keep moving forward.
    fn apply_layout(&mut self, next: Snapshot, cx: &mut Context<Self>) {
        self.device.update(cx, |view, cx| {
            view.edit_snapshot(
                |s| {
                    let src = &next.scene.devices[0];
                    let dst = &mut s.scene.devices[0];
                    dst.panels = src.panels.clone();
                    for c in &mut dst.controls {
                        if let Some(from) = src.controls.iter().find(|f| f.id == c.id) {
                            c.rect = from.rect.clone();
                            from.module_id.clone_into(&mut c.module_id);
                            c.matrix = from.matrix.clone();
                        }
                    }
                },
                cx,
            )
        });
        cx.notify();
    }
    fn key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(editor) = self.editor.as_mut() {
            let snapshot = self.device.read(cx).snapshot().clone();
            let (handled, next) = editor.key(event, &snapshot);
            if let Some(next) = next {
                self.apply_layout(next, cx);
            }
            if handled {
                self.sync(cx);
                cx.stop_propagation();
                return;
            }
        }
        if self.listening.is_none() && super::window::close_key(event) {
            window.remove_window();
            cx.stop_propagation();
            return;
        }
        if event.keystroke.key == "escape" {
            if self.cancel_value_entry(cx) {
                self.focus(window, cx);
                cx.stop_propagation();
            } else if self.listening.is_some() {
                self.stop_listening(cx);
                self.say("Shortcut unchanged.", cx);
                cx.stop_propagation();
            } else if self.test.active {
                self.toggle_test(cx);
                cx.stop_propagation();
            }
            return;
        }
        if self.listening.is_some() {
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

fn cycle<T: Copy>(
    items: &[(T, &'static str)],
    is_current: impl Fn(&(T, &'static str)) -> bool,
    delta: isize,
) -> (T, &'static str) {
    let index = items.iter().position(is_current).unwrap_or(0) as isize;
    items[(index + delta).rem_euclid(items.len() as isize) as usize]
}
#[derive(Clone, Copy)]
enum MicStep {
    By(isize),
    To(usize),
}
const MIC_SETTLE: Duration = Duration::from_millis(900);

/// The model fields the device shows, copied once per sync.
struct ModelView {
    preview: bool,
    engine_running: bool,
    hotkeys: HotkeyConfig,
    input_device: Option<String>,
    input_pending: Option<Option<String>>,
    input_error: Option<String>,
    hotkey_mode: DictationHotkeyMode,
    escape_cancels: bool,
    profiles: bool,
    language: String,
    insertion: InsertionMode,
    pause_ms: u32,
    threshold: f32,
    wake: Option<f32>,
    server: String,
    servers: usize,
    setup_mode: bool,
    connection: ears::dictation::ConnectionState,
    partial: String,
}
impl AppModel {
    fn clone_view(&self) -> ModelView {
        let wake = &self.core.dictation.wake;
        ModelView {
            preview: self.configuration_is_preview,
            engine_running: self.engine_running(),
            hotkeys: self.core.hotkeys.clone(),
            input_device: self.core.dictation.input_device.clone(),
            input_pending: self.input_pending.clone(),
            input_error: self.input_error.clone(),
            hotkey_mode: self.hotkey_mode,
            escape_cancels: self.ui.dictation.escape_cancels,
            profiles: self.ui.dictation.profiles,
            language: self.language.clone().unwrap_or_else(|| "auto".into()),
            insertion: self.insertion_mode,
            pause_ms: self.core.dictation.auto_pause.idle_ms,
            threshold: self.core.dictation.auto_pause.speech_threshold,
            wake: (wake.enabled && !wake.command.is_empty()).then(|| wake.threshold.unwrap_or(0.5)),
            server: self.ui.dictation.server.clone(),
            servers: self.core.dictation.servers.len(),
            setup_mode: self.setup_mode,
            connection: self.connection,
            partial: self.partial.clone(),
        }
    }
}

fn write_shortcuts(s: &mut Snapshot, hotkeys: &HotkeyConfig, listening: Option<Binding>) {
    for binding in Binding::ALL {
        let id = format!("shortcut.{}", binding.id());
        let value = binding.get(hotkeys).filter(|v| !v.trim().is_empty());
        let text = if listening == Some(binding) {
            "Press keys".to_owned()
        } else {
            scene::shortcut_text(value)
        };
        scene::text(s, &id, text);
        // Every row keeps its clear key; it is available only when it can act.
        scene::enable(
            s,
            &format!("{id}.clear"),
            binding != Binding::Toggle && value.is_some() && listening.is_none(),
        );
    }
}
fn write_settings(s: &mut Snapshot, model: &ModelView, server_choice: &str) {
    let name = |items: &[(&str, &'static str)], key: &str| {
        items
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(key.to_owned(), |(_, n)| (*n).to_owned())
    };
    let mode = KEY_MODES
        .iter()
        .find(|(m, _)| *m == model.hotkey_mode)
        .map_or("Toggle", |(_, n)| n);
    scene::text(s, "keymode", mode);
    let insert = INSERT
        .iter()
        .find(|(m, _)| *m == model.insertion)
        .map_or("At cursor", |(_, n)| n);
    scene::text(s, "insert", insert);
    scene::text(s, "language", name(&LANGUAGES, &model.language));
    for (id, on) in [
        ("escape", model.escape_cancels),
        ("profiles", model.profiles),
    ] {
        scene::switch(s, id, on);
        scene::label(s, id, if on { "On" } else { "Off" });
    }
    let pause_on = model.pause_ms > 0;
    scene::text(
        s,
        "pause",
        if pause_on {
            format!("{:.1} s", model.pause_ms as f32 / 1000.0)
        } else {
            "Off".into()
        },
    );
    scene::text(s, "threshold", format!("{:.2}", model.threshold));
    scene::text(
        s,
        "wake",
        model.wake.map_or("Off".into(), |v| format!("{v:.2}")),
    );
    for (id, on) in [("server", model.servers > 0)] {
        scene::enable(s, &format!("{id}.prev"), on);
        scene::enable(s, &format!("{id}.next"), on);
    }
    let server = if server_choice.is_empty() {
        "Default"
    } else {
        server_choice
    };
    scene::text(s, "server", server);
    let pending = server_choice != model.server;
    let status = if pending {
        "Not in use"
    } else if model.setup_mode {
        "Not started"
    } else {
        match model.connection {
            ears::dictation::ConnectionState::Connected => "Connected",
            ears::dictation::ConnectionState::Connecting => "Connecting",
            ears::dictation::ConnectionState::Disconnected => "Offline",
        }
    };
    scene::text(s, "connection", status);
    scene::enable(s, "server.use", pending || model.setup_mode);
    let _ = (model.preview, model.engine_running, &model.input_device);
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
                        .child(format!("Renderer unavailable: {error}")),
                )
            })
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseDownEvent, window, cx| {
                    let editing = this.editor.as_ref().is_some_and(|e| e.active);
                    if !editing
                        && this
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
            .children(self.text_overlays(cx))
            .children(self.value_entry_overlay(cx))
            .map(|root| {
                #[cfg(target_os = "macos")]
                {
                    let device = self.device.clone();
                    let buttons = self.window_buttons.clone();
                    return root.on_children_prepainted(move |_, window, cx| {
                        if let Some(buttons) = &buttons {
                            super::window_buttons::sync(device.read(cx), buttons, window);
                        }
                    });
                }
                #[cfg(not(target_os = "macos"))]
                root
            })
            .when_some(self.editor.as_ref(), |root, editor| {
                let device = self.device.read(cx);
                let overlay = editor.overlay(device.projection(), device.snapshot());
                let active = editor.active;
                root.child(
                    div()
                        .id("layout-editor")
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full()
                        .when(active, |layer| {
                            layer
                                .occlude()
                                .on_mouse_down(
                                    gpui::MouseButton::Left,
                                    cx.listener(|this, e: &gpui::MouseDownEvent, window, cx| {
                                        this.device.read(cx).focus(window);
                                        let device = this.device.read(cx);
                                        let (Some(p), s) =
                                            (device.projection(), device.snapshot().clone())
                                        else {
                                            return;
                                        };
                                        if let Some(editor) = this.editor.as_mut() {
                                            editor.mouse_down(
                                                &p,
                                                &s,
                                                e.position.x.into(),
                                                e.position.y.into(),
                                                e.modifiers.shift,
                                            );
                                        }
                                        // Never also start a window drag underneath.
                                        cx.stop_propagation();
                                        cx.notify();
                                    }),
                                )
                                .on_mouse_move(cx.listener(
                                    |this, e: &gpui::MouseMoveEvent, _, cx| {
                                        if !e.dragging() {
                                            return;
                                        }
                                        let device = this.device.read(cx);
                                        let (Some(p), s) =
                                            (device.projection(), device.snapshot().clone())
                                        else {
                                            return;
                                        };
                                        let next = this.editor.as_mut().and_then(|editor| {
                                            editor.mouse_move(
                                                &p,
                                                &s,
                                                e.position.x.into(),
                                                e.position.y.into(),
                                            )
                                        });
                                        if let Some(next) = next {
                                            this.apply_layout(next, cx);
                                        }
                                        cx.stop_propagation();
                                    },
                                ))
                                .on_mouse_up(
                                    gpui::MouseButton::Left,
                                    cx.listener(|this, _, _, cx| {
                                        if let Some(editor) = this.editor.as_mut() {
                                            editor.mouse_up();
                                        }
                                        cx.stop_propagation();
                                        cx.notify();
                                    }),
                                )
                        })
                        .child(overlay),
                )
            })
    }
}

/// Microphone legend for the Test key: capsule, cradle, stem and base, drawn
/// natively so it stays crisp. Amber while a test is running.
fn mic_icon(label: &str, size: f32, ink: u32) -> gpui::AnyElement {
    use gpui::{px, rgb};
    let colour = if label == "Stop test" {
        rgb(0xff9e52)
    } else {
        rgb(ink)
    };
    // The key is small; the glyph fills most of it rather than reading as a dot.
    let u = (size / 6.6).max(1.2);
    let line = (u * 1.1).max(1.0);
    div()
        .flex()
        .flex_col()
        .items_center()
        .child(
            div()
                .w(px(4.0 * u))
                .h(px(6.4 * u))
                .rounded(px(2.0 * u))
                .bg(colour),
        )
        .child(
            div()
                .mt(px(-2.6 * u))
                .w(px(7.0 * u))
                .h(px(4.2 * u))
                .border_l(px(line))
                .border_r(px(line))
                .border_b(px(line))
                .border_color(colour)
                .rounded_bl(px(3.5 * u))
                .rounded_br(px(3.5 * u)),
        )
        .child(div().w(px(line)).h(px(1.4 * u)).bg(colour))
        .child(div().w(px(4.0 * u)).h(px(line)).bg(colour))
        .into_any_element()
}
