//! Metadata-only input picker. Saving a preference never touches the running engine.
use crate::{
    app::AppModel,
    theme::{Theme, theme},
    ui::{ButtonKind, button, hint, row, section},
};
use anyhow::{Result, ensure};
use ears::audio::{InputDeviceInfo, InputDevices};
use ears::dictation::DictationCommand;
use gpui::prelude::FluentBuilder;
use gpui::{
    Context, Entity, FocusHandle, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    Render, SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};
use std::path::Path;
use toml_edit::{DocumentMut, value};

fn updated_document(original: &str, name: Option<&str>) -> Result<String> {
    ensure!(
        name.is_none_or(|s| !s.trim().is_empty() && s.len() <= 1024),
        "Choose a valid microphone name"
    );
    let mut document: DocumentMut = original.parse()?;
    ensure!(
        document
            .get("dictation")
            .is_none_or(|v| v.as_table_like().is_some()),
        "dictation must be a TOML table"
    );
    if let Some(name) = name {
        let item = &mut document["dictation"]["input_device"];
        let decor = item.as_value().map(|v| v.decor().clone());
        *item = value(name);
        if let Some(decor) = decor {
            *item.as_value_mut().unwrap().decor_mut() = decor;
        }
    } else if let Some(table) = document
        .get_mut("dictation")
        .and_then(|v| v.as_table_like_mut())
    {
        table.remove("input_device");
    }
    Ok(document.to_string())
}
pub(crate) fn persist_input_device(path: &Path, name: Option<&str>) -> Result<()> {
    ensure!(
        path.exists(),
        "Configuration disappeared; reopen Settings before saving"
    );
    super::auto_pause::persist_edit(path, |text| updated_document(text, name))
}
/// Shared by both settings hosts: save only after a live engine confirms the switch.
pub(super) fn apply(
    model: &mut AppModel,
    name: Option<String>,
    cx: &mut Context<AppModel>,
) -> Result<()> {
    ensure!(
        model.input_pending.is_none(),
        "A microphone switch is already pending"
    );
    if model.engine_running() {
        model.input_pending = Some(name.clone());
        model.input_error = None;
        model.send(DictationCommand::SetInputDevice(name));
    } else {
        if !model.configuration_is_preview {
            persist_input_device(&ears::config::AppConfig::path()?, name.as_deref())?;
        }
        model.core.dictation.input_device = name;
        model.input_error = None;
    }
    cx.notify();
    Ok(())
}
#[derive(Clone)]
pub(super) struct Choice {
    pub(super) name: Option<String>,
    pub(super) label: String,
    pub(super) selectable: bool,
}
pub(super) fn choices(catalog: Option<&InputDevices>) -> Vec<Choice> {
    let mut out = vec![Choice {
        name: None,
        label: "System default".into(),
        selectable: true,
    }];
    if let Some(catalog) = catalog {
        for device in &catalog.devices {
            let unique = device.name.as_ref().is_some_and(|name| {
                !name.trim().is_empty()
                    && catalog
                        .devices
                        .iter()
                        .filter(|d| d.name.as_ref() == Some(name))
                        .count()
                        == 1
            });
            let name = device.name.clone();
            let label = match &name {
                Some(n) if unique => n.clone(),
                Some(n) => format!("{n} (duplicate name)"),
                None => format!("Unnamed input {}", device.index),
            };
            out.push(Choice {
                name,
                label,
                selectable: unique,
            });
        }
    }
    out
}
pub(super) fn fixture() -> InputDevices {
    if std::env::args().any(|arg| arg == "--preview-settings-mic-empty") {
        return InputDevices {
            default_name: None,
            devices: Vec::new(),
        };
    }
    InputDevices {
        default_name: Some("Built-in microphone".into()),
        devices: vec![
            InputDeviceInfo {
                index: 0,
                name: Some("Built-in microphone".into()),
            },
            InputDeviceInfo {
                index: 1,
                name: Some("USB microphone".into()),
            },
        ],
    }
}

pub struct MicrophonePicker {
    model: Entity<AppModel>,
    focus: FocusHandle,
    catalog: Option<InputDevices>,
    expanded: bool,
    loading: bool,
    error: Option<String>,
    cursor: usize,
}
impl MicrophonePicker {
    pub fn new(model: Entity<AppModel>, cx: &mut Context<Self>) -> Self {
        cx.observe(&model, |_, _, cx| cx.notify()).detach();
        let catalog = model.read(cx).configuration_is_preview.then(fixture);
        Self {
            model,
            focus: cx.focus_handle(),
            catalog,
            expanded: false,
            loading: false,
            error: None,
            cursor: 0,
        }
    }
    fn selected_cursor(&self, cx: &Context<Self>) -> usize {
        let saved = self.model.read(cx).core.dictation.input_device.as_ref();
        choices(self.catalog.as_ref())
            .iter()
            .position(|c| c.selectable && c.name.as_ref() == saved)
            .unwrap_or(0)
    }

    fn open(&mut self, cx: &mut Context<Self>) {
        self.expanded = true;
        self.cursor = self.selected_cursor(cx);
        if self.catalog.is_none() {
            self.refresh(cx);
        }
        cx.notify();
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        if self.loading {
            return;
        }
        self.error = None;
        if self.model.read(cx).configuration_is_preview {
            self.catalog = Some(fixture());
            self.cursor = self.selected_cursor(cx);
            cx.notify();
            return;
        }
        self.loading = true;
        cx.notify();
        let task = cx
            .background_executor()
            .spawn(async { ears::audio::input_devices() });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ =
                this.update(cx, |this, cx| {
                    this.loading = false;
                    match result {
                        Ok(catalog) => this.catalog = Some(catalog),
                        Err(_) => this.error = Some(
                            "Could not list microphones. Check device connections, then Refresh."
                                .into(),
                        ),
                    }
                    this.cursor = this.selected_cursor(cx);
                    cx.notify();
                });
        })
        .detach();
    }
    fn choose(&mut self, index: usize, cx: &mut Context<Self>) {
        let options = choices(self.catalog.as_ref());
        let Some(choice) = options.get(index).filter(|c| c.selectable) else {
            return;
        };
        let name = choice.name.clone();
        if let Err(error) = self.model.update(cx, |m, cx| apply(m, name, cx)) {
            self.error = Some(format!("Input not changed: {error:#}"));
            cx.notify();
            return;
        }
        self.error = None;
        self.expanded = false;
        cx.notify();
    }
    fn key(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        let options = choices(self.catalog.as_ref());
        match key {
            "escape" => self.expanded = false,
            "r" => self.refresh(cx),
            "enter" | "space" if self.expanded => self.choose(self.cursor, cx),
            "enter" | "space" | "down" if !self.expanded => {
                self.open(cx);
            }
            "up" | "down" if self.expanded => {
                let step = if key == "down" { 1 } else { options.len() - 1 };
                for _ in 0..options.len() {
                    self.cursor = (self.cursor + step) % options.len();
                    if options[self.cursor].selectable {
                        break;
                    }
                }
            }
            _ => return false,
        }
        cx.notify();
        true
    }
}
impl Render for MicrophonePicker {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme: Theme = theme(cx).clone();
        let saved = self.model.read(cx).core.dictation.input_device.clone();
        let preview = self.model.read(cx).configuration_is_preview;
        let (pending, active, input_error, running) = {
            let model = self.model.read(cx);
            (
                model.input_pending.clone(),
                model.input_active.clone(),
                model.input_error.clone(),
                model.engine_running(),
            )
        };
        let selected = pending
            .clone()
            .unwrap_or(saved.clone())
            .unwrap_or_else(|| "System default".into());
        let focus_color = theme.roles.primary;
        let mut control = div()
            .id("microphone-picker")
            .track_focus(&self.focus)
            .tab_index(0)
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .gap(theme.space(2.0))
            .border_1()
            .border_color(Theme::alpha(focus_color, 0.0))
            .rounded(theme.radius.sm())
            .focus(move |style| style.border_color(focus_color))
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                if this.key(&event.keystroke.key, cx) {
                    cx.stop_propagation();
                }
            }))
            .child(
                div()
                    .flex()
                    .w_full()
                    .min_w_0()
                    .gap(theme.space(2.0))
                    .child(
                        button(
                            "mic-open",
                            SharedString::from(selected),
                            ButtonKind::Secondary,
                            &theme,
                        )
                        .flex_1()
                        .min_w_0()
                        .overflow_hidden()
                        .whitespace_nowrap()
                        .on_click(cx.listener(|this, _, window, cx| {
                            window.focus(&this.focus);
                            if this.expanded {
                                this.expanded = false;
                            } else {
                                this.open(cx);
                            }
                            cx.notify();
                        })),
                    )
                    .child(
                        button(
                            "mic-refresh",
                            if self.loading { "Wait" } else { "Refresh" },
                            ButtonKind::Secondary,
                            &theme,
                        )
                        .w(px(64.0))
                        .flex_shrink_0()
                        .on_click(cx.listener(|this, _, _, cx| this.refresh(cx))),
                    ),
            );
        if self.expanded {
            let mut list = div()
                .id("microphone-options")
                .w_full()
                .max_h(px(160.0))
                .overflow_y_scroll()
                .flex()
                .flex_col()
                .gap(theme.space(1.0));
            for (index, choice) in choices(self.catalog.as_ref()).into_iter().enumerate() {
                let active = choice.name == saved && choice.selectable;
                let hovered = self.cursor == index;
                let kind = if active || hovered {
                    ButtonKind::Secondary
                } else {
                    ButtonKind::Ghost
                };
                list = list.child(
                    button(
                        ("microphone-choice", index),
                        SharedString::from(choice.label),
                        kind,
                        &theme,
                    )
                    .w_full()
                    .min_w_0()
                    .overflow_hidden()
                    .when(hovered, |b| b.border_color(theme.roles.primary))
                    .when(!choice.selectable, |b| b.opacity(0.45))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        window.focus(&this.focus);
                        this.choose(index, cx);
                    })),
                )
            }
            control = control.child(list);
            if let Some(name) = self.catalog.as_ref().and_then(|c| c.default_name.as_ref()) {
                control = control.child(hint(format!("System default: {name}"), &theme));
            }
        }
        if let Some(name) = &saved {
            if self.catalog.as_ref().is_some_and(|c| {
                !choices(Some(c))
                    .iter()
                    .any(|v| v.selectable && v.name.as_ref() == Some(name))
            }) {
                control=control.child(hint("Saved microphone is unavailable or its name is ambiguous. Choose another input or System default.",&theme));
            }
        }
        if self.catalog.as_ref().is_some_and(|c| c.devices.is_empty()) {
            control = control.child(hint(
                "No input devices found. Connect a microphone, then Refresh.",
                &theme,
            ));
        }
        if let Some(error) = &self.error {
            control = control.child(hint(error.clone(), &theme).text_color(theme.roles.danger));
        }
        if let Some(error) = input_error {
            control = control.child(hint(error, &theme).text_color(theme.roles.danger));
        } else if pending.is_some() {
            control = control.child(hint("Switching…", &theme));
        } else if let (true, Some(active)) = (running, active) {
            control = control.child(hint(format!("In use: {active}"), &theme));
        }
        if preview {
            control = control.child(hint(
                "Preview only; no device access or settings write.",
                &theme,
            ));
        }
        section("Input", &theme).child(row(
            "Microphone",
            Some("Switching applies immediately. Refresh never records.".into()),
            control,
            &theme,
        ))
    }
}

#[cfg(test)]
#[path = "microphone_tests.rs"]
mod tests;
