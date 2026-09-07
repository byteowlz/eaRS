//! The settings window: dictation, overlay, dictionaries and context
//! profiles. All edits write through [`AppModel`] (persisted to `ui.toml` or
//! `profiles.toml`) so the headless engine and the tray see the same state
//! immediately.

use crate::app::AppModel;
use crate::config::OverlayAnchor;
use crate::runtime;
use crate::theme::Theme;
use crate::ui::ButtonKind;
use crate::ui::badge;
use crate::ui::button;
use crate::ui::card;
use crate::ui::hint;
use crate::ui::label;
use crate::ui::row;
use crate::ui::section;
use crate::ui::segmented;
use crate::ui::text_input::{TextInput, TextInputEvent, init};
use crate::ui::toggle;
use ears::config::DictationHotkeyMode;
use ears::dictation::ConnectionState;
use ears::dictation::DictationCommand;
use ears::dictation::InsertionMode;
use ears::profiles::ContextProfile;
use ears::replacement::ReplacementDictionary;
use ears::replacement::ReplacementEngine;
use ears::replacement::dictionary_paths;
use gpui::AnyElement;
use gpui::App;
use gpui::AppContext;
use gpui::Context;
use gpui::Entity;
use gpui::FocusHandle;
use gpui::Focusable;
use gpui::FontWeight;
use gpui::InteractiveElement;
use gpui::IntoElement;
use gpui::ParentElement;
use gpui::Render;
use gpui::SharedString;
use gpui::StatefulInteractiveElement;
use gpui::Styled;
use gpui::Window;
use gpui::div;
use gpui::prelude::FluentBuilder;
use gpui::px;

const INPUT_W: f32 = 300.0;
const NUM_W: f32 = 110.0;
const LANGUAGES: [&str; 6] = ["en", "de", "fr", "es", "it", "ja"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Dictation,
    Overlay,
    Dictionaries,
    Profiles,
}

impl Tab {
    const ALL: [Tab; 4] = [
        Tab::Dictation,
        Tab::Overlay,
        Tab::Dictionaries,
        Tab::Profiles,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Dictation => "Dictation",
            Tab::Overlay => "Overlay",
            Tab::Dictionaries => "Dictionaries",
            Tab::Profiles => "Profiles",
        }
    }
}

/// Draft state for the profile editor (editing an existing profile or
/// creating a new one).
struct ProfileDraft {
    original: Option<String>,
    name: Entity<TextInput>,
    priority: Entity<TextInput>,
    bundle_id: Entity<TextInput>,
    binary: Entity<TextInput>,
    window_title: Entity<TextInput>,
    dictionaries: Entity<TextInput>,
    language: Entity<TextInput>,
    insertion_mode: Option<InsertionMode>,
}

pub struct SettingsView {
    model: Entity<AppModel>,
    focus_handle: FocusHandle,
    tab: Tab,
    server: Entity<TextInput>,
    language: Option<String>,
    test_input: Entity<TextInput>,
    test_output: String,
    dict_replacement: Entity<TextInput>,
    dict_phrases: Entity<TextInput>,
    margin: Entity<TextInput>,
    linger: Entity<TextInput>,
    opacity: Entity<TextInput>,
    radius: Entity<TextInput>,
    draft: Option<ProfileDraft>,
}

fn input(
    cx: &mut Context<SettingsView>,
    placeholder: &str,
    text: &str,
    mono: bool,
) -> Entity<TextInput> {
    let placeholder = placeholder.to_string();
    let text = text.to_string();
    cx.new(move |cx| {
        let mut field = TextInput::new(cx, placeholder);
        if !text.is_empty() {
            field = field.with_text(text);
        }
        if mono {
            field = field.mono();
        }
        field
    })
}

fn split_csv(text: &str) -> Vec<String> {
    text.split(',')
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

fn join_csv(items: &[String]) -> String {
    items.join(", ")
}

impl SettingsView {
    pub fn new(model: Entity<AppModel>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        init(cx);
        let (server_value, language, overlay, radius) = model.read_with(cx, |m, _| {
            (
                m.ui.dictation.server.clone(),
                m.language.clone(),
                m.ui.overlay.clone(),
                m.ui.theme.radius,
            )
        });
        let server = input(cx, "ws://host:port or alias", &server_value, true);
        let test_input = input(cx, "Type a phrase containing a trigger…", "", false);
        let dict_replacement = input(cx, "Canonical text", "", false);
        let dict_phrases = input(cx, "Observed phrase(s), comma separated", "", false);
        let margin = input(cx, "12", &format!("{}", overlay.margin), true);
        let linger = input(cx, "900", &format!("{}", overlay.linger_ms), true);
        let opacity = input(cx, "0.96", &format!("{:.2}", overlay.opacity), true);
        let radius_input = input(cx, "8", &format!("{}", radius), true);

        // Re-render when the engine pushes events (connection badge, profile).
        cx.observe(&model, |_, _, cx| cx.notify()).detach();

        // Live dictionary test box.
        cx.subscribe(&test_input, |this, _, _: &TextInputEvent, cx| {
            let text = this.test_input.read(cx).text().to_string();
            this.test_output = run_test(cx, &text);
            cx.notify();
        })
        .detach();

        // Numeric overlay fields apply on every valid edit.
        let numeric_model = model.clone();
        cx.subscribe(&margin, move |this, _, _: &TextInputEvent, cx| {
            let raw = this.margin.read(cx).text().to_string();
            if let Ok(value) = raw.trim().parse::<f32>() {
                numeric_model.update(cx, |m, cx| {
                    m.update_ui(cx, |ui| ui.overlay.margin = value.clamp(0.0, 400.0));
                });
            }
        })
        .detach();
        let numeric_model = model.clone();
        cx.subscribe(&linger, move |this, _, _: &TextInputEvent, cx| {
            let raw = this.linger.read(cx).text().to_string();
            if let Ok(value) = raw.trim().parse::<u64>() {
                numeric_model.update(cx, |m, cx| {
                    m.update_ui(cx, |ui| ui.overlay.linger_ms = value.min(60_000));
                });
            }
        })
        .detach();
        let numeric_model = model.clone();
        cx.subscribe(&opacity, move |this, _, _: &TextInputEvent, cx| {
            let raw = this.opacity.read(cx).text().to_string();
            if let Ok(value) = raw.trim().parse::<f32>() {
                numeric_model.update(cx, |m, cx| {
                    m.update_ui(cx, |ui| ui.overlay.opacity = value.clamp(0.2, 1.0));
                });
            }
        })
        .detach();
        let numeric_model = model.clone();
        cx.subscribe(&radius_input, move |this, _, _: &TextInputEvent, cx| {
            let raw = this.radius.read(cx).text().to_string();
            if let Ok(value) = raw.trim().parse::<f32>() {
                numeric_model.update(cx, |m, cx| {
                    m.update_ui(cx, |ui| ui.theme.radius = value.clamp(0.0, 32.0));
                });
                runtime::apply_theme(cx, None);
            }
        })
        .detach();

        Self {
            model,
            focus_handle: cx.focus_handle(),
            tab: Tab::Dictation,
            server,
            language,
            test_input,
            test_output: String::new(),
            dict_replacement,
            dict_phrases,
            margin,
            linger,
            opacity,
            radius: radius_input,
            draft: None,
        }
    }

    // ---- actions ---------------------------------------------------------

    fn apply_server(&mut self, cx: &mut Context<Self>) {
        let value = self.server.read(cx).text().trim().to_string();
        self.model.update(cx, |m, cx| {
            m.update_ui(cx, |ui| {
                ui.dictation.server = value.clone();
            });
        });
        runtime::restart_engine(cx);
    }

    fn start_profile_draft(&mut self, existing: Option<ContextProfile>, cx: &mut Context<Self>) {
        let (name, priority, bundle_id, binary, window_title, dictionaries, language, mode) =
            match &existing {
                Some(profile) => (
                    profile.name.clone(),
                    profile.priority.to_string(),
                    join_csv(&profile.matcher.bundle_id),
                    join_csv(&profile.matcher.binary),
                    profile.matcher.window_title.clone().unwrap_or_default(),
                    join_csv(&profile.dictionaries),
                    profile.language.clone().unwrap_or_default(),
                    profile.insertion_mode,
                ),
                None => (
                    String::new(),
                    "0".to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    None,
                ),
            };
        self.draft = Some(ProfileDraft {
            original: existing.as_ref().map(|p| p.name.clone()),
            name: input(cx, "Unique name", &name, false),
            priority: input(cx, "0", &priority, true),
            bundle_id: input(cx, "com.apple.Terminal, …", &bundle_id, true),
            binary: input(cx, "Code, firefox, …", &binary, false),
            window_title: input(cx, "Regex (optional)", &window_title, false),
            dictionaries: input(cx, "/path/to.toml, …", &dictionaries, true),
            language: input(cx, "en (empty = inherit)", &language, false),
            insertion_mode: mode,
        });
        cx.notify();
    }

    fn save_profile_draft(&mut self, cx: &mut Context<Self>) {
        let Some(draft) = &self.draft else {
            return;
        };
        let name = draft.name.read(cx).text().trim().to_string();
        if name.is_empty() {
            return;
        }
        let profile = ContextProfile {
            name: name.clone(),
            priority: draft.priority.read(cx).text().trim().parse().unwrap_or(0),
            matcher: ears::profiles::ProfileMatch {
                bundle_id: split_csv(draft.bundle_id.read(cx).text()),
                binary: split_csv(draft.binary.read(cx).text()),
                window_title: {
                    let title = draft.window_title.read(cx).text().trim().to_string();
                    (!title.is_empty()).then_some(title)
                },
            },
            dictionaries: split_csv(draft.dictionaries.read(cx).text()),
            language: {
                let lang = draft.language.read(cx).text().trim().to_string();
                (!lang.is_empty()).then_some(lang)
            },
            insertion_mode: draft.insertion_mode,
        };
        let rename_from = draft.original.clone().filter(|original| *original != name);
        self.model.update(cx, |m, cx| {
            if let Some(original) = &rename_from {
                m.profiles.remove(original);
            }
            m.profiles.upsert(profile);
            if let Err(err) = m.profiles.save() {
                eprintln!("ears-ui: cannot save profiles.toml: {err:#}");
            }
            m.reload(cx);
        });
        self.draft = None;
        cx.notify();
    }

    fn delete_profile(&mut self, name: &str, cx: &mut Context<Self>) {
        self.model.update(cx, |m, cx| {
            m.profiles.remove(name);
            if let Err(err) = m.profiles.save() {
                eprintln!("ears-ui: cannot save profiles.toml: {err:#}");
            }
            if m.profile.as_deref() == Some(name) {
                m.set_profile(None, cx);
            }
            m.reload(cx);
        });
        cx.notify();
    }

    fn add_dictionary_entry(&mut self, cx: &mut Context<Self>) {
        let replacement = self.dict_replacement.read(cx).text().trim().to_string();
        let phrases = split_csv(self.dict_phrases.read(cx).text());
        if replacement.is_empty() || phrases.is_empty() {
            return;
        }
        self.model.update(cx, |m, _| {
            if let Some(path) = dictionary_paths(&m.core.replacement).first().cloned() {
                match ReplacementDictionary::load_or_create(&path) {
                    Ok(mut dictionary) => {
                        dictionary.add_entry(replacement, phrases);
                        if let Err(err) = dictionary.save(&path) {
                            eprintln!("ears-ui: cannot save dictionary: {err:#}");
                        }
                    }
                    Err(err) => eprintln!("ears-ui: cannot load dictionary: {err:#}"),
                }
            }
        });
        self.model.update(cx, |m, cx| m.reload(cx));
        self.dict_replacement.update(cx, |f, cx| f.set_text("", cx));
        self.dict_phrases.update(cx, |f, cx| f.set_text("", cx));
        cx.notify();
    }

    fn remove_dictionary_entry(&mut self, replacement: &str, cx: &mut Context<Self>) {
        self.model.update(cx, |m, _| {
            if let Some(path) = dictionary_paths(&m.core.replacement).first().cloned() {
                match ReplacementDictionary::load_or_create(&path) {
                    Ok(mut dictionary) => {
                        dictionary.remove_replacement(replacement);
                        if let Err(err) = dictionary.save(&path) {
                            eprintln!("ears-ui: cannot save dictionary: {err:#}");
                        }
                    }
                    Err(err) => eprintln!("ears-ui: cannot load dictionary: {err:#}"),
                }
            }
        });
        self.model.update(cx, |m, cx| m.reload(cx));
        cx.notify();
    }

    // ---- panes -----------------------------------------------------------

    fn view(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.global::<Theme>().clone();
        let r = &theme.roles;

        let tab_bar = segmented(
            "settings-tab",
            Tab::ALL
                .into_iter()
                .map(|tab| (tab, SharedString::from(tab.label())))
                .collect(),
            self.tab,
            &theme,
            {
                let entity = cx.entity();
                move |tab: Tab, _, cx| {
                    entity.update(cx, |this, cx| {
                        this.tab = tab;
                        cx.notify();
                    });
                }
            },
        );

        let pane = match self.tab {
            Tab::Dictation => self.dictation_pane(&theme, cx),
            Tab::Overlay => self.overlay_pane(&theme, cx),
            Tab::Dictionaries => self.dictionaries_pane(&theme, cx),
            Tab::Profiles => self.profiles_pane(&theme, cx),
        };

        let footer = "Dictation runs headless in ears-dictation / the tray engine; \
                      these settings edit the same config files the CLI uses.";

        div()
            .id("settings-root")
            .size_full()
            .flex()
            .flex_col()
            .bg(r.background)
            .text_color(r.foreground)
            .font_family(theme.font_sans.clone())
            .track_focus(&self.focus_handle)
            .child(
                div()
                    .id("settings-scroll")
                    .flex()
                    .flex_col()
                    .gap(theme.space(4.0))
                    .p(theme.space(5.0))
                    .flex_1()
                    .overflow_y_scroll()
                    .child(
                        div()
                            .text_size(px(crate::ui::FONT_XL))
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("eaRS Settings"),
                    )
                    .child(tab_bar)
                    .child(pane),
            )
            .child(
                div()
                    .px(theme.space(5.0))
                    .py(theme.space(2.0))
                    .border_t_1()
                    .border_color(r.border)
                    .text_size(px(crate::ui::FONT_SM))
                    .text_color(r.muted_foreground)
                    .child(footer),
            )
            .into_any_element()
    }

    fn dictation_pane(&mut self, theme: &Theme, cx: &mut Context<Self>) -> AnyElement {
        let model = self.model.read(cx);
        let r = &theme.roles;

        let (state_text, state_color) = match model.connection {
            ConnectionState::Connected => ("Connected", r.success),
            ConnectionState::Connecting => ("Connecting…", r.warning),
            ConnectionState::Disconnected => ("Disconnected", r.danger),
        };
        let backend_line = match &model.backend {
            Some(backend) => {
                let latency = backend
                    .latency_ms
                    .map(|ms| format!("{ms} ms"))
                    .unwrap_or_else(|| "–".to_string());
                let alias = backend
                    .alias
                    .clone()
                    .unwrap_or_else(|| "custom URL".to_string());
                format!("{} ({alias}) · latency {latency}", backend.url)
            }
            None => "not connected".to_string(),
        };
        let server_value = model.ui.dictation.server.clone();
        let aliases: Vec<String> = {
            let mut names: Vec<String> = model.core.dictation.servers.keys().cloned().collect();
            names.sort();
            names
        };
        let hotkey = model.core.hotkeys.toggle.clone();
        let insertion_mode = model.insertion_mode;
        let hotkey_mode = model.hotkey_mode;
        let escape_cancels = model.ui.dictation.escape_cancels;
        let profiles_enabled = model.ui.dictation.profiles;
        let current_language = model.language.clone();

        let mut pane = div().flex().flex_col().gap(theme.space(5.0));

        // Backend ---------------------------------------------------------
        let mut backend = section("Backend", theme).child(
            div()
                .flex()
                .items_center()
                .gap(theme.space(2.0))
                .child(badge(state_text, state_color, theme))
                .child(
                    div()
                        .text_size(px(crate::ui::FONT_SM))
                        .text_color(r.muted_foreground)
                        .child(SharedString::from(backend_line)),
                ),
        );
        backend = backend.child(row(
            "Server",
            Some("Alias from config.toml or a ws:// URL. Applied by restarting the engine.".into()),
            div()
                .flex()
                .gap(theme.space(2.0))
                .items_center()
                .child(div().w(px(INPUT_W)).child(self.server.clone()))
                .child(
                    button("apply-server", "Apply", ButtonKind::Secondary, theme).on_click({
                        let entity = cx.entity();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.apply_server(cx));
                        }
                    }),
                ),
            theme,
        ));
        if !aliases.is_empty() {
            let mut options: Vec<(usize, SharedString)> = Vec::new();
            let mut selected_alias = 0usize;
            for (index, alias) in aliases.iter().enumerate() {
                if *alias == server_value {
                    selected_alias = index;
                }
                options.push((index, SharedString::from(alias.clone())));
            }
            backend = backend.child(row(
                "Configured servers",
                None,
                segmented("server-alias", options, selected_alias, theme, {
                    let entity = cx.entity();
                    let aliases = aliases.clone();
                    move |index: usize, _, cx| {
                        let alias = aliases[index].clone();
                        entity.update(cx, |this, cx| {
                            this.server.update(cx, |field, cx| {
                                field.set_text(&alias, cx);
                            });
                        });
                    }
                }),
                theme,
            ));
        }
        pane = pane.child(backend);

        // Dictation ---------------------------------------------------------
        let mut behaviour = section("Dictation", theme).child(row(
            "Insertion mode",
            Some("What happens with finished text.".into()),
            segmented(
                "insertion-mode",
                vec![
                    (InsertionMode::InsertAtCursor, "Insert".into()),
                    (InsertionMode::Clipboard, "Copy".into()),
                    (InsertionMode::SendAsPrompt, "Prompt".into()),
                ],
                insertion_mode,
                theme,
                {
                    let entity = cx.entity();
                    move |mode: InsertionMode, _, cx| {
                        entity.update(cx, |this, cx| {
                            this.model
                                .update(cx, |m, cx| m.set_insertion_mode(mode, cx));
                        });
                    }
                },
            ),
            theme,
        ));
        behaviour = behaviour.child(row(
            "Hotkey mode",
            Some(format!("Toggle combo: {hotkey}").into()),
            segmented(
                "hotkey-mode",
                vec![
                    (DictationHotkeyMode::Toggle, "Toggle".into()),
                    (DictationHotkeyMode::PushToTalk, "Push to talk".into()),
                    (DictationHotkeyMode::Hybrid, "Hybrid".into()),
                ],
                hotkey_mode,
                theme,
                {
                    let entity = cx.entity();
                    move |mode: DictationHotkeyMode, _, cx| {
                        entity.update(cx, |this, cx| {
                            this.model.update(cx, |m, cx| m.set_hotkey_mode(mode, cx));
                        });
                    }
                },
            ),
            theme,
        ));
        behaviour = behaviour.child(row(
            "Escape cancels",
            Some("Discard pending words instead of pausing.".into()),
            toggle("escape-cancels", escape_cancels, theme, {
                let entity = cx.entity();
                move |on: bool, _, cx| {
                    entity.update(cx, |this, cx| {
                        this.model.update(cx, |m, cx| {
                            m.update_ui(cx, |ui| ui.dictation.escape_cancels = on);
                            m.send(DictationCommand::SetEscapeCancels(on));
                        });
                    });
                }
            }),
            theme,
        ));
        behaviour = behaviour.child(row(
            "Context profiles",
            Some("Match dictionaries per frontmost application.".into()),
            toggle("profiles-enabled", profiles_enabled, theme, {
                let entity = cx.entity();
                move |on: bool, _, cx| {
                    entity.update(cx, |this, cx| {
                        this.model.update(cx, |m, cx| {
                            m.update_ui(cx, |ui| ui.dictation.profiles = on);
                        });
                    });
                }
            }),
            theme,
        ));

        let mut language_options: Vec<(usize, SharedString)> = vec![(0, "Auto".into())];
        let mut selected_language = 0usize;
        for (index, lang) in LANGUAGES.into_iter().enumerate() {
            if current_language.as_deref() == Some(lang) {
                selected_language = index + 1;
            }
            language_options.push((index + 1, lang.into()));
        }
        behaviour = behaviour.child(row(
            "Language",
            Some("Session transcription language.".into()),
            segmented("language", language_options, selected_language, theme, {
                let entity = cx.entity();
                move |index: usize, _, cx| {
                    let lang = if index == 0 {
                        "auto".to_string()
                    } else {
                        LANGUAGES[index - 1].to_string()
                    };
                    entity.update(cx, |this, cx| {
                        this.language = (index != 0).then(|| lang.clone());
                        this.model.update(cx, |m, cx| m.set_language(lang, cx));
                    });
                }
            }),
            theme,
        ));
        pane = pane.child(behaviour);
        pane.into_any_element()
    }

    fn overlay_pane(&mut self, theme: &Theme, cx: &mut Context<Self>) -> AnyElement {
        let entity = cx.entity();
        let (overlay, scheme) = self
            .model
            .read_with(cx, |m, _| (m.ui.overlay.clone(), m.ui.theme.scheme.clone()));
        let island_hint = if cfg!(target_os = "macos") {
            "macOS: hug the camera notch like a dynamic island."
        } else {
            "macOS only; ignored elsewhere."
        };

        let mut pane = div().flex().flex_col().gap(theme.space(5.0));

        pane = pane.child(
            section("Island", theme)
                .child(row(
                    "Enabled",
                    Some("Show the floating pill while dictating.".into()),
                    toggle("overlay-enabled", overlay.enabled, theme, {
                        let entity = entity.clone();
                        move |on: bool, _, cx| {
                            entity.update(cx, |this, cx| {
                                this.model.update(cx, |m, cx| {
                                    m.update_ui(cx, |ui| ui.overlay.enabled = on);
                                });
                            });
                        }
                    }),
                    theme,
                ))
                .child(row(
                    "Dormant dot",
                    Some("Keep a tiny dot visible while paused.".into()),
                    toggle("dormant-dot", overlay.dormant_dot, theme, {
                        let entity = entity.clone();
                        move |on: bool, _, cx| {
                            entity.update(cx, |this, cx| {
                                this.model.update(cx, |m, cx| {
                                    m.update_ui(cx, |ui| ui.overlay.dormant_dot = on);
                                });
                            });
                        }
                    }),
                    theme,
                ))
                .child(row(
                    "Anchor",
                    Some("Screen edge the island hugs; drag it to fine-tune.".into()),
                    segmented(
                        "anchor",
                        OverlayAnchor::ALL
                            .into_iter()
                            .map(|anchor| (anchor, SharedString::from(anchor.label())))
                            .collect(),
                        overlay.anchor,
                        theme,
                        {
                            let entity = entity.clone();
                            move |anchor: OverlayAnchor, _, cx| {
                                entity.update(cx, |this, cx| {
                                    this.model.update(cx, |m, cx| {
                                        m.update_ui(cx, |ui| ui.overlay.anchor = anchor);
                                    });
                                });
                            }
                        },
                    ),
                    theme,
                ))
                .child(row(
                    "Island mode",
                    Some(island_hint.into()),
                    toggle("island-mode", overlay.island_mode, theme, {
                        let entity = entity.clone();
                        move |on: bool, _, cx| {
                            entity.update(cx, |this, cx| {
                                this.model.update(cx, |m, cx| {
                                    m.update_ui(cx, |ui| ui.overlay.island_mode = on);
                                });
                            });
                        }
                    }),
                    theme,
                ))
                .child(row(
                    "Click through",
                    Some("Let clicks pass through the island.".into()),
                    toggle("click-through", overlay.click_through, theme, {
                        let entity = entity.clone();
                        move |on: bool, _, cx| {
                            entity.update(cx, |this, cx| {
                                this.model.update(cx, |m, cx| {
                                    m.update_ui(cx, |ui| ui.overlay.click_through = on);
                                });
                            });
                        }
                    }),
                    theme,
                ))
                .child(row(
                    "Animation",
                    Some("Morph between states. Off when the OS asks for reduced motion.".into()),
                    toggle("animation", overlay.animation, theme, {
                        let entity = entity.clone();
                        move |on: bool, _, cx| {
                            entity.update(cx, |this, cx| {
                                this.model.update(cx, |m, cx| {
                                    m.update_ui(cx, |ui| ui.overlay.animation = on);
                                });
                            });
                        }
                    }),
                    theme,
                )),
        );

        pane = pane.child(
            section("Appearance", theme)
                .child(row(
                    "Scheme",
                    Some("Auto follows the OS; Dark / Light use the built-in oqto schemes.".into()),
                    segmented(
                        "scheme",
                        vec![
                            ("auto", "Auto".into()),
                            ("oqto-dark", "Dark".into()),
                            ("oqto-light", "Light".into()),
                        ],
                        scheme_scheme(&scheme),
                        theme,
                        {
                            let entity = entity.clone();
                            move |scheme: &'static str, _, cx| {
                                entity.update(cx, |this, cx| {
                                    this.model.update(cx, |m, cx| {
                                        m.update_ui(cx, |ui| ui.theme.scheme = scheme.to_string());
                                    });
                                    runtime::apply_theme(cx, None);
                                });
                            }
                        },
                    ),
                    theme,
                ))
                .child(row(
                    "Corner radius",
                    Some("The radius dial in px; 0 is sharp everywhere.".into()),
                    div().w(px(NUM_W)).child(self.radius.clone()),
                    theme,
                ))
                .child(row(
                    "Opacity",
                    Some("Pill opacity, 0.2 – 1.0.".into()),
                    div().w(px(NUM_W)).child(self.opacity.clone()),
                    theme,
                ))
                .child(row(
                    "Margin",
                    Some("Distance from the screen edge, px.".into()),
                    div().w(px(NUM_W)).child(self.margin.clone()),
                    theme,
                ))
                .child(row(
                    "Linger",
                    Some("How long the island stays after dictation, ms.".into()),
                    div().w(px(NUM_W)).child(self.linger.clone()),
                    theme,
                )),
        );
        pane.into_any_element()
    }

    fn dictionaries_pane(&mut self, theme: &Theme, cx: &mut Context<Self>) -> AnyElement {
        let entries = self.model.read_with(cx, |m, _| {
            dictionary_paths(&m.core.replacement)
                .first()
                .cloned()
                .and_then(|path| {
                    ReplacementDictionary::load(&path)
                        .ok()
                        .map(|dictionary| (dictionary.entries, path))
                })
        });

        let mut pane = div().flex().flex_col().gap(theme.space(5.0));

        pane = pane.child(
            section("Try it", theme).child(row(
                "Test phrase",
                Some("Type text containing a trigger to see the replacement live.".into()),
                div()
                    .flex()
                    .flex_col()
                    .gap(theme.space(1.0))
                    .w(px(INPUT_W + 160.0))
                    .child(self.test_input.clone())
                    .child(hint(self.test_output.clone(), theme)),
                theme,
            )),
        );

        pane = pane.child(
            section("Add entry", theme)
                .child(row(
                    "Replacement",
                    None,
                    div().w(px(INPUT_W)).child(self.dict_replacement.clone()),
                    theme,
                ))
                .child(row(
                    "Observed phrase(s)",
                    Some("Comma separated; phrases may span words.".into()),
                    div().w(px(INPUT_W)).child(self.dict_phrases.clone()),
                    theme,
                ))
                .child(
                    button("add-entry", "Add entry", ButtonKind::Primary, theme).on_click({
                        let entity = cx.entity();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.add_dictionary_entry(cx));
                        }
                    }),
                ),
        );

        let mut list = section("Entries", theme);
        match entries {
            Some((entries, path)) => {
                if entries.is_empty() {
                    list = list.child(hint("No entries yet.", theme));
                }
                for entry in entries {
                    list = list.child(
                        card(theme).child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .flex()
                                        .flex_col()
                                        .child(label(entry.replace.clone(), theme))
                                        .child(hint(entry.phrases.join(", "), theme)),
                                )
                                .child(
                                    button(
                                        gpui::ElementId::Name(
                                            format!("del-{}", entry.replace).into(),
                                        ),
                                        "Remove",
                                        ButtonKind::Danger,
                                        theme,
                                    )
                                    .on_click({
                                        let entity = cx.entity();
                                        let replacement = entry.replace.clone();
                                        move |_, _, cx| {
                                            entity.update(cx, |this, cx| {
                                                this.remove_dictionary_entry(&replacement, cx);
                                            });
                                        }
                                    }),
                                ),
                        ),
                    );
                }
                list = list.child(hint(path.display().to_string(), theme));
            }
            None => {
                list = list.child(hint("Dictionary not readable.", theme));
            }
        }
        pane = pane.child(list);
        pane.into_any_element()
    }

    fn profiles_pane(&mut self, theme: &Theme, cx: &mut Context<Self>) -> AnyElement {
        let (profiles, active, pinned) = self.model.read_with(cx, |m, _| {
            (m.profile_names(), m.profile.clone(), m.profile_pinned)
        });

        let mut pane = div().flex().flex_col().gap(theme.space(5.0));

        let mut list = section("Context profiles", theme).child(hint(
            "Higher priority wins. Rules match the frontmost application; the matched \
             dictionary set, language and insertion mode apply while it is frontmost.",
            theme,
        ));
        for profile in &profiles {
            let summary = self.model.read_with(cx, |m, _| {
                m.profiles
                    .get(&profile)
                    .map(matcher_summary)
                    .unwrap_or_default()
            });
            let is_active = pinned && active.as_deref() == Some(profile.as_str());
            list = list.child(
                card(theme).child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(theme.space(0.5))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(theme.space(2.0))
                                        .child(label(profile.clone(), theme))
                                        .when(is_active, |this| {
                                            this.child(badge("active", theme.roles.success, theme))
                                        }),
                                )
                                .child(hint(summary, theme)),
                        )
                        .child(
                            div()
                                .flex()
                                .gap(theme.space(2.0))
                                .child(
                                    button(
                                        gpui::ElementId::Name(format!("edit-{profile}").into()),
                                        "Edit",
                                        ButtonKind::Secondary,
                                        theme,
                                    )
                                    .on_click({
                                        let entity = cx.entity();
                                        let name = profile.clone();
                                        move |_, _, cx| {
                                            entity.update(cx, |this, cx| {
                                                let existing = this.model.read_with(cx, |m, _| {
                                                    m.profiles.get(&name).cloned()
                                                });
                                                this.start_profile_draft(existing, cx);
                                            });
                                        }
                                    }),
                                )
                                .child(
                                    button(
                                        gpui::ElementId::Name(format!("del-{profile}").into()),
                                        "Delete",
                                        ButtonKind::Danger,
                                        theme,
                                    )
                                    .on_click({
                                        let entity = cx.entity();
                                        let name = profile.clone();
                                        move |_, _, cx| {
                                            entity.update(cx, |this, cx| {
                                                this.delete_profile(&name, cx);
                                            });
                                        }
                                    }),
                                ),
                        ),
                ),
            );
        }
        if profiles.is_empty() {
            list = list.child(hint("No profiles yet.", theme));
        }
        list = list.child(
            button("new-profile", "New profile", ButtonKind::Primary, theme).on_click({
                let entity = cx.entity();
                move |_, _, cx| {
                    entity.update(cx, |this, cx| this.start_profile_draft(None, cx));
                }
            }),
        );
        pane = pane.child(list);

        if let Some(draft) = &self.draft {
            pane = pane.child(self.draft_pane(draft, theme, cx));
        }

        pane.into_any_element()
    }

    fn draft_pane(&self, draft: &ProfileDraft, theme: &Theme, cx: &Context<Self>) -> AnyElement {
        let entity = cx.entity();
        let title = draft
            .original
            .clone()
            .map(|name| format!("Edit: {name}"))
            .unwrap_or_else(|| "New profile".to_string());
        let mut pane = section(title, theme)
            .child(row(
                "Name",
                None,
                div().w(px(INPUT_W)).child(draft.name.clone()),
                theme,
            ))
            .child(row(
                "Priority",
                Some("Higher wins.".into()),
                div().w(px(NUM_W)).child(draft.priority.clone()),
                theme,
            ))
            .child(row(
                "Bundle ids",
                Some("macOS bundle identifiers, comma separated.".into()),
                div().w(px(INPUT_W)).child(draft.bundle_id.clone()),
                theme,
            ))
            .child(row(
                "App names",
                Some("Binary or application names, comma separated.".into()),
                div().w(px(INPUT_W)).child(draft.binary.clone()),
                theme,
            ))
            .child(row(
                "Window title",
                Some("Regex against the focused window title (optional).".into()),
                div().w(px(INPUT_W)).child(draft.window_title.clone()),
                theme,
            ))
            .child(row(
                "Dictionaries",
                Some("Extra dictionary .toml paths while active, comma separated.".into()),
                div().w(px(INPUT_W)).child(draft.dictionaries.clone()),
                theme,
            ))
            .child(row(
                "Language",
                Some("e.g. de; empty inherits the session language.".into()),
                div().w(px(NUM_W)).child(draft.language.clone()),
                theme,
            ))
            .child(row(
                "Insertion mode",
                Some("Empty inherits the session setting.".into()),
                segmented(
                    "draft-mode",
                    vec![
                        (None, "Inherit".into()),
                        (Some(InsertionMode::InsertAtCursor), "Insert".into()),
                        (Some(InsertionMode::Clipboard), "Copy".into()),
                        (Some(InsertionMode::SendAsPrompt), "Prompt".into()),
                    ],
                    draft.insertion_mode,
                    theme,
                    {
                        let entity = entity.clone();
                        move |mode: Option<InsertionMode>, _, cx| {
                            entity.update(cx, |this, cx| {
                                if let Some(draft) = &mut this.draft {
                                    draft.insertion_mode = mode;
                                }
                                cx.notify();
                            });
                        }
                    },
                ),
                theme,
            ))
            .child(
                div()
                    .flex()
                    .gap(theme.space(2.0))
                    .child(
                        button("save-profile", "Save", ButtonKind::Primary, theme).on_click({
                            let entity = entity.clone();
                            move |_, _, cx| {
                                entity.update(cx, |this, cx| this.save_profile_draft(cx));
                            }
                        }),
                    )
                    .child(
                        button("cancel-profile", "Cancel", ButtonKind::Ghost, theme).on_click(
                            move |_, _, cx| {
                                entity.update(cx, |this, cx| {
                                    this.draft = None;
                                    cx.notify();
                                });
                            },
                        ),
                    ),
            );
        pane = card(theme).child(pane);
        pane.into_any_element()
    }
}

fn run_test(cx: &Context<SettingsView>, text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    let config = crate::runtime::model(cx).read_with(cx, |m, _| m.core.replacement.clone());
    match ReplacementEngine::from_config(&config) {
        Ok(engine) => engine.replace(text),
        Err(err) => format!("dictionary error: {err:#}"),
    }
}

fn scheme_scheme(scheme: &str) -> &'static str {
    match scheme {
        "oqto-dark" => "oqto-dark",
        "oqto-light" => "oqto-light",
        _ => "auto",
    }
}

fn matcher_summary(profile: &ContextProfile) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !profile.matcher.bundle_id.is_empty() {
        parts.push(format!("bundle: {}", profile.matcher.bundle_id.join(", ")));
    }
    if !profile.matcher.binary.is_empty() {
        parts.push(format!("apps: {}", profile.matcher.binary.join(", ")));
    }
    if let Some(title) = &profile.matcher.window_title {
        parts.push(format!("title: /{title}/"));
    }
    if !profile.dictionaries.is_empty() {
        parts.push(format!("dicts: {}", profile.dictionaries.len()));
    }
    if let Some(lang) = &profile.language {
        parts.push(format!("lang: {lang}"));
    }
    parts.join(" · ")
}

impl Focusable for SettingsView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for SettingsView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.view(window, cx)
    }
}
