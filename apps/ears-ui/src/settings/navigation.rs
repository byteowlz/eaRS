//! Four incumbent destinations; navigation never participates in pane scrolling.
use super::*;
use gpui::KeyDownEvent;
#[cfg(test)]
use gpui::ScrollHandle;

impl Tab {
    pub(super) fn index(self) -> usize {
        Self::ALL.iter().position(|tab| *tab == self).unwrap_or(0)
    }
    fn key(self, key: &str) -> Option<Self> {
        let index = self.index();
        let count = Self::ALL.len();
        let next = match key {
            "right" => (index + 1) % count,
            "left" => (index + count - 1) % count,
            "home" => 0,
            "end" => count - 1,
            _ => return None,
        };
        Some(Self::ALL[next])
    }
}
impl SettingsView {
    fn select_tab(&mut self, tab: Tab, cx: &mut Context<Self>) {
        self.tab = tab;
        self.model.update(cx, |model, _| model.settings_tab = tab);
        cx.notify();
    }
    pub(super) fn navigation(
        &self,
        theme: &Theme,
        window: &Window,
        cx: &Context<Self>,
    ) -> AnyElement {
        let focused = self.navigation_focus.clone();
        let mut tabs = div()
            .on_children_prepainted(crate::preview_proof::bounds(
                self.model.read(cx).configuration_is_preview,
                "settings-tabs",
                self.tab.index() as u32,
            ))
            .id("settings-tabs")
            .w_full()
            .flex()
            .gap(px(2.0))
            .p(px(2.0))
            .rounded(theme.radius.md())
            .bg(theme.roles.surface_sunken)
            .border_1()
            .border_color(theme.roles.border)
            .track_focus(&focused)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                if !this.navigation_focus.is_focused(window)
                    || event.keystroke.modifiers != gpui::Modifiers::default()
                {
                    return;
                }
                if let Some(tab) = this.tab.key(&event.keystroke.key) {
                    this.select_tab(tab, cx);
                    cx.stop_propagation();
                }
            }))
            .when(focused.is_focused(window), |d| {
                d.border_color(theme.roles.foreground)
            });
        for tab in Tab::ALL {
            let active = tab == self.tab;
            tabs = tabs.child(
                div()
                    .id(tab.index())
                    .flex_1()
                    .min_w_0()
                    .h(theme.space(8.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(theme.radius.sm())
                    .text_size(px(crate::ui::FONT_MD))
                    .text_color(theme.roles.foreground)
                    .when(active, |d| {
                        d.bg(theme.roles.surface).font_weight(FontWeight::SEMIBOLD)
                    })
                    .when(!active, |d| d.hover(|d| d.bg(theme.roles.surface)))
                    .cursor_pointer()
                    .on_click({
                        let entity = cx.entity();
                        move |_, window, cx| {
                            entity.update(cx, |this, cx| {
                                this.navigation_focus.focus(window);
                                this.select_tab(tab, cx);
                            })
                        }
                    })
                    .child(tab.label()),
            );
        }
        tabs.into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn navigation_keys_cover_all_panes_and_do_not_handle_text_editing() {
        assert_eq!(Tab::Dictation.key("left"), Some(Tab::Profiles));
        assert_eq!(Tab::Profiles.key("right"), Some(Tab::Dictation));
        for tab in Tab::ALL {
            assert_eq!(tab.key("home"), Some(Tab::Dictation));
            assert_eq!(tab.key("end"), Some(Tab::Profiles));
            assert_eq!(tab.key("a"), None);
        }
    }
    #[test]
    fn dictionary_action_requires_actual_content() {
        for (replacement, phrases) in [("", "heard"), ("value", " , , "), ("  ", "heard")] {
            assert!(!can_add_entry(replacement, phrases));
        }
        assert!(can_add_entry("eaRS", "ears, years"));
    }

    #[test]
    fn scroll_handles_belong_to_each_pane() {
        let handles: [ScrollHandle; 4] = std::array::from_fn(|_| ScrollHandle::new());
        handles[Tab::Overlay.index()].set_offset(gpui::point(px(0.0), px(-240.0)));
        assert_eq!(handles[Tab::Dictation.index()].offset().y, px(0.0));
        assert_eq!(handles[Tab::Overlay.index()].offset().y, px(-240.0));
    }
}
