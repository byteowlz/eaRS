//! Small shared components rendered with the design-system roles.
//!
//! Every colour here is a role (never a literal), every radius comes from the
//! dial and every gap from the spacing unit, so swapping the scheme or
//! turning the dial re-skins the whole UI.

pub mod text_input;

use crate::theme::Theme;
use gpui::App;
use gpui::ClickEvent;
use gpui::Div;
use gpui::ElementId;
use gpui::Hsla;
use gpui::InteractiveElement;
use gpui::IntoElement;
use gpui::ParentElement;
use gpui::SharedString;
use gpui::Stateful;
use gpui::StatefulInteractiveElement;
use gpui::Styled;
use gpui::Window;
use gpui::div;
use gpui::prelude::FluentBuilder;
use gpui::px;

pub const FONT_SM: f32 = 12.0;
pub const FONT_MD: f32 = 13.0;
pub const FONT_LG: f32 = 15.0;
pub const FONT_XL: f32 = 18.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    Primary,
    Secondary,
    Ghost,
    Danger,
}

/// A clickable button. Attach `.on_click(...)` on the returned element.
pub fn button(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    kind: ButtonKind,
    theme: &Theme,
) -> Stateful<Div> {
    let r = &theme.roles;
    let (bg, fg, hover, border) = match kind {
        ButtonKind::Primary => (
            r.primary,
            r.primary_foreground,
            Theme::alpha(r.primary, 0.85),
            r.primary,
        ),
        ButtonKind::Secondary => (
            r.secondary,
            r.secondary_foreground,
            Theme::alpha(r.foreground, 0.12),
            r.border,
        ),
        ButtonKind::Ghost => (
            Theme::alpha(r.foreground, 0.0),
            r.foreground,
            Theme::alpha(r.foreground, 0.08),
            Theme::alpha(r.foreground, 0.0),
        ),
        ButtonKind::Danger => (
            Theme::alpha(r.danger, 0.15),
            r.danger,
            Theme::alpha(r.danger, 0.28),
            Theme::alpha(r.danger, 0.4),
        ),
    };
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_center()
        .h(theme.space(7.0))
        .px(theme.space(3.0))
        .rounded(theme.radius.md())
        .border_1()
        .border_color(border)
        .bg(bg)
        .text_color(fg)
        .text_size(px(FONT_MD))
        .font_family(theme.font_sans.clone())
        .cursor_pointer()
        .hover(move |s| s.bg(hover))
        .child(label.into())
}

/// A small pill-shaped toggle switch.
pub fn toggle(
    id: impl Into<ElementId>,
    on: bool,
    theme: &Theme,
    on_change: impl Fn(bool, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    let r = &theme.roles;
    let track = if on {
        r.primary
    } else {
        Theme::alpha(r.foreground, 0.2)
    };
    let knob = if on {
        r.primary_foreground
    } else {
        r.background
    };
    div()
        .id(id)
        .relative()
        .w(px(36.0))
        .h(px(20.0))
        .rounded_full()
        .bg(track)
        .cursor_pointer()
        .on_click(move |_: &ClickEvent, window, cx| on_change(!on, window, cx))
        .child(
            div()
                .absolute()
                .top(px(2.0))
                .left(if on { px(18.0) } else { px(2.0) })
                .w(px(16.0))
                .h(px(16.0))
                .rounded_full()
                .bg(knob),
        )
}

/// Segmented control: one option highlighted.
pub fn segmented<T: Copy + PartialEq + 'static>(
    id: impl Into<ElementId>,
    options: Vec<(T, SharedString)>,
    selected: T,
    theme: &Theme,
    on_select: impl Fn(T, &mut Window, &mut App) + Clone + 'static,
) -> impl IntoElement {
    let r = &theme.roles;
    let mut container = div()
        .id(id)
        .flex()
        .items_center()
        .p(px(2.0))
        .gap(px(2.0))
        .rounded(theme.radius.md())
        .bg(r.surface_sunken)
        .border_1()
        .border_color(r.border);
    let sm = theme.radius.sm();
    let (fg, muted, surface, hover) = (
        r.foreground,
        r.muted_foreground,
        r.surface,
        Theme::alpha(r.foreground, 0.06),
    );
    for (index, (value, label)) in options.into_iter().enumerate() {
        let active = value == selected;
        let on_select = on_select.clone();
        container = container.child(
            div()
                .id(index)
                .px(theme.space(2.5))
                .py(px(3.0))
                .rounded(sm)
                .text_size(px(FONT_SM))
                .text_color(if active { fg } else { muted })
                .when(active, |s| s.bg(surface))
                .when(!active, |s| s.hover(move |s| s.bg(hover)))
                .cursor_pointer()
                .on_click(move |_: &ClickEvent, window, cx| on_select(value, window, cx))
                .child(label),
        );
    }
    container
}

/// Section header for settings panes.
pub fn section(title: impl Into<SharedString>, theme: &Theme) -> Div {
    div().flex().flex_col().gap(theme.space(3.0)).child(
        div()
            .text_size(px(FONT_LG))
            .font_weight(gpui::FontWeight::SEMIBOLD)
            .text_color(theme.roles.foreground)
            .child(title.into()),
    )
}

/// Form row: label + hint on the left, control on the right.
pub fn row(
    label: impl Into<SharedString>,
    hint: Option<SharedString>,
    control: impl IntoElement,
    theme: &Theme,
) -> Div {
    let mut text = div().flex().flex_col().gap(px(2.0)).child(
        div()
            .text_size(px(FONT_MD))
            .text_color(theme.roles.foreground)
            .child(label.into()),
    );
    if let Some(hint) = hint {
        text = text.child(
            div()
                .text_size(px(FONT_SM))
                .text_color(theme.roles.muted_foreground)
                .child(hint),
        );
    }
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap(theme.space(4.0))
        .py(theme.space(2.0))
        .child(text)
        .child(control)
}

pub fn label(text: impl Into<SharedString>, theme: &Theme) -> Div {
    div()
        .text_size(px(FONT_MD))
        .text_color(theme.roles.foreground)
        .child(text.into())
}

pub fn hint(text: impl Into<SharedString>, theme: &Theme) -> Div {
    div()
        .text_size(px(FONT_SM))
        .text_color(theme.roles.muted_foreground)
        .child(text.into())
}

/// Coloured status badge.
pub fn badge(text: impl Into<SharedString>, color: Hsla, theme: &Theme) -> Div {
    div()
        .px(theme.space(2.0))
        .py(px(1.0))
        .rounded(theme.radius.sm())
        .bg(Theme::alpha(color, 0.16))
        .text_color(color)
        .text_size(px(FONT_SM))
        .child(text.into())
}

/// Card surface (`surface` role, `border` role, `lg` radius).
pub fn card(theme: &Theme) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(theme.space(2.0))
        .p(theme.space(3.0))
        .rounded(theme.radius.lg())
        .bg(theme.roles.surface)
        .border_1()
        .border_color(theme.roles.border)
}
