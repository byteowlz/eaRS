//! Overlay, Dictionaries and Profiles pages on the same 12-slot grid as Dictation.
//! Every page reuses the top band; only the two modules change with the tab.
use super::*;

/// Visible rows per page of dictionary entries / profiles.
pub(in crate::settings) const DICT_ROWS: usize = 10;
pub(in crate::settings) const PROFILE_ROWS: usize = 9;

impl Column<'_> {
    /// A bank of latching keys with lamps spanning the column.
    fn keys(&mut self, keys: &[(&str, &str, &str)]) {
        let y = self.next_row();
        let (full, module) = (self.full, self.module);
        for (span, (id, label, help)) in full.tracks(keys.len().max(3), GAP).into_iter().zip(keys) {
            self.b.add(
                id,
                module,
                "toggle",
                label,
                help,
                span,
                y,
                ROW,
                json!(false),
            );
        }
    }
    /// Label, then an editable text display across the control column.
    fn text(&mut self, id: &str, label: &str, help: &str) {
        let y = self.next_row();
        self.legend(id, label, help, y);
        let (control, module) = (self.control, self.module);
        self.b
            .add(id, module, "text", label, help, control, y, ROW, json!(""));
    }
    /// Label, then a read-only display across the control column.
    fn result(&mut self, id: &str, label: &str, help: &str) {
        let y = self.next_row();
        self.legend(id, label, help, y);
        let (control, module) = (self.control, self.module);
        self.b.add(
            id,
            module,
            "readout",
            label,
            help,
            control,
            y,
            ROW,
            json!(""),
        );
    }
    fn wide_readout(&mut self, id: &str, help: &str) {
        let y = self.next_row();
        let (full, module) = (self.full, self.module);
        self.b
            .add(id, module, "readout", help, help, full, y, ROW, json!(""));
    }
    /// A list entry: wide display plus a key at the column edge.
    fn list_row(&mut self, id: &str, key: &str, key_help: &str) {
        let y = self.next_row();
        let (full, module) = (self.full, self.module);
        let display = Span {
            right: full.right - SIDE_KEY - INNER,
            ..full
        };
        self.b
            .add(id, module, "readout", "", "", display, y, ROW, json!(""));
        self.b.add(
            &format!("{id}.remove"),
            module,
            "button",
            key,
            key_help,
            full.last(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
    }
    fn wide_toggle(&mut self, id: &str, help: &str) {
        let y = self.next_row();
        let (full, module) = (self.full, self.module);
        self.b
            .add(id, module, "toggle", "", help, full, y, ROW, json!(false));
    }
    /// [◀] page display [▶] across the column.
    fn pager(&mut self, id: &str) {
        let y = self.next_row();
        let (full, module) = (self.full, self.module);
        self.b.add(
            &format!("{id}.prev"),
            module,
            "button",
            "◀",
            "Previous page",
            full.first(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
        self.b.add(
            id,
            module,
            "readout",
            "Page",
            "Page",
            full.inset(SIDE_KEY + INNER),
            y,
            ROW,
            json!(""),
        );
        self.b.add(
            &format!("{id}.next"),
            module,
            "button",
            "▶",
            "Next page",
            full.last(SIDE_KEY),
            y,
            ROW,
            Json::Null,
        );
    }
    /// Momentary keys sharing one row across the control column or the full column.
    fn actions(&mut self, keys: &[(&str, &str, &str)], wide: bool) {
        let y = self.next_row();
        let (span, module) = (if wide { self.full } else { self.control }, self.module);
        for (s, (id, label, help)) in span.tracks(keys.len(), GAP).into_iter().zip(keys) {
            self.b
                .add(id, module, "button", label, help, s, y, ROW, Json::Null);
        }
    }
}

pub(super) fn overlay(b: &mut Builder, left: Span, right: Span, top: f64) {
    b.page = "overlay";
    {
        let mut c = Column::new(b, "ears.left", left, top);
        c.heading("ov.head.island", "ISLAND");
        c.keys(&[
            (
                "ov.show",
                "Show",
                "Show the island and optional recording notch.",
            ),
            (
                "ov.dot",
                "Dormant dot",
                "Keep a tiny dot visible while paused.",
            ),
            (
                "ov.hug",
                "Hug notch",
                "macOS: hug the camera notch like a dynamic island.",
            ),
        ]);
        c.keys(&[
            (
                "ov.rec",
                "Rec notch",
                "macOS: recordable notch on built-in and external displays.",
            ),
            (
                "ov.through",
                "Click-thru",
                "Let clicks pass through the island.",
            ),
            (
                "ov.details",
                "Details",
                "Show a second row when the notched island is active.",
            ),
        ]);
        c.keys(&[
            ("ov.animate", "Animate", "Animate the island and indicator."),
            (
                "ov.always",
                "Always",
                "Animate even when the system asks for reduced motion.",
            ),
        ]);
        c.stepper(
            "ov.anchor",
            "Anchor",
            "Screen edge the island hugs; drag the island to fine-tune.",
            ("◀", "▶"),
        );
        c.stepper(
            "ov.indicator",
            "Indicator",
            "Indicator drawn in the island.",
            ("◀", "▶"),
        );
        c.stepper(
            "ov.shape",
            "Notch motion",
            "How the shape widens on a notched Mac.",
            ("◀", "▶"),
        );
        c.stepper(
            "ov.content",
            "Notch content",
            "How content appears after the notch widens.",
            ("◀", "▶"),
        );
        c.heading("ov.head.word", "WORD PILL");
        c.stepper(
            "ov.wordwidth",
            "Width",
            "Floating word pill width. Word mode only.",
            ("−", "+"),
        );
        c.stepper(
            "ov.wordsize",
            "Size",
            "Maximum word size. Word mode only.",
            ("−", "+"),
        );
        c.switch(
            "ov.wordfade",
            "Quiet fade",
            "Fade the word during real silence. Word mode only.",
        );
    }
    {
        let mut c = Column::new(b, "ears.right", right, top);
        c.heading("ov.head.transcript", "TRANSCRIPT");
        c.stepper(
            "ov.pill",
            "Pill",
            "Pill look. Hardware notches stay black.",
            ("◀", "▶"),
        );
        c.stepper(
            "ov.top",
            "Top text",
            "Phrase, centred word, indicator only, or hidden.",
            ("◀", "▶"),
        );
        c.stepper(
            "ov.preview",
            "Preview",
            "Passive full-text preview; never an insertion target.",
            ("◀", "▶"),
        );
        c.stepper(
            "ov.font",
            "Typeface",
            "Overlay text font. Bundled; no download.",
            ("◀", "▶"),
        );
        c.heading("ov.head.look", "APPEARANCE");
        c.stepper(
            "ov.scheme",
            "Scheme",
            "Auto follows the system.",
            ("◀", "▶"),
        );
        c.stepper(
            "ov.radius",
            "Corner radius",
            "0 is sharp everywhere.",
            ("−", "+"),
        );
        c.stepper("ov.opacity", "Opacity", "Pill opacity.", ("−", "+"));
        c.stepper(
            "ov.margin",
            "Margin",
            "Distance from the screen edge.",
            ("−", "+"),
        );
        c.stepper(
            "ov.linger",
            "Linger",
            "How long the island stays after dictation.",
            ("−", "+"),
        );
    }
}

pub(super) fn dictionaries(b: &mut Builder, left: Span, right: Span, top: f64) {
    b.page = "dictionaries";
    {
        let mut c = Column::new(b, "ears.left", left, top);
        c.heading("dict.head.try", "TRY IT");
        c.text(
            "dict.phrase",
            "Phrase",
            "Type text with a trigger to see the replacement live.",
        );
        c.result("dict.result", "Result", "The phrase after replacement.");
        c.heading("dict.head.add", "ADD ENTRY");
        c.text("dict.write", "Write", "The text eaRS should write.");
        c.text(
            "dict.heard",
            "Heard as",
            "What you say or what is misheard; comma separated.",
        );
        c.actions(
            &[("dict.add", "Add entry", "Add the entry to the dictionary")],
            false,
        );
        c.heading("dict.head.file", "FILE");
        c.wide_readout("dict.file", "Dictionary file");
    }
    {
        let mut c = Column::new(b, "ears.right", right, top);
        c.heading("dict.head.entries", "ENTRIES");
        for i in 0..DICT_ROWS {
            c.list_row(&format!("dict.entry.{i}"), "×", "Remove this entry");
        }
        c.pager("dict.page");
    }
}

pub(super) fn profiles(b: &mut Builder, left: Span, right: Span, top: f64) {
    b.page = "profiles";
    {
        let mut c = Column::new(b, "ears.left", left, top);
        c.heading("prof.head.list", "PROFILES");
        for i in 0..PROFILE_ROWS {
            c.wide_toggle(&format!("prof.item.{i}"), "Edit this profile");
        }
        c.pager("prof.page");
        c.actions(&[("prof.new", "New profile", "Start a new profile")], true);
    }
    {
        let mut c = Column::new(b, "ears.right", right, top);
        c.heading("prof.head.edit", "PROFILE");
        c.text("prof.name", "Name", "Unique profile name.");
        c.stepper(
            "prof.priority",
            "Priority",
            "Higher wins when several match.",
            ("−", "+"),
        );
        c.text(
            "prof.bundles",
            "Bundle ids",
            "macOS bundle identifiers, comma separated.",
        );
        c.text(
            "prof.apps",
            "App names",
            "Binary or application names, comma separated.",
        );
        c.text(
            "prof.title",
            "Window title",
            "Regex against the focused window title (optional).",
        );
        c.text(
            "prof.dicts",
            "Dictionaries",
            "Extra dictionary .toml paths, comma separated.",
        );
        c.stepper(
            "prof.lang",
            "Language",
            "Inherit uses the session language.",
            ("◀", "▶"),
        );
        c.stepper(
            "prof.insert",
            "Insert as",
            "Inherit uses the session setting.",
            ("◀", "▶"),
        );
        c.actions(
            &[
                ("prof.save", "Save", "Save this profile"),
                ("prof.delete", "Delete", "Delete this profile"),
                ("prof.revert", "Revert", "Discard unsaved changes"),
            ],
            true,
        );
    }
}
