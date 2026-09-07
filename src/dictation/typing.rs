//! Text side of the engine: live word buffering, dictionary replacement with
//! hot reload, transcript history and insertion targets.

use super::DictationEvent;
use super::InsertionMode;
use super::Shared;
use super::vlog;
use crate::replacement::ReplacementConfig;
use crate::replacement::ReplacementEngine;
use crate::replacement::TranscriptHistoryConfig;
use crate::replacement::dictionary_paths;
use crate::replacement::transcript_history_dir;
use crate::virtual_keyboard::SpecialKey;
use crate::virtual_keyboard::VirtualKeyboard;
use crate::virtual_keyboard::create_virtual_keyboard;
use anyhow::Context;
use anyhow::Result;
use serde_json::Value;
use serde_json::json;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;

/// Replacement engine that reloads its dictionaries when their files change.
///
/// A context profile may temporarily override the dictionary set; the
/// override is applied on top of the base configuration from `config.toml`.
pub struct ReplacementReloader {
    base: ReplacementConfig,
    config: ReplacementConfig,
    engine: ReplacementEngine,
    stamp: Vec<(PathBuf, Option<SystemTime>)>,
}

impl ReplacementReloader {
    pub fn new(config: ReplacementConfig) -> Self {
        let engine = ReplacementEngine::from_config(&config).unwrap_or_else(|err| {
            eprintln!("warning: failed to load replacement dictionary: {err}");
            ReplacementEngine::empty()
        });
        let stamp = dictionary_stamp(&config);
        Self {
            base: config.clone(),
            config,
            engine,
            stamp,
        }
    }

    /// Use `paths` instead of the configured dictionaries (empty = base set).
    pub fn set_override(&mut self, paths: &[String]) {
        let mut config = self.base.clone();
        if !paths.is_empty() {
            config.enabled = true;
            config.dictionary_paths = paths.to_vec();
        }
        if config.dictionary_paths == self.config.dictionary_paths
            && config.enabled == self.config.enabled
        {
            return;
        }
        self.config = config;
        self.stamp.clear();
        self.reload_if_changed();
    }

    pub fn config(&self) -> &ReplacementConfig {
        &self.config
    }

    pub fn replace(&mut self, text: &str) -> String {
        self.reload_if_changed();
        self.engine.replace(text)
    }

    pub fn ends_with_phrase_prefix(&mut self, text: &str) -> bool {
        self.reload_if_changed();
        self.engine.ends_with_phrase_prefix(text)
    }

    pub fn reload_if_changed(&mut self) {
        let stamp = dictionary_stamp(&self.config);
        if stamp == self.stamp {
            return;
        }
        match ReplacementEngine::from_config(&self.config) {
            Ok(engine) => {
                self.engine = engine;
                self.stamp = stamp;
                vlog!("[dictionary] reloaded replacement dictionary");
            }
            Err(err) => {
                eprintln!("warning: failed to hot-reload replacement dictionary: {err}");
            }
        }
    }
}

fn dictionary_stamp(config: &ReplacementConfig) -> Vec<(PathBuf, Option<SystemTime>)> {
    dictionary_paths(config)
        .into_iter()
        .map(|path| {
            let modified = fs::metadata(&path).and_then(|meta| meta.modified()).ok();
            (path, modified)
        })
        .collect()
}

/// Holds live words briefly so multi-word dictionary phrases can be replaced
/// before anything is typed.
pub struct LiveWordBuffer {
    max_phrase_hold: Duration,
    pub(crate) chunks: Vec<String>,
    pub(crate) last_push: Option<Instant>,
    /// Whether the previous dictation write added a trailing space. This lets
    /// punctuation remove only our own auto-space, never arbitrary editor text.
    trailing_space_inserted: bool,
}

impl LiveWordBuffer {
    pub fn new(max_phrase_hold: Duration) -> Self {
        Self {
            max_phrase_hold,
            chunks: Vec::new(),
            last_push: None,
            trailing_space_inserted: false,
        }
    }

    pub fn push(&mut self, chunk: String) {
        self.chunks.push(chunk);
        self.last_push = Some(Instant::now());
    }

    /// Words currently held (raw, space separated).
    pub fn pending_text(&self) -> String {
        self.chunks.join(" ")
    }

    /// Drop everything without typing.
    pub fn discard(&mut self) {
        self.chunks.clear();
        self.last_push = None;
    }

    pub fn flush_ready(
        &mut self,
        keyboard: &mut Box<dyn VirtualKeyboard>,
        replacement_reloader: &mut ReplacementReloader,
        transcript_recorder: &mut TranscriptRecorder,
    ) -> Result<Option<String>> {
        let raw = self.chunks.join(" ");
        if replacement_reloader.ends_with_phrase_prefix(&raw) {
            return Ok(None);
        }
        self.flush_all(keyboard, replacement_reloader, transcript_recorder)
    }

    pub fn flush_if_expired(
        &mut self,
        keyboard: &mut Box<dyn VirtualKeyboard>,
        replacement_reloader: &mut ReplacementReloader,
        transcript_recorder: &mut TranscriptRecorder,
    ) -> Result<Option<String>> {
        if self
            .last_push
            .is_some_and(|last_push| last_push.elapsed() >= self.max_phrase_hold)
        {
            return self.flush_all(keyboard, replacement_reloader, transcript_recorder);
        }
        Ok(None)
    }

    /// Type every held word. Returns the formatted text that was typed.
    pub fn flush_all(
        &mut self,
        keyboard: &mut Box<dyn VirtualKeyboard>,
        replacement_reloader: &mut ReplacementReloader,
        transcript_recorder: &mut TranscriptRecorder,
    ) -> Result<Option<String>> {
        self.flush_count(
            self.chunks.len(),
            keyboard,
            replacement_reloader,
            transcript_recorder,
        )
    }

    fn flush_count(
        &mut self,
        count: usize,
        keyboard: &mut Box<dyn VirtualKeyboard>,
        replacement_reloader: &mut ReplacementReloader,
        transcript_recorder: &mut TranscriptRecorder,
    ) -> Result<Option<String>> {
        if count == 0 {
            return Ok(None);
        }
        let raw = self.chunks.drain(..count).collect::<Vec<_>>().join(" ");
        if self.chunks.is_empty() {
            self.last_push = None;
        }
        if raw.trim().is_empty() {
            return Ok(None);
        }
        let replaced = replacement_reloader.replace(&raw);
        let formatted = self.type_formatted(keyboard, &replaced)?;
        vlog!("[TYPING WORD] {}", formatted);
        transcript_recorder.record("word", &raw, &formatted)?;
        Ok(Some(formatted))
    }

    pub fn type_formatted(
        &mut self,
        keyboard: &mut Box<dyn VirtualKeyboard>,
        text: &str,
    ) -> Result<String> {
        let formatted = normalize_punctuation_spacing(text);
        if formatted.is_empty() {
            return Ok(formatted);
        }
        if starts_with_closing_punctuation(&formatted) && self.trailing_space_inserted {
            keyboard.press_key(SpecialKey::Backspace)?;
        }
        keyboard.type_text(&formatted)?;
        keyboard.press_key(SpecialKey::Space)?;
        self.trailing_space_inserted = true;
        Ok(formatted)
    }
}

/// Punctuation tokens are sometimes emitted separately from their preceding
/// word. Normalise only the safe, common rule: no whitespace before closing
/// punctuation. Opening quotes/brackets deliberately remain untouched because
/// their language-specific spacing rules are more nuanced.
pub fn normalize_punctuation_spacing(text: &str) -> String {
    const NO_SPACE_BEFORE: [char; 10] = ['.', ',', '!', '?', ';', ':', '%', '…', ')', ']'];
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if NO_SPACE_BEFORE.contains(&ch) {
            while out.ends_with(' ') {
                out.pop();
            }
            out.push(ch);
            pending_space = false;
            continue;
        }
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        out.push(ch);
        pending_space = false;
    }
    out
}

pub fn starts_with_closing_punctuation(text: &str) -> bool {
    matches!(
        text.chars().next(),
        Some('.' | ',' | '!' | '?' | ';' | ':' | '%' | '…' | ')' | ']')
    )
}

/// Appends dictation history to `$XDG_DATA_HOME/ears/transcripts/<date>.jsonl`.
pub struct TranscriptRecorder {
    config: TranscriptHistoryConfig,
    pending: Vec<String>,
    last_flush: Instant,
}

impl TranscriptRecorder {
    pub fn new(config: TranscriptHistoryConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            last_flush: Instant::now(),
        }
    }

    pub fn record(&mut self, event_type: &str, raw: &str, replaced: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let now = time::OffsetDateTime::now_utc();
        let changed = raw != replaced;
        let mut event = serde_json::Map::new();
        event.insert("ts".to_string(), json!(now.unix_timestamp()));
        event.insert("type".to_string(), json!(event_type));
        event.insert("changed".to_string(), json!(changed));
        if self.config.store_raw {
            event.insert("raw".to_string(), json!(raw));
        }
        if self.config.store_replaced && (changed || self.config.store_unchanged_replaced) {
            event.insert("replaced".to_string(), json!(replaced));
        }
        self.pending.push(Value::Object(event).to_string());
        if self.pending.len() >= self.config.flush_max_events.max(1) {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush_if_due(&mut self) -> Result<()> {
        if self.last_flush.elapsed() >= Duration::from_millis(self.config.flush_interval_ms) {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() || !self.config.enabled {
            return Ok(());
        }
        let dir = transcript_history_dir(&self.config.path);
        fs::create_dir_all(&dir)?;
        let date = time::OffsetDateTime::now_utc().date().to_string();
        let path = dir.join(format!("{date}.jsonl"));
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        for line in self.pending.drain(..) {
            writeln!(file, "{}", line)?;
        }
        self.last_flush = Instant::now();
        Ok(())
    }
}

/// A [`VirtualKeyboard`] that accumulates text on the system clipboard
/// instead of typing it. Backing store for [`InsertionMode::Clipboard`] and
/// [`InsertionMode::SendAsPrompt`].
pub struct ClipboardKeyboard {
    buffer: String,
    clipboard: Option<arboard::Clipboard>,
}

impl ClipboardKeyboard {
    pub fn new() -> Self {
        let clipboard = match arboard::Clipboard::new() {
            Ok(clipboard) => Some(clipboard),
            Err(err) => {
                eprintln!("warning: clipboard unavailable, text is kept in memory only: {err}");
                None
            }
        };
        Self {
            buffer: String::new(),
            clipboard,
        }
    }

    pub fn text(&self) -> &str {
        &self.buffer
    }

    pub fn take(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    fn sync(&mut self) -> Result<()> {
        if let Some(clipboard) = self.clipboard.as_mut() {
            clipboard
                .set_text(self.buffer.trim_end().to_string())
                .context("failed to write clipboard")?;
        }
        Ok(())
    }
}

impl Default for ClipboardKeyboard {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualKeyboard for ClipboardKeyboard {
    fn type_text(&mut self, text: &str) -> Result<()> {
        self.buffer.push_str(text);
        self.sync()
    }

    fn press_key(&mut self, key: SpecialKey) -> Result<()> {
        match key {
            SpecialKey::Space => self.buffer.push(' '),
            SpecialKey::Enter => self.buffer.push('\n'),
            SpecialKey::Tab => self.buffer.push('\t'),
            SpecialKey::Backspace => {
                self.buffer.pop();
            }
            _ => {}
        }
        self.sync()
    }
}

/// Insertion target with a switchable mode.
pub struct Inserter {
    mode: InsertionMode,
    keyboard: Box<dyn VirtualKeyboard>,
    /// Kept when the mode is a clipboard mode so the session text can be read.
    clipboard: Option<Arc<std::sync::Mutex<String>>>,
}

struct SharedClipboardKeyboard {
    inner: ClipboardKeyboard,
    mirror: Arc<std::sync::Mutex<String>>,
}

impl VirtualKeyboard for SharedClipboardKeyboard {
    fn type_text(&mut self, text: &str) -> Result<()> {
        self.inner.type_text(text)?;
        *self.mirror.lock().unwrap() = self.inner.text().to_string();
        Ok(())
    }

    fn press_key(&mut self, key: SpecialKey) -> Result<()> {
        self.inner.press_key(key)?;
        *self.mirror.lock().unwrap() = self.inner.text().to_string();
        Ok(())
    }
}

impl Inserter {
    pub fn new(mode: InsertionMode) -> Result<Self> {
        let mut inserter = Self {
            mode,
            keyboard: Box::new(NullKeyboard),
            clipboard: None,
        };
        inserter.set_mode(mode)?;
        Ok(inserter)
    }

    pub fn mode(&self) -> InsertionMode {
        self.mode
    }

    pub fn keyboard(&mut self) -> &mut Box<dyn VirtualKeyboard> {
        &mut self.keyboard
    }

    /// Switch the insertion target. Re-creates the keyboard, so pending
    /// clipboard text is dropped.
    pub fn set_mode(&mut self, mode: InsertionMode) -> Result<()> {
        match mode {
            InsertionMode::InsertAtCursor => {
                self.keyboard = create_virtual_keyboard().context(
                    "Failed to initialize virtual keyboard. \
                     On Linux/Wayland, ensure you are in the 'input' group.",
                )?;
                self.clipboard = None;
            }
            InsertionMode::Clipboard | InsertionMode::SendAsPrompt => {
                let mirror = Arc::new(std::sync::Mutex::new(String::new()));
                self.keyboard = Box::new(SharedClipboardKeyboard {
                    inner: ClipboardKeyboard::new(),
                    mirror: mirror.clone(),
                });
                self.clipboard = Some(mirror);
            }
        }
        self.mode = mode;
        Ok(())
    }

    /// Text accumulated in a clipboard mode since the session began.
    pub fn session_text(&self) -> Option<String> {
        self.clipboard
            .as_ref()
            .map(|mirror| mirror.lock().unwrap().trim().to_string())
    }

    /// Called when capture starts: clipboard modes start a fresh buffer.
    pub fn begin_session(&mut self) -> Result<()> {
        if matches!(
            self.mode,
            InsertionMode::Clipboard | InsertionMode::SendAsPrompt
        ) {
            self.set_mode(self.mode)?;
        }
        Ok(())
    }

    /// Called when capture stops. Emits the prompt text in
    /// [`InsertionMode::SendAsPrompt`].
    pub(crate) fn end_session(&mut self, shared: &Shared) {
        if self.mode == InsertionMode::SendAsPrompt {
            if let Some(text) = self.session_text().filter(|t| !t.is_empty()) {
                shared.emit(DictationEvent::PromptReady { text });
            }
        }
    }
}

/// Placeholder used before the real keyboard is created.
struct NullKeyboard;

impl VirtualKeyboard for NullKeyboard {
    fn type_text(&mut self, _text: &str) -> Result<()> {
        Ok(())
    }

    fn press_key(&mut self, _key: SpecialKey) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::replacement::ReplacementDictionary;
    use crate::replacement::ReplacementEntry;
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    pub(crate) struct RecordingKeyboard {
        pub(crate) typed: Arc<Mutex<String>>,
    }

    impl VirtualKeyboard for RecordingKeyboard {
        fn type_text(&mut self, text: &str) -> Result<()> {
            self.typed.lock().unwrap().push_str(text);
            Ok(())
        }

        fn press_key(&mut self, key: SpecialKey) -> Result<()> {
            if matches!(key, SpecialKey::Space) {
                self.typed.lock().unwrap().push(' ');
            }
            Ok(())
        }
    }

    pub(crate) fn test_reloader(dir: &std::path::Path) -> ReplacementReloader {
        let path = dir.join("global.toml");
        ReplacementDictionary {
            version: 1,
            entries: vec![ReplacementEntry {
                replace: "trx issue".to_string(),
                phrases: vec!["tricks issue".to_string()],
            }],
        }
        .save(&path)
        .unwrap();
        ReplacementReloader::new(ReplacementConfig {
            enabled: true,
            dictionary_paths: vec![path.to_string_lossy().to_string()],
            case_sensitive: false,
        })
    }

    pub(crate) fn silent_recorder() -> TranscriptRecorder {
        let mut history_config = TranscriptHistoryConfig::default();
        history_config.enabled = false;
        TranscriptRecorder::new(history_config)
    }

    #[test]
    fn buffers_dictionary_phrase_across_stream_events() {
        let dir =
            std::env::temp_dir().join(format!("ears-live-dict-test-{}", uuid::Uuid::new_v4()));
        let mut reloader = test_reloader(&dir);
        let mut recorder = silent_recorder();
        let mut buffer = LiveWordBuffer::new(Duration::from_millis(700));
        let recording_keyboard = RecordingKeyboard::default();
        let typed = Arc::clone(&recording_keyboard.typed);
        let mut keyboard: Box<dyn VirtualKeyboard> = Box::new(recording_keyboard);

        buffer.push("tricks".to_string());
        buffer
            .flush_ready(&mut keyboard, &mut reloader, &mut recorder)
            .unwrap();
        assert_eq!(buffer.chunks, vec!["tricks"]);

        buffer.push("issue".to_string());
        buffer
            .flush_ready(&mut keyboard, &mut reloader, &mut recorder)
            .unwrap();
        assert!(buffer.chunks.is_empty());
        assert_eq!(buffer.last_push, None);
        assert_eq!(&*typed.lock().unwrap(), "trx issue ");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn override_swaps_dictionaries_and_restores_base() {
        let dir =
            std::env::temp_dir().join(format!("ears-live-dict-test-{}", uuid::Uuid::new_v4()));
        let mut reloader = test_reloader(&dir);
        let other = dir.join("other.toml");
        ReplacementDictionary {
            version: 1,
            entries: vec![ReplacementEntry {
                replace: "GPUI".to_string(),
                phrases: vec!["gee pee you eye".to_string()],
            }],
        }
        .save(&other)
        .unwrap();

        assert_eq!(reloader.replace("tricks issue"), "trx issue");
        reloader.set_override(&[other.to_string_lossy().to_string()]);
        assert_eq!(reloader.replace("gee pee you eye"), "GPUI");
        assert_eq!(reloader.replace("tricks issue"), "tricks issue");
        reloader.set_override(&[]);
        assert_eq!(reloader.replace("tricks issue"), "trx issue");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn normalizes_space_before_closing_punctuation() {
        assert_eq!(
            normalize_punctuation_spacing("Das ist . Wirklich , ja !"),
            "Das ist. Wirklich, ja!"
        );
        assert_eq!(normalize_punctuation_spacing("  Hallo …  "), "Hallo…");
        assert!(starts_with_closing_punctuation("."));
        assert!(!starts_with_closing_punctuation("Hallo."));
    }

    #[test]
    fn clipboard_keyboard_accumulates_text() {
        let mut keyboard = ClipboardKeyboard {
            buffer: String::new(),
            clipboard: None,
        };
        keyboard.type_text("hello").unwrap();
        keyboard.press_key(SpecialKey::Space).unwrap();
        keyboard.type_text("world").unwrap();
        keyboard.press_key(SpecialKey::Space).unwrap();
        keyboard.press_key(SpecialKey::Backspace).unwrap();
        keyboard.type_text(".").unwrap();
        assert_eq!(keyboard.text(), "hello world.");
    }
}
