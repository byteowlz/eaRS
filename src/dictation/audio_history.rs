//! Opt-in local retention of dictation audio (trx-63fk).
//!
//! When `[dictation.audio_history] enabled = true`, every committed dictation
//! utterance is kept as a mono 16-bit WAV of the audio the client sent to the
//! server, plus one JSON line of metadata (transcripts, typed text, engine,
//! target app). It is evidence for the owner's own training and evaluation
//! data, nothing more: retention is local only, never uploaded, and a record
//! is never marked training-eligible automatically.
//!
//! Layout under the history directory (default
//! `$XDG_STATE_HOME/ears/audio-history`, next to the text history):
//!
//! ```text
//! index.jsonl                    one UtteranceRecord per line
//! audio/<YYYY-MM-DD>/<id>.wav    the utterance audio
//! ```
//!
//! Directories are 0700 and files 0600. With retention disabled no audio is
//! buffered at all, not even in memory.
//!
//! Utterance spans are cut by arrival time: from the first recognised word
//! minus `pre_roll_ms` to the server's end-of-speech boundary. That is fine
//! for training data, but it is not an authoritative PCM binding, so it must
//! never be used as speaker-gate evidence.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Sample rate of the dictation capture stream (resampled by `audio`).
pub const SAMPLE_RATE: u32 = 24_000;
const INDEX_FILE: &str = "index.jsonl";
const RECOGNITION_FILE: &str = "recognition.jsonl";
const RECORD_VERSION: u32 = 1;

/// Bundle ids / binaries of password managers and credential prompts where
/// audio is never retained, even with retention enabled.
fn default_excluded_apps() -> Vec<String> {
    [
        "com.1password.1password",
        "com.agilebits.onepassword7",
        "com.bitwarden.desktop",
        "com.apple.keychainaccess",
        "com.apple.Passwords",
        "org.keepassxc.keepassxc",
        "com.lastpass.LastPass",
        "1password",
        "bitwarden",
        "keepassxc",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AudioHistoryConfig {
    /// Keep the audio of committed utterances. Off by default; never enabled
    /// implicitly.
    pub enabled: bool,
    /// Optional client-owned acoustic segmentation. Requires `client-vad` build.
    pub client_vad: Option<ClientVadConfig>,
    /// History directory, or "default" for `$XDG_STATE_HOME/ears/audio-history`.
    /// Must not be inside a git working tree.
    pub path: String,
    /// Total size cap; the oldest utterances are deleted first.
    pub max_total_mb: u64,
    /// Utterances older than this are deleted (0 keeps them indefinitely).
    pub max_age_days: u32,
    /// Legacy: before the first recognized word. Client VAD: before acoustic
    /// onset, rounded up to whole detector frames.
    pub pre_roll_ms: u32,
    /// Longest utterance kept; longer ones are truncated to their last part.
    pub max_utterance_secs: u32,
    /// Apps (bundle id, binary or name, case-insensitive) during which nothing
    /// is retained.
    pub excluded_apps: Vec<String>,
    /// Pause retention while macOS Secure Event Input is active (password
    /// fields).
    pub pause_on_secure_input: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClientVadConfig {
    pub model_path: String,
    pub model_sha256: String,
    pub runtime_path: String,
}

#[cfg(feature = "client-vad")]
impl ClientVadConfig {
    pub(crate) fn load_detector(&self) -> Result<super::client_vad::SileroVad> {
        let model = crate::replacement::expand_tilde(&self.model_path);
        if hex(&Sha256::digest(fs::read(&model)?)) != self.model_sha256 {
            bail!("client VAD model hash does not match configuration");
        }
        let runtime = crate::replacement::expand_tilde(&self.runtime_path);
        super::client_vad::SileroVad::load(&model, &runtime)
    }
}

impl Default for AudioHistoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            client_vad: None,
            path: "default".to_string(),
            max_total_mb: 2048,
            max_age_days: 90,
            pre_roll_ms: 1000,
            max_utterance_secs: 120,
            excluded_apps: default_excluded_apps(),
            pause_on_secure_input: true,
        }
    }
}

pub fn history_dir(configured_path: &str) -> PathBuf {
    if configured_path != "default" {
        return crate::replacement::expand_tilde(configured_path);
    }
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(dirs::state_dir)
        .or_else(|| dirs::home_dir().map(|home| home.join(".local/state")))
        .unwrap_or_else(|| PathBuf::from(".local/state"));
    base.join("ears").join("audio-history")
}

/// One retained utterance. Field names are the export manifest contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtteranceRecord {
    pub version: u32,
    pub id: String,
    /// UTC, RFC 3339.
    pub recorded_at: String,
    /// One dictation capture session (listening on -> off).
    pub session_id: String,
    pub engine: Option<String>,
    pub backend: Option<String>,
    pub language: Option<String>,
    pub device: Option<String>,
    pub sample_rate: u32,
    pub samples: u64,
    pub duration_secs: f64,
    pub bytes: u64,
    pub sha256: String,
    /// Relative to the history directory.
    pub audio_path: String,
    pub raw_transcript: String,
    pub replaced_transcript: String,
    pub inserted_text: String,
    /// Bundle id (macOS) or binary of the frontmost app; never a window title.
    pub target_app: Option<String>,
    pub label_status: String,
    pub training_eligible: bool,
    /// None for legacy arrival-time clips. Client-VAD clips are untranscribed;
    /// recognition events are stored separately, never guessed onto a clip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_boundary: Option<CaptureBoundary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaptureBoundary {
    pub source: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub complete: bool,
}

#[derive(Serialize, Deserialize)]
struct RecognitionEvent {
    recorded_at: String,
    session_id: String,
    at_capture_sample: u64,
    engine: Option<String>,
    backend: Option<String>,
    language: Option<String>,
    raw_transcript: String,
    replaced_transcript: String,
    inserted_text: String,
    alignment: String,
    training_eligible: bool,
}

impl UtteranceRecord {
    fn date(&self) -> &str {
        self.recorded_at.get(..10).unwrap_or("")
    }
}

/// Rolling buffer of the most recent captured audio, addressed by absolute
/// sample index so utterance spans survive buffer rotation.
#[derive(Debug)]
pub struct AudioTap {
    inner: Mutex<TapInner>,
}

#[derive(Debug)]
struct TapInner {
    samples: VecDeque<f32>,
    capacity: usize,
    /// Absolute index of the next sample pushed.
    position: u64,
}

impl AudioTap {
    pub fn new(capacity_secs: u32) -> Arc<Self> {
        let capacity = capacity_secs.max(1) as usize * SAMPLE_RATE as usize;
        Arc::new(Self {
            inner: Mutex::new(TapInner {
                samples: VecDeque::with_capacity(capacity),
                capacity,
                position: 0,
            }),
        })
    }

    pub fn push(&self, chunk: &[f32]) {
        let mut inner = self.inner.lock().unwrap();
        inner.samples.extend(chunk.iter().copied());
        let overflow = inner.samples.len().saturating_sub(inner.capacity);
        inner.samples.drain(..overflow);
        inner.position += chunk.len() as u64;
    }

    pub fn position(&self) -> u64 {
        self.inner.lock().unwrap().position
    }

    /// Samples in `[start, end)` that are still buffered.
    pub(super) fn slice(&self, start: u64, end: u64) -> Vec<f32> {
        let inner = self.inner.lock().unwrap();
        let oldest = inner.position - inner.samples.len() as u64;
        let start = start.max(oldest);
        let end = end.min(inner.position);
        if start >= end {
            return Vec::new();
        }
        inner
            .samples
            .range((start - oldest) as usize..(end - oldest) as usize)
            .copied()
            .collect()
    }
}

impl AudioTap {
    fn clear(&self) {
        self.inner.lock().unwrap().samples.clear();
    }
}

/// What the engine knows about the moment an utterance is committed.
#[derive(Debug, Clone, Default)]
pub struct CommitContext {
    pub raw: String,
    pub replaced: String,
    pub language: Option<String>,
    /// Bundle id, binary and name of the frontmost app, for exclusion and the
    /// `target_app` field.
    pub app_bundle_id: Option<String>,
    pub app_binary: Option<String>,
    pub app_name: Option<String>,
    pub secure_input: bool,
}

/// Why an utterance was not retained.
#[derive(Debug, Clone, PartialEq)]
pub enum Skip {
    ExcludedApp(String),
    SecureInput,
    NoAudio,
}

/// Per-engine retention state. Built only when retention is enabled.
pub struct AudioHistory {
    config: AudioHistoryConfig,
    dir: PathBuf,
    tap: Arc<AudioTap>,
    session_id: String,
    utterance_start: Option<u64>,
    last_end: u64,
    inserted: String,
    engine: Option<String>,
    backend: Option<String>,
    device: Option<String>,
    #[cfg(feature = "client-vad")]
    capture_vad: Option<super::recording_vad::RecordingVad>,
    capture_app: Option<String>,
    capture_needs_quiet: bool,
}

impl AudioHistory {
    /// `None` when retention is disabled. Refuses a directory inside a git
    /// working tree so recordings can never be committed by accident.
    pub fn from_config(
        config: &AudioHistoryConfig,
        device: Option<String>,
        backend: Option<String>,
    ) -> Result<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }
        let dir = history_dir(&config.path);
        ensure_private_dir(&dir)?;
        if let Some(repo) = enclosing_git_repo(&dir) {
            bail!(
                "audio history directory {} is inside the git repository {}; choose a path outside any repository",
                dir.display(),
                repo.display()
            );
        }
        #[cfg(not(feature = "client-vad"))]
        if config.client_vad.is_some() {
            bail!("client VAD retention requires an eaRS build with client-vad");
        }
        #[cfg(feature = "client-vad")]
        let capture_vad = config
            .client_vad
            .as_ref()
            .map(|vad| -> Result<_> {
                let detector = vad.load_detector()?;
                super::recording_vad::RecordingVad::new(
                    Box::new(detector),
                    config.max_utterance_secs,
                    config.pre_roll_ms,
                )
            })
            .transpose()?;
        let capacity = config.max_utterance_secs + config.pre_roll_ms / 1000 + 10;
        let history = Self {
            config: config.clone(),
            dir,
            tap: AudioTap::new(capacity),
            session_id: new_id(),
            utterance_start: None,
            last_end: 0,
            inserted: String::new(),
            engine: None,
            backend,
            device,
            #[cfg(feature = "client-vad")]
            capture_vad,
            capture_app: None,
            capture_needs_quiet: false,
        };
        enforce_limits(&history.dir, &history.config)?;
        Ok(Some(history))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The tap the capture thread feeds while dictation is listening.
    pub fn tap(&self) -> Arc<AudioTap> {
        self.tap.clone()
    }

    pub fn set_engine(&mut self, engine: Option<String>) {
        self.engine = engine;
    }

    /// Listening started: a new capture session.
    pub fn begin_session(&mut self) {
        self.session_id = new_id();
        self.discard();
        self.capture_app = None;
        self.capture_needs_quiet = false;
        self.tap.clear();
        #[cfg(feature = "client-vad")]
        if let Some(vad) = &mut self.capture_vad {
            vad.reset(self.tap.position(), false);
        }
    }

    /// The first word of an utterance arrived.
    pub fn begin_utterance(&mut self) {
        if self.utterance_start.is_some() {
            return;
        }
        let pre_roll = self.config.pre_roll_ms as u64 * SAMPLE_RATE as u64 / 1000;
        let start = self.tap.position().saturating_sub(pre_roll);
        self.utterance_start = Some(start.max(self.last_end));
    }

    /// Text typed into the focused app for the current utterance. Typed
    /// chunks arrive without the separating spaces the keyboard inserts.
    pub fn note_inserted(&mut self, text: &str) {
        let text = text.trim();
        if self.utterance_start.is_none() || text.is_empty() {
            return;
        }
        if !self.inserted.is_empty() {
            self.inserted.push(' ');
        }
        self.inserted.push_str(text);
    }

    /// Drop the current utterance (cancel, connection loss, push-to-talk
    /// release without a boundary).
    pub fn discard(&mut self) {
        if self.utterance_start.take().is_some() {
            self.last_end = self.tap.position();
        }
        self.inserted.clear();
        #[cfg(feature = "client-vad")]
        if let Some(vad) = &mut self.capture_vad {
            self.capture_needs_quiet = true;
            vad.reset(self.tap.position(), true);
        }
    }

    /// Write the current utterance. Returns the stored record, or why it was
    /// skipped. Never writes anything for excluded apps or secure input.
    pub fn commit(
        &mut self,
        ctx: &CommitContext,
    ) -> Result<std::result::Result<UtteranceRecord, Skip>> {
        let end = self.tap.position();
        let Some(start) = self.utterance_start.take() else {
            return Ok(Err(Skip::NoAudio));
        };
        self.last_end = end;
        let inserted = std::mem::take(&mut self.inserted);

        if self.config.pause_on_secure_input && ctx.secure_input {
            return Ok(Err(Skip::SecureInput));
        }
        if let Some(app) = self.excluded_app(ctx) {
            return Ok(Err(Skip::ExcludedApp(app)));
        }
        let max = self.config.max_utterance_secs.max(1) as u64 * SAMPLE_RATE as u64;
        let start = start.max(end.saturating_sub(max));
        let pcm = self.tap.slice(start, end);
        if pcm.is_empty() {
            return Ok(Err(Skip::NoAudio));
        }

        self.store_clip(ctx, &pcm, inserted, None).map(Ok)
    }

    fn store_clip(
        &mut self,
        ctx: &CommitContext,
        pcm: &[f32],
        inserted: String,
        boundary: Option<CaptureBoundary>,
    ) -> Result<UtteranceRecord> {
        let now = time::OffsetDateTime::now_utc();
        let recorded_at = now
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_default();
        let id = new_id();
        let date = now.date().to_string();
        let relative = format!("audio/{date}/{id}.wav");
        let wav = encode_wav(&pcm, SAMPLE_RATE);
        let sha256 = hex(&Sha256::digest(&wav));
        let audio_path = self.dir.join(&relative);
        ensure_private_dir(audio_path.parent().expect("audio path has a parent"))?;
        write_private(&audio_path, &wav)?;

        let record = UtteranceRecord {
            version: RECORD_VERSION,
            id,
            recorded_at,
            session_id: self.session_id.clone(),
            engine: self.engine.clone(),
            backend: self.backend.clone(),
            language: ctx.language.clone(),
            device: self.device.clone(),
            sample_rate: SAMPLE_RATE,
            samples: pcm.len() as u64,
            duration_secs: pcm.len() as f64 / SAMPLE_RATE as f64,
            bytes: wav.len() as u64,
            sha256,
            audio_path: relative,
            raw_transcript: ctx.raw.clone(),
            replaced_transcript: ctx.replaced.clone(),
            inserted_text: inserted,
            target_app: ctx.app_bundle_id.clone().or_else(|| ctx.app_binary.clone()),
            label_status: "unreviewed".to_string(),
            training_eligible: false,
            capture_boundary: boundary,
        };
        append_record(&self.dir, &record)?;
        enforce_limits(&self.dir, &self.config)?;
        Ok(record)
    }

    pub fn client_vad_enabled(&self) -> bool {
        self.config.client_vad.is_some()
    }

    /// Feed original 24 kHz capture PCM on the engine thread, before transport.
    /// Privacy is observed at each capture batch, not inferred from an ASR final.
    pub fn record_capture(&mut self, pcm: &[f32], ctx: &CommitContext) -> Result<()> {
        if !self.client_vad_enabled() {
            self.tap.push(pcm);
            return Ok(());
        }
        #[cfg(feature = "client-vad")]
        {
            let app = ctx
                .app_bundle_id
                .as_ref()
                .or(ctx.app_binary.as_ref())
                .or(ctx.app_name.as_ref());
            let allowed = app.is_some()
                && self.excluded_app(ctx).is_none()
                && !(self.config.pause_on_secure_input && ctx.secure_input);
            if !allowed {
                self.tap.clear();
                self.capture_app = None;
                self.capture_needs_quiet = true;
                self.capture_vad
                    .as_mut()
                    .context("client VAD unavailable")?
                    .reset(self.tap.position(), true);
                return Ok(());
            }
            if self.capture_app.as_ref() != app {
                let wait = self.capture_needs_quiet || self.capture_app.is_some();
                self.capture_needs_quiet = wait;
                self.tap.clear();
                self.capture_vad
                    .as_mut()
                    .context("client VAD unavailable")?
                    .reset(self.tap.position(), wait);
                self.capture_app = app.cloned();
            }
            self.tap.push(pcm);
            let clips = self
                .capture_vad
                .as_mut()
                .context("client VAD unavailable")?
                .push(pcm, &self.tap)?;
            for clip in clips {
                self.store_capture_clip(clip, ctx)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "client-vad"))]
        {
            let _ = ctx;
            bail!("client VAD unavailable")
        }
    }

    /// Explicit capture release retains available PCM but marks it incomplete.
    /// Cancel/disconnect must call discard instead, never manufacture a boundary.
    pub fn end_capture(&mut self, ctx: &CommitContext) -> Result<()> {
        #[cfg(feature = "client-vad")]
        if self.client_vad_enabled() {
            let app = ctx
                .app_bundle_id
                .as_ref()
                .or(ctx.app_binary.as_ref())
                .or(ctx.app_name.as_ref());
            if app.is_none()
                || self.capture_app.as_ref() != app
                || self.excluded_app(ctx).is_some()
                || (self.config.pause_on_secure_input && ctx.secure_input)
            {
                self.discard();
                return Ok(());
            }
            if let Some(clip) = self
                .capture_vad
                .as_mut()
                .context("client VAD unavailable")?
                .finish(&self.tap)?
            {
                self.store_capture_clip(clip, ctx)?;
            }
        }
        let _ = ctx;
        Ok(())
    }

    #[cfg(feature = "client-vad")]
    fn store_capture_clip(
        &mut self,
        clip: super::recording_vad::Clip,
        ctx: &CommitContext,
    ) -> Result<()> {
        let context = CommitContext {
            raw: String::new(),
            replaced: String::new(),
            ..ctx.clone()
        };
        self.store_clip(
            &context,
            &clip.pcm,
            String::new(),
            Some(CaptureBoundary {
                source: "client_silero".into(),
                start_sample: clip.start,
                end_sample: clip.end,
                complete: clip.complete,
            }),
        )?;
        Ok(())
    }

    /// ASR output is retained as an explicitly unaligned suggestion, not a label
    /// guessed onto the next VAD clip. Original recognition remains inspectable.
    pub fn record_recognition(&mut self, ctx: &CommitContext) -> Result<()> {
        let inserted = std::mem::take(&mut self.inserted);
        self.utterance_start = None;
        if self.capture_needs_quiet
            || self.excluded_app(ctx).is_some()
            || (self.config.pause_on_secure_input && ctx.secure_input)
            || ctx
                .app_bundle_id
                .as_ref()
                .or(ctx.app_binary.as_ref())
                .or(ctx.app_name.as_ref())
                .is_none()
        {
            return Ok(());
        }
        let event = RecognitionEvent {
            recorded_at: time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)?,
            session_id: self.session_id.clone(),
            at_capture_sample: self.tap.position(),
            engine: self.engine.clone(),
            backend: self.backend.clone(),
            language: ctx.language.clone(),
            raw_transcript: ctx.raw.clone(),
            replaced_transcript: ctx.replaced.clone(),
            inserted_text: inserted,
            alignment: "unassigned".into(),
            training_eligible: false,
        };
        let mut file = private_options()
            .create(true)
            .append(true)
            .open(self.dir.join(RECOGNITION_FILE))?;
        writeln!(file, "{}", serde_json::to_string(&event)?)?;
        enforce_limits(&self.dir, &self.config)?;
        Ok(())
    }

    fn excluded_app(&self, ctx: &CommitContext) -> Option<String> {
        let candidates = [&ctx.app_bundle_id, &ctx.app_binary, &ctx.app_name];
        for app in candidates.into_iter().flatten() {
            if self
                .config
                .excluded_apps
                .iter()
                .any(|excluded| excluded.eq_ignore_ascii_case(app))
            {
                return Some(app.clone());
            }
        }
        None
    }
}

fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Mono 16-bit PCM WAV.
pub fn encode_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in pcm {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn ensure_private_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to restrict {}", dir.display()))?;
    }
    Ok(())
}

fn private_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = private_options()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.write_all(bytes)?;
    Ok(())
}

/// Nearest ancestor (including `dir`) that holds a `.git` entry.
fn enclosing_git_repo(dir: &Path) -> Option<PathBuf> {
    let dir = fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    dir.ancestors()
        .find(|ancestor| ancestor.join(".git").exists())
        .map(Path::to_path_buf)
}

fn append_record(dir: &Path, record: &UtteranceRecord) -> Result<()> {
    let mut file = private_options()
        .create(true)
        .append(true)
        .open(dir.join(INDEX_FILE))
        .context("failed to open audio history index")?;
    writeln!(file, "{}", serde_json::to_string(record)?)?;
    Ok(())
}

/// All records, oldest first. Unparseable lines are skipped.
pub fn load_records(dir: &Path) -> Result<Vec<UtteranceRecord>> {
    let path = dir.join(INDEX_FILE);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err).with_context(|| format!("failed to read {}", path.display())),
    };
    let mut records = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if let Ok(record) = serde_json::from_str::<UtteranceRecord>(&line) {
            records.push(record);
        }
    }
    Ok(records)
}

/// Resolve a retained utterance to a verified local audio file for review.
/// The ID must uniquely identify an indexed record; missing, escaped or modified
/// audio is rejected rather than handed to a player.
pub fn review_audio_path(dir: &Path, id: &str) -> Result<PathBuf> {
    let records = load_records(dir)?;
    let mut matches = records.iter().filter(|record| record.id == id);
    let record = matches.next().context("utterance not found")?;
    if matches.next().is_some() {
        bail!("utterance ID is ambiguous");
    }
    let root = fs::canonicalize(dir).context("audio history directory unavailable")?;
    let path =
        fs::canonicalize(root.join(&record.audio_path)).context("utterance audio unavailable")?;
    if !path.starts_with(root.join("audio")) || !path.is_file() {
        bail!("utterance audio is outside the history audio directory");
    }
    let bytes = fs::read(&path).context("failed to read utterance audio")?;
    if hex(&Sha256::digest(&bytes)) != record.sha256 {
        bail!("utterance audio does not match its recorded sha256");
    }
    Ok(path)
}

/// Replace the index atomically with `records`.
fn write_records(dir: &Path, records: &[UtteranceRecord]) -> Result<()> {
    let tmp = dir.join(format!("{INDEX_FILE}.tmp"));
    let mut body = String::new();
    for record in records {
        body.push_str(&serde_json::to_string(record)?);
        body.push('\n');
    }
    write_private(&tmp, body.as_bytes())?;
    fs::rename(&tmp, dir.join(INDEX_FILE)).context("failed to replace audio history index")?;
    Ok(())
}

/// Which records `purge` removes. All set conditions must match.
#[derive(Debug, Clone, Default)]
pub struct PurgeFilter {
    pub all: bool,
    /// `YYYY-MM-DD`: records from before this UTC date.
    pub before: Option<String>,
    pub session_id: Option<String>,
    pub id: Option<String>,
}

impl PurgeFilter {
    fn is_empty(&self) -> bool {
        !self.all && self.before.is_none() && self.session_id.is_none() && self.id.is_none()
    }

    fn matches(&self, record: &UtteranceRecord) -> bool {
        if self.all {
            return true;
        }
        self.before
            .as_deref()
            .is_none_or(|date| record.date() < date)
            && self
                .session_id
                .as_deref()
                .is_none_or(|s| record.session_id == s)
            && self.id.as_deref().is_none_or(|id| record.id == id)
    }
}

/// Delete matching records together with their audio. Returns the removed
/// records.
pub fn purge(dir: &Path, filter: &PurgeFilter) -> Result<Vec<UtteranceRecord>> {
    if filter.is_empty() {
        bail!("refusing to purge without a filter; pass --all to delete everything");
    }
    if let Some(date) = &filter.before {
        validate_date(date)?;
    }
    let (removed, kept): (Vec<_>, Vec<_>) = load_records(dir)?
        .into_iter()
        .partition(|record| filter.matches(record));
    remove_audio(dir, &removed)?;
    if !removed.is_empty() {
        write_records(dir, &kept)?;
    }
    if filter.all {
        remove_orphaned_audio(dir)?;
    }
    prune_recognition(dir, &removed, filter)?;
    Ok(removed)
}

fn prune_recognition(dir: &Path, removed: &[UtteranceRecord], filter: &PurgeFilter) -> Result<()> {
    let path = dir.join(RECOGNITION_FILE);
    if !path.exists()
        || (removed.is_empty()
            && !filter.all
            && filter.before.is_none()
            && filter.session_id.is_none())
    {
        return Ok(());
    }
    if filter.all {
        fs::remove_file(path)?;
        return Ok(());
    }
    let sessions: std::collections::HashSet<_> =
        removed.iter().map(|record| &record.session_id).collect();
    let mut kept = String::new();
    for line in BufReader::new(File::open(&path)?).lines() {
        let line = line?;
        let event: RecognitionEvent =
            serde_json::from_str(&line).context("invalid recognition event; purge incomplete")?;
        let remove = sessions.contains(&event.session_id)
            || filter.session_id.as_ref() == Some(&event.session_id)
            || filter.before.as_ref().is_some_and(|date| {
                event
                    .recorded_at
                    .get(..10)
                    .is_some_and(|day| day < date.as_str())
            });
        if !remove {
            kept.push_str(&line);
            kept.push('\n');
        }
    }
    let tmp = dir.join(format!("recognition-{}.tmp", new_id()));
    write_private(&tmp, kept.as_bytes())?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn validate_date(date: &str) -> Result<()> {
    let bytes = date.as_bytes();
    let shape = bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit());
    if !shape {
        bail!("expected a date as YYYY-MM-DD, got {date}");
    }
    Ok(())
}

fn remove_audio(dir: &Path, records: &[UtteranceRecord]) -> Result<()> {
    for record in records {
        let path = dir.join(&record.audio_path);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| format!("failed to delete {}", path.display()));
            }
        }
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir(parent); // only succeeds once the day is empty
        }
    }
    Ok(())
}

fn remove_orphaned_audio(dir: &Path) -> Result<()> {
    let audio = dir.join("audio");
    if audio.exists() {
        fs::remove_dir_all(&audio)
            .with_context(|| format!("failed to delete {}", audio.display()))?;
    }
    Ok(())
}

/// Apply the age and size limits, deleting the oldest records first.
pub fn enforce_limits(dir: &Path, config: &AudioHistoryConfig) -> Result<Vec<UtteranceRecord>> {
    apply_limits(
        dir,
        config.max_age_days,
        config.max_total_mb.saturating_mul(1024 * 1024),
    )
}

fn apply_limits(dir: &Path, max_age_days: u32, cap: u64) -> Result<Vec<UtteranceRecord>> {
    let mut records = load_records(dir)?;
    let mut removed = Vec::new();
    if max_age_days > 0 {
        let cutoff = time::OffsetDateTime::now_utc() - time::Duration::days(max_age_days as i64);
        let cutoff = cutoff.date().to_string();
        let (old, keep): (Vec<_>, Vec<_>) = records
            .into_iter()
            .partition(|r| r.date() < cutoff.as_str());
        removed.extend(old);
        records = keep;
    }
    let mut total: u64 = records.iter().map(|r| r.bytes).sum::<u64>()
        + fs::metadata(dir.join(RECOGNITION_FILE))
            .map(|file| file.len())
            .unwrap_or(0);
    let mut evict = 0;
    while total > cap && evict < records.len() {
        total -= records[evict].bytes;
        evict += 1;
    }
    removed.extend(records.drain(..evict));
    if !removed.is_empty() {
        remove_audio(dir, &removed)?;
        write_records(dir, &records)?;
    }
    let before = if max_age_days > 0 {
        Some(
            (time::OffsetDateTime::now_utc() - time::Duration::days(max_age_days as i64))
                .date()
                .to_string(),
        )
    } else {
        None
    };
    prune_recognition(
        dir,
        &removed,
        &PurgeFilter {
            all: total > cap && records.is_empty(),
            before,
            ..PurgeFilter::default()
        },
    )?;
    Ok(removed)
}

/// Session-disjoint split hint: every utterance of a session lands in the
/// same split, so train and test never share a session.
pub fn split_for_session(session_id: &str) -> &'static str {
    match Sha256::digest(session_id.as_bytes())[0] % 10 {
        0 => "test",
        1 => "dev",
        _ => "train",
    }
}

#[derive(Debug, Serialize)]
struct ManifestEntry<'a> {
    #[serde(flatten)]
    record: &'a UtteranceRecord,
    split: &'static str,
    split_group: &'a str,
}

/// Copy all retained audio to `out` and write `out/manifest.jsonl`, grouped by
/// session. Returns the number of exported utterances.
pub fn export(dir: &Path, out: &Path) -> Result<usize> {
    let mut records = load_records(dir)?;
    records.sort_by(|a, b| (&a.session_id, &a.recorded_at).cmp(&(&b.session_id, &b.recorded_at)));
    ensure_private_dir(out)?;
    let mut manifest = String::new();
    for record in &records {
        let src = dir.join(&record.audio_path);
        let dst = out.join(&record.audio_path);
        ensure_private_dir(dst.parent().expect("audio path has a parent"))?;
        let bytes = fs::read(&src).with_context(|| format!("failed to read {}", src.display()))?;
        if hex(&Sha256::digest(&bytes)) != record.sha256 {
            bail!("{} does not match its recorded sha256", src.display());
        }
        write_private(&dst, &bytes)?;
        manifest.push_str(&serde_json::to_string(&ManifestEntry {
            record,
            split: split_for_session(&record.session_id),
            split_group: &record.session_id,
        })?);
        manifest.push('\n');
    }
    let recognition = dir.join(RECOGNITION_FILE);
    if recognition.exists() {
        let mut events = String::new();
        for line in BufReader::new(File::open(recognition)?).lines() {
            let line = line?;
            let event: RecognitionEvent = serde_json::from_str(&line)?;
            if records
                .iter()
                .any(|record| record.session_id == event.session_id)
            {
                events.push_str(&line);
                events.push('\n');
            }
        }
        write_private(&out.join(RECOGNITION_FILE), events.as_bytes())?;
    }
    write_private(&out.join("manifest.jsonl"), manifest.as_bytes())?;
    Ok(records.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ears-audio-history-{name}-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn enabled_config(dir: &Path) -> AudioHistoryConfig {
        AudioHistoryConfig {
            enabled: true,
            path: dir.to_string_lossy().to_string(),
            ..AudioHistoryConfig::default()
        }
    }

    fn tone(secs: f32) -> Vec<f32> {
        (0..(secs * SAMPLE_RATE as f32) as usize)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect()
    }

    fn ctx(raw: &str) -> CommitContext {
        CommitContext {
            raw: raw.into(),
            replaced: raw.into(),
            language: Some("de".into()),
            app_bundle_id: Some("com.apple.TextEdit".into()),
            ..CommitContext::default()
        }
    }

    pub(crate) fn record_utterance(
        history: &mut AudioHistory,
        secs: f32,
        raw: &str,
    ) -> UtteranceRecord {
        history.tap().push(&tone(0.5));
        history.begin_utterance();
        history.tap().push(&tone(secs));
        history.note_inserted(raw);
        history.commit(&ctx(raw)).unwrap().unwrap()
    }

    #[test]
    fn disabled_config_creates_nothing() {
        let root = temp_dir("off");
        let dir = root.join("history");
        let mut config = enabled_config(&dir);
        config.enabled = false;
        assert!(
            AudioHistory::from_config(&config, None, None)
                .unwrap()
                .is_none()
        );
        assert!(!dir.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn configured_client_vad_never_falls_back_when_unavailable() {
        let root = temp_dir("client-unavailable");
        let model = root.join("invalid.onnx");
        fs::write(&model, b"not a model").unwrap();
        let mut config = enabled_config(&root);
        config.client_vad = Some(ClientVadConfig {
            model_path: model.to_string_lossy().into_owned(),
            model_sha256: "wrong hash".into(),
            runtime_path: "missing-runtime".into(),
        });
        assert!(AudioHistory::from_config(&config, None, None).is_err());
        assert!(!root.join(INDEX_FILE).exists());
        assert!(!root.join("audio").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn commit_writes_one_private_wav_with_matching_metadata() {
        let root = temp_dir("commit");
        let mut history = AudioHistory::from_config(
            &enabled_config(&root),
            Some("Mic".into()),
            Some("local".into()),
        )
        .unwrap()
        .unwrap();
        history.set_engine(Some("parakeet-rs".into()));
        let record = record_utterance(&mut history, 2.0, "hallo welt");

        let wav = fs::read(root.join(&record.audio_path)).unwrap();
        assert_eq!(hex(&Sha256::digest(&wav)), record.sha256);
        assert_eq!(wav.len() as u64, record.bytes);
        // 2 s of speech plus the full 0.5 s already buffered as pre-roll.
        assert!(
            (record.duration_secs - 2.5).abs() < 1e-6,
            "{}",
            record.duration_secs
        );
        assert_eq!(record.samples, (2.5 * SAMPLE_RATE as f64) as u64);
        assert_eq!(record.engine.as_deref(), Some("parakeet-rs"));
        assert_eq!(record.device.as_deref(), Some("Mic"));
        assert_eq!(record.inserted_text, "hallo welt");
        assert_eq!(record.target_app.as_deref(), Some("com.apple.TextEdit"));
        assert_eq!(record.label_status, "unreviewed");
        assert!(!record.training_eligible);
        assert_eq!(load_records(&root).unwrap(), vec![record.clone()]);
        let wavs = fs::read_dir(root.join(&record.audio_path).parent().unwrap())
            .unwrap()
            .count();
        assert_eq!(wavs, 1);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&root), 0o700);
            assert_eq!(mode(root.join(&record.audio_path).parent().unwrap()), 0o700);
            assert_eq!(mode(&root.join(&record.audio_path)), 0o600);
            assert_eq!(mode(&root.join(INDEX_FILE)), 0o600);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn review_audio_path_rejects_missing_modified_and_escaping_audio() {
        let root = temp_dir("review-path");
        let mut history = AudioHistory::from_config(&enabled_config(&root), None, None)
            .unwrap()
            .unwrap();
        let mut record = record_utterance(&mut history, 0.2, "test");
        let path = review_audio_path(&root, &record.id).unwrap();
        assert_eq!(
            path,
            fs::canonicalize(root.join(&record.audio_path)).unwrap()
        );
        assert!(review_audio_path(&root, "unknown").is_err());
        fs::write(&path, b"modified").unwrap();
        assert!(review_audio_path(&root, &record.id).is_err());
        fs::remove_file(&path).unwrap();
        assert!(review_audio_path(&root, &record.id).is_err());
        let outside = root.join("outside.wav");
        fs::write(&outside, b"outside").unwrap();
        record.audio_path = "outside.wav".into();
        record.sha256 = hex(&Sha256::digest(b"outside"));
        write_records(&root, &[record.clone()]).unwrap();
        assert!(review_audio_path(&root, &record.id).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, root.join("audio/link.wav")).unwrap();
            record.audio_path = "audio/link.wav".into();
            write_records(&root, &[record.clone()]).unwrap();
            assert!(review_audio_path(&root, &record.id).is_err());
        }
        write_records(&root, &[record.clone(), record.clone()]).unwrap();
        assert!(review_audio_path(&root, &record.id).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(feature = "client-vad")]
    fn client_history(root: &Path) -> AudioHistory {
        // Only synthetic tests use marker energy as an explicit probability
        // source. Production always loads the hash-verified Silero model.
        struct MarkerScorer;
        impl super::super::client_vad::SpeechScorer for MarkerScorer {
            fn reset(&mut self) {}
            fn score(&mut self, frame: &[f32]) -> Result<f32> {
                Ok(
                    if frame.iter().map(|v| v.abs()).sum::<f32>() / frame.len() as f32 > 0.05 {
                        0.9
                    } else {
                        0.0
                    },
                )
            }
        }
        let mut history = AudioHistory::from_config(&enabled_config(root), None, None)
            .unwrap()
            .unwrap();
        history.config.client_vad = Some(ClientVadConfig {
            model_path: String::new(),
            model_sha256: String::new(),
            runtime_path: String::new(),
        });
        history.capture_vad = Some(
            super::super::recording_vad::RecordingVad::new(Box::new(MarkerScorer), 120, 1000)
                .unwrap(),
        );
        history
    }

    #[cfg(feature = "client-vad")]
    #[test]
    fn client_vad_records_original_pcm_despite_delayed_recognition() {
        let root = temp_dir("client-delayed-asr");
        let mut history = client_history(&root);
        let mut audio = vec![0.0; 12_000];
        let onset = audio.len();
        audio.extend(vec![0.7; 2400]);
        audio.extend(vec![0.2; 48_000]);
        audio.extend(vec![-0.6; 2400]);
        let end = audio.len();
        audio.extend(vec![0.0; 24_000]);
        for chunk in audio.chunks(1920) {
            history.record_capture(chunk, &ctx("")).unwrap();
        }
        let records = load_records(&root).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        let span = record.capture_boundary.as_ref().unwrap();
        assert!(
            span.complete && span.start_sample <= onset as u64 && span.end_sample >= end as u64
        );
        let expected = encode_wav(
            &audio[span.start_sample as usize..span.end_sample as usize],
            SAMPLE_RATE,
        );
        assert_eq!(fs::read(root.join(&record.audio_path)).unwrap(), expected);
        assert!(record.raw_transcript.is_empty() && !record.training_eligible);
        // An ASR result delayed past the entire utterance cannot move its audio.
        history.begin_utterance();
        history.note_inserted("delayed text");
        history.record_recognition(&ctx("delayed text")).unwrap();
        assert_eq!(load_records(&root).unwrap(), records);
        let events = fs::read_to_string(root.join(RECOGNITION_FILE)).unwrap();
        assert!(events.contains("unassigned") && events.contains("delayed text"));
        let export_root = temp_dir("client-export");
        export(&root, &export_root).unwrap();
        assert_eq!(
            fs::read_to_string(export_root.join(RECOGNITION_FILE)).unwrap(),
            events
        );
        purge(
            &root,
            &PurgeFilter {
                id: Some(record.id.clone()),
                ..PurgeFilter::default()
            },
        )
        .unwrap();
        assert!(
            fs::read_to_string(root.join(RECOGNITION_FILE))
                .unwrap()
                .is_empty()
        );
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(export_root).unwrap();
    }

    #[cfg(feature = "client-vad")]
    #[test]
    fn client_vad_callback_partitions_do_not_change_retained_pcm() {
        let first = temp_dir("client-whole");
        let second = temp_dir("client-partitioned");
        let mut a = client_history(&first);
        let mut b = client_history(&second);
        let mut audio = vec![0.0; 12_000];
        audio.extend(vec![0.3; 24_000]);
        audio.extend(vec![0.0; 24_000]);
        a.record_capture(&audio, &ctx("")).unwrap();
        for chunk in audio.chunks(113) {
            b.record_capture(chunk, &ctx("")).unwrap();
        }
        let ar = load_records(&first).unwrap();
        let br = load_records(&second).unwrap();
        assert_eq!(ar.len(), 1);
        assert_eq!(br.len(), 1);
        assert_eq!(ar[0].capture_boundary, br[0].capture_boundary);
        assert_eq!(
            fs::read(first.join(&ar[0].audio_path)).unwrap(),
            fs::read(second.join(&br[0].audio_path)).unwrap()
        );
        fs::remove_dir_all(first).unwrap();
        fs::remove_dir_all(second).unwrap();
    }

    #[cfg(feature = "client-vad")]
    #[test]
    fn client_vad_cancel_discards_prefix_and_requires_new_onset() {
        let root = temp_dir("client-cancel");
        let mut history = client_history(&root);
        for _ in 0..20 {
            history.record_capture(&vec![0.7; 1920], &ctx("")).unwrap();
        }
        history.discard();
        for _ in 0..20 {
            history.record_capture(&vec![0.7; 1920], &ctx("")).unwrap();
        }
        for _ in 0..12 {
            history.record_capture(&vec![0.0; 1920], &ctx("")).unwrap();
        }
        assert!(load_records(&root).unwrap().is_empty());
        for _ in 0..20 {
            history.record_capture(&vec![0.2; 1920], &ctx("")).unwrap();
        }
        for _ in 0..12 {
            history.record_capture(&vec![0.0; 1920], &ctx("")).unwrap();
        }
        let records = load_records(&root).unwrap();
        assert_eq!(records.len(), 1);
        let wav = fs::read(root.join(&records[0].audio_path)).unwrap();
        assert!(
            wav[44..]
                .chunks_exact(2)
                .all(|b| i16::from_le_bytes([b[0], b[1]]).abs() < 7000)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(feature = "client-vad")]
    #[test]
    fn client_vad_privacy_transition_cancels_and_requires_quiet() {
        let root = temp_dir("client-privacy");
        let mut history = client_history(&root);
        for _ in 0..20 {
            history.record_capture(&vec![0.3; 1920], &ctx("")).unwrap();
        }
        let mut secure = ctx("");
        secure.secure_input = true;
        history.record_capture(&vec![0.99; 1920], &secure).unwrap();
        // Returning mid-word must not save a chopped suffix as a new utterance.
        for _ in 0..20 {
            history.record_capture(&vec![0.3; 1920], &ctx("")).unwrap();
        }
        assert!(load_records(&root).unwrap().is_empty());
        for _ in 0..12 {
            history.record_capture(&vec![0.0; 1920], &ctx("")).unwrap();
        }
        for _ in 0..20 {
            history.record_capture(&vec![0.2; 1920], &ctx("")).unwrap();
        }
        for _ in 0..12 {
            history.record_capture(&vec![0.0; 1920], &ctx("")).unwrap();
        }
        let records = load_records(&root).unwrap();
        assert_eq!(records.len(), 1);
        let wav = fs::read(root.join(&records[0].audio_path)).unwrap();
        let max = wav[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]).abs())
            .max()
            .unwrap();
        assert!(max < 7000); // No pre-transition or protected marker in this clip.
        history
            .record_recognition(&ctx("possibly stale private result"))
            .unwrap();
        assert!(!root.join(RECOGNITION_FILE).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(feature = "client-vad")]
    #[test]
    fn client_vad_release_is_marked_incomplete_and_unknown_context_is_not_saved() {
        let root = temp_dir("client-release");
        let mut history = client_history(&root);
        history
            .record_capture(&vec![0.8; 24_000], &CommitContext::default())
            .unwrap();
        assert_eq!(history.tap.position(), 0);
        history.begin_session();
        for _ in 0..20 {
            history.record_capture(&vec![0.2; 1920], &ctx("")).unwrap();
        }
        history.end_capture(&ctx("")).unwrap();
        let records = load_records(&root).unwrap();
        assert_eq!(records.len(), 1);
        let boundary = records[0].capture_boundary.as_ref().unwrap();
        assert!(!boundary.complete && boundary.end_sample == 20 * 1920);
        assert!(!records[0].training_eligible);
        purge(
            &root,
            &PurgeFilter {
                all: true,
                ..PurgeFilter::default()
            },
        )
        .unwrap();
        assert!(!root.join(RECOGNITION_FILE).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn consecutive_utterances_do_not_overlap() {
        let root = temp_dir("overlap");
        let mut history = AudioHistory::from_config(&enabled_config(&root), None, None)
            .unwrap()
            .unwrap();
        let first = record_utterance(&mut history, 1.0, "eins");
        // The next pre-roll would reach back into the first utterance.
        history.begin_utterance();
        history.tap().push(&tone(1.0));
        let second = history.commit(&ctx("zwei")).unwrap().unwrap();
        assert!((first.duration_secs - 1.5).abs() < 1e-6);
        assert!(
            (second.duration_secs - 1.0).abs() < 1e-6,
            "{}",
            second.duration_secs
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn excluded_apps_and_secure_input_write_nothing() {
        let root = temp_dir("excluded");
        let mut history = AudioHistory::from_config(&enabled_config(&root), None, None)
            .unwrap()
            .unwrap();

        history.begin_utterance();
        history.tap().push(&tone(1.0));
        let mut excluded = ctx("secret");
        excluded.app_bundle_id = Some("com.1password.1password".into());
        assert_eq!(
            history.commit(&excluded).unwrap(),
            Err(Skip::ExcludedApp("com.1password.1password".into()))
        );

        history.begin_utterance();
        history.tap().push(&tone(1.0));
        let mut secure = ctx("secret");
        secure.secure_input = true;
        assert_eq!(history.commit(&secure).unwrap(), Err(Skip::SecureInput));

        assert!(load_records(&root).unwrap().is_empty());
        assert!(!root.join("audio").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn discarded_utterance_writes_nothing() {
        let root = temp_dir("discard");
        let mut history = AudioHistory::from_config(&enabled_config(&root), None, None)
            .unwrap()
            .unwrap();
        history.begin_utterance();
        history.tap().push(&tone(1.0));
        history.discard();
        assert_eq!(history.commit(&ctx("late")).unwrap(), Err(Skip::NoAudio));
        assert!(load_records(&root).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn size_and_age_limits_evict_oldest_first() {
        let root = temp_dir("limit");
        let mut history = AudioHistory::from_config(&enabled_config(&root), None, None)
            .unwrap()
            .unwrap();
        let records: Vec<_> = (0..4)
            .map(|i| record_utterance(&mut history, 1.0, &format!("u{i}")))
            .collect();
        let two = records[2].bytes + records[3].bytes;

        let removed = apply_limits(&root, 0, two).unwrap();
        assert_eq!(removed, records[..2].to_vec());
        assert_eq!(load_records(&root).unwrap(), records[2..].to_vec());
        assert!(!root.join(&records[0].audio_path).exists());
        assert!(root.join(&records[3].audio_path).exists());

        // Age limit: rewrite one record as old and let the policy drop it.
        let mut aged = load_records(&root).unwrap();
        aged[0].recorded_at = "2020-01-01T00:00:00Z".into();
        write_records(&root, &aged).unwrap();
        let removed = apply_limits(&root, 30, u64::MAX).unwrap();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].id, records[2].id);
        assert_eq!(load_records(&root).unwrap(), records[3..].to_vec());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn purge_removes_audio_and_metadata_together() {
        let root = temp_dir("purge");
        let mut history = AudioHistory::from_config(&enabled_config(&root), None, None)
            .unwrap()
            .unwrap();
        let a = record_utterance(&mut history, 0.5, "a");
        history.begin_session();
        let b = record_utterance(&mut history, 0.5, "b");
        let c = record_utterance(&mut history, 0.5, "c");

        assert!(
            purge(&root, &PurgeFilter::default()).is_err(),
            "unfiltered purge must refuse"
        );

        let removed = purge(
            &root,
            &PurgeFilter {
                id: Some(a.id.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(removed, vec![a.clone()]);
        assert!(!root.join(&a.audio_path).exists());

        let removed = purge(
            &root,
            &PurgeFilter {
                session_id: Some(b.session_id.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(removed, vec![b.clone(), c.clone()]);
        assert!(load_records(&root).unwrap().is_empty());
        assert!(!root.join(&c.audio_path).exists());

        let d = record_utterance(&mut history, 0.5, "d");
        let tomorrow = (time::OffsetDateTime::now_utc() + time::Duration::days(1))
            .date()
            .to_string();
        let removed = purge(
            &root,
            &PurgeFilter {
                before: Some(tomorrow),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(removed, vec![d.clone()]);
        assert!(
            purge(
                &root,
                &PurgeFilter {
                    before: Some("26-10-2026".into()),
                    ..Default::default()
                }
            )
            .is_err()
        );

        record_utterance(&mut history, 0.5, "e");
        purge(
            &root,
            &PurgeFilter {
                all: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(load_records(&root).unwrap().is_empty());
        assert!(!root.join("audio").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn export_groups_by_session_and_verifies_audio() {
        let root = temp_dir("export");
        let out = temp_dir("export-out").join("set");
        let mut history = AudioHistory::from_config(&enabled_config(&root), None, None)
            .unwrap()
            .unwrap();
        let a = record_utterance(&mut history, 0.5, "a");
        history.begin_session();
        let b = record_utterance(&mut history, 0.5, "b");

        assert_eq!(export(&root, &out).unwrap(), 2);
        let manifest: Vec<serde_json::Value> = fs::read_to_string(out.join("manifest.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(manifest.len(), 2);
        for entry in &manifest {
            let record = [&a, &b].into_iter().find(|r| r.id == entry["id"]).unwrap();
            assert_eq!(entry["split_group"], record.session_id);
            assert_eq!(entry["split"], split_for_session(&record.session_id));
            assert_eq!(entry["sha256"], record.sha256);
            assert_eq!(entry["training_eligible"], false);
            let copied = fs::read(out.join(entry["audio_path"].as_str().unwrap())).unwrap();
            assert_eq!(hex(&Sha256::digest(&copied)), record.sha256);
        }
        // Entries of one session are contiguous.
        let sessions: Vec<_> = manifest.iter().map(|e| e["session_id"].clone()).collect();
        assert_ne!(sessions[0], sessions[1]);

        // A tampered file fails the export instead of shipping bad data.
        fs::write(root.join(&a.audio_path), b"tampered").unwrap();
        assert!(export(&root, &out).is_err());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(out.parent().unwrap()).unwrap();
    }

    #[test]
    fn refuses_a_directory_inside_a_git_repository() {
        let root = temp_dir("git");
        fs::create_dir_all(root.join(".git")).unwrap();
        let err = AudioHistory::from_config(&enabled_config(&root.join("history")), None, None)
            .err()
            .unwrap();
        assert!(err.to_string().contains("git repository"), "{err}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tap_slices_by_absolute_position_across_rotation() {
        let tap = AudioTap::new(1);
        tap.push(&vec![0.1; SAMPLE_RATE as usize]);
        tap.push(&vec![0.2; SAMPLE_RATE as usize / 2]);
        assert_eq!(tap.position(), SAMPLE_RATE as u64 * 3 / 2);
        // The first half second has rotated out.
        let all = tap.slice(0, tap.position());
        assert_eq!(all.len(), SAMPLE_RATE as usize);
        assert_eq!(all[0], 0.1);
        assert_eq!(*all.last().unwrap(), 0.2);
    }
}
