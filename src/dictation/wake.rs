//! Wake-phrase triggers for dictation through an external detector process.
//!
//! eaRS stays the only microphone owner. While dictation is paused, the captured audio
//! (24 kHz) is resampled to 16 kHz and streamed to a detector command speaking
//! `wake-stream/1` (see the `wake` repository, `src/protocol.rs`): binary PCM frames on
//! stdin, JSON-line detections on stdout. Which model runs is entirely the command's
//! business; eaRS learns the phrases from the `ready` line and maps them to actions.
//!
//! A detection is evidence that a phrase was probably spoken. The only action it can
//! take here is what the toggle hotkey can already do: start capture.

use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, unbounded};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::Duration;

pub const PROTOCOL: &str = "wake-stream/1";
const CAPTURE_RATE: usize = 24_000;
const DETECTOR_RATE: usize = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeAction {
    /// Same as pressing the toggle hotkey while paused.
    StartDictation,
    /// Recognised but deliberately unused (e.g. logged for evaluation).
    Ignore,
}

/// `[dictation.wake]`. Off by default; nothing runs unless `enabled` and `command` are set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WakeConfig {
    pub enabled: bool,
    /// Detector process and arguments, e.g.
    /// `["wake", "listen", "--manifest", "/path/to/release"]`.
    pub command: Vec<String>,
    /// Phrase reported by the detector -> action. Phrases not listed are ignored.
    pub actions: HashMap<String, WakeAction>,
    /// Optional extra floor on top of the detector's own threshold.
    pub min_score: Option<f32>,
    /// Audio before the detection sent to the server after a wake, so the first words
    /// after the phrase are not lost. The phrase itself is then stripped (`strip_words`).
    pub preroll_ms: u32,
    /// Leading words dropped once after a wake (the wake phrase as the ASR spells it).
    pub strip_words: Vec<String>,
    /// Whole utterances that pause capture without typing (e.g. "oqto stopp"). Works
    /// with or without a detector; matched after punctuation is removed.
    pub stop_phrases: Vec<String>,
    /// Write every detection and near miss (no audio) to a local JSONL log.
    pub log: bool,
    /// Log path; empty = $XDG_STATE_HOME/ears/wake-log.jsonl.
    pub log_path: String,
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            command: Vec::new(),
            actions: HashMap::new(),
            min_score: None,
            preroll_ms: 1500,
            strip_words: [
                "hey", "hi", "he", "hej", "ey", "oqto", "okto", "octo", "otto", "ok", "to", "okay",
            ]
            .map(String::from)
            .to_vec(),
            stop_phrases: Vec::new(),
            log: true,
            log_path: String::new(),
        }
    }
}

impl WakeConfig {
    pub fn active(&self) -> bool {
        self.enabled && !self.command.is_empty()
    }

    /// The action for a detection, if any.
    pub fn decide(&self, phrase: &str, score: f32) -> Option<WakeAction> {
        if self.min_score.is_some_and(|floor| score < floor) {
            return None;
        }
        self.actions
            .get(phrase)
            .copied()
            .filter(|a| *a != WakeAction::Ignore)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Detection {
    pub phrase: String,
    pub score: f32,
    pub at_sample: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Ready {
    pub protocol: String,
    pub sample_rate: u32,
    pub phrases: Vec<String>,
    #[serde(default)]
    pub detector: String,
    #[serde(default)]
    pub release: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WakeMessage {
    Detection(Detection),
    /// Near miss below the detector's threshold: logged, never acted upon.
    Candidate(Detection),
    Error(String),
    Exited,
}

/// Streaming windowed-sinc resampler from the 24 kHz capture rate to 16 kHz.
pub struct Resampler {
    history: Vec<f32>,
    position: f64, // input index of the next output sample, relative to `history`
    step: f64,
    cutoff: f64,
    half_width: usize,
}

impl Resampler {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            history: Vec::new(),
            position: 0.0,
            step: from as f64 / to as f64,
            cutoff: 0.95 * (to as f64 / from as f64).min(1.0),
            half_width: 16,
        }
    }

    fn kernel(&self, d: f64) -> f64 {
        let w = self.half_width as f64;
        if d.abs() >= w {
            return 0.0;
        }
        let x = std::f64::consts::PI * d * self.cutoff;
        let sinc = if d.abs() < 1e-9 { 1.0 } else { x.sin() / x };
        let hann = 0.5 + 0.5 * (std::f64::consts::PI * d / w).cos();
        self.cutoff * sinc * hann
    }

    pub fn push(&mut self, input: &[f32]) -> Vec<f32> {
        self.history.extend_from_slice(input);
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 1);
        let hw = self.half_width as f64;
        while self.position + hw < self.history.len() as f64 {
            let center = self.position.floor() as isize;
            let mut acc = 0.0;
            for k in (center - self.half_width as isize + 1)..=(center + self.half_width as isize) {
                if k < 0 || k as usize >= self.history.len() {
                    continue;
                }
                acc += self.history[k as usize] as f64 * self.kernel(self.position - k as f64);
            }
            out.push(acc as f32);
            self.position += self.step;
        }
        let keep_from = (self.position.floor() as usize).saturating_sub(self.half_width);
        if keep_from > 0 {
            self.history.drain(..keep_from);
            self.position -= keep_from as f64;
        }
        out
    }
}

/// A running detector process.
pub struct WakeClient {
    child: Child,
    stdin: ChildStdin,
    resampler: Resampler,
    pub ready: Ready,
    pub messages: Receiver<WakeMessage>,
}

fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

impl WakeClient {
    pub fn spawn(command: &[String], ready_timeout: Duration) -> Result<Self> {
        let (program, args) = command
            .split_first()
            .ok_or_else(|| anyhow!("wake command is empty"))?;
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("starting wake detector {program}"))?;
        let stdin = child.stdin.take().context("detector stdin")?;
        let stdout = child.stdout.take().context("detector stdout")?;
        let (tx, rx): (Sender<WakeMessage>, Receiver<WakeMessage>) = unbounded();
        let (ready_tx, ready_rx) = unbounded::<Result<Ready, String>>();
        std::thread::Builder::new()
            .name("ears-wake-reader".into())
            .spawn(move || read_events(BufReader::new(stdout), ready_tx, tx))
            .context("spawning wake reader thread")?;
        let ready = match ready_rx.recv_timeout(ready_timeout) {
            Ok(Ok(ready)) => ready,
            Ok(Err(message)) => {
                let _ = child.kill();
                bail!("wake detector did not start: {message}");
            }
            Err(_) => {
                let _ = child.kill();
                bail!("wake detector sent no ready line within {ready_timeout:?}");
            }
        };
        if ready.protocol != PROTOCOL || ready.sample_rate as usize != DETECTOR_RATE {
            let _ = child.kill();
            bail!(
                "wake detector speaks {} at {} Hz; eaRS needs {PROTOCOL} at {DETECTOR_RATE} Hz",
                ready.protocol,
                ready.sample_rate
            );
        }
        Ok(Self {
            child,
            stdin,
            resampler: Resampler::new(CAPTURE_RATE, DETECTOR_RATE),
            ready,
            messages: rx,
        })
    }

    /// Feed captured 24 kHz audio.
    pub fn push_capture(&mut self, pcm: &[f32]) -> Result<()> {
        let resampled = self.resampler.push(pcm);
        if resampled.is_empty() {
            return Ok(());
        }
        let bytes: Vec<u8> = resampled
            .iter()
            .flat_map(|&x| ((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
            .collect();
        self.stdin.write_all(&frame(b'A', &bytes))?;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.resampler = Resampler::new(CAPTURE_RATE, DETECTOR_RATE);
        self.stdin.write_all(&frame(b'R', &[]))?;
        Ok(())
    }

    pub fn set_playback(&mut self, active: bool) -> Result<()> {
        self.stdin.write_all(&frame(b'P', &[active as u8]))?;
        Ok(())
    }
}

impl Drop for WakeClient {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(&frame(b'Q', &[]));
        let _ = self.stdin.flush();
        std::thread::sleep(Duration::from_millis(50));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_events(
    stdout: impl BufRead,
    ready: Sender<Result<Ready, String>>,
    events: Sender<WakeMessage>,
) {
    let mut readied = false;
    for line in stdout.lines() {
        let Ok(line) = line else { break };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue; // detectors log to stderr; tolerate stray stdout lines
        };
        match value.get("type").and_then(|t| t.as_str()) {
            Some("ready") if !readied => {
                readied = true;
                let _ = ready.send(serde_json::from_value(value).map_err(|e| e.to_string()));
            }
            Some(kind @ ("detection" | "candidate")) => {
                let candidate = kind == "candidate";
                if let Ok(d) = serde_json::from_value::<Detection>(value) {
                    let _ = events.send(if candidate {
                        WakeMessage::Candidate(d)
                    } else {
                        WakeMessage::Detection(d)
                    });
                }
            }
            Some("error") => {
                let message = value
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("detector error")
                    .to_string();
                if readied {
                    let _ = events.send(WakeMessage::Error(message));
                } else {
                    let _ = ready.send(Err(message));
                    return;
                }
            }
            _ => {}
        }
    }
    if !readied {
        let _ = ready.send(Err("detector exited before ready".into()));
    }
    let _ = events.send(WakeMessage::Exited);
}

/// Local JSONL log of detections and near misses (no audio). Failures disable it quietly.
struct WakeLog(Option<std::fs::File>);

impl WakeLog {
    fn open(config: &WakeConfig) -> Self {
        if !config.log {
            return Self(None);
        }
        let path = if config.log_path.is_empty() {
            let state = std::env::var_os("XDG_STATE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| dirs::home_dir().map(|h| h.join(".local/state")));
            match state {
                Some(dir) => dir.join("ears").join("wake-log.jsonl"),
                None => return Self(None),
            }
        } else {
            std::path::PathBuf::from(&config.log_path)
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        Self(options.open(&path).ok())
    }

    fn write(&mut self, kind: &str, d: &Detection, outcome: &str) {
        let line = serde_json::json!({
            "at": time::OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
            "type": kind, "phrase": d.phrase, "score": d.score, "at_sample": d.at_sample, "outcome": outcome,
        });
        if let Some(file) = self.0.as_mut() {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Pre-roll position requested by a wake: detector-domain (16 kHz) sample of the detection.
type PrerollRequest = std::sync::Arc<std::sync::Mutex<Option<u64>>>;

/// Start the detector and the tee. While capture is paused, captured audio goes to the
/// detector and into a short ring buffer, not to the engine. When a mapped phrase starts
/// capture, the ring from `preroll_ms` before the detection is sent to the engine first,
/// so words right after the wake phrase are not lost; while capturing, every chunk goes
/// to the engine as before. If the detector cannot start, audio flows to the engine
/// exactly as without wake support and the failure is reported.
pub(crate) fn spawn_tee(
    config: &WakeConfig,
    capture_rx: Receiver<Vec<f32>>,
    engine_tx: Sender<Vec<f32>>,
    shared: std::sync::Arc<super::Shared>,
) -> Result<()> {
    let client = match WakeClient::spawn(&config.command, Duration::from_secs(30)) {
        Ok(client) => {
            let missing: Vec<&String> = config
                .actions
                .keys()
                .filter(|p| !client.ready.phrases.contains(p))
                .collect();
            eprintln!(
                "wake detector ready: {} {} phrases {:?}{}",
                client.ready.detector,
                client.ready.release,
                client.ready.phrases,
                if missing.is_empty() {
                    String::new()
                } else {
                    format!(" (actions for unknown phrases {missing:?})")
                }
            );
            Some(client)
        }
        Err(err) => {
            shared.emit(super::DictationEvent::Error {
                message: format!("wake detector unavailable: {err:#}"),
            });
            None
        }
    };
    let preroll: PrerollRequest = Default::default();

    if let Some(client) = &client {
        let messages = client.messages.clone();
        let config = config.clone();
        let shared = shared.clone();
        let preroll = preroll.clone();
        let mut log = WakeLog::open(&config);
        std::thread::Builder::new()
            .name("ears-wake-events".into())
            .spawn(move || {
                for message in messages {
                    match message {
                        WakeMessage::Detection(d) => {
                            let outcome = if shared.is_capturing() {
                                "already_capturing"
                            } else {
                                match config.decide(&d.phrase, d.score) {
                                    Some(WakeAction::StartDictation) => {
                                        *preroll.lock().unwrap() = Some(d.at_sample);
                                        shared.mark_wake_started();
                                        if shared.set_capturing(true) {
                                            shared.emit(super::DictationEvent::Wake {
                                                phrase: d.phrase.clone(),
                                                score: d.score,
                                            });
                                            "started"
                                        } else {
                                            "already_capturing"
                                        }
                                    }
                                    _ if config.actions.get(&d.phrase)
                                        == Some(&WakeAction::Ignore) =>
                                    {
                                        "ignored"
                                    }
                                    _ if config.actions.contains_key(&d.phrase) => {
                                        "below_min_score"
                                    }
                                    _ => "unmapped",
                                }
                            };
                            eprintln!("wake: {} {:.2} -> {outcome}", d.phrase, d.score);
                            log.write("detection", &d, outcome);
                        }
                        WakeMessage::Candidate(d) => {
                            if !shared.is_capturing() {
                                eprintln!("wake: near miss {} {:.2}", d.phrase, d.score);
                                log.write("candidate", &d, "none");
                            }
                        }
                        WakeMessage::Error(message) => shared.emit(super::DictationEvent::Error {
                            message: format!("wake detector: {message}"),
                        }),
                        WakeMessage::Exited => {
                            shared.emit(super::DictationEvent::Error {
                                message: "wake detector exited".into(),
                            });
                            break;
                        }
                    }
                }
            })
            .context("spawning wake event thread")?;
    }

    let preroll_samples = config.preroll_ms as u64 * CAPTURE_RATE as u64 / 1000;
    std::thread::Builder::new()
        .name("ears-wake-tee".into())
        .spawn(move || {
            let mut client = client;
            let mut was_capturing = true; // reset the detector on the first paused chunk
            let mut ring: std::collections::VecDeque<f32> = std::collections::VecDeque::new();
            let ring_cap = CAPTURE_RATE * 4;
            let mut fed: u64 = 0; // capture-rate samples fed to the detector since its last reset
            for chunk in capture_rx {
                let capturing = shared.is_capturing();
                if client.is_none() {
                    // No detector: behave exactly like eaRS without wake support.
                    if engine_tx.send(chunk).is_err() {
                        break;
                    }
                    continue;
                }
                if capturing {
                    if !was_capturing {
                        if let Some(at) = preroll.lock().unwrap().take() {
                            let at_capture = at * CAPTURE_RATE as u64 / DETECTOR_RATE as u64;
                            let start = at_capture.saturating_sub(preroll_samples);
                            let ring_start = fed - ring.len() as u64;
                            let skip = start.saturating_sub(ring_start) as usize;
                            let audio: Vec<f32> = ring.iter().skip(skip).copied().collect();
                            for piece in audio.chunks(CAPTURE_RATE / 50) {
                                if engine_tx.send(piece.to_vec()).is_err() {
                                    return;
                                }
                            }
                        }
                        ring.clear();
                    }
                    was_capturing = true;
                    if engine_tx.send(chunk).is_err() {
                        break;
                    }
                    continue;
                }
                let detector = client.as_mut().unwrap();
                let mut result = Ok(());
                if was_capturing {
                    result = detector.reset();
                    ring.clear();
                    fed = 0;
                    preroll.lock().unwrap().take();
                }
                if result.is_ok() {
                    result = detector.push_capture(&chunk);
                }
                if let Err(err) = result {
                    shared.emit(super::DictationEvent::Error {
                        message: format!("wake detector stopped: {err}"),
                    });
                    client = None;
                }
                fed += chunk.len() as u64;
                ring.extend(chunk.iter().copied());
                while ring.len() > ring_cap {
                    ring.pop_front();
                }
                was_capturing = false;
            }
        })
        .context("spawning wake tee thread")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_maps_phrases_and_respects_floor() {
        let mut config = WakeConfig {
            enabled: true,
            command: vec!["wake".into()],
            ..Default::default()
        };
        config
            .actions
            .insert("hey_oqto".into(), WakeAction::StartDictation);
        config.actions.insert("oqto".into(), WakeAction::Ignore);
        assert_eq!(
            config.decide("hey_oqto", 0.9),
            Some(WakeAction::StartDictation)
        );
        assert_eq!(config.decide("oqto", 0.9), None);
        assert_eq!(config.decide("stopp", 0.9), None);
        config.min_score = Some(0.7);
        assert_eq!(config.decide("hey_oqto", 0.6), None);
        assert!(!WakeConfig::default().active());
    }

    #[test]
    fn config_parses_from_toml() {
        let config: WakeConfig = toml::from_str(
            "enabled = true\ncommand = [\"wake\", \"listen\", \"--manifest\", \"/m\"]\n[actions]\nhey_oqto = \"start_dictation\"\n",
        )
        .unwrap();
        assert!(config.active());
        assert_eq!(config.actions["hey_oqto"], WakeAction::StartDictation);
    }

    #[test]
    fn resampler_keeps_rate_and_tone() {
        let mut r = Resampler::new(24_000, 16_000);
        let tone: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24_000.0).sin() * 0.5)
            .collect();
        let mut out = Vec::new();
        for chunk in tone.chunks(480) {
            out.extend(r.push(chunk)); // capture-sized chunks
        }
        assert!(
            (out.len() as i64 - 16_000).abs() < 40,
            "length {}",
            out.len()
        );
        // 1 kHz survives at the right frequency: correlate with a 16 kHz reference tone
        let body = &out[200..out.len() - 200];
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for (i, &x) in body.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * 1000.0 * (i + 200) as f64 / 16_000.0;
            s += x as f64 * phase.sin();
            c += x as f64 * phase.cos();
        }
        let amplitude = 2.0 * (s * s + c * c).sqrt() / body.len() as f64;
        assert!((amplitude - 0.5).abs() < 0.03, "amplitude {amplitude}");
        // 10 kHz (above the new Nyquist) is suppressed rather than aliased to 6 kHz
        let mut r = Resampler::new(24_000, 16_000);
        let high: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * 10_000.0 * i as f32 / 24_000.0).sin() * 0.5)
            .collect();
        let out = r.push(&high);
        let rms = (out[100..out.len() - 100]
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            / (out.len() - 200) as f64)
            .sqrt();
        assert!(rms < 0.02, "alias rms {rms}");
    }

    fn fake_detector(dir: &std::path::Path, detections: &str) -> Vec<String> {
        // Waits for some audio, then reports the given detections: proves the tee feeds it.
        let script = dir.join("fake_detector.py");
        std::fs::write(
            &script,
            format!(
                "import sys, json\nprint(json.dumps({{'type':'ready','protocol':'wake-stream/1','sample_rate':16000,'phrases':['hey_oqto','oqto'],'detector':'fake','release':'t'}}), flush=True)\n\
                 n = 0\nwhile n < 32000:\n    head = sys.stdin.buffer.read(5)\n    if len(head) < 5: sys.exit(0)\n    size = int.from_bytes(head[1:], 'little')\n    sys.stdin.buffer.read(size)\n    n += size if head[:1] == b'A' else 0\n\
                 for line in {detections:?}.splitlines(): print(line, flush=True)\nsys.stdin.buffer.read()\n"
            ),
        )
        .unwrap();
        vec!["python3".into(), script.to_string_lossy().into_owned()]
    }

    /// Runs the tee with a fake detector. Each captured sample's value encodes its index
    /// (index * 1e-6), so the engine side can tell exactly which audio it received.
    fn run_tee(
        command: Vec<String>,
        actions: &[(&str, WakeAction)],
        preroll_ms: u32,
    ) -> (bool, Vec<super::super::DictationEvent>, Vec<f32>) {
        use super::super::{DictationEvent, Shared};
        let mut wake = WakeConfig {
            enabled: true,
            command,
            preroll_ms,
            log: false,
            ..Default::default()
        };
        for (phrase, action) in actions {
            wake.actions.insert((*phrase).into(), *action);
        }
        let config = crate::config::AppConfig::default();
        let (events_tx, events_rx) = unbounded();
        let (commands_tx, _commands_rx) = unbounded();
        let shared = std::sync::Arc::new(Shared::new(&config, true, events_tx, commands_tx, false));
        let (capture_tx, capture_rx) = unbounded();
        let (engine_tx, engine_rx) = unbounded();
        spawn_tee(&wake, capture_rx, engine_tx, shared.clone()).unwrap();
        let mut index = 0usize;
        let mut chunk = || {
            let c: Vec<f32> = (index..index + 480).map(|i| i as f32 * 1e-6).collect();
            index += 480;
            c
        };
        for _ in 0..100 {
            capture_tx.send(chunk()).unwrap(); // 2 s paused
        }
        std::thread::sleep(Duration::from_millis(800)); // the fake detector reports after 1 s of audio
        for _ in 0..10 {
            capture_tx.send(chunk()).unwrap(); // 0.2 s more, after the wake (if any)
        }
        std::thread::sleep(Duration::from_millis(300));
        let received: Vec<f32> = engine_rx.try_iter().flatten().collect();
        let events: Vec<DictationEvent> = events_rx.try_iter().collect();
        (shared.is_capturing(), events, received)
    }

    #[test]
    fn detection_starts_dictation_with_preroll_and_unmapped_phrases_do_not() {
        use super::super::DictationEvent;
        let dir = std::env::temp_dir().join(format!("ears-wake-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // detection at 16 kHz sample 16000 = capture sample 24000 (1.0 s)
        let line = r#"{"type":"detection","phrase":"hey_oqto","score":0.91,"at_sample":16000}"#;
        let (capturing, events, received) = run_tee(
            fake_detector(&dir, line),
            &[("hey_oqto", WakeAction::StartDictation)],
            500,
        );
        assert!(capturing, "a mapped detection starts capture");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DictationEvent::Wake { phrase, .. } if phrase == "hey_oqto"))
        );
        let first = (received[0] / 1e-6).round() as i64;
        assert!(
            (first - 12_000).abs() < 50,
            "pre-roll starts 500 ms before the detection, got sample {first}"
        );
        let mut previous = first - 1;
        for &x in &received {
            let i = (x / 1e-6).round() as i64;
            assert_eq!(
                i,
                previous + 1,
                "pre-roll then live audio, contiguous, no duplicates"
            );
            previous = i;
        }
        assert_eq!(
            previous,
            110 * 480 - 1,
            "every chunk after the wake reached the engine"
        );

        let other = r#"{"type":"detection","phrase":"oqto","score":0.99,"at_sample":16000}"#;
        let (capturing, events, received) = run_tee(
            fake_detector(&dir, other),
            &[
                ("hey_oqto", WakeAction::StartDictation),
                ("oqto", WakeAction::Ignore),
            ],
            500,
        );
        assert!(!capturing, "an ignored phrase does nothing");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, DictationEvent::Wake { .. }))
        );
        assert!(received.is_empty(), "paused audio never reaches the engine");

        let (capturing, events, received) = run_tee(
            vec!["/nonexistent/wake-detector".into()],
            &[("hey_oqto", WakeAction::StartDictation)],
            500,
        );
        assert!(!capturing);
        assert!(events.iter().any(|e| matches!(e, DictationEvent::Error { message } if message.contains("wake detector unavailable"))));
        assert_eq!(
            received.len(),
            110 * 480,
            "without a detector, audio flows to the engine as before"
        );
    }

    /// Real detector over the real protocol, with 24 kHz input as captured by eaRS.
    /// Set WAKE_BIN, WAKE_MANIFEST and WAKE_CLIP_24K (mono 24 kHz WAV containing a wake phrase).
    #[test]
    fn real_detector_hears_a_24k_clip() {
        let (Ok(bin), Ok(manifest), Ok(clip)) = (
            std::env::var("WAKE_BIN"),
            std::env::var("WAKE_MANIFEST"),
            std::env::var("WAKE_CLIP_24K"),
        ) else {
            eprintln!("skipped: set WAKE_BIN, WAKE_MANIFEST, WAKE_CLIP_24K");
            return;
        };
        // canonical 44-byte header, PCM16 mono (e.g. written by Python's wave module)
        let bytes = std::fs::read(&clip).unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            24_000,
            "clip must be 24 kHz"
        );
        let pcm: Vec<f32> = bytes[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        let mut client = WakeClient::spawn(
            &[bin, "listen".into(), "--manifest".into(), manifest],
            Duration::from_secs(30),
        )
        .unwrap();
        let silence = vec![0.0f32; 24_000];
        client.push_capture(&silence).unwrap();
        for chunk in pcm.chunks(480) {
            client.push_capture(chunk).unwrap();
        }
        client.push_capture(&silence).unwrap();
        let mut found = Vec::new();
        while let Ok(WakeMessage::Detection(d)) =
            client.messages.recv_timeout(Duration::from_secs(3))
        {
            found.push(d);
        }
        eprintln!("detections: {found:?}");
        assert!(
            found
                .iter()
                .any(|d| d.phrase == "hey_oqto" && d.score >= 0.5)
        );
    }

    #[test]
    fn reader_reports_ready_then_detections() {
        let input = "not json\n{\"type\":\"ready\",\"protocol\":\"wake-stream/1\",\"sample_rate\":16000,\"phrases\":[\"hey_oqto\"]}\n{\"type\":\"detection\",\"phrase\":\"hey_oqto\",\"score\":0.9,\"at_sample\":32000}\n";
        let (ready_tx, ready_rx) = unbounded();
        let (tx, rx) = unbounded();
        read_events(BufReader::new(input.as_bytes()), ready_tx, tx);
        let ready = ready_rx.recv().unwrap().unwrap();
        assert_eq!(ready.phrases, vec!["hey_oqto"]);
        assert_eq!(
            rx.recv().unwrap(),
            WakeMessage::Detection(Detection {
                phrase: "hey_oqto".into(),
                score: 0.9,
                at_sample: 32000
            })
        );
        assert_eq!(rx.recv().unwrap(), WakeMessage::Exited);
    }
}
