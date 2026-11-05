//! Parakeet Ultra (moondream) as a server engine.
//!
//! Ultra is the same 0.6B TDT architecture as v3 but ships as safetensors for a
//! Python runtime, and TDT is not cache-aware, so chunked streaming degrades.
//! eaRS therefore treats it as an utterance-level engine: audio is buffered for
//! one utterance, written to a temporary WAV, and handed to an operator-provided
//! command that prints `{"text": "..."}` (optionally `{"words": [...]}`).
//! Nothing is downloaded, no model is loaded into this process, and a missing or
//! failing command is reported instead of guessed.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, bounded};

use super::SessionSink;
use super::engine::{Engine, EngineKind, EngineSession, queue_command};
use crate::{TranscriptionSink, WebSocketMessage, WordTimestamp};

/// 24 kHz mono f32, as the capture client sends.
const RATE: u32 = 24_000;

#[derive(Debug, Clone, Default)]
pub struct ParakeetUltraConfig {
    /// Command that transcribes one WAV file, e.g.
    /// `["python", "/opt/parakeet-ultra/transcribe.py"]`. The WAV path is
    /// appended as the last argument. Empty disables the engine.
    pub command: Vec<String>,
    /// Language hint passed as `--language` when set.
    pub language: Option<String>,
    /// Bound on one transcription, including model load in the worker.
    pub timeout_ms: u64,
    /// Longest utterance kept (seconds); longer audio is refused, not truncated.
    pub max_utterance_secs: u32,
    /// Where temporary WAVs are written (removed after each utterance).
    pub temp_dir: Option<String>,
}

pub struct ParakeetUltraEngine {
    config: Arc<ParakeetUltraConfig>,
}

impl ParakeetUltraEngine {
    pub fn load(config: ParakeetUltraConfig) -> Result<Self> {
        if config.command.is_empty() {
            bail!("parakeet-ultra needs a transcription command");
        }
        let timeout_ms = if config.timeout_ms == 0 {
            60_000
        } else {
            config.timeout_ms
        };
        let max_utterance_secs = if config.max_utterance_secs == 0 {
            120
        } else {
            config.max_utterance_secs
        };
        let config = ParakeetUltraConfig {
            timeout_ms,
            max_utterance_secs,
            ..config
        };
        eprintln!(
            "[parakeet-ultra] utterance engine via {:?}, timeout {} ms, max {} s",
            config.command, config.timeout_ms, config.max_utterance_secs
        );
        Ok(Self {
            config: Arc::new(config),
        })
    }
}

impl Engine for ParakeetUltraEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::ParakeetUltra
    }

    fn allocate(&self, sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
        let (tx, rx) = bounded(16);
        let config = self.config.clone();
        std::thread::Builder::new()
            .name("ultra-session".into())
            .spawn(move || worker(config, sink, rx))?;
        Ok(Some(Box::new(UltraSession { tx })))
    }
}

enum Command {
    /// Captured 24 kHz mono samples.
    Audio(Vec<f32>),
    /// The utterance ended: transcribe what was buffered.
    Finish,
    /// Discard the buffered utterance without transcribing.
    Cancel,
}

struct UltraSession {
    tx: Sender<Command>,
}

impl EngineSession for UltraSession {
    fn engine(&self) -> EngineKind {
        EngineKind::ParakeetUltra
    }

    fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        queue_command(&self.tx, Command::Audio(pcm))
    }

    fn set_language(&self, lang: String) -> Result<()> {
        // Applied on the next utterance by the worker's own config; refusing
        // loudly is better than silently ignoring a requested language.
        bail!("parakeet-ultra language is fixed per server ({lang} requested)")
    }

    fn request_stop(&self) {
        // Dropping the session buffer is enough: nothing is left running.
        let _ = self.tx.try_send(Command::Cancel);
    }

    /// The language is fixed per server: the worker's command owns it.
    fn supports_language(&self) -> bool {
        false
    }

    /// The server's acoustic boundary ends the utterance: TDT is not
    /// cache-aware, so one bounded utterance is transcribed at a time.
    fn send_speech_boundary(&self, active: bool) -> Result<bool> {
        if !active {
            queue_command(&self.tx, Command::Finish)?;
        }
        Ok(false)
    }
}

impl Drop for UltraSession {
    fn drop(&mut self) {
        let _ = self.tx.try_send(Command::Cancel);
    }
}

fn worker(config: Arc<ParakeetUltraConfig>, mut sink: SessionSink, rx: Receiver<Command>) {
    let limit = RATE as usize * config.max_utterance_secs as usize;
    let mut pcm: Vec<f32> = Vec::new();
    for command in rx {
        match command {
            Command::Audio(chunk) => {
                if pcm.len().saturating_add(chunk.len()) > limit {
                    pcm.clear();
                    eprintln!(
                        "[parakeet-ultra] utterance exceeds {} s; discarded",
                        config.max_utterance_secs
                    );
                    continue;
                }
                pcm.extend_from_slice(&chunk);
            }
            Command::Cancel => pcm.clear(),
            Command::Finish => {
                if pcm.is_empty() {
                    continue;
                }
                let audio = std::mem::take(&mut pcm);
                match transcribe(&config, &audio) {
                    Ok((text, words)) => {
                        if !text.trim().is_empty() {
                            sink.handle_message(WebSocketMessage::Final { text, words });
                        }
                    }
                    Err(error) => eprintln!("[parakeet-ultra] {error:#}"),
                }
            }
        }
    }
}

/// Run the configured command for one utterance. Returns the text and word
/// timestamps (empty when the worker reports only text).
fn transcribe(config: &ParakeetUltraConfig, pcm: &[f32]) -> Result<(String, Vec<WordTimestamp>)> {
    let dir = match &config.temp_dir {
        Some(path) => PathBuf::from(path),
        None => std::env::temp_dir(),
    };
    let path = write_wav(&dir, pcm)?;
    let mut command = std::process::Command::new(&config.command[0]);
    command.args(&config.command[1..]);
    if let Some(language) = &config.language {
        command.arg("--language").arg(language);
    }
    command.arg(&path);
    let output = run_with_timeout(command, Duration::from_millis(config.timeout_ms));
    let _ = std::fs::remove_file(&path);
    let output = output?;
    parse_result(&output)
}

fn write_wav(dir: &Path, samples: &[f32]) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let name = format!(
        "ultra-{}.wav",
        std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_micros() as u64)
                .unwrap_or(0)
    );
    let path = dir.join(name);
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let data_len = (samples.len() * 2) as u32;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_len).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?; // PCM
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&RATE.to_le_bytes())?;
    file.write_all(&(RATE * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        file.write_all(&((clamped * 32767.0) as i16).to_le_bytes())?;
    }
    file.flush()?;
    Ok(path)
}

/// Run a command with a bounded wait; the child is killed when it overruns.
fn run_with_timeout(mut command: std::process::Command, timeout: Duration) -> Result<String> {
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("starting {:?}", command.get_program()))?;
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait()?.is_some() {
            let output = child.wait_with_output()?;
            if !output.status.success() {
                bail!(
                    "transcription command failed ({}): {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            return Ok(String::from_utf8_lossy(&output.stdout).to_string());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "transcription command timed out after {} ms",
                timeout.as_millis()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Accept `{"text": "..."}` with optional `words: [{word, start_time, end_time}]`.
fn parse_result(stdout: &str) -> Result<(String, Vec<WordTimestamp>)> {
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).context("transcription command did not print JSON")?;
    let text = value
        .get("text")
        .and_then(|t| t.as_str())
        .context("transcription JSON has no text")?
        .to_string();
    let words = value
        .get("words")
        .and_then(|w| w.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    Some(WordTimestamp {
                        word: item.get("word")?.as_str()?.to_string(),
                        start_time: item.get("start_time")?.as_f64()?,
                        end_time: item.get("end_time").and_then(|e| e.as_f64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Ok((text, words))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(command: Vec<&str>) -> ParakeetUltraConfig {
        ParakeetUltraConfig {
            command: command.into_iter().map(String::from).collect(),
            timeout_ms: 5_000,
            max_utterance_secs: 5,
            ..Default::default()
        }
    }

    #[test]
    fn a_command_that_prints_json_yields_text_and_words() {
        let (text, words) = transcribe(
            &config(vec!["sh", "-c", r#"cat >/dev/null; echo '{"text":"hallo welt","words":[{"word":"hallo","start_time":0.0,"end_time":0.4}]}'"#]),
            &[0.1; 480],
        )
        .unwrap();
        assert_eq!(text, "hallo welt");
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "hallo");
    }

    #[test]
    fn failures_and_garbage_are_reported_not_guessed() {
        let error = transcribe(&config(vec!["sh", "-c", "exit 3"]), &[0.0; 48]).unwrap_err();
        assert!(error.to_string().contains("failed"), "{error}");
        let garbage =
            transcribe(&config(vec!["sh", "-c", "echo not-json"]), &[0.0; 48]).unwrap_err();
        assert!(garbage.to_string().contains("JSON"), "{garbage}");
        let no_text = transcribe(
            &config(vec!["sh", "-c", r#"echo '{"words":[]}'"#]),
            &[0.0; 48],
        )
        .unwrap_err();
        assert!(no_text.to_string().contains("text"), "{no_text}");
        // A missing program never panics.
        assert!(transcribe(&config(vec!["definitely-not-a-program"]), &[0.0; 48]).is_err());
    }

    #[test]
    fn a_hanging_command_times_out_and_is_killed() {
        let mut slow = config(vec!["sh", "-c", "sleep 30; echo '{\"text\":\"late\"}'"]);
        slow.timeout_ms = 250;
        let started = Instant::now();
        let error = transcribe(&slow, &[0.0; 48]).unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn the_wav_is_a_valid_mono_24k_pcm_file_and_is_removed() {
        let dir = std::env::temp_dir().join(format!("ultra-test-{}", std::process::id()));
        let path = write_wav(&dir, &[0.0; 2400]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1, "mono");
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            RATE
        );
        assert_eq!(bytes.len(), 44 + 2400 * 2);
        std::fs::remove_file(path).unwrap();
        let _ = std::fs::remove_dir(dir);
    }
}
