//! Native Moonshine Tiny Streaming adapter. Public German base assets live in
//! HF_HOME/moonshine; explicit local ORT bundles support private fine-tunes.
//! Native tentative lines are revisable `Interim` snapshots. Only completed
//! lines become append-only `Word` events, so dictation never types revisions.

mod model;
#[cfg(test)]
mod trace_tests;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, bounded};
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions, Transcript};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::engine::{Engine, EngineKind, EngineSession, queue_command};
use super::{SessionSink, current_timestamp};
use crate::{TranscriptionSink, WebSocketMessage, WordTimestamp};

#[derive(Debug, Clone, Default)]
pub struct MoonshineEngineConfig {
    /// Native Tiny Streaming ORT bundle; None downloads the public German base.
    pub model_dir: Option<PathBuf>,
}

pub struct MoonshineEngine {
    model: Arc<Transcriber>,
    semaphore: Arc<Semaphore>,
}

impl MoonshineEngine {
    pub fn load(config: MoonshineEngineConfig) -> Result<Self> {
        let dir = model::resolve(config.model_dir.as_deref())?;
        let options = TranscriberOptions::new().with_ort_providers("cpu");
        let model = Transcriber::from_files(&dir, ModelArch::TinyStreaming, Some(&options))
            .with_context(|| format!("loading Moonshine Tiny Streaming from {}", dir.display()))?;
        eprintln!(
            "[moonshine] native runtime {}, CPU, model {}",
            moonshine_rs::get_version(),
            dir.display()
        );
        Ok(Self {
            model: Arc::new(model),
            // Keep inference deterministic and bound native CPU/memory use.
            semaphore: Arc::new(Semaphore::new(1)),
        })
    }
}

impl Engine for MoonshineEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Moonshine
    }

    fn allocate(&self, mut sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
        let Ok(permit) = self.semaphore.clone().try_acquire_owned() else {
            return Ok(None);
        };
        let (tx, rx) = bounded(64);
        let model = self.model.clone();
        std::thread::Builder::new()
            .name("ears-moonshine".into())
            .spawn(move || {
                if let Err(err) = run_session(model, rx, &mut sink, permit) {
                    eprintln!("[moonshine] session failed: {err:#}");
                }
                sink.close();
            })?;
        Ok(Some(Box::new(Session { tx })))
    }
}

enum Command {
    Audio(Vec<f32>),
    Boundary,
    Drain(u64),
    Stop,
}

struct Session {
    tx: Sender<Command>,
}

impl EngineSession for Session {
    fn engine(&self) -> EngineKind {
        EngineKind::Moonshine
    }

    fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        // File transcription bursts faster than real time. This synchronous
        // EngineSession seam backpressures the producer while the independent
        // native thread drains a bounded queue. Admission times out on a stalled
        // worker; already accepted audio is never dropped or falsely ACKed.
        queue_command(&self.tx, Command::Audio(pcm))
    }

    fn set_language(&self, _lang: String) -> Result<()> {
        bail!(
            "Moonshine language is fixed by its loaded model; restart with a different --moonshine-model"
        )
    }

    fn send_speech_boundary(&self, active: bool) -> Result<bool> {
        if active {
            return Ok(false);
        }
        queue_command(&self.tx, Command::Boundary)?;
        Ok(true)
    }

    fn supports_ordered_drain(&self) -> bool {
        true
    }

    fn request_drain(&self, request_id: u64) -> Result<()> {
        if request_id == 0 {
            bail!("drain request_id must be nonzero");
        }
        queue_command(&self.tx, Command::Drain(request_id))
    }

    fn request_stop(&self) {
        // Bound cleanup admission on a stalled worker. Subsequent handle drop
        // disconnects the FIFO, also finalizing previously accepted audio.
        let _ = queue_command(&self.tx, Command::Stop);
    }
    fn supports_language(&self) -> bool {
        false
    }
}

/// Track finalized line IDs separately from tentative text. The IDs are local
/// to a native stream and reset at a boundary; committed word timestamps stay
/// on the outer client-session clock.
#[derive(Default)]
struct Output {
    seen: HashSet<u64>,
    words: Vec<WordTimestamp>,
}

impl Output {
    fn absorb(
        &mut self,
        transcript: Transcript,
        force_final: bool,
        offset: f64,
        sink: &mut SessionSink,
    ) {
        let mut preview = Vec::new();
        for line in transcript.lines {
            if line.text.trim().is_empty() {
                continue;
            }
            preview.push(line.text.clone());
            if !(force_final || line.is_complete) || !self.seen.insert(line.id) {
                continue;
            }
            for word in line.text.split_whitespace() {
                let start_time = offset + f64::from(line.start_time);
                let end_time = Some(start_time + f64::from(line.duration));
                sink.handle_message(WebSocketMessage::Word {
                    word: word.to_string(),
                    start_time,
                    end_time,
                });
                self.words.push(WordTimestamp {
                    word: word.to_string(),
                    start_time,
                    end_time,
                });
            }
        }
        if !preview.is_empty() {
            sink.handle_message(WebSocketMessage::Interim {
                text: preview.join(" "),
            });
        }
    }
}

fn run_session(
    model: Arc<Transcriber>,
    rx: Receiver<Command>,
    sink: &mut SessionSink,
    _permit: OwnedSemaphorePermit,
) -> Result<()> {
    let mut output = Output::default();
    let mut elapsed = 0.0;
    let mut offset = 0.0;
    let mut received_audio = false;
    let mut stream = model.create_stream()?;
    loop {
        match rx.recv() {
            Ok(Command::Audio(pcm)) => {
                if pcm.is_empty() {
                    continue;
                }
                elapsed += pcm.len() as f64 / 24_000.0;
                stream.add_audio(&pcm, 24_000)?;
                output.absorb(stream.poll(false)?, false, offset, sink);
                received_audio = true;
            }
            Ok(Command::Boundary) => {
                if received_audio {
                    output.absorb(stream.finalize()?, true, offset, sink);
                    stream = model.create_stream()?;
                    output.seen.clear();
                    offset = elapsed;
                    received_audio = false;
                }
                sink.handle_message(WebSocketMessage::Speech {
                    active: false,
                    timestamp: current_timestamp(),
                });
            }
            terminal @ (Ok(Command::Stop | Command::Drain(_)) | Err(_)) => {
                if received_audio {
                    output.absorb(stream.finalize()?, true, offset, sink);
                }
                sink.handle_message(WebSocketMessage::Final {
                    text: output
                        .words
                        .iter()
                        .map(|word| word.word.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    words: output.words,
                });
                if let Ok(Command::Drain(request_id)) = terminal {
                    sink.handle_message(WebSocketMessage::Drained { request_id });
                }
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonshine_rs::TranscriptLine;

    fn line(id: u64, text: &str, complete: bool) -> Transcript {
        Transcript {
            lines: vec![TranscriptLine {
                id,
                text: text.into(),
                start_time: 0.5,
                duration: 1.0,
                is_complete: complete,
                is_updated: true,
                is_new: true,
                has_text_changed: true,
                have_speakers_changed: false,
                last_transcription_latency_ms: 0,
                words: vec![],
                speaker_spans: vec![],
            }],
        }
    }

    #[test]
    fn file_bursts_backpressure_without_dropping_audio_or_reordering_controls() {
        let (tx, rx) = bounded(2);
        let session = Session { tx };
        let reader = std::thread::spawn(move || {
            for expected in 0..128 {
                match rx.recv().unwrap() {
                    Command::Audio(pcm) => assert_eq!(pcm, vec![expected as f32]),
                    _ => panic!("control overtook queued audio"),
                }
            }
            assert!(matches!(rx.recv().unwrap(), Command::Boundary));
            assert!(matches!(rx.recv().unwrap(), Command::Stop));
        });
        for i in 0..128 {
            session.send_audio(vec![i as f32]).unwrap();
        }
        assert!(session.send_speech_boundary(false).unwrap());
        session.request_stop();
        reader.join().unwrap();
    }

    #[test]
    fn drain_uses_audio_fifo_and_survives_handle_disconnect() {
        let (tx, rx) = bounded(4);
        let session = Session { tx };
        assert!(session.request_drain(0).is_err());
        assert!(rx.is_empty());
        session.send_audio(vec![0.1]).unwrap();
        session.send_speech_boundary(false).unwrap();
        session.request_drain(42).unwrap();
        drop(session);
        assert!(matches!(rx.recv().unwrap(), Command::Audio(pcm) if pcm == [0.1]));
        assert!(matches!(rx.recv().unwrap(), Command::Boundary));
        assert!(matches!(rx.recv().unwrap(), Command::Drain(42)));
        assert!(rx.recv().is_err());
    }

    #[test]
    fn drain_admission_failure_does_not_queue_a_barrier() {
        let (tx, rx) = bounded(1);
        let session = Session { tx };
        session.send_audio(vec![0.1]).unwrap();
        assert!(session.request_drain(42).is_err());
        // Cleanup must not block behind the full worker queue.
        session.request_stop();
        drop(session);
        assert!(matches!(rx.recv().unwrap(), Command::Audio(_)));
        assert!(rx.recv().is_err());
    }

    #[test]
    fn tentative_revisions_are_never_typed_and_completed_lines_are_not_duplicated() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut sink = SessionSink::new(tx);
        let mut output = Output::default();
        output.absorb(line(1, "wrong", false), false, 0.0, &mut sink);
        output.absorb(line(1, "correct words", false), false, 0.0, &mut sink);
        assert!(output.words.is_empty());
        output.absorb(line(1, "correct words", true), false, 2.0, &mut sink);
        output.absorb(line(1, "correct words", true), true, 2.0, &mut sink);
        assert_eq!(
            output
                .words
                .iter()
                .map(|word| word.word.as_str())
                .collect::<Vec<_>>(),
            ["correct", "words"]
        );
        assert_eq!(output.words[0].start_time, 2.5);
        output.seen.clear();
        output.absorb(line(1, "next", false), true, 5.0, &mut sink);
        assert_eq!(output.words.len(), 3);
        assert_eq!(output.words[2].start_time, 5.5);
    }
}
