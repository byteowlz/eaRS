use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
#[cfg(any(feature = "moonshine", feature = "parakeet-rs"))]
use crossbeam_channel::{SendTimeoutError, Sender};

use crate::{TranscriptionSink, WebSocketMessage};

use super::SessionSink;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum)]
pub enum EngineKind {
    Kyutai,
    #[cfg(feature = "moonshine")]
    Moonshine,
    #[cfg(feature = "parakeet-rs")]
    ParakeetRs,
    #[cfg(feature = "transcribe-cpp")]
    TranscribeCpp,
}

impl EngineKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineKind::Kyutai => "kyutai",
            #[cfg(feature = "moonshine")]
            EngineKind::Moonshine => "moonshine",
            #[cfg(feature = "parakeet-rs")]
            EngineKind::ParakeetRs => "parakeet-rs",
            #[cfg(feature = "transcribe-cpp")]
            EngineKind::TranscribeCpp => "transcribe-cpp",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "kyutai" | "moshi" => Some(EngineKind::Kyutai),
            #[cfg(feature = "moonshine")]
            "moonshine" => Some(EngineKind::Moonshine),
            #[cfg(feature = "parakeet-rs")]
            "parakeet-rs" | "parakeet_rs" | "nemotron" => Some(EngineKind::ParakeetRs),
            #[cfg(feature = "transcribe-cpp")]
            "transcribe-cpp" | "transcribe_cpp" | "tcpp" => Some(EngineKind::TranscribeCpp),
            _ => None,
        }
    }
}

pub trait EngineSession: Send {
    #[allow(dead_code)]
    fn engine(&self) -> EngineKind;
    fn send_audio(&self, pcm: Vec<f32>) -> Result<()>;
    fn set_language(&self, lang: String) -> Result<()>;
    /// Notify an engine of an acoustic speech-boundary transition. Returns true
    /// when the engine takes ownership of emitting the corresponding `Speech`
    /// event (for example, after flushing an utterance); false lets the server
    /// emit it immediately. Most engines need no special handling.
    fn send_speech_boundary(&self, _active: bool) -> Result<bool> {
        Ok(false)
    }
    /// True only when `request_drain` shares the engine's accepted-audio FIFO.
    fn supports_ordered_drain(&self) -> bool {
        false
    }
    /// Queue a terminal barrier after all prior accepted audio. On success the
    /// worker must finalize, emit a cumulative-session `Final` (see its protocol
    /// contract in `WebSocketMessage`), emit `Drained { request_id }`, then close
    /// its sink. Final text must preserve the exact previously emitted Word-token
    /// prefix; it is not just the last utterance. Failure must never synthesize
    /// an ACK. Admission must be bounded.
    fn request_drain(&self, _request_id: u64) -> Result<()> {
        anyhow::bail!("ordered drain is not supported by this engine")
    }
    fn request_stop(&self);
    fn supports_language(&self) -> bool;
}

/// Shared bounded admission policy for engines with an accepted-audio FIFO.
/// A failed enqueue is not accepted input and must never result in a drain ACK.
/// Keep the distinction between a stalled and disconnected worker explicit.
#[cfg(any(feature = "moonshine", feature = "parakeet-rs"))]
pub(super) fn queue_command<T>(sender: &Sender<T>, command: T) -> Result<()> {
    match sender.send_timeout(command, std::time::Duration::from_secs(1)) {
        Ok(()) => Ok(()),
        Err(SendTimeoutError::Timeout(_)) => anyhow::bail!("engine command queue timed out"),
        Err(SendTimeoutError::Disconnected(_)) => {
            anyhow::bail!("engine command queue disconnected")
        }
    }
}

pub trait Engine: Send + Sync {
    fn kind(&self) -> EngineKind;
    fn allocate(&self, sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>>;
}

#[cfg(test)]
mod tests {
    use super::EngineKind;

    #[test]
    fn engine_kind_from_str_variants() {
        assert_eq!(EngineKind::from_str("kyutai"), Some(EngineKind::Kyutai));
        assert_eq!(EngineKind::from_str("KYUTAI"), Some(EngineKind::Kyutai));
        assert_eq!(EngineKind::from_str("moshi"), Some(EngineKind::Kyutai));
        // Removed legacy engines must not silently select Nemotron.
        assert_eq!(EngineKind::from_str("parakeet"), None);
        assert_eq!(EngineKind::from_str("prkt"), None);
        #[cfg(feature = "parakeet-rs")]
        {
            assert_eq!(
                EngineKind::from_str("nemotron"),
                Some(EngineKind::ParakeetRs)
            );
            assert_eq!(
                EngineKind::from_str("parakeet-rs"),
                Some(EngineKind::ParakeetRs)
            );
        }
        assert_eq!(EngineKind::from_str("unknown"), None);
    }
}

#[derive(Clone)]
pub struct EngineManager {
    engines: Arc<HashMap<EngineKind, Arc<dyn Engine>>>,
}

impl EngineManager {
    pub fn new() -> Self {
        Self {
            engines: Arc::new(HashMap::new()),
        }
    }

    pub fn register(&mut self, engine: Arc<dyn Engine>) {
        Arc::make_mut(&mut self.engines).insert(engine.kind(), engine);
    }

    pub fn available(&self) -> Vec<EngineKind> {
        self.engines.keys().copied().collect()
    }

    pub fn has(&self, kind: EngineKind) -> bool {
        self.engines.contains_key(&kind)
    }

    pub fn allocate(
        &self,
        kind: EngineKind,
        sink: SessionSink,
    ) -> Result<Option<Box<dyn EngineSession>>> {
        if let Some(engine) = self.engines.get(&kind) {
            engine.allocate(sink)
        } else {
            Ok(None)
        }
    }
}

pub fn send_engine_changed(sink: &mut SessionSink, kind: EngineKind) {
    let message = WebSocketMessage::EngineChanged {
        engine: kind.as_str().to_string(),
    };
    sink.handle_message(message);
}
