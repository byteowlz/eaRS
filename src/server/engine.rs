use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::{TranscriptionSink, WebSocketMessage};

use super::SessionSink;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineKind {
    Kyutai,
    #[cfg(feature = "parakeet")]
    Parakeet,
}

impl EngineKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineKind::Kyutai => "kyutai",
            #[cfg(feature = "parakeet")]
            EngineKind::Parakeet => "parakeet",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "kyutai" | "moshi" => Some(EngineKind::Kyutai),
            #[cfg(feature = "parakeet")]
            "parakeet" | "prkt" => Some(EngineKind::Parakeet),
            _ => None,
        }
    }
}

pub trait EngineSession: Send {
    #[allow(dead_code)]
    fn engine(&self) -> EngineKind;
    fn send_audio(&self, pcm: Vec<f32>) -> Result<()>;
    fn set_language(&self, lang: String) -> Result<()>;
    fn request_stop(&self);
    fn supports_language(&self) -> bool;
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
        #[cfg(feature = "parakeet")]
        {
            assert_eq!(EngineKind::from_str("parakeet"), Some(EngineKind::Parakeet));
            assert_eq!(EngineKind::from_str("PRKT"), Some(EngineKind::Parakeet));
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
