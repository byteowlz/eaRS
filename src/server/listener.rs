use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use crate::WebSocketMessage;

#[derive(Clone)]
pub struct StreamRegistry {
    streams: Arc<Mutex<HashMap<u64, StreamBroadcaster>>>,
}

pub struct StreamBroadcaster {
    pub session_id: u64,
    subscribers: Vec<mpsc::UnboundedSender<Message>>,
}

impl StreamRegistry {
    pub fn new() -> Self {
        Self {
            streams: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn register_stream(&self, session_id: u64) -> Result<()> {
        let mut streams = self.streams.lock().unwrap();
        streams.insert(
            session_id,
            StreamBroadcaster {
                session_id,
                subscribers: Vec::new(),
            },
        );
        Ok(())
    }

    pub fn unregister_stream(&self, session_id: u64) {
        let mut streams = self.streams.lock().unwrap();
        streams.remove(&session_id);
    }

    pub fn add_listener(
        &self,
        session_id: u64,
        sender: mpsc::UnboundedSender<Message>,
    ) -> Result<()> {
        let mut streams = self.streams.lock().unwrap();
        let stream = streams
            .get_mut(&session_id)
            .ok_or_else(|| anyhow!("Stream {} not found", session_id))?;
        stream.subscribers.push(sender);
        Ok(())
    }

    pub fn broadcast_message(&self, session_id: u64, message: &WebSocketMessage) -> Result<()> {
        let mut streams = self.streams.lock().unwrap();
        if let Some(stream) = streams.get_mut(&session_id) {
            if let Ok(json) = serde_json::to_string(message) {
                let ws_msg = Message::text(json);
                stream
                    .subscribers
                    .retain(|sender| sender.send(ws_msg.clone()).is_ok());
            }
        }
        Ok(())
    }

    pub fn list_active_streams(&self) -> Vec<u64> {
        let streams = self.streams.lock().unwrap();
        streams.keys().copied().collect()
    }
}

#[derive(Clone)]
pub struct TokenValidator {
    valid_tokens: Vec<String>,
}

impl TokenValidator {
    pub fn new(tokens: Vec<String>) -> Self {
        Self {
            valid_tokens: tokens,
        }
    }

    pub fn validate(&self, token: &str) -> bool {
        self.valid_tokens.iter().any(|t| t == token)
    }

    pub fn generate_token() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        format!("{:x}", timestamp ^ 0xDEADBEEF)
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ListenerCommand {
    Authenticate { token: String },
    Subscribe { stream_id: u64 },
    ListStreams,
}
