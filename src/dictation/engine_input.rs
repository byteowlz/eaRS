//! The engine's single microphone stream, replaceable while dictation runs.
//!
//! A switch opens the new input muted while the old one keeps running, then
//! stops the old stream before unmuting the new one, so the downstream PCM
//! never interleaves two devices. A failed open leaves the old input untouched.
use crate::audio::{Capture, ErrorHook, InputSelection};
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread;
use std::time::Duration;

use crate::dictation::{DictationEvent, Shared};

pub(crate) trait InputStream: Send {
    fn device(&self) -> &str;
    fn enable(&self);
}

impl InputStream for Capture {
    fn device(&self) -> &str {
        Capture::device(self)
    }
    fn enable(&self) {
        Capture::enable(self)
    }
}

/// Opens one stream for a selection; `enabled` controls whether audio flows yet.
pub(crate) type Opener =
    Box<dyn Fn(InputSelection, bool) -> Result<Box<dyn InputStream>> + Send + Sync>;
type Emit = Box<dyn Fn(DictationEvent) + Send + Sync>;

pub(crate) struct InputSlot {
    open: Opener,
    emit: Emit,
    current: Mutex<Option<Box<dyn InputStream>>>,
    retry_delay: Duration,
}

impl InputSlot {
    pub(crate) fn new(open: Opener, emit: Emit, retry_delay: Duration) -> Self {
        Self {
            open,
            emit,
            current: Mutex::new(None),
            retry_delay,
        }
    }

    /// No microphone: every open fails. For engine tests.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        Self::new(
            Box::new(|_, _| Err(anyhow!("no audio in tests"))),
            Box::new(|_| {}),
            Duration::ZERO,
        )
    }

    /// Real microphone capture feeding `audio_tx`.
    pub(crate) fn for_capture(
        audio_tx: crossbeam_channel::Sender<Vec<f32>>,
        shared: Arc<Shared>,
    ) -> Self {
        let failure = shared.clone();
        let on_error: ErrorHook = Arc::new(move |message| {
            eprintln!("Audio stream error: {message}");
            failure.emit(DictationEvent::Error {
                message: format!("microphone stopped: {message}"),
            });
        });
        Self::new(
            Box::new(move |selection, enabled| {
                let capture =
                    Capture::open(audio_tx.clone(), selection, on_error.clone(), enabled)?;
                Ok(Box::new(capture) as Box<dyn InputStream>)
            }),
            Box::new(move |event| shared.emit(event)),
            Duration::from_secs(1),
        )
    }

    /// Open the configured input in the background, retrying transient failures.
    pub(crate) fn start(self: &Arc<Self>, requested: Option<String>, attempts: u32) {
        let slot = self.clone();
        let spawned = thread::Builder::new()
            .name("ears-audio-start".into())
            .spawn(move || slot.start_now(requested, attempts));
        if let Err(error) = spawned {
            (self.emit)(DictationEvent::Error {
                message: format!("microphone unavailable: {error}"),
            });
        }
    }

    pub(crate) fn start_now(&self, requested: Option<String>, attempts: u32) {
        let mut current = self.current.lock().unwrap_or_else(|p| p.into_inner());
        let selection = InputSelection::from_name(requested.as_deref());
        let mut attempt = 1;
        let result = loop {
            match (self.open)(selection.clone(), true) {
                Ok(stream) => break Ok(stream),
                Err(error) if attempt >= attempts.max(1) => break Err(error),
                Err(error) => {
                    eprintln!("Audio capture failed (attempt {attempt}/{attempts}): {error}");
                    thread::sleep(self.retry_delay);
                    attempt += 1;
                }
            }
        };
        match result {
            Ok(stream) => {
                let active = Some(stream.device().to_owned());
                *current = Some(stream);
                (self.emit)(DictationEvent::InputDevice {
                    requested,
                    active,
                    error: None,
                });
            }
            Err(error) => {
                eprintln!("Audio capture error: {error}");
                (self.emit)(DictationEvent::Error {
                    message: format!("microphone unavailable: {error}"),
                });
                (self.emit)(DictationEvent::InputDevice {
                    requested,
                    active: None,
                    error: Some(format!("{error:#}")),
                });
            }
        }
    }

    /// Replace the running input. On failure the previous input keeps running.
    pub(crate) fn switch(&self, selection: InputSelection) -> Result<String> {
        let mut current = match self.current.try_lock() {
            Ok(current) => current,
            Err(TryLockError::WouldBlock) => {
                return Err(anyhow!("the microphone is still starting; try again"));
            }
            Err(TryLockError::Poisoned(p)) => p.into_inner(),
        };
        let next = (self.open)(selection, current.is_none())?;
        let device = next.device().to_owned();
        // Stop (and join) the old stream before the new one is heard.
        drop(current.replace(next));
        if let Some(stream) = current.as_ref() {
            stream.enable();
        }
        Ok(device)
    }

    /// Device currently open; `None` while starting or after a failed start.
    pub(crate) fn device(&self) -> Option<String> {
        let current = self.current.try_lock().ok()?;
        Some(current.as_ref()?.device().to_owned())
    }
}

#[cfg(test)]
#[path = "engine_input_tests.rs"]
mod tests;
