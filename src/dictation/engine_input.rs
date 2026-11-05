//! The engine's single microphone stream, replaceable while dictation runs.
//!
//! A switch opens the new input muted while the old one keeps running, then
//! stops the old stream before unmuting the new one, so the downstream PCM
//! never interleaves two devices. A failed open leaves the old input untouched.
//!
//! A supervisor keeps the chosen input alive: a stream that dies (device
//! unplugged) is dropped and the *same* selection is reopened once it is back;
//! a named microphone never silently becomes another device. With System
//! default selected, a change of the OS default input is followed.
use crate::audio::{Capture, ErrorHook, InputSelection};
use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, Weak};
use std::thread;
use std::time::Duration;

use crate::dictation::{DictationEvent, Shared};

/// Inputs to try when the system default cannot be opened, in order: the
/// enumerated devices except the broken default PCM itself. Linux only; also
/// used by tests without touching audio.
pub(crate) fn default_fallbacks(devices: &[String]) -> Vec<InputSelection> {
    devices
        .iter()
        .filter(|name| name.as_str() != "default" && name.as_str() != "pipewire")
        .map(|name| InputSelection::Named(name.clone()))
        .collect()
}

/// An opener that can be called repeatedly (retry, fallback, supervisor).
type SharedOpener = Arc<dyn Fn(InputSelection, bool) -> Result<Box<dyn InputStream>> + Send + Sync>;

/// On Linux the PipeWire-backed ALSA `default` PCM can be unusable while the
/// hardware device works (observed on Arch: `snd_pcm_hw_params` fails for
/// `default`/`pipewire`, `hw:`/`plughw:`/`sysdefault:` capture fine). Only the
/// system default falls back, in enumeration order, and the device actually
/// used is reported. A microphone chosen by name never falls back.
fn open_default_fallback(
    open: &(dyn Fn(InputSelection, bool) -> Result<Box<dyn InputStream>> + Send + Sync),
    selection: InputSelection,
    error: anyhow::Error,
    enabled: bool,
) -> Result<Box<dyn InputStream>> {
    if !cfg!(target_os = "linux") || !matches!(selection, InputSelection::Default) {
        return Err(error);
    }
    for candidate in default_fallbacks(&crate::audio::input_device_names()) {
        let InputSelection::Named(name) = candidate.clone() else {
            continue;
        };
        if let Ok(stream) = open(candidate, enabled) {
            eprintln!("input: system default unusable ({error:#}); using {name}");
            return Ok(stream);
        }
    }
    Err(error)
}

pub(crate) trait InputStream: Send {
    fn device(&self) -> &str;
    fn enable(&self);
    /// The stream stopped because of a device error.
    fn failed(&self) -> bool;
}

impl InputStream for Capture {
    fn device(&self) -> &str {
        Capture::device(self)
    }
    fn enable(&self) {
        Capture::enable(self)
    }
    fn failed(&self) -> bool {
        Capture::failed(self)
    }
}

/// Opens one stream for a selection; `enabled` controls whether audio flows yet.
pub(crate) type Opener =
    Box<dyn Fn(InputSelection, bool) -> Result<Box<dyn InputStream>> + Send + Sync>;
type Emit = Box<dyn Fn(DictationEvent) + Send + Sync>;
/// Name of the OS default input right now (metadata only, no stream).
type DefaultName = Box<dyn Fn() -> Option<String> + Send + Sync>;

/// Shown while waiting for a lost microphone to return.
pub(crate) const RECONNECTING: &str = "reconnecting automatically when it is available";

pub(crate) struct InputSlot {
    open: Opener,
    emit: Emit,
    default_name: DefaultName,
    current: Mutex<Option<Box<dyn InputStream>>>,
    selection: Mutex<InputSelection>,
    /// The loss was reported; avoids repeating it every supervisor tick.
    lost_reported: AtomicBool,
    retry_delay: Duration,
}

fn requested(selection: &InputSelection) -> Option<String> {
    match selection {
        InputSelection::Named(name) => Some(name.clone()),
        _ => None,
    }
}

impl InputSlot {
    pub(crate) fn new(
        open: Opener,
        emit: Emit,
        default_name: DefaultName,
        retry_delay: Duration,
    ) -> Self {
        Self {
            open,
            emit,
            default_name,
            current: Mutex::new(None),
            selection: Mutex::new(InputSelection::Default),
            lost_reported: AtomicBool::new(false),
            retry_delay,
        }
    }

    /// No microphone: every open fails. For engine tests.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        Self::new(
            Box::new(|_, _| Err(anyhow!("no audio in tests"))),
            Box::new(|_| {}),
            Box::new(|| None),
            Duration::ZERO,
        )
    }

    /// Real microphone capture feeding `audio_tx`.
    pub(crate) fn for_capture(
        audio_tx: crossbeam_channel::Sender<Vec<f32>>,
        shared: Arc<Shared>,
    ) -> Self {
        let on_error: ErrorHook = Arc::new(|message| {
            // The supervisor notices the failed stream and reports/reconnects.
            eprintln!("Audio stream error: {message}");
        });
        let raw: SharedOpener = {
            let audio_tx = audio_tx.clone();
            let on_error = on_error.clone();
            Arc::new(move |selection, enabled| {
                Ok(Box::new(Capture::open(
                    audio_tx.clone(),
                    selection,
                    on_error.clone(),
                    enabled,
                )?) as Box<dyn InputStream>)
            })
        };
        let open = raw.clone();
        Self::new(
            Box::new(
                move |selection, enabled| match open(selection.clone(), enabled) {
                    Ok(stream) => Ok(stream),
                    Err(error) => open_default_fallback(&*raw, selection, error, enabled),
                },
            ),
            Box::new(move |event| shared.emit(event)),
            Box::new(crate::audio::default_input_device_name),
            Duration::from_secs(1),
        )
    }

    /// Open the configured input in the background, retrying transient
    /// failures, then keep it alive with the supervisor.
    fn selection(&self) -> InputSelection {
        self.selection
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn try_current(&self) -> Result<MutexGuard<'_, Option<Box<dyn InputStream>>>> {
        match self.current.try_lock() {
            Ok(current) => Ok(current),
            Err(TryLockError::WouldBlock) => {
                Err(anyhow!("the microphone is busy starting; try again"))
            }
            Err(TryLockError::Poisoned(p)) => Ok(p.into_inner()),
        }
    }

    pub(crate) fn start(self: &Arc<Self>, requested: Option<String>, attempts: u32) {
        let slot = self.clone();
        let spawned = thread::Builder::new()
            .name("ears-audio-start".into())
            .spawn(move || {
                slot.start_now(requested, attempts);
                // Only a weak reference loops, so stopping the engine releases
                // the slot and with it the microphone.
                let weak = Arc::downgrade(&slot);
                drop(slot);
                Self::supervise_forever(weak, Duration::from_secs(1));
            });
        if let Err(error) = spawned {
            (self.emit)(DictationEvent::Error {
                message: format!("microphone unavailable: {error}"),
            });
        }
    }

    /// Supervise until the slot (and with it the engine) is gone.
    fn supervise_forever(slot: Weak<Self>, interval: Duration) {
        loop {
            thread::sleep(interval);
            let Some(slot) = slot.upgrade() else {
                return;
            };
            slot.supervise();
        }
    }

    pub(crate) fn start_now(&self, requested: Option<String>, attempts: u32) {
        let mut current = self.current.lock().unwrap_or_else(|p| p.into_inner());
        let selection = InputSelection::from_name(requested.as_deref());
        *self.selection.lock().unwrap_or_else(|p| p.into_inner()) = selection.clone();
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
                // The supervisor keeps trying; report the wait once.
                eprintln!("Audio capture error: {error}");
                self.lost_reported.store(true, Ordering::SeqCst);
                (self.emit)(DictationEvent::InputDevice {
                    requested,
                    active: None,
                    error: Some(format!("unavailable ({error:#}); {RECONNECTING}")),
                });
            }
        }
    }

    /// Replace the running input. On failure the previous input keeps running.
    pub(crate) fn switch(&self, selection: InputSelection) -> Result<String> {
        let mut current = self.try_current()?;
        let next = (self.open)(selection.clone(), current.is_none())?;
        let device = next.device().to_owned();
        // Stop (and join) the old stream before the new one is heard.
        drop(current.replace(next));
        if let Some(stream) = current.as_ref() {
            stream.enable();
        }
        *self.selection.lock().unwrap_or_else(|p| p.into_inner()) = selection;
        self.lost_reported.store(false, Ordering::SeqCst);
        Ok(device)
    }

    /// One supervisor step: drop a dead stream, reopen the chosen input when it
    /// is available, and follow the OS default when that is the selection.
    /// Never blocks a concurrent switch or startup: skips the tick instead.
    pub(crate) fn supervise(&self) {
        let Ok(mut current) = self.try_current() else {
            return;
        };
        let selection = self.selection();
        if current.as_ref().is_some_and(|stream| stream.failed()) {
            let lost = current.take().map(|s| s.device().to_owned());
            eprintln!(
                "input: {} disconnected; waiting for it",
                lost.as_deref().unwrap_or("microphone")
            );
        }
        match current.as_ref() {
            None => self.reconnect(&mut current, selection),
            Some(stream) if selection == InputSelection::Default => {
                let Some(default) = (self.default_name)() else {
                    return;
                };
                if stream.device() == default {
                    return;
                }
                // Follow the new OS default; on failure keep the old input.
                if let Ok(next) = (self.open)(InputSelection::Default, false) {
                    let device = next.device().to_owned();
                    drop(current.replace(next));
                    if let Some(stream) = current.as_ref() {
                        stream.enable();
                    }
                    eprintln!("input: system default changed; now using {device}");
                    (self.emit)(DictationEvent::InputDevice {
                        requested: None,
                        active: Some(device),
                        error: None,
                    });
                }
            }
            Some(_) => {}
        }
    }

    fn reconnect(&self, current: &mut Option<Box<dyn InputStream>>, selection: InputSelection) {
        let wanted = requested(&selection);
        match (self.open)(selection, true) {
            Ok(stream) => {
                let device = stream.device().to_owned();
                *current = Some(stream);
                self.lost_reported.store(false, Ordering::SeqCst);
                eprintln!("input: reconnected to {device}");
                (self.emit)(DictationEvent::InputDevice {
                    requested: wanted,
                    active: Some(device),
                    error: None,
                });
            }
            Err(_) if self.lost_reported.swap(true, Ordering::SeqCst) => {}
            Err(_) => (self.emit)(DictationEvent::InputDevice {
                requested: wanted,
                active: None,
                error: Some(format!("disconnected; {RECONNECTING}")),
            }),
        }
    }

    /// Device currently open; `None` while starting or while disconnected.
    pub(crate) fn device(&self) -> Option<String> {
        let current = self.current.try_lock().ok()?;
        Some(current.as_ref()?.device().to_owned())
    }
}

#[cfg(test)]
#[path = "engine_input_tests.rs"]
mod tests;
