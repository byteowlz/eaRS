//! Capture ownership, retention and the transport cut; no detached audio consumer.
use super::*;

impl Engine {
    /// Connecting/retry work must service local capture too. Offline audio is
    /// never replayed into a later target; only the independent VAD clock runs.
    pub(super) fn service_disconnected_capture(
        &mut self,
        audio_rx: &Receiver<Vec<f32>>,
    ) -> Result<bool> {
        if !self.shared.is_running() || matches!(self.poll_commands()?, Some(Flow::Exit)) {
            return Ok(false);
        }
        let capturing = self.shared.is_capturing();
        let generation = self.shared.capture_generation();
        if capturing != self.was_capturing
            || (capturing && self.auto_pause.capture_generation != Some(generation))
        {
            self.buffer.discard();
            self.router.discard_pending();
            self.utterance.clear();
            self.on_capture_edge(capturing)?;
        }
        if capturing {
            if let Some(auto) = &mut self.auto_pause.runtime {
                if !auto.control.active(generation) {
                    auto.start(generation);
                }
            }
        }
        for _ in 0..256 {
            let Ok(chunk) = audio_rx.try_recv() else {
                break;
            };
            if !capturing
                || !self.shared.is_capturing()
                || self.shared.capture_generation() != generation
            {
                continue;
            }
            if let Some(err) = self
                .auto_pause
                .runtime
                .as_mut()
                .and_then(|auto| auto.observe(&chunk).err())
            {
                self.auto_pause.runtime = None;
                self.shared.emit(DictationEvent::Error {
                    message: format!("auto-pause disabled: {err:#}"),
                });
            }
        }
        if capturing
            && audio_rx.is_empty()
            && self
                .auto_pause
                .runtime
                .as_ref()
                .is_some_and(|auto| auto.control.due(generation))
        {
            if self.shared.pause_capture_if_current(generation).is_some() {
                self.on_capture_edge(false)?;
                self.shared.emit(DictationEvent::Finalization {
                    capture_generation: generation,
                    state: FinalizationState::Unconfirmed,
                });
                self.shared.emit(DictationEvent::Error {
                    message:
                        "dictation paused locally while offline; no ASR finalization was available"
                            .into(),
                });
            }
        }
        self.discard_history();
        Ok(true)
    }

    /// Called both before work and after an awaited read, before any stale effects.
    pub(super) fn interrupted_drain(&mut self) -> Result<Option<Flow>> {
        if self.auto_pause.local_pause.is_some() {
            return Ok(
                if self.auto_pause.accepts(self.shared.capture_generation()) {
                    None
                } else {
                    Some(self.finish_local_pause(
                        false,
                        Some("capture generation superseded pending ASR finalization"),
                    ))
                },
            );
        }
        if !self.auto_pause.draining()
            || (self.shared.is_capturing()
                && self.auto_pause.accepts(self.shared.capture_generation()))
        {
            return Ok(None);
        }
        if !self.shared.is_capturing() && self.was_capturing {
            self.on_capture_edge(false)?;
        }
        self.abort_auto_pause();
        Ok(Some(Flow::Restart))
    }

    pub(super) fn advance_capture(
        &mut self,
        audio_rx: &Receiver<Vec<f32>>,
    ) -> Result<Option<Flow>> {
        if let Some(flow) = self.interrupted_drain()? {
            return Ok(Some(flow));
        }
        if self.auto_pause.local_pause.is_some() {
            return Ok(self.advance_pending_capture());
        }
        let capturing = self.shared.is_capturing();
        let generation = self.shared.capture_generation();
        if capturing
            && self.was_capturing
            && self.auto_pause.runtime.is_some()
            && self
                .auto_pause
                .capture_generation
                .is_some_and(|old| old != generation)
        {
            for _ in audio_rx.try_iter() {} // Untagged old-generation audio is not replayable.
            self.abort_auto_pause();
            return Ok(Some(Flow::Restart));
        }
        if capturing != self.was_capturing {
            self.on_capture_edge(capturing)?;
            if !capturing && self.auto_pause.runtime.is_some() {
                return Ok(Some(Flow::Restart));
            }
        }
        if capturing {
            self.evaluate_profiles(false)?;
        }
        Ok(None)
    }

    /// A replay has already passed these observers; it must not advance either clock twice.
    fn observe_capture(&mut self, chunk: &[f32], context: &CommitContext) -> Option<Flow> {
        let retention_error = self
            .history
            .as_mut()
            .and_then(|history| history.record_capture(chunk, context).err());
        if let Some(err) = retention_error {
            self.history = None; // Never fall back to known-clipped slicing.
            self.shared.audio_history.store(false, Ordering::SeqCst);
            let message = format!("retention stopped: {err:#}");
            self.shared.emit(DictationEvent::AudioHistory {
                active: false,
                note: Some(message.clone()),
            });
            self.shared.emit(DictationEvent::Error { message });
        }
        let was_draining = self.auto_pause.draining();
        let activity_error = self
            .auto_pause
            .runtime
            .as_mut()
            .and_then(|auto| auto.observe(chunk).err());
        if let Some(err) = activity_error {
            self.auto_pause.runtime = None;
            if was_draining {
                return Some(self.auto_pause_failure(&format!(
                    "speech detector failed during finalization: {err:#}"
                )));
            }
            self.shared.emit(DictationEvent::Error {
                message: format!("auto-pause disabled: {err:#}"),
            });
        }
        None
    }

    pub(super) fn capture_batch(
        &mut self,
        audio_rx: &Receiver<Vec<f32>>,
        encoder: &mut Option<kaudio::ogg_opus::Encoder>,
    ) -> Result<Option<Flow>> {
        let capturing = self.shared.is_capturing();
        let generation = self.shared.capture_generation();
        let context = if self
            .history
            .as_ref()
            .is_some_and(AudioHistory::client_vad_enabled)
        {
            capture_context(self.effective_lang.clone())
        } else {
            CommitContext::default()
        };
        // Callback fragments may be sub-millisecond. Bound work without limiting
        // throughput below normal small-buffer capture rates.
        for _ in 0..256 {
            if capturing
                && (!self.shared.is_capturing() || self.shared.capture_generation() != generation)
            {
                break;
            }
            let replay = if !self.auto_pause.draining() {
                self.auto_pause.replay.pop_front()
            } else {
                None
            };
            let (chunk, observed) = match replay {
                Some(chunk) => (chunk, self.auto_pause.replay_observed),
                None => match audio_rx.try_recv() {
                    Ok(chunk) => (chunk, false),
                    Err(_) => break,
                },
            };
            if !capturing {
                continue;
            }
            if self.auto_pause.local_pause.is_some() {
                if let Some(flow) = self.buffer_pending_capture(chunk) {
                    return Ok(Some(flow));
                }
                continue;
            }
            if !observed {
                if let Some(flow) = self.observe_capture(&chunk, &context) {
                    return Ok(Some(flow));
                }
            }
            let bytes = match encoder.as_mut() {
                Some(enc) => enc.encode_page(&chunk).context("opus encode failed")?,
                None => encode_chunk(&chunk),
            };
            if !bytes.is_empty() {
                if let Some(writer) = &self.writer {
                    let _ = writer.send(WriterCommand::Audio(bytes));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn begin_idle_drain(
        &mut self,
        audio_rx: &Receiver<Vec<f32>>,
        encoder: &mut Option<kaudio::ogg_opus::Encoder>,
    ) -> Result<Option<Flow>> {
        // Snapshot ownership once: a concurrent new capture may invalidate this
        // barrier, but must not turn a due(old)/begin(new) race into a fatal error.
        let generation = self.shared.capture_generation();
        if !self.shared.is_capturing()
            || !self.auto_pause.replay.is_empty()
            || !audio_rx.is_empty()
            || self.shared.ptt_held()
            || !self
                .auto_pause
                .runtime
                .as_ref()
                .is_some_and(|auto| auto.control.due(generation))
        {
            return Ok(None);
        }
        let request_id = self
            .auto_pause
            .runtime
            .as_mut()
            .unwrap()
            .control
            .begin(generation, Instant::now())?;
        if !self.begin_local_pause(generation, request_id)? {
            return Ok(None);
        }
        if let Some(encoder) = encoder.as_mut() {
            let bytes = match flush_opus_for_drain(encoder) {
                Ok(bytes) => bytes,
                Err(err) => {
                    return Ok(Some(self.auto_pause_failure(&format!(
                        "Opus tail finalization failed: {err:#}"
                    ))));
                }
            };
            if self
                .writer
                .as_ref()
                .is_none_or(|writer| writer.send(WriterCommand::Audio(bytes)).is_err())
            {
                return Ok(Some(self.auto_pause_failure(
                    "transcription writer disconnected during Opus tail flush",
                )));
            }
        }
        if !self.auto_pause.supported {
            return Ok(None);
        }
        let request = serde_json::json!({"type":"drain", "request_id":request_id}).to_string();
        if self
            .writer
            .as_ref()
            .is_none_or(|writer| writer.send(WriterCommand::Text(request)).is_err())
        {
            return Ok(Some(self.auto_pause_failure(
                "transcription writer disconnected before drain",
            )));
        }
        eprintln!("auto-pause: local capture paused; ordered ASR finalization pending");
        Ok(None)
    }
}
