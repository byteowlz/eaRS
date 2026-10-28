//! Capture ownership, retention and the transport cut; no detached audio consumer.
use super::*;

impl Engine {
    /// Called both before work and after an awaited read, before any stale effects.
    pub(super) fn interrupted_drain(&mut self) -> Result<Option<Flow>> {
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
        if self.auto_pause.draining() && self.shared.ptt_held() {
            if let Some(auto) = &mut self.auto_pause.runtime {
                auto.control.resume(self.shared.capture_generation());
            }
        }
        if self
            .auto_pause
            .runtime
            .as_ref()
            .is_some_and(|auto| auto.control.timed_out(Instant::now()))
        {
            return Ok(Some(self.auto_pause_failure(
                "server finalization timed out; pending results were not confirmed",
            )));
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
                Some(chunk) => (chunk, true),
                None => match audio_rx.try_recv() {
                    Ok(chunk) => (chunk, false),
                    Err(_) => break,
                },
            };
            if !capturing {
                continue;
            }
            if !observed {
                if let Some(flow) = self.observe_capture(&chunk, &context) {
                    return Ok(Some(flow));
                }
            }
            if self.auto_pause.draining() {
                if let Err(err) = self.auto_pause.freeze(chunk) {
                    return Ok(Some(self.auto_pause_failure(&err.to_string())));
                }
                continue;
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
        connected_at: Instant,
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
        if !self.auto_pause.supported {
            if !self.auto_pause.warned
                && (self.auto_pause.capabilities_known
                    || connected_at.elapsed() >= Duration::from_secs(2))
            {
                self.auto_pause.warned = true;
                self.shared.emit(DictationEvent::Error { message: "auto-pause unavailable: server/engine does not advertise dictation-drain/1; update the server or pause manually".into() });
            }
            return Ok(None);
        }
        let request_id = self
            .auto_pause
            .runtime
            .as_mut()
            .unwrap()
            .control
            .begin(generation, Instant::now())?;
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
        eprintln!("auto-pause: finalizing captured audio");
        Ok(None)
    }
}
