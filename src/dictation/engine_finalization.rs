//! Local capture cut and the separately owned, bounded ASR completion window.
use super::*;

impl Engine {
    pub(super) fn begin_local_pause(&mut self, generation: u64, request_id: u64) -> Result<bool> {
        let cancel_epoch = self.shared.cancel_epoch.load(Ordering::SeqCst);
        let target = (self.target_reader)();
        // Preserve the old wake-strip ownership before rearming the wake listener.
        if self.shared.take_wake_started() {
            self.router
                .arm_wake_strip(&self.config.dictation.wake.strip_words, 3);
        }
        let Some(paused_generation) = self.shared.pause_capture_if_current(generation) else {
            self.auto_pause.runtime.as_mut().unwrap().stop();
            return Ok(false);
        };
        self.auto_pause.local_pause = Some(LocalPause {
            owner_generation: generation,
            request_id,
            cancel_epoch,
            paused_generation,
            target,
            expect_ack: self.auto_pause.supported,
            deferred: Vec::new(),
        });
        let context = capture_context(self.effective_lang.clone());
        if let Some(history) = self.history.as_mut() {
            if let Err(err) = history.end_capture(&context) {
                self.shared.emit(DictationEvent::Error {
                    message: format!("audio history: {err:#}"),
                });
            }
        }
        self.shared.emit(DictationEvent::Finalization {
            capture_generation: generation,
            state: FinalizationState::Pending,
        });
        eprintln!("auto-pause: local capture paused; wake rearmed; ASR finalization pending");
        Ok(true)
    }

    fn pending_generation_matches(&self) -> bool {
        self.auto_pause.local_pause.as_ref().is_some_and(|pause| {
            pause.accepts(self.shared.capture_generation())
                && pause.cancel_epoch == self.shared.cancel_epoch.load(Ordering::SeqCst)
        })
    }

    fn pending_target_matches(&self) -> bool {
        let Some(pause) = &self.auto_pause.local_pause else {
            return false;
        };
        self.pending_generation_matches()
            && pause.target.is_some()
            && pause.target == (self.target_reader)()
            && !crate::frontmost::secure_input_active()
    }

    fn pending_insertion_allowed(&self) -> bool {
        self.auto_pause
            .local_pause
            .as_ref()
            .is_some_and(|pause| pause.deferred.is_empty())
            && self.pending_target_matches()
            && !self.shared.is_capturing()
            && self
                .auto_pause
                .local_pause
                .as_ref()
                .is_some_and(|pause| pause.paused_generation == self.shared.capture_generation())
    }

    fn accept_pending_word(&mut self, word: &str) -> Result<()> {
        anyhow::ensure!(
            word.len() <= 262_144,
            "pending word exceeds reconciliation bound"
        );
        if word.trim().is_empty() {
            return Ok(());
        }
        self.auto_pause.note_word(word);
        anyhow::ensure!(
            !self.auto_pause.word_overflow,
            "pending transcript exceeds reconciliation bound"
        );
        if self.pending_insertion_allowed() {
            self.on_word(word)?;
        } else {
            self.auto_pause
                .local_pause
                .as_mut()
                .unwrap()
                .deferred
                .extend(word.split_whitespace().map(str::to_owned));
        }
        Ok(())
    }

    pub(super) fn advance_pending_capture(&mut self) -> Option<Flow> {
        self.resume_held_capture();
        let generation = self
            .auto_pause
            .local_pause
            .as_ref()
            .unwrap()
            .owner_generation;
        let auto = self.auto_pause.runtime.as_mut().unwrap();
        if self.shared.is_capturing() {
            auto.control.resume(generation);
        }
        auto.control.timed_out(Instant::now()).then(|| {
            self.auto_pause_failure(
                "server finalization timed out; pending ASR tail is unconfirmed",
            )
        })
    }

    pub(super) fn buffer_pending_capture(&mut self, chunk: Vec<f32>) -> Option<Flow> {
        self.auto_pause.freeze(chunk).err().map(|err| {
            self.shared.set_capturing(false); // Never replay a partial new capture after overflow.
            self.auto_pause_failure(&err.to_string())
        })
    }

    pub(super) fn resume_held_capture(&self) {
        if self
            .auto_pause
            .local_pause
            .as_ref()
            .is_some_and(|pause| pause.paused_generation == self.shared.capture_generation())
            && self.shared.ptt_held()
            && !self.shared.is_capturing()
        {
            self.shared.set_capturing(true);
        }
    }

    pub(super) fn handle_pending_finalization(&mut self, json: &Value) -> Result<Option<Flow>> {
        self.resume_held_capture();
        let kind = json.get("type").and_then(Value::as_str);
        if !self.pending_generation_matches() {
            return Ok(Some(self.finish_local_pause(
                false,
                Some("pending ASR generation canceled or superseded; no late insertion"),
            )));
        }
        let pause = self.auto_pause.local_pause.as_ref().unwrap();
        let generation = pause.owner_generation;
        let expect_ack = pause.expect_ack;
        match kind {
            Some("capabilities") => {
                let supported =
                    json.get("features")
                        .and_then(Value::as_array)
                        .is_some_and(|features| {
                            features.iter().any(|value| {
                                value.as_str() == Some(super::super::auto_pause::DRAIN_CAPABILITY)
                            })
                        });
                self.auto_pause.supported = supported;
                self.auto_pause.capabilities_known = true;
                if expect_ack && !supported {
                    return Ok(Some(
                        self.auto_pause_failure("server withdrew ordered-drain support"),
                    ));
                }
                if !expect_ack && supported {
                    let id = self.auto_pause.local_pause.as_ref().unwrap().request_id;
                    let request = serde_json::json!({"type":"drain", "request_id":id}).to_string();
                    if self
                        .writer
                        .as_ref()
                        .is_none_or(|writer| writer.send(WriterCommand::Text(request)).is_err())
                    {
                        return Ok(Some(self.auto_pause_failure(
                            "writer disconnected before ordered finalization",
                        )));
                    }
                    self.auto_pause.local_pause.as_mut().unwrap().expect_ack = true;
                }
            }
            Some("word") => {
                if let Some(word) = json.get("word").and_then(Value::as_str) {
                    if let Err(err) = self.accept_pending_word(word) {
                        return Ok(Some(self.auto_pause_failure(&err.to_string())));
                    }
                }
            }
            Some("final") => {
                let Some(text) = json.get("text").and_then(Value::as_str) else {
                    return Ok(Some(
                        self.auto_pause_failure("final result is missing text"),
                    ));
                };
                let suffix = match self.auto_pause.missing_final_words(text) {
                    Ok(suffix) => suffix,
                    Err(err) => return Ok(Some(self.auto_pause_failure(&err.to_string()))),
                };
                for word in suffix {
                    if !self.pending_generation_matches() {
                        return Ok(Some(self.auto_pause_failure(
                            "pending ASR capture canceled; no late insertion",
                        )));
                    }
                    if let Err(err) = self.accept_pending_word(&word) {
                        return Ok(Some(self.auto_pause_failure(&err.to_string())));
                    }
                }
                self.auto_pause
                    .runtime
                    .as_mut()
                    .unwrap()
                    .control
                    .final_seen(generation);
            }
            Some("drained") if expect_ack => {
                let Some(id) = json.get("request_id").and_then(Value::as_u64) else {
                    return Ok(Some(
                        self.auto_pause_failure("invalid finalization acknowledgement"),
                    ));
                };
                let outcome = self
                    .auto_pause
                    .runtime
                    .as_mut()
                    .unwrap()
                    .control
                    .acknowledge(id, generation);
                match outcome {
                    Ok(super::super::auto_pause::Completion::Stale) => {}
                    Ok(_) => return Ok(Some(self.finish_local_pause(true, None))),
                    Err(err) => return Ok(Some(self.auto_pause_failure(&err.to_string()))),
                }
            }
            Some("enginechanged" | "error") => {
                return Ok(Some(self.auto_pause_failure(
                    "server/engine failed or changed during pending ASR finalization",
                )));
            }
            // Speech belongs to the old stream, not to the newly armed wake or
            // buffered new capture. It cannot dispatch a stale control command.
            _ => {}
        }
        Ok(None)
    }

    pub(super) fn finish_local_pause(&mut self, confirmed: bool, reason: Option<&str>) -> Flow {
        let insertion_allowed = self.pending_insertion_allowed();
        let valid_generation = self.pending_generation_matches();
        let pause = self.auto_pause.local_pause.as_ref().unwrap();
        let generation = pause.owner_generation;
        let resuming = self.shared.is_capturing()
            && pause.paused_generation.checked_add(1) == Some(self.shared.capture_generation());
        let queued = std::mem::take(&mut self.auto_pause.frozen);
        let confirmed = confirmed && valid_generation;
        let mut deferred =
            std::mem::take(&mut self.auto_pause.local_pause.as_mut().unwrap().deferred);
        if !insertion_allowed {
            let held = self.buffer.pending_text();
            if !held.is_empty() {
                deferred.insert(0, held);
            }
            self.buffer.discard();
            self.utterance.clear();
        }
        if !valid_generation {
            deferred.clear();
        }
        // A confirmed old utterance may finish its own control routing only if
        // no newer capture has started. New wake/hold ownership always wins.
        if confirmed && insertion_allowed && !self.utterance.is_empty() {
            if let Err(err) = self.complete_utterance() {
                self.buffer.discard();
                self.shared.emit(DictationEvent::Error {
                    message: format!("ASR finalization: {err:#}"),
                });
            }
        }
        self.router.discard_pending();
        if let Err(err) = self.finish_capture_session(confirmed && insertion_allowed) {
            self.buffer.discard();
            self.shared.emit(DictationEvent::Error {
                message: format!("ASR capture close: {err:#}"),
            });
        }
        if !deferred.is_empty() {
            let text = deferred.join(" ");
            // Honour only the existing, explicit transcript-retention setting.
            if let Err(err) = self.recorder.record("deferred_no_insertion", &text, &text) {
                self.shared.emit(DictationEvent::Error {
                    message: format!("deferred transcript retention: {err:#}"),
                });
            }
            self.shared.emit(DictationEvent::DeferredTranscript {
                capture_generation: generation,
                text,
                confirmed,
            });
            self.shared.emit(DictationEvent::Error {
                message:
                    "previous ASR text not inserted after capture/focus changed; retained in memory"
                        .into(),
            });
        }
        self.auto_pause.local_pause = None;
        self.auto_pause.capture_generation = None;
        let flow = if resuming {
            match self.on_capture_edge(true) {
                Ok(()) => {
                    self.auto_pause.replay = queued;
                    self.auto_pause.replay_observed = false;
                    Flow::DrainedResume
                }
                Err(err) => {
                    self.shared.set_capturing(false);
                    self.shared.emit(DictationEvent::Error {
                        message: format!("new capture failed: {err:#}"),
                    });
                    Flow::DrainedPause
                }
            }
        } else {
            Flow::DrainedPause
        };
        self.shared.emit(DictationEvent::Finalization {
            capture_generation: generation,
            state: if confirmed {
                FinalizationState::Confirmed
            } else {
                FinalizationState::Unconfirmed
            },
        });
        if !confirmed {
            self.shared.emit(DictationEvent::Error {
                message: format!(
                    "dictation paused locally; ASR tail unconfirmed: {}",
                    reason.unwrap_or("finalization failed")
                ),
            });
        }
        flow
    }
}
