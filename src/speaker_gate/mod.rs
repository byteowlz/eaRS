//! Optional unwanted-speaker suppression. Evidence is NOT authentication or a tool grant.
//!
//! The PCM sealed for ASR is the PCM submitted to the evidence worker. Partial
//! text is never admitted. Consumers must supply their current scope at dispatch,
//! refresh registry revisions on revocation, and separately authorize actual effects.
//! Legacy WebSocket timestamps cannot establish this binding; see docs/speaker-gate.md.

use crossbeam_channel::{Receiver, Sender, bounded};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};
use voqx_core::recognition::{Recognition, RecognitionStatus};

pub mod voqx;

/// Mono PCM at the eaRS ASR ingress sample rate.
pub const SAMPLE_RATE: usize = 24_000;
pub const MAX_SAMPLES: usize = SAMPLE_RATE * 30;
pub const MAX_TEXT_BYTES: usize = 4096;
const MAX_PENDING: usize = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Disabled,
    /// Reference observations only; never dispatches instructions.
    Annotate,
    Allowlist,
}

/// Per-consumer policy, not an identity registry. No default security thresholds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    pub mode: Mode,
    pub allowed_voice_ids: Vec<String>,
    pub revision: u64,
    pub deadline_ms: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            mode: Mode::Disabled,
            allowed_voice_ids: Vec::new(),
            revision: 0,
            deadline_ms: 2000,
        }
    }
}

/// Current consumer/target lifecycle and authoritative recognition-profile snapshot.
/// Change generation on cancel, reconnect, focus/target or ASR model change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    pub consumer: String,
    pub target: String,
    pub generation: u64,
    pub policy: Policy,
    pub recognition: Revision,
}

/// Checkpoint and canonical recognition-profile snapshot, not synthesis profiles.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Revision {
    pub engine: String,
    pub model: String,
    pub profiles: String,
}

/// Opaque process-local utterance identity. Callers cannot allocate/reuse IDs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct UtteranceId(String);

/// Bounded, privacy-safe reasons. Never include provider error strings or biometrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    ProviderUnavailable,
    ProviderError,
    Timeout,
    Overloaded,
    InvalidAudio,
    InsufficientAudio,
    UnsupportedCoverage,
    IncompatibleModel,
    Unknown,
    Tentative,
    AmbiguousSpeaker,
    SpeakerCheckUnavailable,
    Disallowed,
    Stale,
    PayloadConflict,
    AlreadyConsumed,
    NotFinal,
    NotEligible,
    Unauthorized,
    UnsupportedAudioBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Pending,
    Eligible,
    /// Reference-only observation, optionally explaining uncertainty/failure.
    Observed(Option<Reason>),
    Denied(Reason),
}

/// Structural capability is not empirical/live qualification. There is no
/// automatic "ready" state derived from configuration or provider presence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    Unavailable,
    EvidenceOnly,
    ProviderContractUnqualified,
}

/// No voice names, vectors, scores, PCM, transcript, target or profile paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Status {
    pub mode: Mode,
    pub provider_present: bool,
    pub readiness: Readiness,
    pub pending: usize,
}

/// Detector evidence over the complete PCM, not a guarantee of speaker purity,
/// authentication or liveness. Consistent requires a qualified detector with
/// documented coverage, calibration and uncertainty handling, not cosine alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerConsistency {
    Unsupported,
    Uncertain,
    Overlap,
    SpeakerChange,
    Consistent,
}

/// Trusted provider output; never deserialize this from an untrusted ASR stream.
/// Deliberately not Debug/Serialize: Recognition includes private display names.
pub struct Evidence {
    pub recognition: Recognition,
    pub revision: Revision,
    pub covered_samples: usize,
    /// A single embedding or confident cosine cannot establish consistency.
    pub speaker_consistency: SpeakerConsistency,
}

/// A local worker seam. Implementations must not silently change models/providers.
/// Cancellation is cooperative; deadlines fence even an uncooperative provider.
pub trait EvidenceProvider: Send + 'static {
    fn revision(&self) -> Revision;
    fn instruction_capable(&self) -> bool;
    fn evaluate(&mut self, pcm: &[f32], cancelled: &AtomicBool) -> Result<Evidence, Reason>;
}

/// Immutable PCM token shared by ASR and recognition. No public constructor.
#[derive(Clone)]
pub struct AsrInput {
    id: UtteranceId,
    epoch: u64,
    pcm: Arc<[f32]>,
    audio_hash: [u8; 32],
}

/// An authoritative final produced from exactly one sealed PCM token.
/// Corrections must increase revision and rerun the same bound ASR input.
pub struct FinalTranscript {
    input: AsrInput,
    revision: u64,
    text: String,
}

impl AsrInput {
    pub fn id(&self) -> &UtteranceId {
        &self.id
    }

    /// Run on an ASR/inference thread, not the capture/render thread. The supplied
    /// transcriber receives exactly the evidence PCM, mono 24 kHz. Apply dictionary
    /// normalization in this closure before returning the final, never partial, text.
    pub fn transcribe<E>(
        &self,
        revision: u64,
        transcriber: impl FnOnce(&[f32]) -> Result<String, E>,
    ) -> Result<FinalTranscript, E> {
        Ok(FinalTranscript {
            input: self.clone(),
            revision,
            text: transcriber(&self.pcm)?,
        })
    }

    /// Concrete existing eaRS Kyutai completed-audio adapter. No model is loaded
    /// here; the caller supplies its exclusively owned, unbatched prepared model.
    /// Reset around each input so earlier captures/priming cannot leak into this
    /// final. Never use a shared server/batched model. No microphone/IO.
    pub fn transcribe_kyutai(
        &self,
        revision: u64,
        model: &mut crate::Model,
        normalize: impl FnOnce(&str) -> String,
    ) -> anyhow::Result<FinalTranscript> {
        self.transcribe(revision, |pcm| {
            if model.state.batch_size() != 1 {
                anyhow::bail!("speaker-bound ASR requires an exclusively owned unbatched model");
            }
            model.reset_stream_state()?;
            let result = model.transcribe_pcm(pcm.to_vec());
            let reset = model.reset_stream_state();
            let result = result?;
            reset?;
            Ok(normalize(&result.text))
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Binding {
    id: UtteranceId,
    epoch: u64,
    audio: [u8; 32],
    transcript: [u8; 32],
    transcript_revision: u64,
    attempt: u64,
}

struct Entry {
    audio: Vec<f32>,
    sealed: Option<AsrInput>,
    binding: Option<Binding>,
    text: String,
    outcome: Outcome,
    deadline: Option<Instant>,
    cancelled: Arc<AtomicBool>,
}

struct Job {
    binding: Binding,
    scope: Scope,
    pcm: Arc<[f32]>,
    cancelled: Arc<AtomicBool>,
}
struct Verdict {
    binding: Binding,
    evidence: Result<Evidence, Reason>,
}

/// Bounded join/dispatch module: at most four utterances and one queued evidence
/// job. A hung worker can deny availability, never approve a later utterance.
pub struct Gate {
    scope: Scope,
    epoch: u64,
    attempt: u64,
    entries: HashMap<UtteranceId, Entry>,
    jobs: Option<Sender<Job>>,
    results: Option<Receiver<Verdict>>,
    capable: bool,
}

impl Gate {
    pub fn new(scope: Scope, provider: Option<Box<dyn EvidenceProvider>>) -> Self {
        let mut gate = Self {
            scope,
            epoch: 0,
            attempt: 0,
            entries: HashMap::new(),
            jobs: None,
            results: None,
            capable: false,
        };
        if let Some(provider) = provider {
            // An incompatible prepared provider must not be substituted silently.
            if provider.revision() != gate.scope.recognition {
                return gate;
            }
            gate.capable = provider.instruction_capable();
            let (jobs_tx, jobs_rx) = bounded::<Job>(1);
            let (results_tx, results_rx) = bounded(1);
            let spawned = std::thread::Builder::new()
                .name("ears-speaker-evidence".into())
                .spawn(move || evidence_worker(provider, jobs_rx, results_tx));
            if spawned.is_ok() {
                gate.jobs = Some(jobs_tx);
                gate.results = Some(results_rx);
            }
        }
        gate
    }

    pub fn status(&self) -> Status {
        Status {
            mode: self.scope.policy.mode,
            provider_present: self.jobs.is_some(),
            readiness: if self.jobs.is_none() {
                Readiness::Unavailable
            } else if self.capable {
                Readiness::ProviderContractUnqualified
            } else {
                Readiness::EvidenceOnly
            },
            pending: self.entries.len(),
        }
    }

    /// Fences every in-flight result, even if a caller reused a revision number.
    /// Invoke for revocation, cancellation, reconnect and target/focus changes.
    pub fn reconfigure(&mut self, scope: Scope) {
        self.cancel_all();
        self.epoch = self.epoch.wrapping_add(1);
        self.scope = scope;
    }

    fn cancel_all(&mut self) {
        for entry in self.entries.values() {
            entry.cancelled.store(true, Ordering::Release);
        }
        self.entries.clear();
    }

    pub fn begin(&mut self) -> Result<UtteranceId, Reason> {
        if self.entries.len() >= MAX_PENDING {
            return Err(Reason::Overloaded);
        }
        let id = UtteranceId(uuid::Uuid::new_v4().to_string());
        self.entries.insert(
            id.clone(),
            Entry {
                audio: Vec::new(),
                sealed: None,
                binding: None,
                text: String::new(),
                outcome: Outcome::Pending,
                deadline: None,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );
        Ok(id)
    }

    /// Feed the SAME chunks that form this utterance's ASR input, before sealing.
    /// An oversized/invalid chunk poisons the whole utterance; no prefix approval.
    pub fn push_pcm(&mut self, id: &UtteranceId, pcm: &[f32]) -> Result<(), Reason> {
        let entry = self.entries.get_mut(id).ok_or(Reason::Stale)?;
        if let Outcome::Denied(reason) = entry.outcome {
            return Err(reason);
        }
        if entry.sealed.is_some() {
            // Appending after a completed boundary cannot authorize just the
            // earlier prefix, even if a caller ignores this error.
            deny(entry, Reason::PayloadConflict);
            return Err(Reason::PayloadConflict);
        }
        if entry.outcome != Outcome::Pending {
            return Err(Reason::NotFinal);
        }
        if pcm.len() > MAX_SAMPLES.saturating_sub(entry.audio.len())
            || pcm.iter().any(|s| !s.is_finite())
        {
            entry.audio.clear();
            entry.outcome = Outcome::Denied(Reason::InvalidAudio);
            return Err(Reason::InvalidAudio);
        }
        entry.audio.extend_from_slice(pcm);
        Ok(())
    }

    /// Completed acoustic boundary. Duplicate/split VAD notifications return the
    /// same token, not another instruction. Audio cannot change after this call.
    pub fn seal_audio(&mut self, id: &UtteranceId) -> Result<AsrInput, Reason> {
        let entry = self.entries.get_mut(id).ok_or(Reason::Stale)?;
        if let Outcome::Denied(reason) = entry.outcome {
            return Err(reason);
        }
        if let Some(input) = &entry.sealed {
            return Ok(input.clone());
        }
        if entry.audio.is_empty() {
            entry.outcome = Outcome::Denied(Reason::InsufficientAudio);
            return Err(Reason::InsufficientAudio);
        }
        let mut hash = Sha256::new();
        for sample in &entry.audio {
            hash.update(sample.to_bits().to_le_bytes());
        }
        let input = AsrInput {
            id: id.clone(),
            epoch: self.epoch,
            pcm: std::mem::take(&mut entry.audio).into(),
            audio_hash: hash.finalize().into(),
        };
        entry.sealed = Some(input.clone());
        Ok(input)
    }

    /// Join authoritative final and full PCM. A newer correction invalidates the
    /// earlier verdict; reusing a revision for a changed payload poisons the ID.
    pub fn submit_final(
        &mut self,
        final_text: FinalTranscript,
        now: Instant,
    ) -> Result<(), Reason> {
        let input = final_text.input;
        if input.epoch != self.epoch {
            return Err(Reason::Stale);
        }
        let entry = self.entries.get_mut(&input.id).ok_or(Reason::Stale)?;
        if entry
            .sealed
            .as_ref()
            .is_none_or(|saved| saved.audio_hash != input.audio_hash)
        {
            return Err(Reason::Stale);
        }
        if final_text.text.is_empty() || final_text.text.len() > MAX_TEXT_BYTES {
            deny(entry, Reason::PayloadConflict);
            return Err(Reason::PayloadConflict);
        }
        let transcript: [u8; 32] = Sha256::digest(final_text.text.as_bytes()).into();
        if let Some(previous) = &entry.binding {
            if final_text.revision < previous.transcript_revision {
                return Err(Reason::Stale);
            }
            if final_text.revision == previous.transcript_revision {
                if transcript == previous.transcript {
                    return Ok(());
                }
                deny(entry, Reason::PayloadConflict);
                return Err(Reason::PayloadConflict);
            }
        }
        // Terminal denials are not retryable by changing text or revision.
        if let Outcome::Denied(reason) = entry.outcome {
            return Err(reason);
        }
        entry.cancelled.store(true, Ordering::Release);
        entry.cancelled = Arc::new(AtomicBool::new(false));
        self.attempt = self.attempt.wrapping_add(1);
        let binding = Binding {
            id: input.id,
            epoch: self.epoch,
            audio: input.audio_hash,
            transcript,
            transcript_revision: final_text.revision,
            attempt: self.attempt,
        };
        entry.binding = Some(binding.clone());
        entry.text = final_text.text;
        if self.scope.policy.mode == Mode::Disabled {
            entry.outcome = Outcome::Eligible;
            return Ok(());
        }
        let deadline = *entry.deadline.get_or_insert_with(|| {
            now + Duration::from_millis(self.scope.policy.deadline_ms.min(30_000))
        });
        if now >= deadline {
            deny(entry, Reason::Timeout);
            return Err(Reason::Timeout);
        }
        entry.outcome = Outcome::Pending;
        let Some(jobs) = &self.jobs else {
            deny(entry, Reason::ProviderUnavailable);
            return Err(Reason::ProviderUnavailable);
        };
        let job = Job {
            binding,
            scope: self.scope.clone(),
            pcm: input.pcm,
            cancelled: entry.cancelled.clone(),
        };
        if let Err(error) = jobs.try_send(job) {
            let reason = if error.is_full() {
                Reason::Overloaded
            } else {
                Reason::ProviderUnavailable
            };
            deny(entry, reason);
            return Err(reason);
        }
        Ok(())
    }

    /// Nonblocking; invoke from the consumer event loop. No inference/IO here.
    pub fn poll(&mut self, now: Instant) {
        if let Some(results) = &self.results {
            while let Ok(verdict) = results.try_recv() {
                let Some(entry) = self.entries.get_mut(&verdict.binding.id) else {
                    continue;
                };
                if entry.binding.as_ref() != Some(&verdict.binding)
                    || entry.outcome != Outcome::Pending
                {
                    continue;
                }
                if entry.deadline.is_none_or(|deadline| now >= deadline) {
                    deny(entry, Reason::Timeout);
                    continue;
                }
                let samples = entry.sealed.as_ref().map_or(0, |input| input.pcm.len());
                let result = verdict
                    .evidence
                    .and_then(|evidence| evaluate(&self.scope, samples, &evidence));
                entry.outcome = if self.scope.policy.mode == Mode::Annotate {
                    Outcome::Observed(result.err())
                } else {
                    match result {
                        Ok(()) if !self.capable => Outcome::Denied(Reason::SpeakerCheckUnavailable),
                        Ok(()) => Outcome::Eligible,
                        Err(reason) => Outcome::Denied(reason),
                    }
                };
            }
        }
        for entry in self.entries.values_mut() {
            if matches!(entry.outcome, Outcome::Pending | Outcome::Eligible)
                && entry.deadline.is_some_and(|deadline| now >= deadline)
            {
                deny(entry, Reason::Timeout);
            }
        }
    }

    pub fn outcome(&self, id: &UtteranceId) -> Result<Outcome, Reason> {
        self.entries
            .get(id)
            .map(|entry| entry.outcome)
            .ok_or(Reason::AlreadyConsumed)
    }

    pub fn discard(&mut self, id: &UtteranceId) {
        if let Some(entry) = self.entries.remove(id) {
            entry.cancelled.store(true, Ordering::Release);
        }
    }

    /// One-shot dispatch, exact normalized text and current scope join. The
    /// independent permission check runs immediately before the caller's effect.
    /// This module implements no robot/controller/tool authority or actuation.
    pub fn dispatch(
        &mut self,
        id: &UtteranceId,
        expected_text: &str,
        current: &Scope,
        now: Instant,
        authorize: impl FnOnce() -> bool,
        effect: impl FnOnce(&str),
    ) -> Result<(), Reason> {
        self.poll(now);
        // Consume BEFORE authorization/effect (including panics and failed grants).
        let entry = self.entries.remove(id).ok_or(Reason::AlreadyConsumed)?;
        entry.cancelled.store(true, Ordering::Release);
        if current != &self.scope {
            return Err(Reason::Stale);
        }
        if expected_text != entry.text {
            return Err(Reason::PayloadConflict);
        }
        if entry.outcome != Outcome::Eligible {
            return Err(match entry.outcome {
                Outcome::Denied(reason) => reason,
                _ => Reason::NotEligible,
            });
        }
        if current.policy.mode != Mode::Disabled
            && entry.deadline.is_none_or(|deadline| now >= deadline)
        {
            return Err(Reason::Timeout);
        }
        if !authorize() {
            return Err(Reason::Unauthorized);
        }
        effect(&entry.text);
        Ok(())
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.cancel_all();
    }
}

/// Own the prepared provider for the worker lifetime; never create per-request
/// threads or restart a hung native provider to bypass the admission bound.
fn evidence_worker(
    mut provider: Box<dyn EvidenceProvider>,
    jobs: Receiver<Job>,
    results: Sender<Verdict>,
) {
    while let Ok(job) = jobs.recv() {
        if job.cancelled.load(Ordering::Acquire) {
            continue;
        }
        let evidence = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if provider.revision() != job.scope.recognition {
                return Err(Reason::IncompatibleModel);
            }
            provider.evaluate(&job.pcm, &job.cancelled)
        }))
        .unwrap_or(Err(Reason::ProviderError));
        if job.cancelled.load(Ordering::Acquire) {
            continue;
        }
        // Bounded result queue. Consumer drop disconnects this worker.
        if results
            .send(Verdict {
                binding: job.binding,
                evidence,
            })
            .is_err()
        {
            break;
        }
    }
}

fn deny(entry: &mut Entry, reason: Reason) {
    entry.cancelled.store(true, Ordering::Release);
    entry.outcome = Outcome::Denied(reason);
}

fn evaluate(scope: &Scope, samples: usize, evidence: &Evidence) -> Result<(), Reason> {
    if evidence.revision != scope.recognition
        || evidence.recognition.engine != scope.recognition.engine
        || scope.recognition.engine.is_empty()
        || scope.recognition.model.is_empty()
        || scope.recognition.profiles.is_empty()
    {
        return Err(Reason::IncompatibleModel);
    }
    if evidence.covered_samples != samples {
        return Err(Reason::UnsupportedCoverage);
    }
    match evidence.recognition.status {
        RecognitionStatus::Confident => {}
        RecognitionStatus::Tentative => return Err(Reason::Tentative),
        RecognitionStatus::Unknown => return Err(Reason::Unknown),
        RecognitionStatus::InsufficientAudio => return Err(Reason::InsufficientAudio),
        RecognitionStatus::IncompatibleModel => return Err(Reason::IncompatibleModel),
    }
    if evidence
        .recognition
        .score
        .is_none_or(|score| !score.is_finite() || !(-1.0..=1.0).contains(&score))
    {
        return Err(Reason::Unknown);
    }
    match evidence.speaker_consistency {
        SpeakerConsistency::Consistent => {}
        SpeakerConsistency::Unsupported => return Err(Reason::SpeakerCheckUnavailable),
        _ => return Err(Reason::AmbiguousSpeaker),
    }
    let Some(voice_id) = &evidence.recognition.voice_id else {
        return Err(Reason::Unknown);
    };
    if scope.policy.mode == Mode::Allowlist && !scope.policy.allowed_voice_ids.contains(voice_id) {
        return Err(Reason::Disallowed);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
