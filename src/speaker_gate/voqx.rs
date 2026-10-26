//! Concrete canonical voqx recognition adapter. No registry/model is loaded here.
//! Current voqx/speakrs cannot supply a full-audio calibrated speaker-consistency detector or
//! attest the loaded checkpoint. Consequently this adapter is evidence-only and
//! NEVER produces allowlist-eligible instructions, even for a confident match.

use super::{Reason, Revision};
use sha2::{Digest, Sha256};
use voqx_core::{
    recognition::{Recognition, Thresholds},
    registry::VoiceRegistry,
};

/// An explicitly supplied canonical recognition snapshot. Synthesis profiles are
/// removed, not interpreted as speaker permission. No on-disk second registry.
pub struct PreparedRegistry {
    registry: VoiceRegistry,
    thresholds: Thresholds,
    revision: Revision,
}

impl PreparedRegistry {
    /// Thresholds and calibration provenance must be explicitly supplied by the
    /// operator. Validation is NOT empirical calibration or a security guarantee.
    /// Model version must exactly match the checkpoint-scoped recognition profiles.
    pub fn new(
        mut registry: VoiceRegistry,
        engine: &str,
        model_version: &str,
        thresholds: Thresholds,
        calibration_id: &str,
    ) -> Result<Self, Reason> {
        if engine.is_empty()
            || model_version.is_empty()
            || calibration_id.is_empty()
            || !thresholds.confident.is_finite()
            || !thresholds.tentative.is_finite()
            || thresholds.tentative < -1.0
            || thresholds.confident > 1.0
            || thresholds.tentative >= thresholds.confident
        {
            return Err(Reason::IncompatibleModel);
        }
        for identity in &mut registry.identities {
            identity.recognition_profiles.retain(|profile| {
                profile.engine == engine && profile.model_version == model_version
            });
            if identity.recognition_profiles.len() > 1
                || identity.recognition_profiles.iter().any(|profile| {
                    profile.centroid.is_empty()
                        || profile.centroid.iter().any(|value| !value.is_finite())
                        || !profile.centroid.iter().any(|value| *value != 0.0)
                })
            {
                return Err(Reason::IncompatibleModel);
            }
            identity.synthesis_profiles.clear();
            identity.wake_profiles.clear();
            identity.sources.clear();
        }
        registry
            .identities
            .retain(|identity| !identity.recognition_profiles.is_empty());
        registry.identities.sort_by(|a, b| a.id.cmp(&b.id));
        if registry.identities.is_empty()
            || registry
                .identities
                .windows(2)
                .any(|pair| pair[0].id == pair[1].id)
        {
            return Err(Reason::IncompatibleModel);
        }
        // Canonical recognition profiles + explicit thresholds/provenance only.
        // Hashes are private internal revision fences, never telemetry.
        let profiles: Vec<_> = registry
            .identities
            .iter()
            .map(|identity| (&identity.id, &identity.recognition_profiles))
            .collect();
        let encoded = serde_json::to_vec(&(profiles, thresholds, calibration_id))
            .map_err(|_| Reason::IncompatibleModel)?;
        let revision = Revision {
            engine: engine.into(),
            model: model_version.into(),
            profiles: format!("{:x}", Sha256::digest(encoded)),
        };
        Ok(Self {
            registry,
            thresholds,
            revision,
        })
    }

    pub fn revision(&self) -> Revision {
        self.revision.clone()
    }

    /// Uses voqx's canonical matcher; embedding matching is not authorization.
    pub fn identify_embedding(
        &self,
        embedding: &[f32],
        engine: &str,
    ) -> Result<Recognition, Reason> {
        if embedding.is_empty() || embedding.iter().any(|value| !value.is_finite()) {
            return Err(Reason::ProviderError);
        }
        if engine != self.revision.engine
            || self.registry.identities.iter().any(|identity| {
                identity
                    .recognition_profiles
                    .iter()
                    .any(|profile| profile.centroid.len() != embedding.len())
            })
        {
            return Err(Reason::IncompatibleModel);
        }
        Ok(self.registry.identify(embedding, engine, self.thresholds))
    }
}

/// Current speakrs 0.4.2's exact unpadded/untruncated input window at 24 kHz.
/// Reject ALL other lengths before inference; never authorize a long prefix.
pub fn supported_window(samples_24k: usize) -> Result<(), Reason> {
    match samples_24k.cmp(&240_000) {
        std::cmp::Ordering::Less => Err(Reason::InsufficientAudio),
        std::cmp::Ordering::Greater => Err(Reason::UnsupportedCoverage),
        std::cmp::Ordering::Equal => Ok(()),
    }
}

#[cfg(feature = "speaker-voqx")]
pub struct VoqxProvider {
    embedder: voqx_recognition::Embedder,
    profiles: PreparedRegistry,
}

#[cfg(feature = "speaker-voqx")]
impl VoqxProvider {
    /// Accept only explicitly prepared, owned resources. Does not download,
    /// initialize a model, inspect a real registry or change acceleration modes.
    pub fn from_prepared(
        embedder: voqx_recognition::Embedder,
        profiles: PreparedRegistry,
    ) -> Result<Self, Reason> {
        if profiles.revision.engine != voqx_recognition::EMBEDDING_MODEL_ID {
            return Err(Reason::IncompatibleModel);
        }
        Ok(Self { embedder, profiles })
    }
}

#[cfg(feature = "speaker-voqx")]
impl super::EvidenceProvider for VoqxProvider {
    fn revision(&self) -> Revision {
        self.profiles.revision()
    }
    fn instruction_capable(&self) -> bool {
        false
    }
    fn evaluate(
        &mut self,
        pcm: &[f32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<super::Evidence, Reason> {
        use std::sync::atomic::Ordering;
        supported_window(pcm.len())?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Reason::Stale);
        }
        let samples = kaudio::resample(pcm, 24_000, 16_000).map_err(|_| Reason::ProviderError)?;
        if samples.len() != 160_000 {
            return Err(Reason::UnsupportedCoverage);
        }
        let embedding = self
            .embedder
            .embed(&samples)
            .map_err(|_| Reason::ProviderError)?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Reason::Stale);
        }
        let recognition = self
            .profiles
            .identify_embedding(&embedding, voqx_recognition::EMBEDDING_MODEL_ID)?;
        Ok(super::Evidence {
            recognition,
            revision: self.profiles.revision(),
            covered_samples: pcm.len(),
            speaker_consistency: super::SpeakerConsistency::Unsupported,
        })
    }
}
