//! Shared, versioned vector motion packages and deterministic playback.
mod bars;
mod runtime;
use oqtopus_core::{decode_pack, Pack};
pub use oqtopus_core::{FramePaths, PathCommand};
pub use runtime::{MotionFrame, MotionPlayer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

const MAGIC: &[u8; 8] = b"BWMOTION";
const HEADER: usize = 54;
const MAX_MANIFEST: usize = 1024 * 1024;
const MAX_PAYLOAD: usize = 64 * 1024 * 1024 + 52;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionError(pub String);
impl fmt::Display for MotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for MotionError {}
fn error(s: &str) -> MotionError {
    MotionError(s.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bar {
    pub center: [f32; 2],
    pub width: f32,
    pub length: f32,
    pub gain: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaveState {
    pub id: String,
    pub amplitude: f32,
    pub period: f32,
    pub stagger: f32,
    pub dwell: f32,
    pub speed: f32,
    pub direction: f32,
    pub mirror: bool,
    pub wave: Wave,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Wave {
    Sine,
    EasedTriangle,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BarsWave {
    pub view_box: [f32; 2],
    pub padding: f32,
    pub rotation_degrees: f32,
    pub travel_degrees: f32,
    pub detune: f32,
    pub bars: Vec<Bar>,
    pub states: Vec<WaveState>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
pub enum MotionAction {
    Idle,
    Play { clip: String },
    Hold { clip: String },
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StateRule {
    pub id: String,
    pub action: MotionAction,
    /// Permit a missing optional clip and explicitly fall back to idle.
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum SourceSpec {
    BarsWave(BarsWave),
    VectorPack {
        #[serde(default)]
        states: Vec<StateRule>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    version: u32,
    id: String,
    source: SourceSpec,
}

#[derive(Debug, Clone)]
pub enum MotionSource {
    BarsWave(BarsWave),
    VectorPack(Pack),
}
#[derive(Debug, Clone)]
pub struct MotionPackage {
    manifest: Manifest,
    source: MotionSource,
    payload: Vec<u8>,
}
impl MotionPackage {
    pub fn id(&self) -> &str {
        &self.manifest.id
    }
    pub fn source(&self) -> &MotionSource {
        &self.source
    }
    pub fn states(&self) -> Vec<&str> {
        match &self.source {
            MotionSource::BarsWave(config) => config.states.iter().map(|s| s.id.as_str()).collect(),
            MotionSource::VectorPack(_) => {
                self.state_rules().iter().map(|s| s.id.as_str()).collect()
            }
        }
    }
    /// Compile a validated manifest and payload into a self-contained package.
    pub fn from_manifest(data: &[u8], payload: Vec<u8>) -> Result<Self, MotionError> {
        if data.len() > MAX_MANIFEST || payload.len() > MAX_PAYLOAD {
            return Err(error("motion package exceeds size limit"));
        }
        let manifest: Manifest = serde_json::from_slice(data)
            .map_err(|e| MotionError(format!("invalid motion manifest: {e}")))?;
        if manifest.format != "byteowlz.motion" || manifest.version != 1 || !valid_id(&manifest.id)
        {
            return Err(error("invalid motion format, version or id"));
        }
        let source = match &manifest.source {
            SourceSpec::BarsWave(config) => {
                if !payload.is_empty() {
                    return Err(error("procedural source must not contain a payload"));
                }
                validate_bars(config)?;
                MotionSource::BarsWave(config.clone())
            }
            SourceSpec::VectorPack { states } => {
                let pack = decode_pack(&payload).map_err(|e| MotionError(e.to_string()))?;
                validate_rules(states, &pack)?;
                MotionSource::VectorPack(pack)
            }
        };
        Ok(Self {
            manifest,
            source,
            payload,
        })
    }
    pub fn state_rules(&self) -> &[StateRule] {
        match &self.manifest.source {
            SourceSpec::VectorPack { states } => states,
            _ => &[],
        }
    }
    /// Set caller-defined state rules without changing any artwork.
    pub fn with_state_rules(mut self, states: Vec<StateRule>) -> Result<Self, MotionError> {
        let MotionSource::VectorPack(pack) = &self.source else {
            return Err(error("state rules require a vector pack"));
        };
        validate_rules(&states, pack)?;
        self.manifest.source = SourceSpec::VectorPack { states };
        Ok(self)
    }
    pub fn vector_pack(id: &str, payload: Vec<u8>) -> Result<Self, MotionError> {
        let manifest = Manifest {
            format: "byteowlz.motion".into(),
            version: 1,
            id: id.into(),
            source: SourceSpec::VectorPack { states: vec![] },
        };
        Self::from_manifest(
            &serde_json::to_vec(&manifest).map_err(|e| MotionError(e.to_string()))?,
            payload,
        )
    }
}
fn validate_rules(states: &[StateRule], pack: &Pack) -> Result<(), MotionError> {
    if states.len() > 128 {
        return Err(error("too many state rules"));
    }
    let mut ids = BTreeSet::new();
    let clips = pack
        .manifest
        .get("clips")
        .and_then(|v| v.as_object())
        .ok_or_else(|| error("invalid clips"))?;
    for state in states {
        if !valid_id(&state.id) || !ids.insert(&state.id) {
            return Err(error("invalid or duplicate state rule"));
        }
        if let MotionAction::Play { clip } | MotionAction::Hold { clip } = &state.action {
            let exists = clips.iter().any(|(name, meta)| {
                name == clip || meta.get("action").and_then(|v| v.as_str()) == Some(clip)
            });
            if !exists && !state.optional {
                return Err(MotionError(format!(
                    "state {} references unknown clip {clip}",
                    state.id
                )));
            }
        }
    }
    Ok(())
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_./".contains(&c))
        && !id.split('/').any(|s| s.is_empty() || s == "." || s == "..")
}
fn validate_bars(config: &BarsWave) -> Result<(), MotionError> {
    let [w, h] = config.view_box;
    if !w.is_finite()
        || !h.is_finite()
        || w <= 0.0
        || h <= 0.0
        || !config.padding.is_finite()
        || config.padding < 0.0
        || config.padding * 2.0 >= w.min(h)
        || !config.rotation_degrees.is_finite()
        || !config.travel_degrees.is_finite()
        || !config.detune.is_finite()
        || !(0.0..1.0).contains(&config.detune)
        || config.bars.is_empty()
        || config.bars.len() > 128
        || config.states.is_empty()
        || config.states.len() > 128
    {
        return Err(error("invalid bar-wave configuration"));
    }
    for b in &config.bars {
        if !b.center.iter().all(|v| v.is_finite())
            || !b.width.is_finite()
            || b.width <= 0.0
            || !b.length.is_finite()
            || b.length <= 0.0
            || !b.gain.is_finite()
            || b.gain < 0.0
        {
            return Err(error("invalid bar geometry"));
        }
    }
    let mut ids = BTreeSet::new();
    for s in &config.states {
        if !valid_id(&s.id)
            || !ids.insert(&s.id)
            || ![
                s.amplitude,
                s.period,
                s.stagger,
                s.dwell,
                s.speed,
                s.direction,
            ]
            .iter()
            .all(|v| v.is_finite())
            || s.amplitude < 0.0
            || s.period <= 0.0
            || s.speed <= 0.0
            || !(0.0..=0.95).contains(&s.dwell)
        {
            return Err(error("invalid bar-wave state"));
        }
    }
    Ok(())
}
/// Serialize a self-contained version-1 motion package with a SHA-256 checksum.
pub fn encode_package(package: &MotionPackage) -> Result<Vec<u8>, MotionError> {
    let manifest = serde_json::to_vec(&package.manifest).map_err(|e| MotionError(e.to_string()))?;
    if manifest.len() > MAX_MANIFEST || package.payload.len() > MAX_PAYLOAD {
        return Err(error("motion package exceeds size limit"));
    }
    let mut digest = Sha256::new();
    digest.update(&manifest);
    digest.update(&package.payload);
    let mut out = Vec::with_capacity(HEADER + manifest.len() + package.payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
    out.extend_from_slice(&(package.payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&digest.finalize());
    out.extend_from_slice(&manifest);
    out.extend_from_slice(&package.payload);
    Ok(out)
}
/// Read generic v1 packages; existing OQTPACK v1 files remain a supported input.
pub fn decode_package(data: &[u8]) -> Result<MotionPackage, MotionError> {
    if data.starts_with(b"OQTPACK\0") {
        if data.len() > MAX_PAYLOAD {
            return Err(error("motion package exceeds size limit"));
        }
        return MotionPackage::vector_pack("legacy-oqtopus", data.to_vec());
    }
    if data.len() < HEADER || &data[..8] != MAGIC || u16::from_le_bytes([data[8], data[9]]) != 1 {
        return Err(error("unsupported motion package header"));
    }
    let manifest_len = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
    let payload_len = u64::from_le_bytes(data[14..22].try_into().unwrap());
    if manifest_len > MAX_MANIFEST || payload_len > MAX_PAYLOAD as u64 {
        return Err(error("motion package exceeds size limit"));
    }
    let expected = HEADER
        .checked_add(manifest_len)
        .and_then(|v| v.checked_add(payload_len as usize))
        .ok_or_else(|| error("invalid motion package length"))?;
    if expected != data.len() {
        return Err(error("motion package length mismatch"));
    }
    let mut digest = Sha256::new();
    digest.update(&data[HEADER..]);
    if digest.finalize().as_slice() != &data[22..54] {
        return Err(error("motion package checksum mismatch"));
    }
    MotionPackage::from_manifest(
        &data[HEADER..HEADER + manifest_len],
        data[HEADER + manifest_len..].to_vec(),
    )
}
