//! Draft native decoder, vector geometry and single-clip sampling primitives.
//! The full graph player and native renderers are not implemented.
mod breath;
mod geometry;
mod player;
mod sampler;
pub use breath::{breathing_frame, BreathParams};
use flate2::{Decompress, FlushDecompress, Status};
pub use geometry::{decode_frame, FramePaths, PathCommand};
pub use player::{BreathMode, Mode, Player, PlayerSample, PlayerState, RestMode};
pub use sampler::{ClipSampler, Layer, Sample};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const HEADER: usize = 52;
const MAX_RAW: u64 = 128 * 1024 * 1024;
const MAX_PACK: usize = 64 * 1024 * 1024;
const MAX_ITEMS: u32 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackError(pub String);
impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PackError {}
type Result<T> = std::result::Result<T, PackError>;
fn error(message: &str) -> PackError {
    PackError(message.to_owned())
}

/// Original delta-encoded tenths of a vector pixel, plus untouched frame metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub metadata: Map<String, Value>,
    pub body: Vec<Vec<i32>>,
    pub stripe: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorDocument {
    pub metadata: Map<String, Value>,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pack {
    pub manifest: Map<String, Value>,
    pub rest: Frame,
    /// Named `clip:<id>` and `bridge:<id>`; original document metadata is retained.
    pub documents: BTreeMap<String, VectorDocument>,
}

// serde_json::Value alone accepts duplicate object keys. Reject them recursively.
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON without duplicate keys")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::from(v)))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::from(v)))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
                Number::from_f64(v)
                    .map(|n| Unique(Value::Number(n)))
                    .ok_or_else(|| E::custom("invalid number"))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::String(v.to_owned())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::String(v)))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Null))
            }
            fn visit_none<E: de::Error>(self) -> std::result::Result<Unique, E> {
                self.visit_unit()
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut values = Vec::new();
                while let Some(Unique(value)) = seq.next_element()? {
                    values.push(value);
                }
                Ok(Unique(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut values = Map::new();
                while let Some((key, Unique(value))) = map.next_entry::<String, Unique>()? {
                    if values.insert(key, value).is_some() {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(Unique(Value::Object(values)))
            }
        }
        d.deserialize_any(V)
    }
}

fn json_object(data: &[u8]) -> Result<Map<String, Value>> {
    match serde_json::from_slice::<Unique>(data)
        .map_err(|e| PackError(format!("invalid JSON: {e}")))?
        .0
    {
        Value::Object(value) => Ok(value),
        _ => Err(error("expected JSON object")),
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn uint(&mut self) -> Result<u32> {
        let mut value = 0u64;
        for i in 0..5 {
            let byte = *self
                .data
                .get(self.pos)
                .ok_or_else(|| error("truncated varint"))?;
            self.pos += 1;
            value |= u64::from(byte & 127) << (7 * i);
            if byte & 128 == 0 {
                if value > u64::from(u32::MAX) || (i > 0 && byte == 0) {
                    return Err(error("invalid varint"));
                }
                return Ok(value as u32);
            }
        }
        Err(error("varint too long"))
    }
    fn blob(&mut self) -> Result<&'a [u8]> {
        let n = self.uint()? as usize;
        let end = self
            .pos
            .checked_add(n)
            .filter(|&v| v <= self.data.len())
            .ok_or_else(|| error("truncated blob"))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }
    fn loops(&mut self) -> Result<Vec<Vec<i32>>> {
        let count = self.uint()?;
        if count > MAX_ITEMS {
            return Err(error("too many loops"));
        }
        let mut loops = Vec::new();
        for _ in 0..count {
            let n = self.uint()?;
            if !(6..=MAX_ITEMS).contains(&n) || n % 2 != 0 {
                return Err(error("invalid vector loop"));
            }
            let mut points = Vec::new();
            for _ in 0..n {
                let v = self.uint()?;
                points.push(((v >> 1) as i32) ^ -((v & 1) as i32));
            }
            loops.push(points);
        }
        Ok(loops)
    }
    fn frame(&mut self) -> Result<Frame> {
        let metadata = json_object(self.blob()?)?;
        if metadata.contains_key("b") || metadata.contains_key("s") {
            return Err(error("invalid frame metadata"));
        }
        Ok(Frame {
            metadata,
            body: self.loops()?,
            stripe: self.loops()?,
        })
    }
}

fn unpack(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < HEADER || data.len() > MAX_PACK + HEADER {
        return Err(error("invalid pack size"));
    }
    if &data[..8] != b"OQTPACK\0"
        || u16::from_le_bytes([data[8], data[9]]) != 1
        || u16::from_le_bytes([data[10], data[11]]) != 1
    {
        return Err(error("unsupported pack header"));
    }
    let size = u64::from_le_bytes(data[12..20].try_into().expect("fixed header"));
    if size > MAX_RAW {
        return Err(error("payload exceeds size limit"));
    }
    let compressed = &data[HEADER..];
    let mut stream = Decompress::new(true);
    let mut raw = vec![0; size as usize + 1];
    let status = stream
        .decompress(compressed, &mut raw, FlushDecompress::Finish)
        .map_err(|_| error("invalid compressed payload"))?;
    if status != Status::StreamEnd
        || stream.total_out() != size
        || stream.total_in() != compressed.len() as u64
    {
        return Err(error("payload length or trailing data mismatch"));
    }
    raw.truncate(size as usize);
    if Sha256::digest(&raw).as_slice() != &data[20..52] {
        return Err(error("payload checksum mismatch"));
    }
    Ok(raw)
}

fn object<'a>(value: Option<&'a Value>, label: &str) -> Result<&'a Map<String, Value>> {
    value.and_then(Value::as_object).ok_or_else(|| error(label))
}
fn list<'a>(value: Option<&'a Value>, label: &str) -> Result<&'a Vec<Value>> {
    value.and_then(Value::as_array).ok_or_else(|| error(label))
}
fn id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
}
fn document_matches(
    doc: &VectorDocument,
    info: &Map<String, Value>,
    manifest: &Map<String, Value>,
) -> bool {
    !doc.frames.is_empty()
        && info.get("frames").and_then(Value::as_u64) == Some(doc.frames.len() as u64)
        && doc.metadata.get("fps") == manifest.get("fps")
        && doc.metadata.get("size") == manifest.get("vector_size")
}
fn frame_index(value: Option<&Value>, limit: usize) -> bool {
    value
        .and_then(Value::as_u64)
        .is_some_and(|v| v < limit as u64)
}

fn validate_manifest(m: &Map<String, Value>) -> Result<()> {
    if m.get("format_version").and_then(Value::as_u64) != Some(1)
        || !m.get("fps").and_then(Value::as_u64).is_some_and(|n| n > 0)
    {
        return Err(error("invalid manifest version or fps"));
    }
    let size = list(m.get("vector_size"), "invalid vector size")?;
    if size.len() != 2 || size.iter().any(|v| !v.as_u64().is_some_and(|n| n > 0)) {
        return Err(error("invalid vector size"));
    }
    match m.get("state_mapping") {
        Some(Value::Null | Value::Object(_)) => {}
        _ => return Err(error("invalid state mapping")),
    }
    let breath = object(m.get("breath"), "invalid breath parameters")?;
    if !matches!(
        breath.get("default").and_then(Value::as_str),
        Some("off" | "subtle" | "soft")
    ) {
        return Err(error("invalid breath default"));
    }
    for key in ["breath_seconds", "sway_seconds"] {
        if !breath
            .get(key)
            .and_then(Value::as_f64)
            .is_some_and(|v| v.is_finite() && v > 0.0)
        {
            return Err(error("invalid breath period"));
        }
    }
    for mode in ["subtle", "soft"] {
        let config = object(breath.get(mode), "invalid breath mode")?;
        for key in ["swell", "sway"] {
            if !config
                .get(key)
                .and_then(Value::as_f64)
                .is_some_and(|v| v.is_finite() && v >= 0.0)
            {
                return Err(error("invalid breath mode"));
            }
        }
    }
    Ok(())
}

fn validate_clips(pack: &Pack) -> Result<BTreeSet<String>> {
    let m = &pack.manifest;
    let clips = object(m.get("clips"), "invalid clips")?;
    if clips.is_empty() {
        return Err(error("empty clips"));
    }
    let mut expected = BTreeSet::new();
    for (name, value) in clips {
        if !id(name) {
            return Err(error("invalid clip id"));
        }
        let key = format!("clip:{name}");
        let doc = pack
            .documents
            .get(&key)
            .ok_or_else(|| error("missing clip"))?;
        if !document_matches(doc, object(Some(value), "invalid clip metadata")?, m) {
            return Err(error("invalid clip document"));
        }
        expected.insert(key);
    }
    Ok(expected)
}

fn validate_transitions(pack: &Pack) -> Result<()> {
    let m = &pack.manifest;
    let clips = object(m.get("clips"), "invalid clips")?;
    let transitions = object(m.get("transitions"), "invalid transitions")?;
    let edges = object(transitions.get("edges"), "invalid transition edges")?;
    if edges.len() != clips.len() || edges.keys().any(|name| !clips.contains_key(name)) {
        return Err(error("transition sources do not match clips"));
    }
    for (source, entries) in edges {
        let source_len = pack.documents[&format!("clip:{source}")].frames.len();
        for edge in list(Some(entries), "invalid transition list")? {
            let edge = list(Some(edge), "invalid transition edge")?;
            if edge.len() != 4 {
                return Err(error("invalid transition edge"));
            }
            let target = edge[1]
                .as_str()
                .ok_or_else(|| error("invalid transition target"))?;
            let target_doc = pack
                .documents
                .get(&format!("clip:{target}"))
                .filter(|_| clips.contains_key(target))
                .ok_or_else(|| error("missing transition target"))?;
            if !frame_index(edge.first(), source_len)
                || !frame_index(edge.get(2), target_doc.frames.len())
                || !edge[3].as_f64().is_some_and(|v| v.is_finite() && v >= 0.0)
            {
                return Err(error("invalid transition reference"));
            }
        }
    }
    Ok(())
}

fn validate_bridges(pack: &Pack, mut expected: BTreeSet<String>) -> Result<()> {
    let m = &pack.manifest;
    let clips = object(m.get("clips"), "invalid clips")?;
    let bridges = object(m.get("bridges"), "invalid bridges")?;
    for bridge in list(bridges.get("bridges"), "invalid bridges")? {
        let info = object(Some(bridge), "invalid bridge")?;
        let name = info
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| id(s))
            .ok_or_else(|| error("invalid bridge id"))?;
        let source = info
            .get("from")
            .and_then(Value::as_str)
            .filter(|s| clips.contains_key(*s))
            .ok_or_else(|| error("invalid bridge source"))?;
        let key = format!("bridge:{name}");
        let doc = pack
            .documents
            .get(&key)
            .ok_or_else(|| error("missing bridge"))?;
        if !expected.insert(key)
            || !frame_index(
                info.get("exit"),
                pack.documents[&format!("clip:{source}")].frames.len(),
            )
            || !document_matches(doc, info, m)
        {
            return Err(error("invalid bridge reference"));
        }
    }
    if expected.len() != pack.documents.len()
        || pack.documents.keys().any(|key| !expected.contains(key))
    {
        return Err(error("unexpected vector record"));
    }
    Ok(())
}

fn validate(pack: &Pack) -> Result<()> {
    validate_manifest(&pack.manifest)?;
    let expected = validate_clips(pack)?;
    validate_transitions(pack)?;
    validate_bridges(pack, expected)
}

/// Decode and validate the entire self-contained v1 pack with bounded decompression.
pub fn decode_pack(data: &[u8]) -> Result<Pack> {
    let raw = unpack(data)?;
    let mut reader = Reader { data: &raw, pos: 0 };
    let manifest = json_object(reader.blob()?)?;
    let count = reader.uint()?;
    if count > MAX_ITEMS {
        return Err(error("too many records"));
    }
    let mut rest = None;
    let mut documents = BTreeMap::new();
    for _ in 0..count {
        let name = std::str::from_utf8(reader.blob()?)
            .map_err(|_| error("invalid record name"))?
            .to_owned();
        let metadata = json_object(reader.blob()?)?;
        let n = reader.uint()?;
        if n == 0 || n > MAX_ITEMS {
            return Err(error("invalid frame count"));
        }
        let mut frames = Vec::new();
        for _ in 0..n {
            frames.push(reader.frame()?);
        }
        if name == "rest" {
            if rest.is_some() || n != 1 || !metadata.is_empty() {
                return Err(error("invalid rest record"));
            }
            rest = frames.pop();
        } else if metadata.contains_key("frames")
            || documents
                .insert(name, VectorDocument { metadata, frames })
                .is_some()
        {
            return Err(error("invalid or duplicate vector record"));
        }
    }
    if reader.pos != raw.len() {
        return Err(error("trailing payload"));
    }
    let pack = Pack {
        manifest,
        rest: rest.ok_or_else(|| error("missing rest"))?,
        documents,
    };
    validate(&pack)?;
    Ok(pack)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_json_keys_at_all_depths() {
        assert!(json_object(br#"{"x":1,"x":2}"#).is_err());
        assert!(json_object(br#"{"nested":{"x":1,"x":2}}"#).is_err());
    }

    #[test]
    fn rejects_graph_forged_with_valid_checksum() {
        use flate2::{write::ZlibEncoder, Compression};
        use std::io::Write;
        let hex = include_str!("../tests/synthetic-pack.hex")
            .trim()
            .as_bytes();
        let bytes: Vec<u8> = hex
            .chunks_exact(2)
            .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
            .collect();
        let mut raw = unpack(&bytes).unwrap();
        let needle = b"\"idle-b\",1,0.1";
        let index = raw
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        raw[index + 6] = b'x';
        let mut compressed = ZlibEncoder::new(Vec::new(), Compression::default());
        compressed.write_all(&raw).unwrap();
        let mut forged = Vec::from(&b"OQTPACK\0"[..]);
        forged.extend_from_slice(&1u16.to_le_bytes());
        forged.extend_from_slice(&1u16.to_le_bytes());
        forged.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        forged.extend_from_slice(&Sha256::digest(&raw));
        forged.extend_from_slice(&compressed.finish().unwrap());
        assert!(decode_pack(&forged).is_err());
    }

    #[test]
    fn rejects_truncated_oversized_and_noncanonical_varints() {
        for bytes in [
            &[0x80, 0x00][..],
            &[0x80],
            &[0xff, 0xff, 0xff, 0xff, 0x10],
            &[0x80; 6],
        ] {
            assert!(Reader {
                data: bytes,
                pos: 0
            }
            .uint()
            .is_err());
        }
    }
}
