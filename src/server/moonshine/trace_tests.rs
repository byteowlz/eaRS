//! Opt-in, offline revision probe. Nothing in this file is shipped in the
//! engine loop. Inputs/traces must stay outside git (especially private audio).
//!
//! MOONSHINE_ORT_SINGLE_THREAD=1 EARS_MOONSHINE_TRACE_MODEL=/local/ort-bundle
//! EARS_MOONSHINE_TRACE_INPUTS=/local/inputs.json
//! EARS_MOONSHINE_TRACE_OUT=/local/trace.jsonl
//! cargo test --features moonshine --lib moonshine_revision_trace -- --ignored

use super::*;
use crate::server::vad::BoundaryVad;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    id: String,
    path: PathBuf,
}

fn snapshot(transcript: Transcript) -> Vec<serde_json::Value> {
    transcript
        .lines
        .into_iter()
        .map(|line| {
            json!({
                "id": line.id,
                "text": line.text,
                "complete": line.is_complete,
                "updated": line.is_updated,
                "text_changed": line.has_text_changed,
            })
        })
        .collect()
}

fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn model_hashes(dir: &Path) -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file())
        .map(|path| {
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                json!(hash(&path)),
            )
        })
        .collect()
}

/// Actual native polling, plus the same ingress boundary detector and forced
/// flush used by the production adapter. Audio-time replay, not live wall time.
#[test]
#[ignore = "requires native models and explicitly selected local recordings"]
fn moonshine_revision_trace() {
    let dir = PathBuf::from(std::env::var("EARS_MOONSHINE_TRACE_MODEL").unwrap());
    let inputs_path = std::env::var("EARS_MOONSHINE_TRACE_INPUTS").unwrap();
    let inputs: Vec<Input> = serde_json::from_slice(&std::fs::read(inputs_path).unwrap()).unwrap();
    let packet_ms: usize = std::env::var("EARS_MOONSHINE_TRACE_PACKET_MS")
        .unwrap_or_else(|_| "80".into())
        .parse()
        .unwrap();
    assert!((10..=500).contains(&packet_ms));
    let tail_ms: usize = std::env::var("EARS_MOONSHINE_TRACE_TAIL_MS")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap();
    assert!(tail_ms <= 5000);
    let boundary_vad =
        std::env::var("EARS_MOONSHINE_TRACE_BOUNDARY_VAD").map_or(true, |value| value != "0");
    let mut output = File::options();
    output.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        output.mode(0o600);
    }
    let mut out = output
        .open(std::env::var("EARS_MOONSHINE_TRACE_OUT").unwrap())
        .unwrap();
    let files = model_hashes(&dir);
    writeln!(
        out,
        "{}",
        json!({"kind":"model", "runtime":moonshine_rs::get_version(),
        "packet_ms":packet_ms, "model_files":files, "ingress_hangover_ms":300, "boundary_vad":boundary_vad, "tail_ms":tail_ms})
    )
    .unwrap();
    let options = TranscriberOptions::new().with_ort_providers("cpu");
    let load_started = Instant::now();
    let model = Transcriber::from_files(&dir, ModelArch::TinyStreaming, Some(&options)).unwrap();
    writeln!(
        out,
        "{}",
        json!({"kind":"model_ready", "load_s":load_started.elapsed().as_secs_f64()})
    )
    .unwrap();
    for input in inputs {
        let started = Instant::now();
        let (mut pcm, rate) = kaudio::pcm_decode(&input.path).unwrap();
        if rate != 24_000 {
            pcm = kaudio::resample(&pcm, rate as usize, 24_000).unwrap();
        }
        let duration = pcm.len() as f64 / 24_000.0;
        pcm.resize(pcm.len() + tail_ms * 24, 0.0);
        let mut vad = BoundaryVad::new(boundary_vad, 24_000, 300);
        let mut stream = model.create_stream().unwrap();
        let mut elapsed = 0.0;
        let mut received = false;
        writeln!(
            out,
            "{}",
            json!({"kind":"clip", "clip":input.id,
            "input_sha256":hash(&input.path), "seconds":duration})
        )
        .unwrap();
        for chunk in pcm.chunks(packet_ms * 24) {
            elapsed += chunk.len() as f64 / 24_000.0;
            stream.add_audio(chunk, 24_000).unwrap();
            let lines = snapshot(stream.poll(false).unwrap());
            writeln!(
                out,
                "{}",
                json!({"kind":"snapshot", "clip":input.id,
                "audio_s":elapsed,"lines":lines})
            )
            .unwrap();
            received = true;
            if vad.observe(chunk) == Some(false) {
                let lines = snapshot(stream.finalize().unwrap());
                writeln!(
                    out,
                    "{}",
                    json!({"kind":"snapshot", "clip":input.id,
                    "audio_s":elapsed,"lines":lines,"forced_boundary":true})
                )
                .unwrap();
                stream = model.create_stream().unwrap();
                received = false;
            }
        }
        if received {
            let lines = snapshot(stream.finalize().unwrap());
            writeln!(
                out,
                "{}",
                json!({"kind":"snapshot", "clip":input.id,
                "audio_s":elapsed,"lines":lines,"stop":true})
            )
            .unwrap();
        }
        writeln!(
            out,
            "{}",
            json!({"kind":"clip_result", "clip":input.id,
            "wall_s":started.elapsed().as_secs_f64()})
        )
        .unwrap();
        eprintln!(
            "traced {} ({:.1}s audio) in {:.2}s",
            input.id,
            duration,
            started.elapsed().as_secs_f64()
        );
    }
}
