//! Offline source-clock A/B. Explicit model/WAV only; no live config or uploads.
#[allow(dead_code)]
#[path = "../src/server/vad.rs"]
mod boundary_vad;
#[path = "segmentation_compare/segmenter.rs"]
mod segmenter;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use moonshine_rs::{ModelArch, Transcriber, TranscriberOptions, Transcript};
use segmenter::{Crop, Region, Segmenter};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

const RATE: usize = 24_000;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Profile {
    Current,
    Merge800,
}

/// Streaming architecture of the explicit bundle; must match the files.
#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Arch {
    Tiny,
    Small,
}

impl Arch {
    fn model_arch(self) -> ModelArch {
        match self {
            Arch::Tiny => ModelArch::TinyStreaming,
            Arch::Small => ModelArch::SmallStreaming,
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "tiny")]
    arch: Arch,
    #[arg(long)]
    wav: PathBuf,
    #[arg(long, value_enum)]
    profile: Profile,
    #[arg(long, default_value_t = 80)]
    packet_ms: usize,
    #[arg(long, default_value_t = 2000)]
    tail_ms: usize,
}

#[derive(Serialize)]
struct Event {
    text: String,
    emitted_at_s: f64,
    start_s: f64,
    end_s: f64,
    capped: bool,
}
#[derive(Default)]
struct Capture {
    seen: HashSet<u64>,
    events: Vec<Event>,
    inference_s: f64,
}
impl Capture {
    fn absorb(&mut self, transcript: Transcript, force_final: bool, offset: f64, now: f64) {
        for line in transcript.lines {
            if line.text.trim().is_empty()
                || !(force_final || line.is_complete)
                || !self.seen.insert(line.id)
            {
                continue;
            }
            self.events.push(Event {
                text: line.text,
                emitted_at_s: now,
                start_s: offset + f64::from(line.start_time),
                end_s: offset + f64::from(line.start_time + line.duration),
                capped: false,
            });
        }
    }
}

fn digest(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn load(model: &Path, arch: Arch, options: TranscriberOptions) -> Result<Transcriber> {
    for name in [
        "adapter.ort",
        "cross_kv.ort",
        "decoder_kv.ort",
        "encoder.ort",
        "frontend.model.ort",
        "frontend.weights.ort",
        "streaming_config.json",
        "tokenizer.bin",
    ] {
        if !model.join(name).is_file() {
            bail!("Explicit model lacks {name}; no fallback");
        }
    }
    Transcriber::from_files(model, arch.model_arch(), Some(&options))
        .with_context(|| format!("loading explicit {arch:?} streaming model"))
}

fn current(model: &Transcriber, audio: &[f32], packet: usize) -> Result<Capture> {
    let mut result = Capture::default();
    let mut vad = boundary_vad::BoundaryVad::new(true, RATE, 300);
    let mut stream = Some(model.create_stream()?);
    let mut offset = 0.0;
    let mut samples = 0;
    let mut received = false;
    for chunk in audio.chunks(packet) {
        samples += chunk.len();
        let now = samples as f64 / RATE as f64;
        let clock = Instant::now();
        let active = stream.as_mut().context("stream")?;
        active.add_audio(chunk, RATE as u32)?;
        let update = active.poll(false)?;
        result.inference_s += clock.elapsed().as_secs_f64();
        result.absorb(update, false, offset, now);
        received = true;
        if vad.observe(chunk) == Some(false) {
            let clock = Instant::now();
            let final_text = stream.take().context("stream")?.finalize()?;
            result.inference_s += clock.elapsed().as_secs_f64();
            result.absorb(final_text, true, offset, now);
            stream = Some(model.create_stream()?);
            result.seen.clear();
            offset = now;
            received = false;
        }
    }
    if received {
        let clock = Instant::now();
        let final_text = stream.take().context("stream")?.finalize()?;
        result.inference_s += clock.elapsed().as_secs_f64();
        result.absorb(final_text, true, offset, samples as f64 / RATE as f64);
    }
    Ok(result)
}

fn regions(transcript: Transcript, available: usize) -> Vec<Region> {
    transcript
        .lines
        .into_iter()
        .map(|line| Region {
            id: line.id,
            start: (line.start_time.max(0.0) as f64 * RATE as f64).round() as usize,
            end: (((line.start_time + line.duration).max(0.0) as f64 * RATE as f64).round()
                as usize)
                .min(available),
            complete: line.is_complete,
        })
        .collect()
}

fn decode_crop(
    decoder: &Transcriber,
    audio: &[f32],
    crop: Crop,
    result: &mut Capture,
) -> Result<()> {
    if crop.start >= crop.end || crop.end > audio.len() || crop.end > crop.emitted_at {
        bail!("Invalid/future crop");
    }
    let clock = Instant::now();
    let text = decoder
        .transcribe(&audio[crop.start..crop.end], RATE as u32)?
        .text();
    result.inference_s += clock.elapsed().as_secs_f64();
    result.events.push(Event {
        text,
        emitted_at_s: crop.emitted_at as f64 / RATE as f64,
        start_s: crop.start as f64 / RATE as f64,
        end_s: crop.end as f64 / RATE as f64,
        capped: crop.capped,
    });
    Ok(())
}

fn merge800(
    detector: &Transcriber,
    decoder: &Transcriber,
    audio: &[f32],
    packet: usize,
) -> Result<Capture> {
    let mut result = Capture::default();
    let mut coordinator = Segmenter::new(RATE);
    let mut stream = detector.create_stream()?;
    let mut available = 0;
    for chunk in audio.chunks(packet) {
        available += chunk.len();
        let clock = Instant::now();
        stream.add_audio(chunk, RATE as u32)?;
        let observed = regions(stream.poll(true)?, available);
        result.inference_s += clock.elapsed().as_secs_f64();
        if let Some(crop) = coordinator.observe(available, &observed, false) {
            decode_crop(decoder, audio, crop, &mut result)?;
        }
    }
    let clock = Instant::now();
    let observed = regions(stream.finalize()?, available);
    result.inference_s += clock.elapsed().as_secs_f64();
    if let Some(crop) = coordinator.observe(available, &observed, true) {
        decode_crop(decoder, audio, crop, &mut result)?;
    }
    Ok(result)
}

fn last_energy_activity(audio: &[f32]) -> f64 {
    audio
        .chunks(RATE * 20 / 1000)
        .enumerate()
        .filter_map(|(i, chunk)| {
            let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            (rms > 0.0025)
                .then_some(((i + 1) * RATE * 20 / 1000).min(audio.len()) as f64 / RATE as f64)
        })
        .last()
        .unwrap_or(0.0)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(10..=500).contains(&args.packet_ms) || args.tail_ms > 5000 {
        bail!("Invalid packet/tail duration");
    }
    let (mut audio, rate) = kaudio::pcm_decode(&args.wav)?;
    if rate != RATE as u32 {
        audio = kaudio::resample(&audio, rate as usize, RATE)?;
    }
    let source_samples = audio.len();
    let last_energy_s = last_energy_activity(&audio);
    audio.resize(audio.len() + args.tail_ms * RATE / 1000, 0.0);
    let base = TranscriberOptions::new().with_ort_providers("cpu");
    let loading = Instant::now();
    let result = match args.profile {
        Profile::Current => {
            let model = load(&args.model, args.arch, base)?;
            let load_s = loading.elapsed().as_secs_f64();
            (
                current(&model, &audio, args.packet_ms * RATE / 1000)?,
                load_s,
            )
        }
        Profile::Merge800 => {
            let detector = load(
                &args.model,
                args.arch,
                base.clone()
                    .set("skip_transcription", "true")
                    .set("vad_max_segment_duration", "60"),
            )?;
            // Only explicitly Silero-positive crops reach this decoder. No
            // arbitrary non-speech recording is decoded with VAD disabled.
            let decoder = load(
                &args.model,
                args.arch,
                base.set("vad_threshold", "0")
                    .set("vad_max_segment_duration", "60"),
            )?;
            let load_s = loading.elapsed().as_secs_f64();
            (
                merge800(&detector, &decoder, &audio, args.packet_ms * RATE / 1000)?,
                load_s,
            )
        }
    };
    let models: BTreeMap<_, _> = std::fs::read_dir(&args.model)?
        .map(|entry| {
            let path = entry?.path();
            Ok((
                path.file_name()
                    .context("model filename")?
                    .to_string_lossy()
                    .into_owned(),
                digest(&path)?,
            ))
        })
        .collect::<Result<_>>()?;
    let text = result
        .0
        .events
        .iter()
        .map(|e| e.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let final_delay_proxy_s = result
        .0
        .events
        .iter()
        .filter(|e| !e.text.trim().is_empty())
        .last()
        .map(|e| e.emitted_at_s - last_energy_s);
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "profile":args.profile,"arch":args.arch,"packet_ms":args.packet_ms,"tail_ms":args.tail_ms,"input_sha256":digest(&args.wav)?,
            "model_files":models,"source_seconds":source_samples as f64/RATE as f64,"last_energy_s":last_energy_s,
            "final_delay_proxy_s":final_delay_proxy_s,"inference_s":result.0.inference_s,"load_s":result.1,
            "text":text,"events":result.0.events,"runtime":moonshine_rs::get_version(),
            "runtime_provenance":serde_json::from_str::<serde_json::Value>(include_str!("../vendor/moonshine-sys/provenance.json"))?
        }))?
    );
    Ok(())
}
