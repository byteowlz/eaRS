//! Offline client-VAD proof: print sample-clock spans, never modify recordings.

#[cfg(not(feature = "client-vad"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("build with --features client-vad; see docs/client-vad.md")
}

#[cfg(feature = "client-vad")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use ears::dictation::{
        capture_segments::{CaptureSegments, Config},
        client_vad::{FRAME_SAMPLES, SAMPLE_RATE, SileroVad},
    };
    use std::path::Path;

    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: client_vad_segments MODEL.onnx AUDIO (prints JSONL; no writes)"
    );
    let audio = Path::new(&args[1]);
    ensure!(
        std::fs::metadata(audio)?.len() <= 32 * 1024 * 1024,
        "audio input exceeds 32 MiB preview limit"
    );
    let (pcm, rate) = kaudio::pcm_decode(audio).context("failed to decode preview audio")?;
    ensure!(
        rate == SAMPLE_RATE,
        "preview requires 16 kHz input; decoder uses first channel"
    );
    ensure!(
        pcm.len() <= SAMPLE_RATE as usize * 300,
        "preview exceeds five minutes"
    );
    let runtime = std::env::var_os("ORT_DYLIB_PATH")
        .context("set ORT_DYLIB_PATH to the local ONNX Runtime library")?;
    let mut detector = SileroVad::load(Path::new(&args[0]), Path::new(&runtime))?;
    let mut segments = CaptureSegments::new(Config {
        sample_rate_hz: SAMPLE_RATE,
        channels: 1,
        frame_samples: FRAME_SAMPLES,
        speech_threshold: 0.5,
        pre_roll_frames: 32,
        min_speech_frames: 4,
        silence_frames: 16,
        post_roll_frames: 16,
        max_segment_frames: 3750,
    })?;
    // Buffer output until success: an error cannot masquerade as a complete run.
    let mut spans = Vec::new();
    let summary = |segment: ears::dictation::capture_segments::Segment| {
        serde_json::json!({
            "start_sample": segment.span.start.min(pcm.len() as u64),
            "end_sample": segment.span.end.min(pcm.len() as u64),
            "speech_start_sample": segment.speech_span.start.min(pcm.len() as u64),
            "speech_end_sample": segment.speech_span.end.min(pcm.len() as u64),
            "sample_rate": SAMPLE_RATE,
            "end_reason": format!("{:?}", segment.end_reason),
            "preview_only": true,
        })
    };
    for (i, input) in pcm.chunks(FRAME_SAMPLES).enumerate() {
        let mut frame = [0.0; FRAME_SAMPLES];
        frame[..input.len()].copy_from_slice(input);
        let score = detector.score(&frame)?;
        if let Some(segment) = segments.push_frame((i * FRAME_SAMPLES) as u64, &frame, score)? {
            spans.push(summary(segment));
        }
    }
    if let Some(segment) = segments.finish()? {
        spans.push(summary(segment));
    }
    for span in spans {
        println!("{span}");
    }
    Ok(())
}
