//! Continuous microphone conversion to mono 24 kHz PCM.
//!
//! F32/I16/U16 devices at 8–192 kHz are supported; other formats/rates fail
//! before starting the stream. Channels are equally weighted (not speaker-layout
//! aware). Partial interleaved frames, filter history and fractional sample phase
//! survive callbacks. A new stream, not an utterance, starts a new converter.
//!
//! Reuse the wake path's windowed-sinc filter. Limit each downsampling stage to
//! 2:1 so its fixed input-domain window still suppresses aliases at high rates.
//! Feed fixed 16-frame quanta so even floating-point phase bookkeeping is
//! independent of device callback boundaries. Filter lookahead plus that quantum
//! holds back at most 128 output samples (<6 ms), not a callback's worth each
//! time. There is no stop-time flush: continuous capture supplies the lookahead;
//! stream shutdown discards only that bounded tail and any incomplete channel
//! frame. Native 24 kHz bypasses both filtering and quantum buffering entirely.

use crate::dictation::wake::Resampler;
use anyhow::{Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;
use std::thread;

const CAPTURE_RATE: u32 = 24_000;
const RESAMPLE_QUANTUM: usize = 16;

/// Stream-local state shared by all supported sample formats.
struct CaptureInput {
    channels: usize,
    frame_sum: f64,
    frame_len: usize,
    resamplers: Vec<Resampler>,
    pending: Vec<f32>,
}

impl CaptureInput {
    fn new(rate: u32, channels: u16, format: cpal::SampleFormat) -> Result<Self> {
        if !matches!(
            format,
            cpal::SampleFormat::F32 | cpal::SampleFormat::I16 | cpal::SampleFormat::U16
        ) {
            bail!("Unsupported sample format: {format:?}. Supported formats: F32, I16, U16");
        }
        if !(8_000..=192_000).contains(&rate) {
            bail!("Unsupported input sample rate: {rate} Hz. Supported range: 8000..=192000 Hz");
        }
        if channels == 0 {
            bail!("Input device must have at least one channel");
        }
        let mut resamplers = Vec::new();
        let mut from = rate;
        while from > 2 * CAPTURE_RATE {
            let to = from.div_ceil(2);
            resamplers.push(Resampler::new(from as usize, to as usize));
            from = to;
        }
        if from != CAPTURE_RATE {
            resamplers.push(Resampler::new(from as usize, CAPTURE_RATE as usize));
        }
        Ok(Self {
            channels: channels as usize,
            frame_sum: 0.0,
            frame_len: 0,
            resamplers,
            pending: Vec::with_capacity(RESAMPLE_QUANTUM),
        })
    }

    fn push<T>(&mut self, data: &[T]) -> Vec<f32>
    where
        T: cpal::Sample,
        f32: cpal::FromSample<T>,
    {
        let mut mono = Vec::with_capacity((self.frame_len + data.len()) / self.channels);
        for &sample in data {
            let sample = sample.to_sample::<f32>();
            // A malformed floating-point sample must not poison filter history.
            self.frame_sum += if sample.is_finite() {
                sample as f64
            } else {
                0.0
            };
            self.frame_len += 1;
            if self.frame_len == self.channels {
                mono.push((self.frame_sum / self.channels as f64) as f32);
                self.frame_sum = 0.0;
                self.frame_len = 0;
            }
        }
        if self.resamplers.is_empty() {
            return mono;
        }
        self.pending.extend(mono);
        let complete = self.pending.len() / RESAMPLE_QUANTUM * RESAMPLE_QUANTUM;
        let mut output = Vec::new();
        for block in self.pending[..complete].chunks_exact(RESAMPLE_QUANTUM) {
            let mut pcm = self.resamplers[0].push(block);
            for resampler in &mut self.resamplers[1..] {
                pcm = resampler.push(&pcm);
            }
            output.extend(pcm);
        }
        self.pending.drain(..complete);
        output
    }
}

/// Name of the default input device, as used by `start_audio_capture(_, None)`.
pub fn default_input_device_name() -> Option<String> {
    cpal::default_host()
        .default_input_device()
        .and_then(|device| device.name().ok())
}

pub fn list_audio_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Available input devices:");
    for (i, device) in host.input_devices()?.enumerate() {
        let name = device.name().unwrap_or("Unknown".to_string());
        println!("  {}: {}", i, name);
    }
    Ok(())
}

pub fn start_audio_capture(audio_tx: Sender<Vec<f32>>, device_index: Option<usize>) -> Result<()> {
    let max_retries = 3;
    let mut retry_count = 0;
    loop {
        match start_audio_capture_internal(audio_tx.clone(), device_index) {
            Ok(()) => break,
            Err(e) => {
                retry_count += 1;
                if retry_count >= max_retries {
                    return Err(e);
                }
                eprintln!(
                    "Audio capture failed (attempt {}/{}): {}",
                    retry_count, max_retries, e
                );
                eprintln!("Retrying audio capture in 1 second...");
                thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    Ok(())
}

fn build_capture_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    audio_tx: Sender<Vec<f32>>,
    mut input: CaptureInput,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let pcm = input.push(data);
            if !pcm.is_empty() && audio_tx.send(pcm).is_err() {
                eprintln!("Audio receiver disconnected");
            }
        },
        |err| {
            eprintln!("Audio stream error: {}", err);
            std::process::exit(1);
        },
        None,
    )?)
}

fn start_audio_capture_internal(
    audio_tx: Sender<Vec<f32>>,
    device_index: Option<usize>,
) -> Result<()> {
    let host = cpal::default_host();
    let device = if let Some(index) = device_index {
        host.input_devices()?
            .nth(index)
            .ok_or_else(|| anyhow::anyhow!("Device index {} not found", index))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?
    };
    let config = device.default_input_config()?;
    let format = config.sample_format();
    let input = CaptureInput::new(config.sample_rate().0, config.channels(), format)?;
    eprintln!(
        "Using input device: {}",
        device.name().unwrap_or("Unknown".to_string())
    );
    eprintln!("Sample rate: {}", config.sample_rate().0);
    eprintln!("Sample format: {:?}", format);
    eprintln!("Channels: {}", config.channels());
    let config = config.into();
    let stream = match format {
        cpal::SampleFormat::F32 => build_capture_stream::<f32>(&device, &config, audio_tx, input)?,
        cpal::SampleFormat::I16 => build_capture_stream::<i16>(&device, &config, audio_tx, input)?,
        cpal::SampleFormat::U16 => build_capture_stream::<u16>(&device, &config, audio_tx, input)?,
        _ => bail!("Unsupported sample format: {format:?}"),
    };
    stream.play()?;
    loop {
        thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn converter(rate: u32, channels: u16) -> CaptureInput {
        CaptureInput::new(rate, channels, cpal::SampleFormat::F32).unwrap()
    }

    fn tone(rate: u32, hz: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (TAU * hz as f64 * i as f64 / rate as f64).sin() as f32 * 0.6)
            .collect()
    }

    fn partitioned<T>(input: &mut CaptureInput, data: &[T]) -> Vec<f32>
    where
        T: cpal::Sample,
        f32: cpal::FromSample<T>,
    {
        let sizes = [1, 7, 127, 256, 513, 3, 1024, 31];
        let mut output = Vec::new();
        let mut pos = 0;
        let mut part = 0;
        while pos < data.len() {
            // Empty callbacks must neither flush nor reset state.
            assert!(input.push::<T>(&[]).is_empty());
            let end = (pos + sizes[part % sizes.len()]).min(data.len());
            output.extend(input.push(&data[pos..end]));
            pos = end;
            part += 1;
        }
        output
    }

    fn assert_close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().chain(b).all(|x| x.is_finite()));
        let error = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(error <= tolerance, "max error {error} > {tolerance}");
    }

    #[test]
    fn callback_partition_preserves_phase_duration_and_channels() {
        for rate in [
            8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000,
        ] {
            let mono = tone(rate, 1_000.0, rate as usize + 113);
            let data: Vec<f32> = mono.iter().flat_map(|&x| [x * 0.5, x, x * 1.5]).collect();
            let whole = converter(rate, 3).push(&data);
            let split = partitioned(&mut converter(rate, 3), &data);
            assert_eq!(whole, split);
            let expected = mono.len() as f64 * CAPTURE_RATE as f64 / rate as f64;
            assert!(
                split.len() as f64 <= expected && expected - split.len() as f64 <= 128.0,
                "rate={rate}, len={}",
                split.len()
            );
            let ideal = tone(CAPTURE_RATE, 1_000.0, split.len());
            // Initial zero extension is a boundary transient, not a phase shift.
            assert_close(&split[100..], &ideal[100..], 0.004);
        }
    }

    #[test]
    fn callback_partition_at_short_and_exact_rate_boundaries() {
        for rate in [
            8_000, 8_001, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 48_001, 96_000, 191_999,
            192_000,
        ] {
            for extra in 0..33 {
                let data = tone(rate, 1_000.0, rate as usize / 100 + extra);
                let whole = converter(rate, 1).push(&data);
                let mut input = converter(rate, 1);
                let split: Vec<f32> = data.chunks(1).flat_map(|part| input.push(part)).collect();
                assert_eq!(
                    whole.len(),
                    split.len(),
                    "rate={rate}, input length={}",
                    data.len()
                );
                assert_eq!(whole, split);
            }
        }
    }

    #[test]
    fn arbitrary_channel_mean_and_partial_frames() {
        for channels in [1, 2, 3, 4, 6, 8, 32] {
            let data: Vec<f32> = (0..channels as usize * 101)
                .map(|i| (i % 9) as f32 / 16.0)
                .collect();
            let expected: Vec<f32> = data
                .chunks(channels as usize)
                .map(|frame| frame.iter().map(|&x| x as f64).sum::<f64>() as f32 / channels as f32)
                .collect();
            assert_close(
                &converter(CAPTURE_RATE, channels).push(&data),
                &expected,
                1e-7,
            );
            assert_close(
                &partitioned(&mut converter(CAPTURE_RATE, channels), &data),
                &expected,
                1e-7,
            );
        }
        let mut input = converter(CAPTURE_RATE, 3);
        assert!(input.push(&[0.0_f32, 0.75]).is_empty());
        assert_eq!(input.push(&[0.75_f32, 0.5]), vec![0.5]);
        assert!(input.push::<f32>(&[]).is_empty());
        assert_eq!(input.push(&[0.25_f32, 0.75]), vec![0.5]);
    }

    #[test]
    fn formats_share_normalization_downmix_and_filter_state() {
        let signed: Vec<i16> = [-32768, -16384, 0, 16384, 32767]
            .into_iter()
            .cycle()
            .take(20_013)
            .collect();
        let unsigned: Vec<u16> = signed.iter().map(|&x| (x as i32 + 32768) as u16).collect();
        let float: Vec<f32> = signed.iter().map(|&x| x as f32 / 32768.0).collect();
        for rate in [24_000, 44_100, 48_000, 192_000] {
            let expected = converter(rate, 3).push(&float);
            let mut i = CaptureInput::new(rate, 3, cpal::SampleFormat::I16).unwrap();
            let mut u = CaptureInput::new(rate, 3, cpal::SampleFormat::U16).unwrap();
            assert_close(&expected, &partitioned(&mut i, &signed), 2e-5);
            assert_close(&expected, &partitioned(&mut u, &unsigned), 2e-5);
        }
        assert_eq!(
            converter(24_000, 1).push(&[-32768_i16, 0, 32767]),
            vec![-1.0, 0.0, 32767.0 / 32768.0]
        );
        assert_eq!(
            converter(24_000, 1).push(&[0_u16, 32768, 65535]),
            vec![-1.0, 0.0, 32767.0 / 32768.0]
        );
    }

    fn steady_rms(pcm: &[f32]) -> f32 {
        let pcm = &pcm[100..pcm.len() - 100];
        (pcm.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / pcm.len() as f64).sqrt() as f32
    }

    #[test]
    fn lowpass_rejects_out_of_band_aliases_and_preserves_speech_band() {
        for rate in [44_100, 48_000, 88_200, 96_000, 176_400, 192_000] {
            let pass = partitioned(&mut converter(rate, 1), &tone(rate, 4_000.0, rate as usize));
            let stop = partitioned(
                &mut converter(rate, 1),
                &tone(rate, 18_000.0, rate as usize),
            );
            let pass_rms = steady_rms(&pass);
            let stop_rms = steady_rms(&stop);
            assert!(
                (pass_rms - 0.6 / 2.0_f32.sqrt()).abs() < 0.005,
                "rate={rate}, pass rms={pass_rms}"
            );
            assert!(stop_rms < 0.005, "rate={rate}, alias rms={stop_rms}");
        }
    }

    #[test]
    fn long_stream_has_bounded_lookahead_not_callback_rounding_loss() {
        let rate = 44_100;
        let data = tone(rate, 1_000.0, rate as usize * 10);
        let output = partitioned(&mut converter(rate, 1), &data);
        assert!(
            output.len() <= 240_000 && 240_000 - output.len() <= 128,
            "length={}",
            output.len()
        );
        let expected = tone(CAPTURE_RATE, 1_000.0, output.len());
        assert_close(&output[100..], &expected[100..], 0.004);
    }

    #[test]
    fn native_rate_has_no_filter_delay_or_amplitude_change() {
        let data = tone(CAPTURE_RATE, 9_000.0, 24_000);
        assert_eq!(partitioned(&mut converter(CAPTURE_RATE, 1), &data), data);
    }

    #[test]
    fn invalid_capture_configuration_fails_before_processing() {
        for rate in [0, 1, 7_999, 192_001, u32::MAX] {
            let error = CaptureInput::new(rate, 1, cpal::SampleFormat::F32)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("Unsupported input sample rate"), "{error}");
        }
        assert!(CaptureInput::new(24_000, 0, cpal::SampleFormat::F32).is_err());
        for format in [
            cpal::SampleFormat::I8,
            cpal::SampleFormat::I24,
            cpal::SampleFormat::I32,
            cpal::SampleFormat::I64,
            cpal::SampleFormat::U8,
            cpal::SampleFormat::U32,
            cpal::SampleFormat::U64,
            cpal::SampleFormat::F64,
        ] {
            let error = CaptureInput::new(24_000, 1, format)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("Unsupported sample format"), "{error}");
        }
    }

    #[test]
    fn nonfinite_float_samples_cannot_poison_following_callbacks() {
        let mut input = converter(48_000, 1);
        let mut output = input.push(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        output.extend(input.push(&tone(48_000, 1_000.0, 48_000)));
        assert!(output.iter().all(|x| x.is_finite()));
        let mut native = converter(24_000, 1);
        assert_eq!(
            native.push(&[f32::NAN, f32::INFINITY, 0.5]),
            vec![0.0, 0.0, 0.5]
        );
    }
}
