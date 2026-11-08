//! Multi-format audio decoding and resampling for transcription input.
//!
//! Enabled by the default `audio` feature. Decodes WAV, MP3, AAC, FLAC, OGG,
//! and M4A via [`symphonia`](https://docs.rs/symphonia) and resamples to the
//! 16 kHz mono `f32` PCM that Moonshine expects via
//! [`rubato`](https://docs.rs/rubato).
//!
//! Most callers only need [`load_audio_for_transcription`]:
//!
//! ```no_run
//! let pcm = moonshine_rs::audio::load_audio_for_transcription("speech.mp3")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::{Error, Result};

/// Decoded PCM audio with its native sample rate and channel count.
#[derive(Debug, Clone)]
pub struct AudioDecoded {
    /// Interleaved `f32` PCM samples in `[-1.0, 1.0]`.
    pub pcm_data: Vec<f32>,
    /// Source sample rate in Hz.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
}

/// Quality/speed tradeoff for [`resample_pcm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResampleQuality {
    /// Faster, linear sinc interpolation. The default.
    #[default]
    Fast,
    /// Slower, higher-quality cubic sinc interpolation.
    High,
}

/// Decodes any supported audio file (WAV, MP3, AAC, FLAC, OGG, M4A, etc.) into normalized float PCM.
pub fn decode_audio_file(path: impl AsRef<Path>) -> Result<AudioDecoded> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::ApiError {
        code: -1,
        message: format!("Failed to open audio file {}: {}", path.display(), e),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| Error::ApiError {
            code: -1,
            message: format!("Failed to probe audio format for {}: {}", path.display(), e),
        })?;

    let mut format = probed.format;

    let track = format.default_track().ok_or_else(|| Error::ApiError {
        code: -1,
        message: format!("No default audio track found in {}", path.display()),
    })?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| Error::ApiError {
            code: -1,
            message: "Audio track missing sample rate".to_string(),
        })?;

    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| Error::ApiError {
            code: -1,
            message: format!("Unsupported audio codec in {}: {}", path.display(), e),
        })?;

    let mut pcm_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                continue;
            }
            Err(e) => {
                return Err(Error::ApiError {
                    code: -1,
                    message: format!("Error reading audio packet from {}: {}", path.display(), e),
                });
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => match decoded {
                AudioBufferRef::F32(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            pcm_samples.push(buf.chan(ch)[i]);
                        }
                    }
                }
                AudioBufferRef::U8(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i] as f32 / 255.0 * 2.0 - 1.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::U16(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i] as f32 / 65535.0 * 2.0 - 1.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::U24(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i].inner() as f32 / 8388607.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::U32(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i] as f32 / 4294967295.0 * 2.0 - 1.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::S8(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i] as f32 / 128.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::S16(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i] as f32 / 32768.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::S24(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i].inner() as f32 / 8388608.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::S32(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            let s = buf.chan(ch)[i] as f32 / 2147483648.0;
                            pcm_samples.push(s);
                        }
                    }
                }
                AudioBufferRef::F64(buf) => {
                    let num_channels = buf.spec().channels.count();
                    for i in 0..buf.frames() {
                        for ch in 0..num_channels {
                            pcm_samples.push(buf.chan(ch)[i] as f32);
                        }
                    }
                }
            },
            Err(SymphoniaError::DecodeError(_)) => {
                continue;
            }
            Err(e) => {
                return Err(Error::ApiError {
                    code: -1,
                    message: format!("Error decoding audio frame in {}: {}", path.display(), e),
                });
            }
        }
    }

    Ok(AudioDecoded {
        pcm_data: pcm_samples,
        sample_rate,
        channels,
    })
}

/// Resamples interleaved PCM audio and mixes it down to mono `f32` PCM.
///
/// If `source_rate == target_rate` the samples are only downmixed to mono.
pub fn resample_pcm(
    pcm: &[f32],
    source_rate: u32,
    target_rate: u32,
    channels: u16,
    quality: ResampleQuality,
) -> Result<Vec<f32>> {
    if pcm.is_empty() {
        return Ok(Vec::new());
    }

    let channels_cnt = channels as usize;
    let num_frames = pcm.len() / channels_cnt;

    // First mix multi-channel down to mono channel
    let mut mono_pcm = Vec::with_capacity(num_frames);
    if channels_cnt == 1 {
        mono_pcm.extend_from_slice(pcm);
    } else {
        for frame in 0..num_frames {
            let mut sum = 0.0f32;
            for ch in 0..channels_cnt {
                sum += pcm[frame * channels_cnt + ch];
            }
            mono_pcm.push(sum / channels_cnt as f32);
        }
    }

    if source_rate == target_rate {
        return Ok(mono_pcm);
    }

    let params = match quality {
        ResampleQuality::Fast => SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.92,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::Hann,
        },
        ResampleQuality::High => SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        },
    };

    let resample_ratio = target_rate as f64 / source_rate as f64;
    let max_chunk_size = 1024;

    let mut resampler = SincFixedIn::<f32>::new(resample_ratio, 2.0, params, max_chunk_size, 1)
        .map_err(|e| Error::ApiError {
            code: -1,
            message: format!("Failed to create rubato resampler: {}", e),
        })?;

    let mut output_mono = Vec::new();
    let mut offset = 0;

    while offset < mono_pcm.len() {
        let chunk_len = max_chunk_size.min(mono_pcm.len() - offset);
        let mut chunk = vec![mono_pcm[offset..offset + chunk_len].to_vec()];

        if chunk_len < max_chunk_size {
            chunk[0].resize(max_chunk_size, 0.0);
        }

        let resampled = resampler
            .process(&chunk, None)
            .map_err(|e| Error::ApiError {
                code: -1,
                message: format!("Resampling error: {}", e),
            })?;

        if !resampled.is_empty() && !resampled[0].is_empty() {
            let expected_samples = (chunk_len as f64 * resample_ratio).round() as usize;
            let take = resampled[0].len().min(expected_samples);
            output_mono.extend_from_slice(&resampled[0][..take]);
        }

        offset += max_chunk_size;
    }

    Ok(output_mono)
}

/// Convenience function that opens any supported audio file (WAV, MP3, AAC, FLAC, OGG, etc.),
/// decodes it, and automatically resamples/mixes down to 16,000 Hz mono PCM float array.
pub fn load_audio_for_transcription(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let decoded = decode_audio_file(path)?;
    resample_pcm(
        &decoded.pcm_data,
        decoded.sample_rate,
        16000,
        decoded.channels,
        ResampleQuality::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        let pcm = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let resampled = resample_pcm(&pcm, 16000, 16000, 1, ResampleQuality::Fast).unwrap();
        assert_eq!(pcm, resampled);
    }

    #[test]
    fn test_resample_stereo_to_mono() {
        let pcm = vec![0.5f32, 0.5, 1.0, -1.0]; // 2 frames stereo
        let resampled = resample_pcm(&pcm, 16000, 16000, 2, ResampleQuality::Fast).unwrap();
        assert_eq!(resampled.len(), 2);
        assert_eq!(resampled[0], 0.5);
        assert_eq!(resampled[1], 0.0);
    }

    #[test]
    fn test_resample_44k_to_16k() {
        let pcm: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.1).sin()).collect();
        let resampled = resample_pcm(&pcm, 44100, 16000, 1, ResampleQuality::Fast).unwrap();
        // 1 second at 44.1kHz -> ~16000 samples at 16kHz
        assert!((resampled.len() as i32 - 16000).abs() < 100);
    }
}
