use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, select, unbounded};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use webrtc_vad::{SampleRate, Vad, VadMode};

use crate::server::SessionSink;
use crate::server::engine::{Engine, EngineKind, EngineSession};
use crate::{TranscriptionOptions, TranscriptionSink, WebSocketMessage, WordTimestamp};

const PARAKEET_SAMPLE_RATE: usize = 16_000;
const SERVER_SAMPLE_RATE: usize = 24_000;
const VAD_FRAME: usize = 480; // 30ms @16k

#[derive(Debug, Clone, Copy)]
pub enum ParakeetDevice {
    Cpu,
    #[cfg(feature = "nvidia")]
    Cuda,
    #[cfg(feature = "apple")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "amd")]
    ROCm,
}

impl ParakeetDevice {
    pub fn default_for_build() -> Self {
        #[cfg(feature = "nvidia")]
        {
            return ParakeetDevice::Cuda;
        }
        #[cfg(all(feature = "amd", not(feature = "nvidia")))]
        {
            return ParakeetDevice::ROCm;
        }
        #[cfg(all(feature = "apple", not(feature = "nvidia"), not(feature = "amd")))]
        {
            return ParakeetDevice::CoreML;
        }
        #[cfg(all(
            feature = "directml",
            not(feature = "nvidia"),
            not(feature = "amd"),
            not(feature = "apple")
        ))]
        {
            return ParakeetDevice::DirectML;
        }
        #[cfg(not(any(
            feature = "nvidia",
            feature = "amd",
            feature = "apple",
            feature = "directml"
        )))]
        ParakeetDevice::Cpu
    }

    fn to_prkt(self) -> prkt::Device {
        match self {
            ParakeetDevice::Cpu => prkt::Device::Cpu,
            #[cfg(feature = "nvidia")]
            ParakeetDevice::Cuda => prkt::Device::Cuda,
            #[cfg(feature = "apple")]
            ParakeetDevice::CoreML => prkt::Device::CoreML,
            #[cfg(feature = "directml")]
            ParakeetDevice::DirectML => prkt::Device::DirectML,
            #[cfg(feature = "amd")]
            ParakeetDevice::ROCm => prkt::Device::ROCm,
        }
    }
}

pub struct ParakeetEngineConfig {
    pub model_repo: String,
    pub model_dir: Option<std::path::PathBuf>,
    pub device: ParakeetDevice,
    pub chunk_seconds: f32,
    pub overlap_seconds: f32,
    pub noise_gate_rms: f32,
}

pub struct ParakeetEngine {
    model: Arc<Mutex<prkt::ParakeetModel>>,
    semaphore: Arc<Semaphore>,
    options: TranscriptionOptions,
    chunk_samples_24k: usize,
    overlap_samples_24k: usize,
    noise_gate_rms: f32,
}

impl ParakeetEngine {
    pub fn load(
        cfg: ParakeetEngineConfig,
        options: TranscriptionOptions,
        max_sessions: usize,
    ) -> Result<Self> {
        let device = cfg.device;
        let model = prkt::ParakeetModel::load_from_hf(
            &cfg.model_repo,
            cfg.model_dir.as_deref(),
            device.to_prkt(),
        )
        .context("failed to load parakeet model")?;

        let capacity = max_sessions.max(1);
        let chunk_24k = (cfg.chunk_seconds.max(0.5) * SERVER_SAMPLE_RATE as f32) as usize;
        let overlap_24k = (cfg.overlap_seconds.max(0.0) * SERVER_SAMPLE_RATE as f32) as usize;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            semaphore: Arc::new(Semaphore::new(capacity)),
            options,
            chunk_samples_24k: chunk_24k,
            overlap_samples_24k: overlap_24k.min(chunk_24k),
            noise_gate_rms: cfg.noise_gate_rms,
        })
    }
}

impl Engine for ParakeetEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Parakeet
    }

    fn allocate(&self, sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
        let permit = match self.semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };

        let (audio_tx, audio_rx) = unbounded();
        let (control_tx, control_rx) = unbounded();

        let session = ParakeetSessionHandle {
            audio_tx,
            control_tx,
        };

        let session_sink = sink.clone();
        let model = self.model.clone();
        let session_opts = ParakeetSessionConfig {
            options: self.options.clone(),
            chunk_samples_24k: self.chunk_samples_24k,
            overlap_samples_24k: self.overlap_samples_24k,
            noise_gate_rms: self.noise_gate_rms,
        };

        std::thread::spawn(move || {
            run_parakeet_session(
                audio_rx,
                control_rx,
                session_sink,
                model,
                permit,
                session_opts,
            );
        });

        Ok(Some(Box::new(session) as Box<dyn EngineSession>))
    }
}

struct ParakeetSessionHandle {
    audio_tx: Sender<Vec<f32>>,
    control_tx: Sender<ParakeetControl>,
}

impl EngineSession for ParakeetSessionHandle {
    fn engine(&self) -> EngineKind {
        EngineKind::Parakeet
    }

    fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.audio_tx
            .send(pcm)
            .context("failed to send audio to parakeet session")?;
        Ok(())
    }

    fn set_language(&self, _lang: String) -> Result<()> {
        Ok(())
    }

    fn request_stop(&self) {
        let _ = self.control_tx.send(ParakeetControl::Stop);
    }

    fn supports_language(&self) -> bool {
        false
    }
}

#[derive(Debug)]
enum ParakeetControl {
    Stop,
}

#[derive(Clone)]
struct ParakeetSessionConfig {
    options: TranscriptionOptions,
    chunk_samples_24k: usize,
    overlap_samples_24k: usize,
    noise_gate_rms: f32,
}

struct VadState {
    vad: Option<Vad>,
    in_speech: bool,
    last_voice_instant: Option<Instant>,
    silence_frames: usize,
}

impl VadState {
    fn new(enabled: bool) -> Self {
        let vad = if enabled {
            let mut v = Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, VadMode::Aggressive);
            v.set_sample_rate(SampleRate::Rate16kHz);
            Some(v)
        } else {
            None
        };

        Self {
            vad,
            in_speech: false,
            last_voice_instant: None,
            silence_frames: 0,
        }
    }

    fn process_frame(&mut self, frame: &[i16]) -> Option<bool> {
        if let Some(vad) = self.vad.as_mut() {
            match vad.is_voice_segment(frame) {
                Ok(is_voice) => Some(is_voice),
                Err(_) => None,
            }
        } else {
            None
        }
    }
}

fn run_parakeet_session(
    audio_rx: Receiver<Vec<f32>>,
    control_rx: Receiver<ParakeetControl>,
    mut sink: SessionSink,
    model: Arc<Mutex<prkt::ParakeetModel>>,
    _permit: OwnedSemaphorePermit,
    config: ParakeetSessionConfig,
) {
    let mut buffer_24k: Vec<f32> = Vec::new();
    let mut buffer_offset_24k: usize = 0;
    let mut total_samples_24k: usize = 0;
    let mut previous_words: Vec<WordTimestamp> = Vec::new();

    let mut vad_state = VadState::new(config.options.vad);
    let vad_timeout = config.options.vad_timeout;
    let mut last_pause_sent: Option<Instant> = None;
    let mut stop_requested = false;

    loop {
        let mut received_audio = false;
        select! {
            recv(audio_rx) -> msg => {
                match msg {
                    Ok(chunk) => {
                        total_samples_24k += chunk.len();
                        buffer_24k.extend_from_slice(&chunk);
                        received_audio = true;
                    }
                    Err(_) => stop_requested = true,
                }
            },
            recv(control_rx) -> msg => {
                match msg {
                    Ok(ParakeetControl::Stop) => stop_requested = true,
                    Err(_) => stop_requested = true,
                }
            },
            default(Duration::from_millis(10)) => {}
        }

        if let Some(timeout) = vad_timeout {
            if let Some(last_voice) = vad_state.last_voice_instant {
                if last_voice.elapsed() > Duration::from_secs_f64(timeout) {
                    stop_requested = true;
                }
            }
        }

        let should_transcribe = buffer_24k.len() >= config.chunk_samples_24k
            || (stop_requested && !buffer_24k.is_empty());

        if should_transcribe {
            let start_time = buffer_offset_24k as f64 / SERVER_SAMPLE_RATE as f64;
            let resampled =
                match kaudio::resample(&buffer_24k, SERVER_SAMPLE_RATE, PARAKEET_SAMPLE_RATE) {
                    Ok(res) => res,
                    Err(err) => {
                        eprintln!("[parakeet] resample failed: {err}");
                        buffer_24k.clear();
                        continue;
                    }
                };

            eprintln!(
                "[parakeet] chunk {:.2}s rms {:.4}",
                resampled.len() as f64 / PARAKEET_SAMPLE_RATE as f64,
                rms(&resampled)
            );

            let has_voice =
                process_vad_frames(&resampled, &mut vad_state, &mut last_pause_sent, &mut sink);
            if !has_voice && !stop_requested {
                eprintln!("[parakeet] chunk skipped: no voice detected");
                buffer_offset_24k = total_samples_24k.saturating_sub(config.overlap_samples_24k);
                if buffer_24k.len() > config.overlap_samples_24k {
                    buffer_24k =
                        buffer_24k.split_off(buffer_24k.len() - config.overlap_samples_24k);
                } else {
                    buffer_24k.clear();
                }
                continue;
            }

            if !config.options.vad && config.noise_gate_rms > 0.0 {
                let chunk_rms = rms(&resampled);
                if chunk_rms < config.noise_gate_rms && !stop_requested {
                    eprintln!(
                        "[parakeet] chunk skipped: below noise gate (rms {:.5} < {:.5})",
                        chunk_rms, config.noise_gate_rms
                    );
                    buffer_offset_24k =
                        total_samples_24k.saturating_sub(config.overlap_samples_24k);
                    if buffer_24k.len() > config.overlap_samples_24k {
                        buffer_24k =
                            buffer_24k.split_off(buffer_24k.len() - config.overlap_samples_24k);
                    } else {
                        buffer_24k.clear();
                    }
                    continue;
                }
            }

            let transcription = {
                let mut locked = match model.lock() {
                    Ok(m) => m,
                    Err(_) => {
                        eprintln!("[parakeet] model lock poisoned");
                        break;
                    }
                };

                locked
                    .transcribe_with_timestamps(&resampled, PARAKEET_SAMPLE_RATE)
                    .map(|result| {
                        let shifted_words = result
                            .words
                            .into_iter()
                            .map(|w| WordTimestamp {
                                word: w.word,
                                start_time: start_time + w.start_time as f64,
                                end_time: Some(start_time + w.end_time as f64),
                            })
                            .collect::<Vec<_>>();
                        (result.text, shifted_words)
                    })
            };

            match transcription {
                Ok((text, words)) => {
                    let new_words = words.clone();
                    let common_prefix = common_prefix_len(&previous_words, &new_words);
                    for word in new_words.iter().skip(common_prefix) {
                        sink.handle_message(WebSocketMessage::Word {
                            word: word.word.clone(),
                            start_time: word.start_time,
                            end_time: None,
                        });
                    }
                    previous_words = new_words.clone();

                    if stop_requested {
                        sink.handle_message(WebSocketMessage::Final {
                            text,
                            words: new_words,
                        });
                        sink.close();
                        break;
                    }
                }
                Err(err) => {
                    eprintln!("[parakeet] transcription failed: {err}");
                }
            }

            if buffer_24k.len() > config.overlap_samples_24k {
                let keep = config.overlap_samples_24k;
                buffer_offset_24k = total_samples_24k.saturating_sub(keep);
                buffer_24k = buffer_24k.split_off(buffer_24k.len() - keep);
            } else {
                buffer_offset_24k = total_samples_24k.saturating_sub(buffer_24k.len());
            }
        }

        if !received_audio && stop_requested {
            if !previous_words.is_empty() {
                let final_text = previous_words
                    .iter()
                    .map(|w| w.word.clone())
                    .collect::<Vec<_>>()
                    .join(" ");
                sink.handle_message(WebSocketMessage::Final {
                    text: final_text,
                    words: previous_words.clone(),
                });
            }
            sink.close();
            break;
        }
    }
}

fn process_vad_frames(
    samples: &[f32],
    state: &mut VadState,
    last_pause_sent: &mut Option<Instant>,
    sink: &mut SessionSink,
) -> bool {
    let mut voice = false;
    if state.vad.is_none() {
        // No VAD configured; treat as voice present
        return true;
    }

    for frame in samples.chunks(VAD_FRAME) {
        if frame.len() < VAD_FRAME {
            break;
        }

        let pcm: Vec<i16> = frame
            .iter()
            .map(|s| (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect();

        if let Some(is_voice) = state.process_frame(&pcm) {
            if is_voice {
                voice = true;
                state.in_speech = true;
                state.silence_frames = 0;
                state.last_voice_instant = Some(Instant::now());
            } else if state.in_speech {
                state.silence_frames += 1;
                if state.silence_frames >= 5 {
                    state.in_speech = false;
                    state.silence_frames = 0;

                    if last_pause_sent
                        .map(|t| t.elapsed() > Duration::from_millis(200))
                        .unwrap_or(true)
                    {
                        let pause_msg = WebSocketMessage::Pause {
                            timestamp: current_timestamp(),
                        };
                        sink.handle_message(pause_msg);
                        *last_pause_sent = Some(Instant::now());
                    }
                }
            }
        }
    }

    voice
}

fn common_prefix_len(a: &[WordTimestamp], b: &[WordTimestamp]) -> usize {
    let mut idx = 0;
    while idx < a.len() && idx < b.len() && a[idx].word == b[idx].word {
        idx += 1;
    }
    idx
}

fn current_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
