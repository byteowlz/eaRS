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

    // Accumulated committed words across all chunks (the full transcript so far)
    let mut committed_words: Vec<WordTimestamp> = Vec::new();
    // Track the audio time boundary: everything before this is "committed"
    let mut committed_until: f64 = 0.0;

    let mut vad_state = VadState::new(config.options.vad);
    let vad_timeout = config.options.vad_timeout;
    let mut last_pause_sent: Option<Instant> = None;
    let mut stop_requested = false;

    let overlap_duration_secs = config.overlap_samples_24k as f64 / SERVER_SAMPLE_RATE as f64;

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

        let should_transcribe = (buffer_24k.len() >= config.chunk_samples_24k
            || (stop_requested && !buffer_24k.is_empty()))
            && buffer_24k.len() >= 1600; // minimum ~67ms at 24kHz, rubato resampler needs sufficient input

        if should_transcribe {
            let chunk_start_time = buffer_offset_24k as f64 / SERVER_SAMPLE_RATE as f64;

            // Only transcribe up to chunk_samples_24k of audio. If the buffer grew
            // beyond this (because previous transcription returned empty), just use
            // the first chunk_samples_24k -- the model works best with ~3s chunks.
            let transcribe_len = buffer_24k.len().min(config.chunk_samples_24k);
            let to_transcribe = &buffer_24k[..transcribe_len];

            let resampled =
                match kaudio::resample(to_transcribe, SERVER_SAMPLE_RATE, PARAKEET_SAMPLE_RATE) {
                    Ok(res) => res,
                    Err(err) => {
                        eprintln!("[parakeet] resample failed: {err}");
                        buffer_24k.clear();
                        continue;
                    }
                };

            let chunk_duration_secs = resampled.len() as f64 / PARAKEET_SAMPLE_RATE as f64;

            eprintln!(
                "[parakeet] chunk {:.2}s rms {:.4} (committed_until={:.2}s)",
                chunk_duration_secs,
                rms(&resampled),
                committed_until,
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
                                start_time: chunk_start_time + w.start_time as f64,
                                end_time: Some(chunk_start_time + w.end_time as f64),
                            })
                            .collect::<Vec<_>>();
                        (result.text, shifted_words)
                    })
            };

            let mut got_words = false;
            match transcription {
                Ok((_text, ref chunk_words)) => {
                    // Parakeet is a non-streaming model: each transcription covers the ENTIRE buffer.
                    // Strategy: REPLACE the full transcript on every chunk, but only emit the
                    // delta (new words) to the client. The last chunk's output IS the final result.
                    //
                    // Overlap handling: since each chunk includes the last N seconds of previous audio,
                    // the words in the overlap region should match what we already committed.
                    // We skip any overlap words that match the tail of committed_words.

                    if !chunk_words.is_empty() {
                        got_words = true;
                        let overlap_end = if committed_until > 0.0 {
                            chunk_start_time + overlap_duration_secs
                        } else {
                            0.0 // First chunk: no overlap
                        };

                        eprintln!(
                            "[parakeet] raw: {} words, overlap_end={:.2}s, words: {}",
                            chunk_words.len(),
                            overlap_end,
                            chunk_words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" "),
                        );

                        // Split into overlap (already-seen audio) and new audio
                        let (overlap_words, new_words): (Vec<_>, Vec<_>) = chunk_words
                            .iter()
                            .partition(|w| w.start_time < overlap_end);

                        let mut to_emit: Vec<WordTimestamp> = Vec::new();

                        // For overlap words: deduplicate against committed tail
                        if !overlap_words.is_empty() {
                            if committed_words.is_empty() {
                                // First chunk with overlap words (shouldn't happen normally)
                                to_emit.extend(overlap_words.iter().cloned().cloned());
                            } else {
                                // Match against the tail of committed words
                                let tail_start = if committed_words.len() > overlap_words.len() * 3 {
                                    committed_words.len() - overlap_words.len() * 3
                                } else {
                                    0
                                };
                                let committed_tail: Vec<&str> = committed_words[tail_start..]
                                    .iter()
                                    .map(|w| w.word.as_str())
                                    .collect();
                                let overlap_strs: Vec<&str> = overlap_words
                                    .iter()
                                    .map(|w| w.word.as_str())
                                    .collect();

                                let ratio = word_match_ratio(&committed_tail, &overlap_strs);

                                if ratio < 0.4 {
                                    // Poor match: the new transcription corrected something.
                                    // Trust the newer one: remove old committed words in overlap
                                    // time range and emit the new ones instead.
                                    let before: Vec<WordTimestamp> = committed_words
                                        .iter()
                                        .filter(|w| w.start_time < chunk_start_time)
                                        .cloned()
                                        .collect();
                                    let removed = committed_words.len() - before.len();
                                    if removed > 0 {
                                        eprintln!(
                                            "[parakeet] overlap correction (ratio={:.2}): -{} +{} words",
                                            ratio, removed, overlap_words.len()
                                        );
                                    }
                                    committed_words = before;
                                    to_emit.extend(overlap_words.iter().cloned().cloned());
                                }
                                // else: good match, overlap words are duplicates -> skip
                            }
                        }

                        // All new-region words are always emitted
                        to_emit.extend(new_words.iter().cloned().cloned());

                        // Send to client
                        for word in &to_emit {
                            sink.handle_message(WebSocketMessage::Word {
                                word: word.word.clone(),
                                start_time: word.start_time,
                                end_time: None,
                            });
                        }

                        committed_words.extend(to_emit);

                        let chunk_end_time = chunk_start_time + chunk_duration_secs;
                        committed_until = chunk_end_time - overlap_duration_secs;

                        eprintln!(
                            "[parakeet] chunk done: {} total committed words, committed_until={:.2}s",
                            committed_words.len(),
                            committed_until,
                        );
                    } else {
                        eprintln!(
                            "[parakeet] chunk produced 0 words (silence/garbage?), NOT trimming buffer"
                        );
                    }

                    if stop_requested {
                        let final_text = committed_words
                            .iter()
                            .map(|w| w.word.clone())
                            .collect::<Vec<_>>()
                            .join(" ");
                        sink.handle_message(WebSocketMessage::Final {
                            text: final_text,
                            words: committed_words.clone(),
                        });
                        sink.close();
                        break;
                    }
                }
                Err(ref err) => {
                    eprintln!("[parakeet] transcription failed: {err}");
                }
            }

            // Trim the transcribed portion from the buffer, keeping overlap for next chunk.
            // We transcribed buffer[0..transcribe_len], so we remove transcribe_len - overlap
            // samples from the front (or as much as we can).
            // CRITICAL: Only trim if transcription produced words. If Parakeet returned empty,
            // trimming would permanently discard untranscribed audio — the root cause of missing
            // words in our v3 dedup results. Instead, keep the audio and retry on the next chunk
            // (the overlap region naturally re-includes it).
            if got_words {
                let consumed = transcribe_len.saturating_sub(config.overlap_samples_24k);
                if consumed > 0 && buffer_24k.len() > consumed {
                    buffer_24k = buffer_24k.split_off(consumed);
                    buffer_offset_24k = total_samples_24k.saturating_sub(buffer_24k.len());
                }
            } else {
                eprintln!(
                    "[parakeet] NOT trimming buffer: transcription produced 0 words, will retry same audio in next chunk"
                );
            }
        }

        if !received_audio && stop_requested {
            if !committed_words.is_empty() {
                let final_text = committed_words
                    .iter()
                    .map(|w| w.word.clone())
                    .collect::<Vec<_>>()
                    .join(" ");
                sink.handle_message(WebSocketMessage::Final {
                    text: final_text,
                    words: committed_words.clone(),
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

/// Word match ratio between two word sequences using LCS.
/// Returns a value between 0.0 (no match) and 1.0 (perfect match).
/// Used to determine if overlap words are duplicates of committed words.
fn word_match_ratio(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let lcs = lcs_len(a, b);
    let shorter = a.len().min(b.len());
    lcs as f64 / shorter as f64
}

/// Length of the Longest Common Subsequence between two word sequences.
/// Space-optimized DP: O(min(m,n)) space.
fn lcs_len(a: &[&str], b: &[&str]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    // Make b the shorter one for space optimization
    let (longer, shorter) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let m = longer.len();
    let n = shorter.len();

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if longer[i - 1] == shorter[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = curr[j - 1].max(prev[j]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[n]
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
