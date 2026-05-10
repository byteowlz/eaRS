use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, select, unbounded};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::server::SessionSink;
use crate::server::engine::{Engine, EngineKind, EngineSession};
use crate::{TranscriptionOptions, TranscriptionSink, WebSocketMessage, WordTimestamp};

const SERVER_SAMPLE_RATE: usize = 24_000;
const SHERPA_SAMPLE_RATE: i32 = 16_000;
const ACCEPT_CHUNK_SAMPLES: usize = 3_200; // 200 ms @ 16 kHz, matches sherpa-onnx example

#[derive(Debug, Clone)]
pub struct SherpaEngineConfig {
    /// `(lang_code, model_dir)` pairs. The first entry is the default language.
    /// `lang_code` is what clients send via `set_language` (e.g., "en", "de").
    pub models: Vec<(String, PathBuf)>,
    pub num_threads: i32,
    pub provider: String,
    pub decoding_method: String,
    pub enable_endpoint: bool,
    pub rule1_min_trailing_silence: f32,
    pub rule2_min_trailing_silence: f32,
    pub rule3_min_utterance_length: f32,
}

impl SherpaEngineConfig {
    pub fn defaults_for(models: Vec<(String, PathBuf)>) -> Self {
        Self {
            models,
            num_threads: 1,
            provider: "cpu".to_string(),
            decoding_method: "greedy_search".to_string(),
            enable_endpoint: true,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: 1.2,
            rule3_min_utterance_length: 30.0,
        }
    }
}

pub struct SherpaEngine {
    recognizers: Arc<HashMap<String, Arc<OnlineRecognizer>>>,
    default_lang: String,
    semaphore: Arc<Semaphore>,
    options: TranscriptionOptions,
}

impl SherpaEngine {
    pub fn load(
        cfg: SherpaEngineConfig,
        options: TranscriptionOptions,
        max_sessions: usize,
    ) -> Result<Self> {
        if cfg.models.is_empty() {
            return Err(anyhow!(
                "sherpa engine requires at least one model (use --sherpa-model-dir or --sherpa-model LANG=PATH)"
            ));
        }

        let default_lang = cfg.models[0].0.clone();
        let mut recognizers: HashMap<String, Arc<OnlineRecognizer>> = HashMap::new();

        for (lang, model_dir) in &cfg.models {
            if recognizers.contains_key(lang) {
                return Err(anyhow!(
                    "duplicate sherpa model for language `{lang}`; only the first is used"
                ));
            }

            let layout = ModelLayout::discover(model_dir)
                .with_context(|| format!("invalid sherpa model dir: {}", model_dir.display()))?;

            let mut rconfig = OnlineRecognizerConfig::default();
            rconfig.model_config.transducer.encoder = Some(path_to_string(&layout.encoder)?);
            rconfig.model_config.transducer.decoder = Some(path_to_string(&layout.decoder)?);
            rconfig.model_config.transducer.joiner = Some(path_to_string(&layout.joiner)?);
            rconfig.model_config.tokens = Some(path_to_string(&layout.tokens)?);
            rconfig.model_config.provider = Some(cfg.provider.clone());
            rconfig.model_config.num_threads = cfg.num_threads.max(1);
            rconfig.decoding_method = Some(cfg.decoding_method.clone());
            rconfig.enable_endpoint = cfg.enable_endpoint;
            rconfig.rule1_min_trailing_silence = cfg.rule1_min_trailing_silence;
            rconfig.rule2_min_trailing_silence = cfg.rule2_min_trailing_silence;
            rconfig.rule3_min_utterance_length = cfg.rule3_min_utterance_length;

            let recognizer = OnlineRecognizer::create(&rconfig).ok_or_else(|| {
                anyhow!(
                    "sherpa-onnx failed to create OnlineRecognizer for lang={lang} ({})",
                    model_dir.display()
                )
            })?;

            eprintln!(
                "[sherpa] loaded model lang={lang} from {} (threads={}, provider={})",
                model_dir.display(),
                rconfig.model_config.num_threads,
                rconfig.model_config.provider.as_deref().unwrap_or("cpu")
            );

            recognizers.insert(lang.clone(), Arc::new(recognizer));
        }

        Ok(Self {
            recognizers: Arc::new(recognizers),
            default_lang,
            semaphore: Arc::new(Semaphore::new(max_sessions.max(1))),
            options,
        })
    }
}

impl Engine for SherpaEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Sherpa
    }

    fn allocate(&self, sink: SessionSink) -> Result<Option<Box<dyn EngineSession>>> {
        let permit = match self.semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };

        let (audio_tx, audio_rx) = unbounded();
        let (control_tx, control_rx) = unbounded();

        let multilingual = self.recognizers.len() > 1;
        let handle = SherpaSessionHandle {
            audio_tx,
            control_tx,
            multilingual,
        };

        let recognizers = self.recognizers.clone();
        let default_lang = self.default_lang.clone();
        let session_sink = sink;
        let _opts = self.options.clone();

        std::thread::spawn(move || {
            run_sherpa_session(
                audio_rx,
                control_rx,
                session_sink,
                recognizers,
                default_lang,
                permit,
            );
        });

        Ok(Some(Box::new(handle) as Box<dyn EngineSession>))
    }
}

struct SherpaSessionHandle {
    audio_tx: Sender<Vec<f32>>,
    control_tx: Sender<SherpaControl>,
    multilingual: bool,
}

impl EngineSession for SherpaSessionHandle {
    fn engine(&self) -> EngineKind {
        EngineKind::Sherpa
    }

    fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.audio_tx
            .send(pcm)
            .context("failed to send audio to sherpa session")?;
        Ok(())
    }

    fn set_language(&self, lang: String) -> Result<()> {
        if !self.multilingual {
            return Ok(());
        }
        self.control_tx
            .send(SherpaControl::SetLanguage(lang))
            .context("failed to send set_language to sherpa session")?;
        Ok(())
    }

    fn request_stop(&self) {
        let _ = self.control_tx.send(SherpaControl::Stop);
    }

    fn supports_language(&self) -> bool {
        self.multilingual
    }
}

#[derive(Debug)]
enum SherpaControl {
    Stop,
    SetLanguage(String),
}

fn run_sherpa_session(
    audio_rx: Receiver<Vec<f32>>,
    control_rx: Receiver<SherpaControl>,
    mut sink: SessionSink,
    recognizers: Arc<HashMap<String, Arc<OnlineRecognizer>>>,
    default_lang: String,
    _permit: OwnedSemaphorePermit,
) {
    let mut current_lang = default_lang.clone();
    let mut recognizer = recognizers
        .get(&current_lang)
        .cloned()
        .expect("default language recognizer must exist");
    let mut stream = recognizer.create_stream();

    // Resampler state for 24 kHz server PCM -> 16 kHz model input.
    // Use kaudio::resample on each chunk. The library handles the per-chunk
    // edge-effects internally (rubato windowing); for streaming this is good enough
    // since model input is feature-frame quantized anyway.
    let mut buffer_24k: Vec<f32> = Vec::new();
    // Threshold to call resample with enough samples to avoid rubato errors.
    // ~67 ms @ 24 kHz.
    const RESAMPLE_MIN_SAMPLES: usize = 1_600;

    let mut committed_words: Vec<WordTimestamp> = Vec::new();
    let mut emitted_in_segment: usize = 0;
    let mut total_input_secs: f64 = 0.0;
    let mut segment_start_secs: f64 = 0.0;
    let mut last_partial_text = String::new();
    let mut stop_requested = false;
    let mut last_drive_at = Instant::now();

    loop {
        let mut received_audio = false;

        select! {
            recv(audio_rx) -> msg => {
                match msg {
                    Ok(chunk) => {
                        buffer_24k.extend_from_slice(&chunk);
                        received_audio = true;
                    }
                    Err(_) => stop_requested = true,
                }
            },
            recv(control_rx) -> msg => {
                match msg {
                    Ok(SherpaControl::Stop) => stop_requested = true,
                    Ok(SherpaControl::SetLanguage(lang)) => {
                        if lang != current_lang {
                            if let Some(new_recognizer) = recognizers.get(&lang).cloned() {
                                // Drain any audio still buffered into the *current* recognizer
                                // so words from the previous language flush before we swap.
                                drain_decode(
                                    &recognizer,
                                    &stream,
                                    segment_start_secs,
                                    &mut emitted_in_segment,
                                    &mut last_partial_text,
                                    &mut committed_words,
                                    &mut sink,
                                );
                                eprintln!(
                                    "[sherpa] language switch {current_lang} -> {lang} at {:.2}s",
                                    total_input_secs
                                );
                                recognizer = new_recognizer;
                                stream = recognizer.create_stream();
                                current_lang = lang;
                                segment_start_secs = total_input_secs;
                                emitted_in_segment = 0;
                                last_partial_text.clear();
                            } else {
                                eprintln!(
                                    "[sherpa] set_language: unknown language `{lang}` (loaded: {})",
                                    recognizers
                                        .keys()
                                        .cloned()
                                        .collect::<Vec<_>>()
                                        .join(",")
                                );
                            }
                        }
                    }
                    Err(_) => stop_requested = true,
                }
            },
            default(Duration::from_millis(10)) => {}
        }

        // Forward as much resampled audio to the recognizer as we can without
        // starving the resampler.
        while buffer_24k.len() >= RESAMPLE_MIN_SAMPLES {
            let take = buffer_24k.len();
            let drained: Vec<f32> = buffer_24k.drain(..take).collect();
            match kaudio::resample(&drained, SERVER_SAMPLE_RATE, SHERPA_SAMPLE_RATE as usize) {
                Ok(samples_16k) => {
                    if !samples_16k.is_empty() {
                        total_input_secs += samples_16k.len() as f64 / SHERPA_SAMPLE_RATE as f64;
                        for piece in samples_16k.chunks(ACCEPT_CHUNK_SAMPLES) {
                            stream.accept_waveform(SHERPA_SAMPLE_RATE, piece);
                        }
                        last_drive_at = Instant::now();
                    }
                }
                Err(err) => {
                    eprintln!("[sherpa] resample failed: {err}");
                    // drop this frame and keep going
                }
            }
        }

        if stop_requested && !buffer_24k.is_empty() {
            // Flush remaining audio with zero-padding so resampler is happy.
            let mut padded = std::mem::take(&mut buffer_24k);
            let pad = RESAMPLE_MIN_SAMPLES.saturating_sub(padded.len());
            padded.extend(std::iter::repeat(0.0).take(pad));
            if let Ok(samples_16k) =
                kaudio::resample(&padded, SERVER_SAMPLE_RATE, SHERPA_SAMPLE_RATE as usize)
            {
                if !samples_16k.is_empty() {
                    total_input_secs += samples_16k.len() as f64 / SHERPA_SAMPLE_RATE as f64;
                    for piece in samples_16k.chunks(ACCEPT_CHUNK_SAMPLES) {
                        stream.accept_waveform(SHERPA_SAMPLE_RATE, piece);
                    }
                }
            }
        }

        if stop_requested {
            // Tail padding (~0.8 s). The upstream example uses 0.3 s, but with the
            // trailing-word holdback we need enough right-context for the streaming
            // Zipformer to actually emit the final word before we flush.
            let tail_len = (SHERPA_SAMPLE_RATE as f32 * 0.8).round() as usize;
            let tail = vec![0.0f32; tail_len];
            stream.accept_waveform(SHERPA_SAMPLE_RATE, &tail);
            stream.input_finished();
        }

        // Drive decoding while the recognizer has enough audio.
        let drive_loop_budget = 32;
        let mut drove = false;
        for _ in 0..drive_loop_budget {
            if !recognizer.is_ready(&stream) {
                break;
            }
            recognizer.decode(&stream);
            drove = true;

            if let Some(result) = recognizer.get_result(&stream) {
                emit_partial_words(
                    &result.text,
                    result.timestamps.as_deref(),
                    result.tokens.as_slice(),
                    segment_start_secs,
                    &mut emitted_in_segment,
                    &mut last_partial_text,
                    &mut committed_words,
                    &mut sink,
                    false,
                );
            }

            if recognizer.is_endpoint(&stream) {
                if let Some(result) = recognizer.get_result(&stream) {
                    // Flush any final words from this segment we haven't emitted yet.
                    emit_partial_words(
                        &result.text,
                        result.timestamps.as_deref(),
                        result.tokens.as_slice(),
                        segment_start_secs,
                        &mut emitted_in_segment,
                        &mut last_partial_text,
                        &mut committed_words,
                        &mut sink,
                        true,
                    );
                }
                recognizer.reset(&stream);
                segment_start_secs = total_input_secs;
                emitted_in_segment = 0;
                last_partial_text.clear();
            }
        }

        if !drove && !received_audio && !stop_requested {
            // Nothing to do; back off briefly.
            if last_drive_at.elapsed() > Duration::from_millis(200) {
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        if stop_requested {
            // Drain any remaining decode steps after input_finished.
            while recognizer.is_ready(&stream) {
                recognizer.decode(&stream);
                if let Some(result) = recognizer.get_result(&stream) {
                    emit_partial_words(
                        &result.text,
                        result.timestamps.as_deref(),
                        result.tokens.as_slice(),
                        segment_start_secs,
                        &mut emitted_in_segment,
                        &mut last_partial_text,
                        &mut committed_words,
                        &mut sink,
                        false,
                    );
                }
            }
            // One last get_result to capture any final hypothesis text.
            if let Some(result) = recognizer.get_result(&stream) {
                emit_partial_words(
                    &result.text,
                    result.timestamps.as_deref(),
                    result.tokens.as_slice(),
                    segment_start_secs,
                    &mut emitted_in_segment,
                    &mut last_partial_text,
                    &mut committed_words,
                    &mut sink,
                    true,
                );
            }

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
}

/// Decode all currently-buffered audio in the recognizer and flush its segment with
/// `final_flush=true` so any pending trailing word is committed. Used when stopping
/// the session or switching languages mid-stream.
fn drain_decode(
    recognizer: &OnlineRecognizer,
    stream: &sherpa_onnx::OnlineStream,
    segment_start_secs: f64,
    emitted_in_segment: &mut usize,
    last_partial_text: &mut String,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
) {
    // Right-context padding: the streaming Zipformer needs ~0.6 s of look-ahead
    // to emit the last word of the current segment.
    let tail_len = (SHERPA_SAMPLE_RATE as f32 * 0.8).round() as usize;
    let tail = vec![0.0f32; tail_len];
    stream.accept_waveform(SHERPA_SAMPLE_RATE, &tail);
    while recognizer.is_ready(stream) {
        recognizer.decode(stream);
    }
    if let Some(result) = recognizer.get_result(stream) {
        emit_partial_words(
            &result.text,
            result.timestamps.as_deref(),
            result.tokens.as_slice(),
            segment_start_secs,
            emitted_in_segment,
            last_partial_text,
            committed_words,
            sink,
            true,
        );
    }
}

/// Emit any new words that appeared since the last partial result for the current segment.
///
/// `text` is the current segment's hypothesis from `get_result`.
/// `tokens` and `timestamps` are per-token; we map each whitespace-separated word
/// to the timestamp of its first contributing token.
///
/// `final_flush` = true means we've reached an endpoint or end-of-stream and should
/// commit even the trailing (still-evolving) word. During partial decoding we hold
/// back the last word because subword tokens may still extend it
/// (e.g. "Mult" → "Multiple").
fn emit_partial_words(
    text: &str,
    timestamps: Option<&[f32]>,
    tokens: &[String],
    segment_start_secs: f64,
    emitted_in_segment: &mut usize,
    last_partial_text: &mut String,
    committed_words: &mut Vec<WordTimestamp>,
    sink: &mut SessionSink,
    final_flush: bool,
) {
    if text == last_partial_text && !final_flush {
        return;
    }
    last_partial_text.clear();
    last_partial_text.push_str(text);

    let words: Vec<&str> = text.split_whitespace().collect();

    // Hold back the trailing word during partials — its tail can still grow as
    // subword tokens come in. Commit it only on endpoint / end-of-stream.
    let commit_upto = if final_flush {
        words.len()
    } else {
        words.len().saturating_sub(1)
    };

    if commit_upto <= *emitted_in_segment {
        return;
    }

    let word_starts = word_start_times(tokens, timestamps);

    for i in *emitted_in_segment..commit_upto {
        let token_ts = word_starts.get(i).copied().unwrap_or(0.0) as f64;
        let abs_start = segment_start_secs + token_ts;
        let ws = WordTimestamp {
            word: words[i].to_string(),
            start_time: abs_start,
            end_time: None,
        };
        sink.handle_message(WebSocketMessage::Word {
            word: ws.word.clone(),
            start_time: ws.start_time,
            end_time: None,
        });
        committed_words.push(ws);
    }
    *emitted_in_segment = commit_upto;
}

/// Word-start timestamps in seconds within the current segment.
///
/// Sentencepiece BPE tokens prefix word starts with U+2581 (▁). We treat the
/// first token as a word start as well. Falls back to per-token timestamps if
/// the model emits one token per word.
fn word_start_times(tokens: &[String], timestamps: Option<&[f32]>) -> Vec<f32> {
    let Some(ts) = timestamps else {
        return Vec::new();
    };
    let mut starts = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        if i >= ts.len() {
            break;
        }
        let is_word_start = i == 0 || tok.starts_with('\u{2581}') || !starts_with_subword(tok);
        if is_word_start {
            starts.push(ts[i]);
        }
    }
    starts
}

fn starts_with_subword(tok: &str) -> bool {
    // Heuristic: subword continuations don't start a new word.
    // Sentencepiece uses ▁ to mark a *new* word, so absence-of-▁ on token > 0
    // means this token continues the previous word.
    !tok.starts_with('\u{2581}')
}

struct ModelLayout {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

impl ModelLayout {
    fn discover(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            return Err(anyhow!(
                "sherpa model dir does not exist or is not a directory: {}",
                dir.display()
            ));
        }
        let encoder = pick_one(dir, "encoder")?;
        let decoder = pick_one(dir, "decoder")?;
        let joiner = pick_one(dir, "joiner")?;
        let tokens = dir.join("tokens.txt");
        if !tokens.is_file() {
            return Err(anyhow!("missing tokens.txt in {}", dir.display()));
        }
        Ok(Self {
            encoder,
            decoder,
            joiner,
            tokens,
        })
    }
}

fn pick_one(dir: &Path, role: &str) -> Result<PathBuf> {
    // Prefer int8 quantized encoder/decoder/joiner if present, fall back to fp32.
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("onnx")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(role))
                    .unwrap_or(false)
        })
        .collect();

    if candidates.is_empty() {
        return Err(anyhow!(
            "no {role}*.onnx file in {} -- expected encoder.onnx / decoder.onnx / joiner.onnx (int8 variants ok)",
            dir.display()
        ));
    }
    // Prefer int8 variant for CPU inference.
    candidates.sort_by_key(|p| {
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        // Lower key = preferred. int8 first.
        if name.contains("int8") { 0 } else { 1 }
    });
    Ok(candidates.remove(0))
}

fn path_to_string(p: &Path) -> Result<String> {
    p.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("non-UTF-8 path: {}", p.display()))
}
