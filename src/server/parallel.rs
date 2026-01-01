use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use candle::Tensor;
use crossbeam_channel::{Receiver, Sender, unbounded};
use moshi::StreamMask;

use crate::{Model, TranscriptionSink, WebSocketMessage, WordTimestamp};
use super::engine::{Engine, EngineKind, EngineSession};

use super::SessionSink;

const FRAME_SIZE: usize = 1920;

#[derive(Debug, Clone)]
pub(crate) struct ParallelEngine {
    command_tx: Sender<EngineCommand>,
    capacity: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionHandle {
    audio_tx: Sender<Vec<f32>>,
    lang_tx: Sender<String>,
    control_tx: Sender<SessionControl>,
}

impl SessionHandle {
    pub fn send_audio(&self, pcm: Vec<f32>) -> Result<()> {
        self.audio_tx
            .send(pcm)
            .context("failed to send audio chunk to engine")
            .map(|_| ())
    }

    pub fn set_language(&self, lang: String) -> Result<()> {
        self.lang_tx
            .send(lang)
            .context("failed to send language command to engine")
    }

    pub fn request_stop(&self) {
        let _ = self.control_tx.send(SessionControl::Stop);
    }

    fn signal_closed(&self) {
        let _ = self.control_tx.send(SessionControl::Closed);
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.signal_closed();
    }
}

impl EngineSession for SessionHandle {
    fn engine(&self) -> EngineKind {
        EngineKind::Kyutai
    }

    fn send_audio(&self, pcm: Vec<f32>) -> anyhow::Result<()> {
        SessionHandle::send_audio(self, pcm)
    }

    fn set_language(&self, lang: String) -> anyhow::Result<()> {
        SessionHandle::set_language(self, lang)
    }

    fn request_stop(&self) {
        SessionHandle::request_stop(self);
    }

    fn supports_language(&self) -> bool {
        true
    }
}

enum EngineCommand {
    Allocate {
        sink: SessionSink,
        response: Sender<Option<SessionAllocation>>,
    },
}

struct SessionAllocation {
    batch_idx: usize,
    audio_tx: Sender<Vec<f32>>,
    lang_tx: Sender<String>,
    control_tx: Sender<SessionControl>,
}

#[derive(Debug)]
enum SessionControl {
    Stop,
    Closed,
}

struct SessionState {
    id: u64,
    audio_rx: Receiver<Vec<f32>>,
    lang_rx: Receiver<String>,
    control_rx: Receiver<SessionControl>,
    sink: SessionSink,
    sample_buffer: Vec<f32>,
    all_audio: Vec<f32>,
    words: Vec<WordTimestamp>,
    current_text: String,
    last_word: Option<(String, f64)>,
    word_sent: bool,
    last_voice_activity: Option<Instant>,
    vad_timeout: Option<f64>,
    vad_enabled: bool,
    timestamps: bool,
    pending_language: Option<String>,
    closed: bool,
    flush_samples_remaining: usize,
}

impl SessionState {
    fn new(
        id: u64,
        sink: SessionSink,
        audio_rx: Receiver<Vec<f32>>,
        lang_rx: Receiver<String>,
        control_rx: Receiver<SessionControl>,
        timestamps: bool,
        vad_enabled: bool,
        vad_timeout: Option<f64>,
        flush_samples: usize,
    ) -> Self {
        Self {
            id,
            audio_rx,
            lang_rx,
            control_rx,
            sink,
            sample_buffer: Vec::with_capacity(FRAME_SIZE * 2),
            all_audio: Vec::new(),
            words: Vec::new(),
            current_text: String::new(),
            last_word: None,
            word_sent: false,
            last_voice_activity: None,
            vad_timeout,
            vad_enabled,
            timestamps,
            pending_language: None,
            closed: false,
            flush_samples_remaining: flush_samples,
        }
    }

    fn process_inputs(&mut self) {
        while let Ok(chunk) = self.audio_rx.try_recv() {
            self.sample_buffer.extend_from_slice(&chunk);
            self.all_audio.extend_from_slice(&chunk);
        }

        while let Ok(lang) = self.lang_rx.try_recv() {
            self.pending_language = Some(lang);
        }

        while let Ok(ctrl) = self.control_rx.try_recv() {
            match ctrl {
                SessionControl::Stop | SessionControl::Closed => {
                    self.closed = true;
                }
            }
        }
    }

    fn prepare_flush(&mut self) {
        if self.closed && self.flush_samples_remaining > 0 && self.sample_buffer.len() < FRAME_SIZE
        {
            let to_add = FRAME_SIZE.min(self.flush_samples_remaining);
            self.sample_buffer
                .extend(std::iter::repeat(0.0).take(to_add));
            self.flush_samples_remaining = self.flush_samples_remaining.saturating_sub(to_add);
        } else if self.closed
            && self.flush_samples_remaining == 0
            && !self.sample_buffer.is_empty()
            && self.sample_buffer.len() < FRAME_SIZE
        {
            let needed = FRAME_SIZE - self.sample_buffer.len();
            self.sample_buffer
                .extend(std::iter::repeat(0.0).take(needed));
        }
    }

    fn take_frame(&mut self) -> Option<Vec<f32>> {
        if self.sample_buffer.len() < FRAME_SIZE {
            return None;
        }
        let frame: Vec<f32> = self.sample_buffer.drain(..FRAME_SIZE).collect();
        Some(frame)
    }

    fn handle_asr_msg(&mut self, msg: moshi::asr::AsrMsg, model: &Model) {
        match msg {
            moshi::asr::AsrMsg::Word {
                tokens, start_time, ..
            } => {
                self.last_voice_activity = Some(Instant::now());
                let word = model
                    .decode_tokens(&tokens)
                    .unwrap_or_else(|_| String::new());
                if word.trim().is_empty() {
                    return;
                }
                self.current_text.push(' ');
                self.current_text.push_str(&word);

                if !self.timestamps && !self.word_sent {
                    self.sink.handle_message(WebSocketMessage::Word {
                        word: word.clone(),
                        start_time,
                        end_time: None,
                    });
                    self.word_sent = true;
                }

                self.last_word = Some((word, start_time));
            }
            moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                self.last_voice_activity = Some(Instant::now());
                if let Some((word, start_time)) = self.last_word.take() {
                    let had_live_emit = self.word_sent;
                    self.words.push(WordTimestamp {
                        word: word.clone(),
                        start_time,
                        end_time: Some(stop_time),
                    });
                    if !self.timestamps && !had_live_emit {
                        self.sink.handle_message(WebSocketMessage::Word {
                            word,
                            start_time,
                            end_time: Some(stop_time),
                        });
                    }
                    self.word_sent = false;
                }
            }
            moshi::asr::AsrMsg::Step { .. } => {}
        }
    }

    fn check_vad_timeout(&mut self) {
        if let (true, Some(timeout), Some(last)) =
            (self.vad_enabled, self.vad_timeout, self.last_voice_activity)
        {
            if last.elapsed() > Duration::from_secs_f64(timeout) {
                self.closed = true;
            }
        }
    }

    fn should_finalize(&self) -> bool {
        self.closed && self.sample_buffer.is_empty() && self.flush_samples_remaining == 0
    }

    fn finalize(&mut self) {
        if let Some((word, start_time)) = self.last_word.take() {
            self.words.push(WordTimestamp {
                word: word.clone(),
                start_time,
                end_time: None,
            });
        }

        let final_text = self.current_text.trim().to_string();
        self.sink.handle_message(WebSocketMessage::Final {
            text: final_text,
            words: self.words.clone(),
        });
        self.word_sent = false;
        self.sink.close();
    }
}

pub(crate) fn spawn_parallel_engine(
    mut model: Model,
    prime_languages: Vec<String>,
) -> ParallelEngine {
    let capacity = model.batch_size();
    let (command_tx, command_rx) = unbounded::<EngineCommand>();
    let engine_tx = command_tx.clone();

    std::thread::spawn(move || {
        if let Err(err) = run_parallel_loop(&mut model, prime_languages, command_rx) {
            eprintln!("[ears-server] parallel engine error: {err:#}");
        }
    });

    ParallelEngine {
        command_tx: engine_tx,
        capacity,
    }
}

impl ParallelEngine {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn allocate_session(&self, sink: SessionSink) -> Result<Option<SessionHandle>> {
        let (response_tx, response_rx) = unbounded();
        self.command_tx
            .send(EngineCommand::Allocate {
                sink,
                response: response_tx,
            })
            .context("failed to request session allocation")?;
        let allocation = response_rx
            .recv()
            .context("failed to receive session allocation response")?;
        Ok(allocation.map(|alloc| SessionHandle {
            audio_tx: alloc.audio_tx,
            lang_tx: alloc.lang_tx,
            control_tx: alloc.control_tx,
        }))
    }
}

impl Engine for ParallelEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Kyutai
    }

    fn allocate(&self, sink: SessionSink) -> anyhow::Result<Option<Box<dyn EngineSession>>> {
        let session = self
            .allocate_session(sink)
            .context("failed to allocate kyutai session")?;
        Ok(session.map(|handle| Box::new(handle) as Box<dyn EngineSession>))
    }
}

fn run_parallel_loop(
    model: &mut Model,
    prime_languages: Vec<String>,
    command_rx: Receiver<EngineCommand>,
) -> Result<()> {
    let capacity = model.batch_size();
    let mut sessions: Vec<Option<SessionState>> =
        std::iter::repeat_with(|| None).take(capacity).collect();
    let base_flush = model.audio_delay_samples() + 24_000;
    let flush_samples = ((base_flush + FRAME_SIZE - 1) / FRAME_SIZE) * FRAME_SIZE;
    let timestamps = model.timestamps_enabled();
    let vad_enabled = model.vad_enabled();
    let vad_timeout = model.vad_timeout_seconds();
    let dev = model.device().clone();

    loop {
        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                EngineCommand::Allocate { sink, response } => {
                    let alloc = allocate_session(
                        &mut sessions,
                        sink,
                        timestamps,
                        vad_enabled,
                        vad_timeout,
                        flush_samples,
                    );
                    if let Some(ref allocation) = alloc {
                        for lang in &prime_languages {
                            if let Err(err) =
                                model.prime_with_lang_code_for_slot(lang, allocation.batch_idx)
                            {
                                eprintln!(
                                    "[ears-server] failed to prime language {} for slot {}: {}",
                                    lang, allocation.batch_idx, err
                                );
                            }
                        }
                    }
                    let _ = response.send(alloc);
                }
            }
        }

        let mut mask_vec = vec![false; capacity];
        let mut pcm_batch = vec![0.0f32; capacity * FRAME_SIZE];

        for (idx, state) in sessions.iter_mut().enumerate() {
            if let Some(session) = state {
                session.process_inputs();
                session.check_vad_timeout();
                if let Some(lang) = session.pending_language.take() {
                    if let Err(err) = model.prime_with_lang_code_for_slot(&lang, idx) {
                        eprintln!(
                            "[ears-server] failed to prime language {} for session {}: {}",
                            lang, session.id, err
                        );
                    } else {
                        session
                            .sink
                            .handle_message(WebSocketMessage::LanguageChanged { lang });
                    }
                }
                session.prepare_flush();
                if let Some(frame) = session.take_frame() {
                    let start = idx * FRAME_SIZE;
                    pcm_batch[start..start + FRAME_SIZE].copy_from_slice(&frame);
                    mask_vec[idx] = true;
                }
            }
        }

        if !mask_vec.iter().any(|&v| v) {
            std::thread::sleep(Duration::from_millis(5));
        } else {
            let mask = StreamMask::new(mask_vec.clone(), &dev)?;
            let pcm = Tensor::from_vec(pcm_batch.clone(), (capacity, 1, FRAME_SIZE), &dev)?;
            let msgs = model.step_pcm_with_mask(pcm, &mask)?;
            for msg in msgs {
                match msg {
                    moshi::asr::AsrMsg::Step { .. } => {}
                    msg @ moshi::asr::AsrMsg::Word { batch_idx, .. }
                    | msg @ moshi::asr::AsrMsg::EndWord { batch_idx, .. } => {
                        if let Some(session) = sessions.get_mut(batch_idx).and_then(Option::as_mut)
                        {
                            session.handle_asr_msg(msg, model);
                        }
                    }
                }
            }
        }

        for (idx, state) in sessions.iter_mut().enumerate() {
            if let Some(session) = state {
                if session.should_finalize() {
                    session.finalize();
                    model.reset_batch_slot(idx)?;
                    *state = None;
                }
            }
        }
    }
}

fn allocate_session(
    sessions: &mut [Option<SessionState>],
    sink: SessionSink,
    timestamps: bool,
    vad_enabled: bool,
    vad_timeout: Option<f64>,
    flush_samples: usize,
) -> Option<SessionAllocation> {
    static ID_GEN: AtomicU64 = AtomicU64::new(1);
    let id = ID_GEN.fetch_add(1, Ordering::Relaxed);

    if let Some(slot) = sessions.iter().position(|s| s.is_none()) {
        let (audio_tx, audio_rx) = unbounded();
        let (lang_tx, lang_rx) = unbounded();
        let (control_tx, control_rx) = unbounded();
        let session = SessionState::new(
            id,
            sink,
            audio_rx,
            lang_rx,
            control_rx,
            timestamps,
            vad_enabled,
            vad_timeout,
            flush_samples,
        );
        sessions[slot] = Some(session);
        Some(SessionAllocation {
            batch_idx: slot,
            audio_tx,
            lang_tx,
            control_tx,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sink() -> SessionSink {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        SessionSink::new(tx)
    }

    #[test]
    fn allocate_session_respects_capacity() {
        let mut sessions = vec![None, None];
        let allocation = allocate_session(&mut sessions, make_sink(), false, false, None, 1920);
        assert!(allocation.is_some());
        let allocation = allocate_session(&mut sessions, make_sink(), false, false, None, 1920);
        assert!(allocation.is_some());
        let allocation = allocate_session(&mut sessions, make_sink(), false, false, None, 1920);
        assert!(allocation.is_none());
    }

    #[test]
    fn session_flush_pads_to_frame() {
        let (audio_tx, audio_rx) = unbounded();
        let (_lang_tx, lang_rx) = unbounded();
        let (_ctrl_tx, control_rx) = unbounded();
        let mut state = SessionState::new(
            1,
            make_sink(),
            audio_rx,
            lang_rx,
            control_rx,
            false,
            false,
            None,
            FRAME_SIZE / 2,
        );
        let _ = audio_tx;
        state.closed = true;
        state.prepare_flush();
        assert!(state.sample_buffer.len() < FRAME_SIZE);
        state.prepare_flush();
        assert_eq!(state.sample_buffer.len(), FRAME_SIZE);
        let frame = state.take_frame();
        assert_eq!(frame.map(|f| f.len()), Some(FRAME_SIZE));
    }
}
