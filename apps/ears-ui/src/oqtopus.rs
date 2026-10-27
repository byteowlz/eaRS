//! Oqtopus indicator: shared vector runtime on a bounded background worker.
//! Only coarse UI state crosses the boundary; no speech or transcripts are sent.
use crate::parametric_mark::State as MarkState;
use gpui::{Hsla, IntoElement, Path, Rgba, Styled, canvas, point, px};
use oqtopus_core::{Pack, Player, PlayerSample, RestMode, decode_frame, decode_pack};
use oqtopus_render::{Composition, PreparedFrame, Region, compose, prepare_frame};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Snapshot {
    pub regions: Vec<Region>,
    pub width: f32,
    pub height: f32,
}
#[derive(Clone, Copy)]
struct Control {
    signal: MarkState,
    animate: bool,
}
struct Shared {
    control: Mutex<Control>,
    snapshot: Mutex<Option<Arc<Snapshot>>>,
    stop: AtomicBool,
}

pub struct Worker {
    shared: Arc<Shared>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
    }
}
impl Worker {
    pub fn start(path: PathBuf) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            control: Mutex::new(Control {
                signal: MarkState::Ready,
                animate: true,
            }),
            snapshot: Mutex::new(None),
            stop: AtomicBool::new(false),
        });
        let background = shared.clone();
        std::thread::Builder::new()
            .name("ears-oqtopus".into())
            .spawn(move || {
                let mut warned = false;
                while !background.stop.load(Ordering::Acquire) {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run(&path, &background)
                    }))
                    .unwrap_or_else(|_| Err("vector geometry failed".into()));
                    match result {
                        Ok(()) => break,
                        Err(error) => {
                            if !warned {
                                eprintln!("ears-ui: Oqtopus unavailable: {error}");
                                warned = true;
                            }
                            if let Ok(mut frame) = background.snapshot.lock() {
                                *frame = None;
                            }
                            // Retry local pack changes without interrupting dictation.
                            for _ in 0..30 {
                                if background.stop.load(Ordering::Acquire) {
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(100));
                            }
                        }
                    }
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(Self { shared })
    }
    pub fn set(&self, signal: MarkState, animate: bool) {
        let signal = if signal == MarkState::Speech {
            MarkState::Listening
        } else {
            signal
        };
        if let Ok(mut c) = self.shared.control.lock() {
            *c = Control { signal, animate };
        }
    }
    pub fn snapshot(&self) -> Option<Arc<Snapshot>> {
        self.shared.snapshot.lock().ok()?.clone()
    }
}

pub fn pack_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("EARS_OQTOPUS_PACK").filter(|v| !v.is_empty()) {
        return Some(path.into());
    }
    std::env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
        .map(|base| base.join("ears/oqtopus.oqp"))
}

fn load_pack(path: &std::path::Path) -> Result<Pack, String> {
    use std::io::Read;
    const MAX: u64 = 64 * 1024 * 1024 + 52;
    let file = std::fs::File::open(path).map_err(|e| format!("cannot read vector pack: {e}"))?;
    let mut bytes = Vec::new();
    file.take(MAX + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read vector pack: {e}"))?;
    if bytes.len() as u64 > MAX {
        return Err("vector pack exceeds size limit".into());
    }
    decode_pack(&bytes).map_err(|e| e.to_string())
}

/// Read-only CLI check: validate the pack, player, and rest-frame geometry.
pub fn check(path: &std::path::Path) -> Result<usize, String> {
    let pack = load_pack(path)?;
    let player = Player::new(&pack, 7).map_err(|e| e.to_string())?;
    let rest = player.rest_paths(true).map_err(|e| e.to_string())?;
    let frame = prepare_frame(&rest).map_err(|e| e.to_string())?;
    let geometry = compose(&frame, None, Composition::Single).map_err(|e| e.to_string())?;
    if geometry.is_empty() {
        return Err("empty rest artwork".into());
    }
    Ok(pack
        .manifest
        .get("clips")
        .and_then(|v| v.as_object())
        .map_or(0, |v| v.len()))
}

fn change_signal(player: &mut Player<'_>, previous: MarkState, current: MarkState) {
    if previous == current {
        return;
    }
    player.release();
    match current {
        MarkState::Processing => {
            if player.hold("thinking").is_err() {
                player.release();
            }
        }
        // The current check mark is time-based, not a confirmed success event.
        MarkState::Done => player.stop(),
        MarkState::Attention => {
            let _ = player.play("surprised");
        }
        MarkState::Ready | MarkState::Listening | MarkState::Speech => {
            player.stop();
        }
    }
}

#[derive(Default)]
struct Cache {
    frames: HashMap<usize, PreparedFrame>,
    order: VecDeque<usize>,
}
impl Cache {
    fn get(&mut self, frame: &oqtopus_core::Frame) -> Result<PreparedFrame, String> {
        let key = frame as *const _ as usize;
        if let Some(value) = self.frames.get(&key) {
            return Ok(value.clone());
        }
        let prepared = prepare_frame(&decode_frame(frame).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if self.order.len() >= 96 {
            if let Some(old) = self.order.pop_front() {
                self.frames.remove(&old);
            }
        }
        self.order.push_back(key);
        self.frames.insert(key, prepared.clone());
        Ok(prepared)
    }
}

fn regions(player: &Player<'_>, cache: &mut Cache, animate: bool) -> Result<Vec<Region>, String> {
    match player.sample(!animate).map_err(|e| e.to_string())? {
        PlayerSample::Rest(_) => {
            let rest = prepare_frame(&player.rest_paths(!animate).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            compose(&rest, None, Composition::Single).map_err(|e| e.to_string())
        }
        PlayerSample::Frame {
            frame, rest_weight, ..
        } if rest_weight <= 0.0 => {
            let current = cache.get(frame)?;
            compose(&current, None, Composition::Single).map_err(|e| e.to_string())
        }
        PlayerSample::Frame {
            frame, rest_weight, ..
        } => {
            let rest = prepare_frame(&player.rest_paths(!animate).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            if rest_weight >= 1.0 {
                return compose(&rest, None, Composition::Single).map_err(|e| e.to_string());
            }
            let current = cache.get(frame)?;
            compose(&rest, Some(&current), Composition::Over(1.0 - rest_weight))
                .map_err(|e| e.to_string())
        }
        PlayerSample::Crossfade { from, to, weight } => {
            let a = cache.get(from)?;
            let b = cache.get(to)?;
            compose(&a, Some(&b), Composition::Crossfade(weight)).map_err(|e| e.to_string())
        }
    }
}

fn run(path: &std::path::Path, shared: &Shared) -> Result<(), String> {
    let pack = load_pack(path)?;
    let size = pack
        .manifest
        .get("vector_size")
        .and_then(|v| v.as_array())
        .ok_or("invalid vector size")?;
    let width = size
        .first()
        .and_then(|v| v.as_u64())
        .ok_or("invalid vector width")? as f32;
    let height = size
        .get(1)
        .and_then(|v| v.as_u64())
        .ok_or("invalid vector height")? as f32;
    let mut player = Player::new(&pack, 7).map_err(|e| e.to_string())?;
    player.set_rest(RestMode::None);
    let mut cache = Cache::default();
    let mut previous = MarkState::Ready;
    let mut last = Instant::now();
    while !shared.stop.load(Ordering::Acquire) {
        let control = *shared
            .control
            .lock()
            .map_err(|_| "indicator control poisoned")?;
        change_signal(&mut player, previous, control.signal);
        previous = control.signal;
        let now = Instant::now();
        let elapsed = now.duration_since(last).as_secs_f64();
        last = now;
        if control.animate {
            player.tick(elapsed).map_err(|e| e.to_string())?;
        }
        let frame = Snapshot {
            regions: regions(&player, &mut cache, control.animate)?,
            width,
            height,
        };
        *shared
            .snapshot
            .lock()
            .map_err(|_| "indicator frame poisoned")? = Some(Arc::new(frame));
        std::thread::sleep(Duration::from_millis(if control.animate {
            33
        } else {
            250
        }));
    }
    Ok(())
}

/// Render vector triangles with the active theme, preserving composite colors.
pub fn mark(snapshot: Arc<Snapshot>, body: Hsla, background: Hsla, size: f32) -> impl IntoElement {
    canvas(
        |_, _, _| (),
        move |bounds, (), window, _| {
            let body: Rgba = body.into();
            let bg: Rgba = background.into();
            let stripe = Rgba {
                r: body.r * 0.8 + bg.r * 0.2,
                g: body.g * 0.8 + bg.g * 0.2,
                b: body.b * 0.8 + bg.b * 0.2,
                a: 1.0,
            };
            let scale = (size / snapshot.width).min(size / snapshot.height);
            let ox = (size - snapshot.width * scale) / 2.0;
            let oy = (size - snapshot.height * scale) / 2.0;
            for region in &snapshot.regions {
                let color = Rgba {
                    r: body.r + (stripe.r - body.r) * region.stripe_mix,
                    g: body.g + (stripe.g - body.g) * region.stripe_mix,
                    b: body.b + (stripe.b - body.b) * region.stripe_mix,
                    a: region.opacity,
                };
                let to_point = |p: [f32; 2]| {
                    point(
                        bounds.origin.x + px(ox + p[0] * scale),
                        bounds.origin.y + px(oy + p[1] * scale),
                    )
                };
                let mut path = Path::new(bounds.origin);
                for t in region.triangles.chunks_exact(3) {
                    path.push_triangle(
                        (to_point(t[0]), to_point(t[1]), to_point(t[2])),
                        (point(0., 1.), point(0., 1.), point(0., 1.)),
                    );
                }
                window.paint_path(path, Hsla::from(color));
            }
        },
    )
    .size(px(size))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn processing_uses_thinking_and_time_based_done_does_not_celebrate() {
        let mut pack =
            decode_pack(include_bytes!("../tests/fixtures/oqtopus-synthetic.oqp")).unwrap();
        pack.manifest["clips"]["idle-b"]["action"] = serde_json::json!("thinking");
        let mut player = Player::new(&pack, 7).unwrap();
        player.set_idle(false);
        change_signal(&mut player, MarkState::Listening, MarkState::Processing);
        assert_eq!(player.state().clip.as_deref(), Some("idle-b"));
        assert!(player.state().hold);
        change_signal(&mut player, MarkState::Processing, MarkState::Done);
        assert!(!player.state().hold);
        assert_ne!(player.state().clip.as_deref(), Some("celebrate"));
    }

    #[test]
    fn missing_thinking_clip_does_not_leave_idle_stuck_in_hold() {
        let pack = decode_pack(include_bytes!("../tests/fixtures/oqtopus-synthetic.oqp")).unwrap();
        let mut player = Player::new(&pack, 7).unwrap();
        change_signal(&mut player, MarkState::Listening, MarkState::Processing);
        assert!(!player.state().hold);
    }

    #[test]
    fn missing_or_corrupt_pack_fails_without_starting_dictation() {
        let path = std::env::temp_dir().join("ears-oqtopus-this-file-must-not-exist.oqp");
        if !path.exists() {
            assert!(check(&path).is_err());
        }
        assert!(decode_pack(b"not a pack").is_err());
    }

    #[test]
    fn worker_recovers_when_a_missing_pack_appears() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ears-oqtopus-retry-{}-{nonce}.oqp",
            std::process::id()
        ));
        let worker = Worker::start(path.clone()).unwrap();
        std::thread::sleep(Duration::from_millis(150));
        assert!(worker.snapshot().is_none());
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(include_bytes!("../tests/fixtures/oqtopus-synthetic.oqp"))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(8);
        while worker.snapshot().is_none() {
            assert!(
                Instant::now() < deadline,
                "worker failed to reload the pack"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        drop(worker);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn synthetic_worker_animates_and_respects_reduced_motion() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ears-oqtopus-test-{}-{nonce}.oqp",
            std::process::id()
        ));
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(include_bytes!("../tests/fixtures/oqtopus-synthetic.oqp"))
            .unwrap();
        let worker = Worker::start(path.clone()).unwrap();
        worker.set(MarkState::Speech, true);
        assert_eq!(
            worker.shared.control.lock().unwrap().signal,
            MarkState::Listening
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        let first = loop {
            if let Some(frame) = worker.snapshot() {
                break format!("{:?}", frame.regions);
            }
            assert!(Instant::now() < deadline, "no synthetic vector frame");
            std::thread::sleep(Duration::from_millis(20));
        };
        loop {
            std::thread::sleep(Duration::from_millis(50));
            let next = format!("{:?}", worker.snapshot().unwrap().regions);
            if next != first {
                break;
            }
            assert!(Instant::now() < deadline, "animation did not advance");
        }
        worker.set(MarkState::Listening, false);
        std::thread::sleep(Duration::from_millis(600));
        let a = format!("{:?}", worker.snapshot().unwrap().regions);
        std::thread::sleep(Duration::from_millis(350));
        let b = format!("{:?}", worker.snapshot().unwrap().regions);
        assert_eq!(a, b, "reduced motion did not remain still");
        drop(worker);
        let _ = std::fs::remove_file(path);
    }
}
