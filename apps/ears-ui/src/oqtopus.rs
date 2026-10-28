//! Oqtopus indicator backed by the shared motion package and vector renderer.
//! Only coarse UI state crosses the worker boundary; no speech or transcripts are sent.
use crate::parametric_mark::State as MarkState;
use motion_core::{MotionPackage, MotionSource, decode_package};
use motion_render::{MotionRenderer, RenderFrame};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Control {
    signal: MarkState,
    animate: bool,
}
struct Shared {
    control: Mutex<Control>,
    snapshot: Mutex<Option<Arc<RenderFrame>>>,
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
            .map_err(|e| e.to_string())?;
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
    pub fn snapshot(&self) -> Option<Arc<RenderFrame>> {
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
        .map(|base| {
            let dir = base.join("ears");
            let new = dir.join("oqtopus.motion");
            let legacy = dir.join("oqtopus.oqp");
            if !new.exists() && legacy.exists() {
                legacy
            } else {
                new
            }
        })
}
fn load_package(path: &Path) -> Result<MotionPackage, String> {
    use std::io::Read;
    const MAX: u64 = 65 * 1024 * 1024 + 128;
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| format!("cannot read motion package: {e}"))?
        .take(MAX + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX {
        return Err("motion package exceeds size limit".into());
    }
    let package = if bytes.starts_with(b"OQTPACK") {
        // The legacy payload has no state mapping; eaRS supplies its provisional mapping.
        MotionPackage::from_manifest(
            include_bytes!("../../../vendor/motion-core/assets/oqtopus.json"),
            bytes,
        )
    } else {
        decode_package(&bytes)
    }
    .map_err(|e| e.to_string())?;
    if !matches!(package.source(), MotionSource::VectorPack(_)) {
        return Err("Oqtopus requires a vector motion package".into());
    }
    for state in ["ready", "listening", "processing", "done", "attention"] {
        if !package.states().contains(&state) {
            return Err(format!("motion package lacks required eaRS state: {state}"));
        }
    }
    Ok(package)
}
/// Read-only CLI check: validate package, state mapping and rest-frame geometry.
pub fn check(path: &Path) -> Result<usize, String> {
    let package = load_package(path)?;
    let clips = match package.source() {
        MotionSource::VectorPack(pack) => pack
            .manifest
            .get("clips")
            .and_then(|v| v.as_object())
            .map_or(0, |v| v.len()),
        _ => 0,
    };
    let mut player = package.player(7).map_err(|e| e.to_string())?;
    player.set_state("ready").map_err(|e| e.to_string())?;
    let frame = MotionRenderer::default()
        .render(player.sample(true).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if frame.regions.is_empty() {
        return Err("empty rest artwork".into());
    }
    Ok(clips)
}
fn run(path: &Path, shared: &Shared) -> Result<(), String> {
    let package = load_package(path)?;
    let mut player = package.player(7).map_err(|e| e.to_string())?;
    player.set_continuous_idle(true);
    let mut renderer = MotionRenderer::default();
    let mut last = Instant::now();
    while !shared.stop.load(Ordering::Acquire) {
        let control = *shared
            .control
            .lock()
            .map_err(|_| "indicator control poisoned")?;
        player
            .set_state(control.signal.id())
            .map_err(|e| e.to_string())?;
        let now = Instant::now();
        let elapsed = now.duration_since(last).as_secs_f64();
        last = now;
        if control.animate {
            player.tick(elapsed).map_err(|e| e.to_string())?;
        }
        let sample = player.sample(!control.animate).map_err(|e| e.to_string())?;
        let frame = renderer.render(sample).map_err(|e| e.to_string())?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn temp_path() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ears-motion-{}-{nonce}.oqp", std::process::id()))
    }
    fn write_fixture(path: &Path) {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap()
            .write_all(include_bytes!("../tests/fixtures/oqtopus-synthetic.oqp"))
            .unwrap();
    }
    #[test]
    fn legacy_and_new_packages_validate_and_missing_or_corrupt_fails() {
        let path = temp_path();
        assert!(check(&path).is_err());
        write_fixture(&path);
        assert!(check(&path).is_ok());
        let package = load_package(&path).unwrap();
        let encoded = motion_core::encode_package(&package).unwrap();
        let packed = temp_path();
        std::fs::write(&packed, encoded).unwrap();
        assert!(check(&packed).is_ok());
        std::fs::write(&packed, b"not a package").unwrap();
        assert!(check(&packed).is_err());
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(packed);
    }
    #[test]
    fn legacy_mapping_uses_thinking_without_treating_done_as_success() {
        let path = temp_path();
        write_fixture(&path);
        let package = load_package(&path).unwrap();
        let rules = package.state_rules();
        let processing = rules.iter().find(|r| r.id == "processing").unwrap();
        assert!(
            matches!(&processing.action, motion_core::MotionAction::Hold { clip } if clip == "thinking")
        );
        assert!(processing.optional);
        let done = rules.iter().find(|r| r.id == "done").unwrap();
        assert!(matches!(done.action, motion_core::MotionAction::Idle));
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn worker_recovers_when_missing_pack_appears() {
        let path = temp_path();
        let worker = Worker::start(path.clone()).unwrap();
        std::thread::sleep(Duration::from_millis(150));
        assert!(worker.snapshot().is_none());
        write_fixture(&path);
        let deadline = Instant::now() + Duration::from_secs(8);
        while worker.snapshot().is_none() {
            assert!(Instant::now() < deadline, "worker did not reload");
            std::thread::sleep(Duration::from_millis(50));
        }
        drop(worker);
        let _ = std::fs::remove_file(path);
    }
    #[test]
    #[ignore = "requires an explicitly supplied local pack"]
    fn local_package_shared_worker_smoke() {
        let path =
            PathBuf::from(std::env::var_os("EARS_OQTOPUS_PACK").expect("set EARS_OQTOPUS_PACK"));
        let worker = Worker::start(path).unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while worker.snapshot().is_none() {
            assert!(
                Instant::now() < deadline,
                "local vector worker did not render"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        for state in [
            MarkState::Listening,
            MarkState::Processing,
            MarkState::Done,
            MarkState::Attention,
        ] {
            worker.set(state, true);
            std::thread::sleep(Duration::from_millis(350));
            assert!(worker.snapshot().is_some_and(|f| !f.regions.is_empty()));
        }
        worker.set(MarkState::Ready, false);
        std::thread::sleep(Duration::from_secs(1));
        let a = format!("{:?}", worker.snapshot().unwrap().regions);
        std::thread::sleep(Duration::from_millis(400));
        let b = format!("{:?}", worker.snapshot().unwrap().regions);
        assert_eq!(a, b);
    }
    #[test]
    fn synthetic_worker_animates_and_respects_reduced_motion() {
        let path = temp_path();
        write_fixture(&path);
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
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        };
        loop {
            std::thread::sleep(Duration::from_millis(50));
            if format!("{:?}", worker.snapshot().unwrap().regions) != first {
                break;
            }
            assert!(Instant::now() < deadline, "animation did not advance");
        }
        worker.set(MarkState::Listening, false);
        std::thread::sleep(Duration::from_millis(600));
        let a = format!("{:?}", worker.snapshot().unwrap().regions);
        std::thread::sleep(Duration::from_millis(350));
        let b = format!("{:?}", worker.snapshot().unwrap().regions);
        assert_eq!(a, b);
        drop(worker);
        let _ = std::fs::remove_file(path);
    }
}
