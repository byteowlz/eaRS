use device_render_wgpu::{OffscreenRenderer, RenderFrame};
use device_scene::ValidatedSnapshot;
use std::sync::{Arc, Condvar, Mutex};

pub struct Completed {
    pub generation: u64,
    pub snapshot: Arc<ValidatedSnapshot>,
    pub frame: Result<RenderFrame, String>,
}
struct Request {
    generation: u64,
    snapshot: Arc<ValidatedSnapshot>,
}
#[derive(Default)]
struct State {
    latest: u64,
    pending: Option<Request>,
    completed: Option<Completed>,
    last_requested: Option<Arc<ValidatedSnapshot>>,
    stopped: bool,
    rendered: u64,
    /// Newest generation offered to the UI. Frames only move forward.
    published: u64,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}
/// One worker, one active request, one replaceable pending request and one completed slot.
/// A finished frame is published when it is newer than the last one published, even if
/// newer input already queued: continuous drags must still show intermediate frames.
/// The notification channel holds no frames. No timer, idle rendering or UI-thread device polling.
pub struct RenderWorker {
    shared: Arc<Shared>,
}
impl RenderWorker {
    pub fn start() -> (Self, async_channel::Receiver<()>) {
        let mut renderer = None;
        Self::start_with(move |snapshot, generation| {
            let renderer = renderer.get_or_insert_with(|| {
                pollster::block_on(OffscreenRenderer::new()).map_err(|e| e.to_string())
            });
            match renderer {
                Ok(renderer) => renderer
                    .render(snapshot, generation)
                    .map_err(|e| e.to_string()),
                Err(error) => Err(error.clone()),
            }
        })
    }
    fn start_with(
        mut render: impl FnMut(&ValidatedSnapshot, u64) -> Result<RenderFrame, String> + Send + 'static,
    ) -> (Self, async_channel::Receiver<()>) {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        });
        let (notify, notifications) = async_channel::bounded(1);
        let thread_shared = shared.clone();
        std::thread::Builder::new()
            .name("device-render".into())
            .spawn(move || {
                loop {
                    let request = {
                        let mut state = thread_shared.state.lock().expect("render state");
                        while state.pending.is_none() && !state.stopped {
                            state = thread_shared.changed.wait(state).expect("render state");
                        }
                        if state.stopped {
                            break;
                        }
                        state.pending.take().expect("pending request")
                    };
                    let frame = render(&request.snapshot, request.generation);
                    let mut state = thread_shared.state.lock().expect("render state");
                    state.rendered += 1;
                    if state.stopped {
                        break;
                    }
                    if request.generation > state.published {
                        state.published = request.generation;
                        state.completed = Some(Completed {
                            generation: request.generation,
                            snapshot: request.snapshot,
                            frame,
                        });
                        let _ = notify.try_send(());
                    }
                }
            })
            .expect("renderer thread creation");
        (Self { shared }, notifications)
    }
    /// Hosts replace scene IDs to start a new lifetime. Equal revisions may only resize the viewport.
    pub fn submit(&self, snapshot: ValidatedSnapshot) -> Result<u64, String> {
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| "renderer state poisoned")?;
        if state.stopped {
            return Err("renderer stopped".into());
        }
        if let Some(last) = &state.last_requested {
            let old = last.snapshot();
            let new = snapshot.snapshot();
            if old.scene.id == new.scene.id {
                if new.scene.revision < old.scene.revision
                    || new.state.revision < old.state.revision
                {
                    return Err("revisions must not go backwards within a scene lifetime".into());
                }
                if new.scene.revision == old.scene.revision && new.scene != old.scene {
                    return Err("scene changed without revision change".into());
                }
                if new.state.revision == old.state.revision && new.state != old.state {
                    return Err("state changed without revision change".into());
                }
            }
        }
        let snapshot = Arc::new(snapshot);
        state.latest = state
            .latest
            .checked_add(1)
            .ok_or("request generation exhausted")?;
        let generation = state.latest;
        state.last_requested = Some(snapshot.clone());
        state.pending = Some(Request {
            generation,
            snapshot,
        });
        self.shared.changed.notify_one();
        Ok(generation)
    }
    pub fn take_completed(&self) -> Option<Completed> {
        self.shared.state.lock().ok()?.completed.take()
    }
    pub fn rendered_requests(&self) -> u64 {
        self.shared.state.lock().map(|s| s.rendered).unwrap_or(0)
    }
}
impl Drop for RenderWorker {
    fn drop(&mut self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.stopped = true;
            state.pending = None;
            state.completed = None;
            state.last_requested = None;
            self.shared.changed.notify_one();
        }
        // No join on UI thread. An active readback has a finite timeout, then the worker exits.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use device_scene::Snapshot;
    use std::{
        sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };
    fn fixture() -> ValidatedSnapshot {
        Snapshot::parse(include_str!(
            "../../../../spec/fixtures/device-scene-v1/ears-input.json"
        ))
        .unwrap()
    }
    #[test]
    fn burst_keeps_only_latest_pending_and_never_renders_idle() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let calls = Arc::new(AtomicUsize::new(0));
        let (a, b, n) = (entered.clone(), release.clone(), calls.clone());
        let (worker, notify) = RenderWorker::start_with(move |_, generation| {
            if n.fetch_add(1, Ordering::SeqCst) == 0 {
                a.wait();
                b.wait();
            }
            Err(format!("synthetic result {generation}"))
        });
        worker.submit(fixture()).unwrap();
        entered.wait();
        let mut latest = 0;
        for width in 800..1000 {
            let mut s = fixture().into_inner();
            s.viewport.width_dip = width as f64;
            latest = worker.submit(ValidatedSnapshot::new(s).unwrap()).unwrap();
        }
        release.wait();
        let completed = loop {
            notify.recv_blocking().unwrap();
            if let Some(c) = worker.take_completed()
                && c.generation == latest
            {
                break c;
            }
        };
        assert_eq!(completed.generation, latest);
        assert_eq!(completed.snapshot.snapshot().viewport.width_dip, 999.0);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(worker.take_completed().is_none());
    }
    #[test]
    fn continuous_input_still_presents_the_newest_finished_frame() {
        // A drag submits faster than a frame renders. Discarding every frame that is
        // no longer the latest request starves the screen until the pointer stops.
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let (a, b) = (entered.clone(), release.clone());
        let (worker, _notify) = RenderWorker::start_with(move |_, generation| {
            a.wait();
            b.wait();
            Err(format!("synthetic result {generation}"))
        });
        let mut s = fixture().into_inner();
        let submit = |s: &mut Snapshot| {
            s.state.revision += 1;
            worker
                .submit(ValidatedSnapshot::new(s.clone()).unwrap())
                .unwrap()
        };
        let first = submit(&mut s);
        let mut shown = Vec::new();
        for _ in 0..3 {
            entered.wait();
            // A newer request arrives while every frame renders, then that frame finishes.
            submit(&mut s);
            release.wait();
            let deadline = std::time::Instant::now() + Duration::from_secs(1);
            let completed = loop {
                if let Some(c) = worker.take_completed() {
                    break Some(c);
                }
                if std::time::Instant::now() > deadline {
                    break None;
                }
                std::thread::sleep(Duration::from_millis(2));
            };
            shown.push(completed.map(|c| c.generation));
        }
        assert_eq!(shown, vec![Some(first), Some(first + 1), Some(first + 2)]);
        entered.wait();
        release.wait();
    }
    #[test]
    fn unchanged_revisions_cannot_change_data_but_can_resize() {
        let (worker, _) = RenderWorker::start_with(|_, _| Err("test".into()));
        worker.submit(fixture()).unwrap();
        let mut s = fixture().into_inner();
        s.state.active_page = "dictation.shortcuts".into();
        assert!(
            worker
                .submit(ValidatedSnapshot::new(s.clone()).unwrap())
                .is_err()
        );
        s.state.revision += 1;
        worker.submit(ValidatedSnapshot::new(s).unwrap()).unwrap();
        assert!(worker.submit(fixture()).is_err());
    }
}
