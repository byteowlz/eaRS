use super::*;
use std::collections::HashSet;
use std::sync::Mutex as StdMutex;

type Log = Arc<StdMutex<Vec<String>>>;

/// A fake set of microphones: plug/unplug devices and change the OS default.
#[derive(Clone, Default)]
struct World {
    log: Log,
    plugged: Arc<StdMutex<HashSet<String>>>,
    default: Arc<StdMutex<Option<String>>>,
    events: Arc<StdMutex<Vec<DictationEvent>>>,
    streams: Arc<StdMutex<Vec<Arc<AtomicBool>>>>,
}

impl World {
    fn with(devices: &[&str], default: &str) -> Self {
        let world = Self::default();
        for d in devices {
            world.plug(d);
        }
        *world.default.lock().unwrap() = Some(default.into());
        world
    }
    fn plug(&self, device: &str) {
        self.plugged.lock().unwrap().insert(device.into());
    }
    /// Unplug: the device disappears and every open stream on it fails.
    fn unplug(&self, device: &str) {
        self.plugged.lock().unwrap().remove(device);
        for failed in self.streams.lock().unwrap().iter() {
            failed.store(true, Ordering::SeqCst);
        }
    }
    fn set_default(&self, device: &str) {
        *self.default.lock().unwrap() = Some(device.into());
    }
    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
    fn clear(&self) {
        self.log.lock().unwrap().clear();
        self.events.lock().unwrap().clear();
    }
    fn events(&self) -> Vec<DictationEvent> {
        self.events.lock().unwrap().clone()
    }
    fn slot(&self) -> InputSlot {
        let world = self.clone();
        let events = self.events.clone();
        let default = self.default.clone();
        InputSlot::new(
            Box::new(move |selection, enabled| {
                let device = match selection {
                    InputSelection::Default => {
                        world.default.lock().unwrap().clone().unwrap_or_default()
                    }
                    InputSelection::Named(name) => name,
                    InputSelection::Index(i) => format!("#{i}"),
                };
                if !world.plugged.lock().unwrap().contains(&device) {
                    world.log.lock().unwrap().push(format!("fail {device}"));
                    return Err(anyhow!("cannot open {device}"));
                }
                world
                    .log
                    .lock()
                    .unwrap()
                    .push(format!("open {device} enabled={enabled}"));
                let failed = Arc::new(AtomicBool::new(false));
                world.streams.lock().unwrap().push(failed.clone());
                Ok(Box::new(Fake {
                    device,
                    log: world.log.clone(),
                    failed,
                }) as Box<dyn InputStream>)
            }),
            Box::new(move |event| events.lock().unwrap().push(event)),
            Box::new(move || default.lock().unwrap().clone()),
            Duration::ZERO,
        )
    }
}

struct Fake {
    device: String,
    log: Log,
    failed: Arc<AtomicBool>,
}
impl Fake {
    fn note(&self, what: &str) {
        self.log
            .lock()
            .unwrap()
            .push(format!("{what} {}", self.device));
    }
}
impl InputStream for Fake {
    fn device(&self) -> &str {
        &self.device
    }
    fn enable(&self) {
        self.note("enable");
    }
    fn failed(&self) -> bool {
        self.failed.load(Ordering::SeqCst)
    }
}
impl Drop for Fake {
    fn drop(&mut self) {
        self.note("stop");
    }
}

fn active(world: &World) -> Vec<Option<String>> {
    world
        .events()
        .into_iter()
        .filter_map(|e| match e {
            DictationEvent::InputDevice { active, .. } => Some(active),
            _ => None,
        })
        .collect()
}

#[test]
fn switching_opens_muted_then_stops_old_before_unmuting_new() {
    let world = World::with(&["Built-in", "USB"], "Built-in");
    let slot = world.slot();
    slot.start_now(None, 3);
    assert_eq!(
        slot.switch(InputSelection::Named("USB".into())).unwrap(),
        "USB"
    );
    assert_eq!(slot.device().as_deref(), Some("USB"));
    assert_eq!(
        world.log(),
        [
            "open Built-in enabled=true",
            "open USB enabled=false",
            "stop Built-in",
            "enable USB"
        ]
    );
}

#[test]
fn failed_switch_keeps_the_running_input() {
    let world = World::with(&["USB"], "USB");
    let slot = world.slot();
    slot.start_now(Some("USB".into()), 1);
    let error = slot
        .switch(InputSelection::Named("Broken".into()))
        .unwrap_err();
    assert!(error.to_string().contains("Broken"));
    assert_eq!(slot.device().as_deref(), Some("USB"));
    assert!(!world.log().iter().any(|l| l == "stop USB"));
}

#[test]
fn an_unplugged_named_microphone_reconnects_when_it_returns_and_never_falls_back() {
    let world = World::with(&["Built-in", "BRIO"], "Built-in");
    let slot = world.slot();
    slot.start_now(Some("BRIO".into()), 1);
    world.clear();
    world.unplug("BRIO");
    for _ in 0..5 {
        slot.supervise();
    }
    assert!(slot.device().is_none());
    assert_eq!(
        world.log().first().map(String::as_str),
        Some("stop BRIO"),
        "dead stream released"
    );
    assert!(
        world.log().iter().all(|l| !l.contains("Built-in")),
        "no silent fallback: {:?}",
        world.log()
    );
    assert_eq!(active(&world), [None], "loss reported once, not every tick");
    world.plug("BRIO");
    slot.supervise();
    assert_eq!(slot.device().as_deref(), Some("BRIO"));
    assert_eq!(active(&world), [None, Some("BRIO".into())]);
}

#[test]
fn system_default_follows_the_os_default_and_recovers_after_loss() {
    let world = World::with(&["Built-in", "Headset"], "Built-in");
    let slot = world.slot();
    slot.start_now(None, 1);
    world.clear();
    slot.supervise();
    assert!(world.log().is_empty(), "unchanged default does nothing");
    world.set_default("Headset");
    slot.supervise();
    assert_eq!(slot.device().as_deref(), Some("Headset"));
    assert_eq!(
        world.log(),
        [
            "open Headset enabled=false",
            "stop Built-in",
            "enable Headset"
        ]
    );
    // Headset unplugged: macOS falls back to the built-in mic, so do we.
    world.unplug("Headset");
    world.set_default("Built-in");
    slot.supervise();
    assert_eq!(slot.device().as_deref(), Some("Built-in"));
}

#[test]
fn a_named_selection_ignores_os_default_changes() {
    let world = World::with(&["Built-in", "USB"], "Built-in");
    let slot = world.slot();
    slot.start_now(Some("USB".into()), 1);
    world.set_default("Built-in");
    world.clear();
    slot.supervise();
    assert_eq!(slot.device().as_deref(), Some("USB"));
    assert!(world.log().is_empty());
}

#[test]
fn a_microphone_missing_at_startup_is_picked_up_when_plugged_in() {
    let world = World::with(&["Built-in"], "Built-in");
    let slot = world.slot();
    slot.start_now(Some("BRIO".into()), 2);
    assert_eq!(
        world.log().iter().filter(|l| *l == "fail BRIO").count(),
        2,
        "retried"
    );
    assert!(world.events().iter().any(|e| matches!(e,
        DictationEvent::InputDevice { active: None, error: Some(err), .. } if err.contains("reconnecting"))));
    slot.supervise();
    assert!(slot.device().is_none());
    assert_eq!(active(&world), [None], "still one report while waiting");
    world.plug("BRIO");
    slot.supervise();
    assert_eq!(slot.device().as_deref(), Some("BRIO"));
}

#[test]
fn dropping_the_slot_stops_capture() {
    let world = World::with(&["Built-in"], "Built-in");
    let slot = world.slot();
    slot.start_now(None, 1);
    drop(slot);
    assert_eq!(
        world.log().last().map(String::as_str),
        Some("stop Built-in")
    );
}

#[test]
fn the_supervisor_exits_once_the_engine_drops_the_slot() {
    let world = World::with(&["Built-in"], "Built-in");
    let slot = Arc::new(world.slot());
    slot.start_now(None, 1);
    let weak = Arc::downgrade(&slot);
    let supervisor =
        thread::spawn(move || InputSlot::supervise_forever(weak, Duration::from_millis(5)));
    thread::sleep(Duration::from_millis(20));
    drop(slot);
    supervisor.join().unwrap();
    assert_eq!(
        world.log().last().map(String::as_str),
        Some("stop Built-in")
    );
}

#[test]
fn a_switch_or_supervisor_tick_during_startup_is_refused_not_blocked() {
    let world = World::with(&["Built-in"], "Built-in");
    let slot = world.slot();
    let held = slot.current.lock().unwrap();
    assert!(
        slot.switch(InputSelection::Default)
            .unwrap_err()
            .to_string()
            .contains("busy")
    );
    slot.supervise(); // returns immediately instead of deadlocking
    drop(held);
}
