use super::*;
use std::sync::Mutex as StdMutex;

type Log = Arc<StdMutex<Vec<String>>>;

struct Fake {
    device: String,
    log: Log,
}
impl InputStream for Fake {
    fn device(&self) -> &str {
        &self.device
    }
    fn enable(&self) {
        self.log
            .lock()
            .unwrap()
            .push(format!("enable {}", self.device));
    }
}
impl Drop for Fake {
    fn drop(&mut self) {
        self.log
            .lock()
            .unwrap()
            .push(format!("stop {}", self.device));
    }
}

fn slot(
    log: &Log,
    events: &Arc<StdMutex<Vec<DictationEvent>>>,
    failing: &'static [&'static str],
) -> InputSlot {
    let opened = log.clone();
    let sink = events.clone();
    InputSlot::new(
        Box::new(move |selection, enabled| {
            let device = match selection {
                InputSelection::Default => "Built-in".to_string(),
                InputSelection::Named(name) => name,
                InputSelection::Index(i) => format!("#{i}"),
            };
            if failing.contains(&device.as_str()) {
                opened.lock().unwrap().push(format!("fail {device}"));
                return Err(anyhow!("cannot open {device}"));
            }
            opened
                .lock()
                .unwrap()
                .push(format!("open {device} enabled={enabled}"));
            Ok(Box::new(Fake {
                device,
                log: opened.clone(),
            }) as Box<dyn InputStream>)
        }),
        Box::new(move |event| sink.lock().unwrap().push(event)),
        Duration::ZERO,
    )
}

#[test]
fn switching_opens_muted_then_stops_old_before_unmuting_new() {
    let log: Log = Default::default();
    let events = Default::default();
    let slot = slot(&log, &events, &[]);
    slot.start_now(None, 3);
    assert_eq!(slot.device().as_deref(), Some("Built-in"));
    assert_eq!(
        slot.switch(InputSelection::Named("USB".into())).unwrap(),
        "USB"
    );
    assert_eq!(slot.device().as_deref(), Some("USB"));
    assert_eq!(
        *log.lock().unwrap(),
        [
            "open Built-in enabled=true",
            "open USB enabled=false",
            "stop Built-in",
            "enable USB",
        ]
    );
}

#[test]
fn failed_switch_keeps_the_running_input() {
    let log: Log = Default::default();
    let events = Default::default();
    let slot = slot(&log, &events, &["Broken"]);
    slot.start_now(Some("USB".into()), 1);
    let error = slot
        .switch(InputSelection::Named("Broken".into()))
        .unwrap_err();
    assert!(error.to_string().contains("Broken"));
    assert_eq!(slot.device().as_deref(), Some("USB"));
    assert!(!log.lock().unwrap().iter().any(|l| l == "stop USB"));
}

#[test]
fn startup_reports_the_active_device_or_the_failure() {
    let log: Log = Default::default();
    let events: Arc<StdMutex<Vec<DictationEvent>>> = Default::default();
    let ok = slot(&log, &events, &[]);
    ok.start_now(Some("USB".into()), 1);
    let broken = slot(&log, &events, &["Gone"]);
    broken.start_now(Some("Gone".into()), 2);
    assert_eq!(
        log.lock()
            .unwrap()
            .iter()
            .filter(|l| *l == "fail Gone")
            .count(),
        2,
        "retried"
    );
    assert!(broken.device().is_none());
    let events = events.lock().unwrap();
    assert!(events.contains(&DictationEvent::InputDevice {
        requested: Some("USB".into()),
        active: Some("USB".into()),
        error: None,
    }));
    assert!(events.iter().any(|e| matches!(e,
        DictationEvent::InputDevice { requested: Some(r), active: None, error: Some(_) } if r == "Gone")));
    // A switch after a failed start opens directly enabled; nothing to mute behind.
    drop(events);
    assert_eq!(broken.switch(InputSelection::Default).unwrap(), "Built-in");
    assert!(
        log.lock()
            .unwrap()
            .contains(&"open Built-in enabled=true".to_string())
    );
}

#[test]
fn dropping_the_slot_stops_capture() {
    let log: Log = Default::default();
    let events = Default::default();
    let slot = slot(&log, &events, &[]);
    slot.start_now(None, 1);
    drop(slot);
    assert_eq!(
        log.lock().unwrap().last().map(String::as_str),
        Some("stop Built-in")
    );
}

#[test]
fn a_switch_during_startup_is_refused_not_blocked() {
    let log: Log = Default::default();
    let events = Default::default();
    let slot = slot(&log, &events, &[]);
    let held = slot.current.lock().unwrap();
    assert!(
        slot.switch(InputSelection::Default)
            .unwrap_err()
            .to_string()
            .contains("still starting")
    );
    drop(held);
}
