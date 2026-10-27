//! Clock-driven clip and transition state machine. It does not render.
use crate::{breathing_frame, decode_frame, BreathParams, Frame, FramePaths, Pack, PackError};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Rest,
    Forward,
    Finish,
    Rewind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestMode {
    None,
    Short,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreathMode {
    Off,
    Subtle,
    Soft,
}
#[derive(Debug, Clone)]
pub struct PlayerState {
    pub clip: Option<String>,
    pub frame: f64,
    pub mode: Mode,
    pub hold: bool,
    pub next: Option<String>,
    pub cut_target: Option<String>,
    pub blending: bool,
}
#[derive(Debug, Clone)]
struct Clip {
    action: String,
    kind: String,
    tier: String,
    frames: usize,
    peak: f64,
}
#[derive(Debug, Clone)]
struct Cut {
    at: f64,
    target: String,
    entry: f64,
    bridge: bool,
}
#[derive(Debug, Clone)]
struct Blend {
    from: String,
    at: f64,
    entry: f64,
}

/// A drawable sample is either grouped rest/clip layers or a crossfade between
/// two complete frames. Stripe clipping and whole-layer compositing are the
/// renderer's responsibility.
#[derive(Debug, Clone)]
pub enum PlayerSample<'a> {
    Rest(&'a Frame),
    Frame {
        rest: &'a Frame,
        frame: &'a Frame,
        rest_weight: f64,
    },
    Crossfade {
        from: &'a Frame,
        to: &'a Frame,
        weight: f64,
    },
}

pub struct Player<'a> {
    pack: &'a Pack,
    clips: BTreeMap<String, Clip>,
    groups: BTreeMap<String, Vec<String>>,
    state: PlayerState,
    cut: Option<Cut>,
    blend: Option<Blend>,
    entry: f64,
    fps: f64,
    blend_frames: f64,
    idle: bool,
    cuts: bool,
    rest_for: f64,
    idle_after: f64,
    rest_mode: RestMode,
    last_idle: Option<String>,
    previous_clip: Option<String>,
    recent_actions: Vec<String>,
    speed: f64,
    breath_mode: BreathMode,
    breath_t: f64,
    breath_env: f64,
    seed: u32,
}

fn err(message: &str) -> PackError {
    PackError(message.into())
}
fn metadata_string<'a>(value: &'a serde_json::Value, key: &str, default: &'a str) -> &'a str {
    value.get(key).and_then(|x| x.as_str()).unwrap_or(default)
}

impl<'a> Player<'a> {
    pub fn new(pack: &'a Pack, seed: u32) -> Result<Self, PackError> {
        let fps = pack
            .manifest
            .get("fps")
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0)
            .ok_or_else(|| err("invalid fps"))? as f64;
        let manifest = pack
            .manifest
            .get("clips")
            .and_then(|v| v.as_object())
            .ok_or_else(|| err("invalid clips"))?;
        let mut clips = BTreeMap::new();
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, metadata) in manifest {
            let doc = pack
                .documents
                .get(&format!("clip:{name}"))
                .filter(|d| !d.frames.is_empty())
                .ok_or_else(|| err("missing clip"))?;
            let action = metadata_string(metadata, "action", name).to_owned();
            let peak = metadata
                .get("peak")
                .and_then(|v| v.as_u64())
                .unwrap_or((doc.frames.len() - 1) as u64) as f64;
            if peak >= doc.frames.len() as f64 {
                return Err(err("invalid peak"));
            }
            let kind = metadata_string(metadata, "kind", "action").to_owned();
            let tier = metadata_string(metadata, "tier", "B").to_owned();
            groups.entry(action.clone()).or_default().push(name.clone());
            clips.insert(
                name.clone(),
                Clip {
                    action,
                    kind,
                    tier,
                    frames: doc.frames.len(),
                    peak,
                },
            );
        }
        if let Some(bridges) = pack
            .manifest
            .get("bridges")
            .and_then(|v| v.get("bridges"))
            .and_then(|v| v.as_array())
        {
            for bridge in bridges {
                let name = bridge
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| err("invalid bridge"))?;
                let doc = pack
                    .documents
                    .get(&format!("bridge:{name}"))
                    .filter(|d| !d.frames.is_empty())
                    .ok_or_else(|| err("missing bridge"))?;
                if clips.contains_key(name) {
                    return Err(err("bridge name collides with clip"));
                }
                clips.insert(
                    name.into(),
                    Clip {
                        action: name.into(),
                        kind: "bridge".into(),
                        tier: "B".into(),
                        frames: doc.frames.len(),
                        peak: (doc.frames.len() - 1) as f64,
                    },
                );
            }
        }
        let blend_frames = pack
            .manifest
            .get("transitions")
            .and_then(|v| v.get("blend"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as f64;
        let breath_mode = match pack
            .manifest
            .get("breath")
            .and_then(|v| v.get("default"))
            .and_then(|v| v.as_str())
        {
            Some("off") => BreathMode::Off,
            Some("subtle") => BreathMode::Subtle,
            Some("soft") => BreathMode::Soft,
            _ => return Err(err("invalid breath default")),
        };
        Ok(Self {
            pack,
            clips,
            groups,
            state: PlayerState {
                clip: None,
                frame: 0.0,
                mode: Mode::Rest,
                hold: false,
                next: None,
                cut_target: None,
                blending: false,
            },
            cut: None,
            blend: None,
            entry: 0.0,
            fps,
            blend_frames,
            idle: true,
            cuts: true,
            rest_for: 0.0,
            idle_after: 3.0,
            rest_mode: RestMode::Normal,
            last_idle: None,
            previous_clip: None,
            recent_actions: Vec::new(),
            speed: 1.0,
            breath_mode,
            breath_t: 0.0,
            breath_env: 0.0,
            seed,
        })
    }
    pub fn state(&self) -> &PlayerState {
        &self.state
    }
    pub fn set_idle(&mut self, value: bool) {
        self.idle = value;
    }
    pub fn set_rest(&mut self, mode: RestMode) {
        self.rest_mode = mode;
        self.idle_after = self.random_rest();
    }
    pub fn set_breath(&mut self, mode: BreathMode) {
        self.breath_mode = mode;
    }
    pub fn set_speed(&mut self, speed: f64) -> Result<(), PackError> {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(err("invalid speed"));
        }
        self.speed = speed;
        Ok(())
    }
    fn random_rest(&mut self) -> f64 {
        let (min, max) = match self.rest_mode {
            RestMode::None => (0.0, 0.0),
            RestMode::Short => (0.5, 1.5),
            RestMode::Normal => (2.0, 6.0),
        };
        min + self.random() * (max - min)
    }
    pub fn set_cuts(&mut self, value: bool) {
        self.cuts = value;
        if !value {
            self.clear_cut();
        }
    }
    fn random(&mut self) -> f64 {
        self.seed = self.seed.wrapping_add(0x6D2B79F5);
        let mut t = (self.seed ^ (self.seed >> 15)).wrapping_mul(self.seed | 1);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61)) ^ t;
        f64::from(t ^ (t >> 14)) / 4294967296.0
    }
    fn start(&mut self, name: String, entry: f64) {
        let clip = &self.clips[&name];
        if clip.kind != "bridge" && self.recent_actions.last() != Some(&clip.action) {
            self.recent_actions.push(clip.action.clone());
            if self.recent_actions.len() > 3 {
                self.recent_actions.remove(0);
            }
        }
        self.state.clip = Some(name);
        self.state.frame = entry;
        self.entry = entry;
        self.state.mode = Mode::Forward;
        self.state.next = None;
        self.rest_for = 0.0;
        self.clear_cut();
    }
    fn clear_cut(&mut self) {
        self.cut = None;
        self.state.cut_target = None;
    }
    fn set_cut(&mut self, cut: Cut) {
        self.state.cut_target = Some(cut.target.clone());
        self.cut = Some(cut);
    }
    fn at_rest(&mut self) {
        self.state.clip = None;
        self.state.mode = Mode::Rest;
        self.state.frame = 0.0;
        self.rest_for = 0.0;
        self.blend = None;
        self.state.blending = false;
        self.clear_cut();
        if let Some(next) = self.state.next.take() {
            self.start(next, 0.0);
        }
    }
    fn variants(&self, action: &str) -> Vec<String> {
        self.groups
            .get(action)
            .cloned()
            .unwrap_or_else(|| vec![action.into()])
    }
    fn bridge_ahead(&self) -> Option<Cut> {
        if !self.cuts || self.state.mode != Mode::Forward {
            return None;
        }
        let name = self.state.clip.as_ref()?;
        if self.clips.get(name)?.kind == "bridge" {
            return None;
        }
        let bridges = self
            .pack
            .manifest
            .get("bridges")?
            .get("bridges")?
            .as_array()?;
        bridges
            .iter()
            .filter(|b| b.get("from").and_then(|x| x.as_str()) == Some(name))
            .filter_map(|b| Some((b.get("exit")?.as_u64()? as f64, b.get("id")?.as_str()?)))
            .filter(|(at, _)| *at >= self.state.frame.ceil() && *at <= self.state.frame + 20.0)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(at, target)| Cut {
                at,
                target: target.into(),
                entry: 0.0,
                bridge: true,
            })
    }
    fn direct_cut(&self, variants: &[String]) -> Option<Cut> {
        if !self.cuts || self.state.mode != Mode::Forward {
            return None;
        }
        let source = self.state.clip.as_ref()?;
        let edges = self
            .pack
            .manifest
            .get("transitions")?
            .get("edges")?
            .get(source)?
            .as_array()?;
        edges
            .iter()
            .filter_map(|v| {
                let e = v.as_array()?;
                Some((
                    e.first()?.as_u64()? as f64,
                    e.get(1)?.as_str()?,
                    e.get(2)?.as_u64()? as f64,
                    e.get(3)?.as_f64()?,
                ))
            })
            .filter(|(at, target, _, _)| {
                variants.iter().any(|v| v == target)
                    && *at >= self.state.frame.ceil()
                    && *at <= self.state.frame + 18.0
            })
            .min_by(|a, b| a.3.total_cmp(&b.3))
            .map(|(at, target, entry, _)| Cut {
                at,
                target: target.into(),
                entry,
                bridge: false,
            })
    }
    fn request(&mut self, action: &str) -> Result<(), PackError> {
        let variants = self.variants(action);
        if variants
            .iter()
            .any(|name| !self.clips.contains_key(name) || self.clips[name].kind == "bridge")
        {
            return Err(err("unknown clip or action"));
        }
        let chosen = if self.groups.contains_key(action) {
            variants[(self.random() * variants.len() as f64) as usize].clone()
        } else {
            variants[0].clone()
        };
        let Some(current) = self.state.clip.clone() else {
            self.start(chosen, 0.0);
            return Ok(());
        };
        if self.state.mode == Mode::Forward
            && self.clips[&current].kind == "action"
            && (current == chosen || self.clips[&current].action == action)
        {
            return Ok(());
        }
        if let Some(cut) = self.direct_cut(&variants) {
            self.set_cut(cut);
            self.state.next = None;
            return Ok(());
        }
        if let Some(cut) = self.bridge_ahead() {
            self.set_cut(cut);
            self.state.next = Some(chosen);
            return Ok(());
        }
        if self.clips[&current].kind == "bridge" && self.state.mode == Mode::Forward {
            self.state.next = Some(chosen);
            return Ok(());
        }
        self.clear_cut();
        self.state.next = Some(chosen);
        let ahead = (self.clips[&current].frames - 1) as f64 - self.state.frame;
        self.state.mode = if self.state.frame <= ahead {
            Mode::Rewind
        } else {
            Mode::Finish
        };
        Ok(())
    }
    pub fn play(&mut self, action: &str) -> Result<(), PackError> {
        self.state.hold = false;
        self.request(action)
    }
    pub fn hold(&mut self, action: &str) -> Result<(), PackError> {
        self.state.hold = true;
        self.request(action)
    }
    pub fn release(&mut self) {
        self.state.hold = false;
    }
    pub fn stop(&mut self) {
        self.state.next = None;
        let Some(current) = self.state.clip.clone() else {
            return;
        };
        if self.clips[&current].kind == "bridge" {
            return;
        }
        if let Some(cut) = self.bridge_ahead() {
            self.set_cut(cut);
            return;
        }
        self.clear_cut();
        self.state.mode =
            if self.state.frame <= (self.clips[&current].frames - 1) as f64 - self.state.frame {
                Mode::Rewind
            } else {
                Mode::Finish
            };
    }
    fn take_cut(&mut self) {
        let Some(cut) = self.cut.take() else {
            return;
        };
        let from = self.state.clip.clone().expect("cut requires clip");
        let over = self.state.frame - cut.at;
        let next = self.state.next.take();
        self.previous_clip = Some(from.clone());
        if self.clips[&cut.target].kind == "idle" || self.clips[&cut.target].kind == "minimal-idle"
        {
            self.last_idle = Some(cut.target.clone());
        }
        let entry = if cut.bridge {
            over.max(0.0)
        } else {
            cut.entry + over
        };
        self.start(cut.target, entry);
        if cut.bridge {
            self.state.next = next;
        }
        self.blend = Some(Blend {
            from,
            at: cut.at,
            entry: cut.entry,
        });
        self.state.blending = true;
    }
    fn plan_idle_cut(&mut self) -> Option<Cut> {
        if !self.cuts
            || self.rest_mode != RestMode::None
            || self.state.next.is_some()
            || self.state.hold
            || !self.idle
        {
            return None;
        }
        let name = self.state.clip.as_ref()?;
        if !matches!(self.clips.get(name)?.kind.as_str(), "idle" | "minimal-idle") {
            return None;
        }
        let from = (self.entry + 36.0).max(self.state.frame.ceil());
        let edges = self
            .pack
            .manifest
            .get("transitions")?
            .get("edges")?
            .get(name)?
            .as_array()?;
        let mut candidates: Vec<(f64, String, f64, f64, u32)> = edges
            .iter()
            .filter_map(|v| {
                let edge = v.as_array()?;
                let at = edge.first()?.as_u64()? as f64;
                let target = edge.get(1)?.as_str()?;
                let entry = edge.get(2)?.as_u64()? as f64;
                let cost = edge.get(3)?.as_f64()?;
                let clip = self.clips.get(target)?;
                if at < from
                    || at > from + 24.0
                    || !matches!(clip.kind.as_str(), "idle" | "minimal-idle")
                    || self.previous_clip.as_deref() == Some(target)
                    || (clip.frames as f64 - entry) < 36.0 + self.blend_frames
                {
                    return None;
                }
                let weight = match clip.tier.as_str() {
                    "S" => 4,
                    "A" => 3,
                    _ => 1,
                };
                Some((at, target.to_owned(), entry, cost, weight))
            })
            .collect();
        if candidates
            .iter()
            .any(|(_, target, _, _, _)| !self.recent_actions.contains(&self.clips[target].action))
        {
            candidates.retain(|(_, target, _, _, _)| {
                !self.recent_actions.contains(&self.clips[target].action)
            });
        }
        let best = candidates.iter().map(|c| c.3).min_by(f64::total_cmp)?;
        candidates.retain(|c| c.3 <= best + 0.02);
        let total: u32 = candidates.iter().map(|c| c.4).sum();
        let mut roll = self.random() * f64::from(total);
        for (at, target, entry, _, weight) in candidates {
            roll -= f64::from(weight);
            if roll < 0.0 {
                return Some(Cut {
                    at,
                    target,
                    entry,
                    bridge: false,
                });
            }
        }
        None
    }
    fn choose_idle(&mut self) -> Option<String> {
        let mut choices: Vec<_> = self
            .clips
            .iter()
            .filter(|(_, c)| c.kind == "idle" || c.kind == "minimal-idle")
            .map(|(name, c)| {
                (
                    name.clone(),
                    match c.tier.as_str() {
                        "S" => 4,
                        "A" => 3,
                        _ => 1,
                    },
                )
            })
            .collect();
        if choices.len() > 1 {
            choices.retain(|(name, _)| Some(name) != self.last_idle.as_ref());
        }
        let total: u32 = choices.iter().map(|(_, w)| *w).sum();
        if total == 0 {
            return None;
        }
        let mut r = self.random() * f64::from(total);
        for (name, weight) in choices {
            r -= f64::from(weight);
            if r < 0.0 {
                return Some(name);
            }
        }
        None
    }
    pub fn tick(&mut self, seconds: f64) -> Result<(), PackError> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(err("invalid elapsed time"));
        }
        let dt = seconds.min(0.1);
        let step = dt * self.fps * self.speed;
        let breath_step = dt * self.speed;
        if !step.is_finite() || !breath_step.is_finite() {
            return Err(err("speed out of range"));
        }
        self.breath_t += breath_step;
        let target = if self.state.mode == Mode::Rest {
            1.0
        } else {
            0.0
        };
        self.breath_env += (target - self.breath_env) * (dt / 0.35).min(1.0);
        match self.state.mode {
            Mode::Rest => {
                self.rest_for += dt;
                if self.idle && self.rest_for > self.idle_after {
                    if let Some(name) = self.choose_idle() {
                        self.last_idle = Some(name.clone());
                        self.start(name, 0.0);
                        self.idle_after = self.random_rest();
                    }
                }
            }
            Mode::Forward => {
                let name = self
                    .state
                    .clip
                    .clone()
                    .ok_or_else(|| err("invalid player state"))?;
                self.state.frame += step;
                if self.state.hold
                    && self.cut.is_none()
                    && self.state.next.is_none()
                    && self.state.frame >= self.clips[&name].peak
                {
                    self.state.frame = self.clips[&name].peak;
                }
                if self.cut.is_none() {
                    if let Some(cut) = self.plan_idle_cut() {
                        self.set_cut(cut);
                    }
                }
                if self
                    .cut
                    .as_ref()
                    .is_some_and(|cut| self.state.frame >= cut.at)
                {
                    self.take_cut();
                } else if self.state.frame >= (self.clips[&name].frames - 1) as f64 {
                    self.at_rest();
                }
            }
            Mode::Finish => {
                self.state.frame += step * 3.0;
                if self
                    .state
                    .clip
                    .as_ref()
                    .is_some_and(|name| self.state.frame >= (self.clips[name].frames - 1) as f64)
                {
                    self.at_rest();
                }
            }
            Mode::Rewind => {
                self.state.frame -= step * 3.0;
                if self.state.frame <= 0.0 {
                    self.at_rest();
                }
            }
        }
        if self.blend.as_ref().is_some_and(|blend| {
            self.state.mode != Mode::Forward || self.state.frame - blend.entry >= self.blend_frames
        }) {
            self.blend = None;
            self.state.blending = false;
        }
        Ok(())
    }
    /// Rest geometry for the current clock/envelope; static when reduced motion is requested.
    pub fn rest_paths(&self, reduced_motion: bool) -> Result<FramePaths, PackError> {
        if reduced_motion || self.breath_mode == BreathMode::Off || self.breath_env <= 0.001 {
            return decode_frame(&self.pack.rest);
        }
        let breath = self
            .pack
            .manifest
            .get("breath")
            .ok_or_else(|| err("missing breath"))?;
        let mode = match self.breath_mode {
            BreathMode::Subtle => "subtle",
            BreathMode::Soft => "soft",
            BreathMode::Off => unreachable!(),
        };
        let config = &breath[mode];
        let value =
            |v: &serde_json::Value| v.as_f64().ok_or_else(|| err("invalid breath metadata"));
        breathing_frame(
            &self.pack.rest,
            self.breath_t,
            self.breath_env,
            BreathParams {
                swell: value(&config["swell"])?,
                sway: value(&config["sway"])?,
                breath_seconds: value(&breath["breath_seconds"])?,
                sway_seconds: value(&breath["sway_seconds"])?,
            },
        )
    }
    fn frame(&self, name: &str, index: f64) -> Result<&'a Frame, PackError> {
        let clip = &self.clips[name];
        let prefix = if clip.kind == "bridge" {
            "bridge"
        } else {
            "clip"
        };
        let doc = self
            .pack
            .documents
            .get(&format!("{prefix}:{name}"))
            .ok_or_else(|| err("missing clip"))?;
        Ok(&doc.frames[(index.round().max(0.0) as usize).min(doc.frames.len() - 1)])
    }
    pub fn sample(&self, reduced_motion: bool) -> Result<PlayerSample<'a>, PackError> {
        if reduced_motion || self.state.clip.is_none() {
            return Ok(PlayerSample::Rest(&self.pack.rest));
        }
        let name = self.state.clip.as_ref().unwrap();
        let frame = self.frame(name, self.state.frame)?;
        if let Some(blend) = &self.blend {
            let t = self.state.frame - blend.entry;
            if self.blend_frames > 0.0 && t < self.blend_frames {
                return Ok(PlayerSample::Crossfade {
                    from: self.frame(&blend.from, blend.at + t)?,
                    to: frame,
                    weight: (t / self.blend_frames).clamp(0.0, 1.0),
                });
            }
        }
        let rest_weight = match frame.metadata.get("m") {
            None => 0.0,
            Some(v) => v
                .as_f64()
                .filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
                .ok_or_else(|| err("invalid rest blend"))?,
        };
        Ok(PlayerSample::Frame {
            rest: &self.pack.rest,
            frame,
            rest_weight,
        })
    }
}
