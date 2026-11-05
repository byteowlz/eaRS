//! Portable device-scene v1 data, semantic validation and projection. No I/O.
mod model;
mod projection;
mod validate;
pub use model::*;
pub use projection::*;
pub use validate::{ValidationError, page_matches, validate_value};

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("invalid snapshot JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(#[from] ValidationError),
}

/// An owned validated snapshot; validation cannot be invalidated through a mutable borrow.
#[derive(Clone, Debug)]
pub struct ValidatedSnapshot(Snapshot);
impl ValidatedSnapshot {
    pub fn new(snapshot: Snapshot) -> Result<Self, ValidationError> {
        snapshot.validate()?;
        Ok(Self(snapshot))
    }
    pub fn snapshot(&self) -> &Snapshot {
        &self.0
    }
    pub fn into_inner(self) -> Snapshot {
        self.0
    }
}
impl Snapshot {
    pub fn parse(json: &str) -> Result<ValidatedSnapshot, ParseError> {
        Ok(ValidatedSnapshot::new(serde_json::from_str(json)?)?)
    }
    pub fn control(&self, id: &str) -> Option<(&Device, &Control, &ControlState)> {
        self.scene.devices.iter().find_map(|d| {
            d.controls
                .iter()
                .find(|c| c.id == id)
                .and_then(|c| self.state.controls.get(id).map(|s| (d, c, s)))
        })
    }
    pub fn control_visible(&self, device: &Device, control: &Control) -> bool {
        self.state
            .controls
            .get(&control.id)
            .is_some_and(|s| s.visible)
            && page_matches(&control.page, &self.state.active_page)
            && device
                .panels
                .iter()
                .find(|p| p.id == control.module_id)
                .is_some_and(|p| page_matches(&p.page, &self.state.active_page))
    }
    pub fn focus_order(&self) -> Vec<&Control> {
        let mut controls: Vec<_> = self
            .scene
            .devices
            .iter()
            .flat_map(|d| {
                d.controls.iter().filter(move |c| {
                    self.control_visible(d, c)
                        && c.tab_index >= 0
                        && self.state.controls[&c.id].enabled
                })
            })
            .collect();
        controls.sort_by(|a, b| a.tab_index.cmp(&b.tab_index).then_with(|| a.id.cmp(&b.id)));
        controls
    }
}
