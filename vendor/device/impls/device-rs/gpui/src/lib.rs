//! GPUI 0.2.2 adapter: bounded asynchronous rendering and atomic image/projection installation.
mod view;
mod worker;
use device_render_wgpu::RenderFrame;
use device_scene::ValidatedSnapshot;
use gpui::{RenderImage, Window};
use std::sync::Arc;
pub use view::{DeviceView, Hover, Icon};
pub use worker::{Completed, RenderWorker};

/// Image and semantic projection are installed together, never from separate generations.
pub struct PresentedFrame {
    pub frame: RenderFrame,
    pub snapshot: Arc<ValidatedSnapshot>,
    pub image: Arc<RenderImage>,
}
#[derive(Default)]
pub struct ImageSlot {
    current: Option<PresentedFrame>,
    pub error: Option<String>,
    expected_generation: u64,
}
impl ImageSlot {
    pub fn expect(&mut self, generation: u64) {
        self.expected_generation = generation;
    }
    pub fn current(&self) -> Option<&PresentedFrame> {
        self.current.as_ref()
    }
    /// False means stale/failed result. Keep old pixels AND old projection on error.
    pub fn install(&mut self, completed: Completed, window: &mut Window) -> bool {
        if completed.generation != self.expected_generation {
            return false;
        }
        let frame = match completed.frame {
            Ok(frame) => frame,
            Err(error) => {
                self.error = Some(error);
                return false;
            }
        };
        let snapshot = completed.snapshot.snapshot();
        if frame.request_generation != completed.generation
            || frame.scene_id != snapshot.scene.id
            || frame.scene_revision != snapshot.scene.revision
            || frame.state_revision != snapshot.state.revision
            || frame.viewport != snapshot.viewport
            || frame.projection.viewport != frame.viewport
            || frame.projection.extent.width_px != frame.width_px
            || frame.projection.extent.height_px != frame.height_px
            || frame.stride_bytes != frame.width_px.saturating_mul(4)
        {
            self.error = Some("incoherent frame metadata; previous frame retained".into());
            return false;
        }
        let Some(buffer) =
            image::RgbaImage::from_raw(frame.width_px, frame.height_px, frame.pixels.clone())
        else {
            self.error = Some("invalid packed frame".into());
            return false;
        };
        // Despite image::Frame's type name, GPUI RenderImage requires the agreed BGRA bytes.
        let image = Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]));
        let previous = self.current.replace(PresentedFrame {
            frame,
            snapshot: completed.snapshot,
            image,
        });
        self.error = None;
        if let Some(previous) = previous
            && let Err(error) = window.drop_image(previous.image)
        {
            self.error = Some(format!("image retirement: {error}"));
        }
        true
    }
    pub fn clear(&mut self, window: &mut Window) {
        if let Some(previous) = self.current.take() {
            let _ = window.drop_image(previous.image);
        }
    }
}
