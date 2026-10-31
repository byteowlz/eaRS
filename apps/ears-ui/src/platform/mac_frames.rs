//! Display-paced, main-run-loop callbacks for passive cursor-preview movement.
//! Independent of GPUI's non-activating popup redraw/ASR-service timer. No input
//! monitors, permission requests, cursor mutation or background-thread UI access.
use objc2::rc::Retained;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::NSScreen;
use objc2_foundation::{NSObject, NSObjectProtocol, NSRunLoop, NSRunLoopCommonModes};
use objc2_quartz_core::{CADisplayLink, CAFrameRateRange};

define_class!(
    // SAFETY: NSObject has no extra subclassing requirements. Callbacks and
    // destruction stay on the main thread; the target has no custom Drop.
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[ivars = Box<dyn Fn()>]
    struct EarsPreviewFrameTarget;
    unsafe impl NSObjectProtocol for EarsPreviewFrameTarget {}
    impl EarsPreviewFrameTarget {
        #[unsafe(method(frame:))]
        fn frame(&self, _link: &CADisplayLink) { (self.ivars())(); }
    }
);

pub struct PreviewFrames {
    link: Retained<CADisplayLink>,
    _target: Retained<EarsPreviewFrameTarget>,
}
impl Drop for PreviewFrames {
    fn drop(&mut self) {
        self.link.invalidate();
    }
}

pub(super) fn start(screen: &NSScreen, callback: impl Fn() + 'static) -> Option<PreviewFrames> {
    // NSScreen display links are public on macOS14+. Older standalone builds
    // keep the existing bounded service-tick fallback rather than crashing.
    if !screen.respondsToSelector(sel!(displayLinkWithTarget:selector:)) {
        return None;
    }
    let target = EarsPreviewFrameTarget::alloc(super::mtm()).set_ivars(Box::new(callback));
    // SAFETY: NSObject initialization; the selector matches frame(CADisplayLink).
    let target: Retained<EarsPreviewFrameTarget> = unsafe { msg_send![super(target), init] };
    let link = unsafe { screen.displayLinkWithTarget_selector(&target, sel!(frame:)) };
    let fps = screen.maximumFramesPerSecond().clamp(30, 240) as f32;
    link.setPreferredFrameRateRange(CAFrameRateRange {
        minimum: fps,
        maximum: fps,
        preferred: fps,
    });
    // SAFETY: main-thread-owned link and main run loop; common modes also keep
    // following responsive during native menu/scroll tracking.
    unsafe {
        link.addToRunLoop_forMode(&NSRunLoop::mainRunLoop(), NSRunLoopCommonModes);
    }
    Some(PreviewFrames {
        link,
        _target: target,
    })
}
