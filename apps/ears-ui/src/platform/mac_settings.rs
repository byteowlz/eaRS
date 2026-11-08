//! Borderless native controls with etched glyphs on the receiver frame. Native
//! hit targets, window actions and accessibility survive without traffic lights.
#[path = "mac_settings_icons.rs"]
mod icons;
use gpui::Window;
use icons::Glyph;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send, rc::Retained, sel};
use objc2_app_kit::{NSAccessibility, NSButton, NSButtonType, NSCellImagePosition, NSView};
use objc2_foundation::{NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize, NSString};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::rc::Rc;

define_class!(
    // SAFETY: NSObject subclass without ivars; creation/action/destruction all
    // stay on the main thread. The sender is our borderless native NSButton.
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[ivars = ()]
    struct EarsSettingsCloseTarget;
    unsafe impl NSObjectProtocol for EarsSettingsCloseTarget {}
    impl EarsSettingsCloseTarget {
        #[unsafe(method(closeSettings:))]
        fn close_settings(&self, sender: &NSButton) {
            // performClose: refuses a borderless NSWindow. close calls GPUI's
            // override, which removes the window and releases its Settings view.
            if let Some(window) = sender.window() {
                window.close();
            }
        }
    }
);

struct Buttons {
    container: Retained<NSView>,
    controls: [Retained<NSButton>; 3],
    _close_target: Retained<EarsSettingsCloseTarget>,
}
impl Drop for Buttons {
    fn drop(&mut self) {
        self.container.removeFromSuperview();
    }
}

#[derive(Clone)]
pub struct SettingsButtons(Rc<Buttons>);

fn content_view(window: &Window) -> Option<&NSView> {
    let handle = HasWindowHandle::window_handle(window).ok()?;
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return None;
    };
    // SAFETY: GPUI owns this NSView for the borrowed window's lifetime; every
    // caller runs on its main-thread view creation/prepaint path.
    Some(unsafe { appkit.ns_view.cast::<NSView>().as_ref() })
}

fn frame(bounds: NSRect, flipped: bool, anchor: [f64; 4], size: NSSize) -> NSRect {
    let top = (anchor[1] + anchor[3] - size.height) / 2.0;
    let y = bounds.origin.y
        + if flipped {
            top
        } else {
            bounds.size.height - top - size.height
        };
    NSRect::new(NSPoint::new(bounds.origin.x + anchor[0], y), size)
}

impl SettingsButtons {
    pub fn new(window: &Window) -> Option<Self> {
        let mtm = MainThreadMarker::new()?;
        let view = content_view(window)?;
        let ns = view.window()?;
        // A sibling of the Metal-backed content view, not a subview whose
        // backing layer would conflict with GPUI's layer-hosting render path.
        // SAFETY: the live window's parent is accessed on the main thread.
        let host = unsafe { view.superview() }?;
        let controls = [NSButton::new(mtm), NSButton::new(mtm), NSButton::new(mtm)];
        let container = NSView::new(mtm);
        container.setHidden(true);
        for (button, glyph) in controls.iter().zip(Glyph::ALL) {
            // Pointer and accessibility controls only: a button that becomes first
            // responder takes keyDown from GPUI, and no Settings field takes text.
            button.setRefusesFirstResponder(true);
            button.setBordered(false);
            button.setButtonType(NSButtonType::MomentaryChange);
            button.setTitle(&NSString::from_str(""));
            button.setImagePosition(NSCellImagePosition::ImageOnly);
            button.setImage(Some(&glyph.image()));
            let label = NSString::from_str(glyph.label());
            button.setToolTip(Some(&label));
            button.setAccessibilityLabel(Some(&label));
            button.setAccessibilitySubrole(Some(glyph.subrole()));
            container.addSubview(button);
        }
        host.addSubview(&container);
        // Adding views can move first responder; keyboard input belongs to GPUI.
        ns.makeFirstResponder(Some(view));
        for (button, action) in
            controls
                .iter()
                .zip([sel!(performClose:), sel!(miniaturize:), sel!(zoom:)])
        {
            // SAFETY: public NSWindow actions all take one object sender. Set
            // targets after adding the controls to the window. NSControl's target
            // is unretained and the window outlives its content/controls.
            unsafe {
                button.setTarget(Some(&ns));
                button.setAction(Some(action));
            }
            button.setEnabled(true);
            button.setFrameSize(NSSize::new(16.0, 16.0));
        }
        // SAFETY: NSObject initialization on the main thread. The target is
        // retained with the controls, and holds no window reference/cycle.
        let close_target: Retained<EarsSettingsCloseTarget> = unsafe {
            msg_send![
                super(EarsSettingsCloseTarget::alloc(mtm).set_ivars(())),
                init
            ]
        };
        unsafe {
            controls[0].setTarget(Some(&close_target));
            controls[0].setAction(Some(sel!(closeSettings:)));
        }
        // Preserve the window's close/minimise/zoom accessibility relationships
        // as well as each etched button's native role, label and AXPress action.
        // SAFETY: AppKit owns the supplied accessibility elements; the header
        // and the retained controls live for the Settings view's lifetime.
        unsafe {
            ns.setAccessibilityCloseButton(Some(&controls[0]));
            ns.setAccessibilityMinimizeButton(Some(&controls[1]));
            ns.setAccessibilityZoomButton(Some(&controls[2]));
        }
        Some(Self(Rc::new(Buttons {
            container,
            controls,
            _close_target: close_target,
        })))
    }

    pub fn position(&self, window: &Window, anchor: Option<[f64; 4]>) {
        let Some(view) = content_view(window) else {
            return;
        };
        let Some(anchor) = anchor else {
            self.0.container.setHidden(true);
            return;
        };
        // SAFETY: the container belongs to this live window and is accessed
        // only on its main thread, after layout/renderer projection updates.
        let Some(host) = (unsafe { self.0.container.superview() }) else {
            return;
        };
        let button_size = NSSize::new(16.0, 16.0);
        let size = NSSize::new(button_size.width + 40.0, button_size.height);
        let content_frame = frame(view.bounds(), view.isFlipped(), anchor, size);
        let target = host.convertRect_fromView(content_frame, Some(view));
        if self.0.container.frame() != target {
            self.0.container.setFrame(target);
        }
        for (index, button) in self.0.controls.iter().enumerate() {
            button.setFrame(NSRect::new(
                NSPoint::new(index as f64 * 20.0, 0.0),
                button_size,
            ));
        }
        if self.0.container.isHidden() {
            self.0.container.setHidden(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_coordinates_preserve_size_and_top_left_anchor() {
        let size = NSSize::new(54.0, 14.0);
        let bounds = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1240.0, 860.0));
        let anchor = [82.0, 70.0, 190.0, 90.0];
        for flipped in [false, true] {
            let r = frame(bounds, flipped, anchor, size);
            assert_eq!(r.origin.x, 82.0);
            assert_eq!(r.size, size);
            let top = if flipped {
                r.origin.y
            } else {
                860.0 - r.origin.y - r.size.height
            };
            assert_eq!(top, 73.0);
            assert!(r.origin.x + r.size.width <= anchor[2]);
        }
    }
}
