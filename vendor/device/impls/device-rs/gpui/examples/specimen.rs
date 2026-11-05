//! Synthetic host of the reusable native view. No app effects or persistence.
use device_gpui::DeviceView;
use device_scene::{Action, Snapshot};
use gpui::{App, Application, Bounds, WindowBounds, WindowOptions, prelude::*, px, size};
use std::borrow::Cow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 3 {
        return Err("usage: specimen <fixture.json> [printed-font-file] [ui-font-file]".into());
    }
    let snapshot = Snapshot::parse(&std::fs::read_to_string(&args[0])?)?.into_inner();
    let fonts = args
        .iter()
        .skip(1)
        .map(std::fs::read)
        .collect::<Result<Vec<_>, _>>()?;
    Application::new().run(move |cx: &mut App| {
        cx.text_system()
            .add_fonts(fonts.into_iter().map(Cow::Owned).collect())
            .expect("fonts");
        let names = cx.text_system().all_font_names();
        let printed = if names.iter().any(|n| n == "Michroma") {
            "Michroma"
        } else {
            "Helvetica Neue"
        };
        let bounds = Bounds::centered(
            None,
            size(
                px(snapshot.viewport.width_dip as f32),
                px(snapshot.viewport.height_dip as f32),
            ),
            cx,
        );
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        cx.open_window(
            WindowOptions {
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("Native device — synthetic fixture".into()),
                    ..Default::default()
                }),
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            move |window, cx| {
                cx.new(|cx| {
                    cx.subscribe_self(|view: &mut DeviceView, action: &Action, cx| {
                        println!("{}", serde_json::to_string(action).expect("action JSON"));
                        view.edit_snapshot(
                            |snapshot| {
                                let (_, control, _) = snapshot.control(&action.control_id).unwrap();
                                if control.action == "navigate" {
                                    let page = action.value.string().unwrap_or("");
                                    let page = if page == "dictation" {
                                        "dictation.input"
                                    } else {
                                        page
                                    };
                                    if ["dictation.input", "dictation.shortcuts"].contains(&page) {
                                        snapshot.state.active_page = page.into();
                                    }
                                } else if control.kind == device_scene::Kind::Select {
                                    snapshot
                                        .state
                                        .controls
                                        .get_mut(&action.control_id)
                                        .unwrap()
                                        .value = action.value.clone();
                                }
                            },
                            cx,
                        );
                    })
                    .detach();
                    DeviceView::new(
                        snapshot,
                        printed.into(),
                        "Helvetica Neue".into(),
                        window,
                        cx,
                    )
                })
            },
        )
        .expect("proof window");
        cx.activate(true);
    });
    Ok(())
}
