use tauri::Manager;

#[derive(serde::Serialize)]
struct SubsStyle {
    border_radius: u32,
    border_thickness: u32,
}

#[tauri::command]
fn get_subs_style() -> Result<SubsStyle, String> {
    let config = ears::config::AppConfig::load()
        .map_err(|e| format!("Failed to load config: {}", e))?;
    
    Ok(SubsStyle {
        border_radius: config.subs.border_radius,
        border_thickness: config.subs.border_thickness,
    })
}

#[tauri::command]
fn toggle_always_on_top(window: tauri::Window, on_top: bool) -> Result<(), String> {
    window.set_always_on_top(on_top).map_err(|e| e.to_string())
}

#[tauri::command]
fn set_window_position(window: tauri::Window, x: i32, y: i32) -> Result<(), String> {
    window.set_position(tauri::Position::Physical(tauri::PhysicalPosition { x, y }))
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn start_drag(window: tauri::Window) -> Result<(), String> {
    window.start_dragging().map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      let config = ears::config::AppConfig::load()
        .map_err(|e| format!("Failed to load config: {}", e))?;
      
      let window = app.get_webview_window("main")
        .ok_or("Failed to get main window")?;

      #[cfg(target_os = "linux")]
      {
        use webkit2gtk::WebViewExt;
        window.with_webview(|webview| {
          let webview = webview.inner().clone();
          webview.connect_permission_request(|_, request| {
            use webkit2gtk::PermissionRequestExt;
            request.allow();
            true
          });
        }).map_err(|e| format!("Failed to set permission handler: {}", e))?;
      }

      let monitor = window.current_monitor()
        .map_err(|e| format!("Failed to get monitor: {}", e))?
        .ok_or("No monitor found")?;
      
      let monitor_size = monitor.size();
      let screen_width = monitor_size.width as f64;
      let screen_height = monitor_size.height as f64;

      let width = (screen_width * config.subs.width as f64 / 100.0) as u32;
      let height = (screen_height * config.subs.heigth as f64 / 100.0) as u32;

      let x = ((screen_width * config.subs.x_position as f64 / 100.0) - (width as f64 / 2.0)) as i32;
      let y = ((screen_height * config.subs.y_position as f64 / 100.0) - (height as f64 / 2.0)) as i32;

      window.set_position(tauri::Position::Physical(tauri::PhysicalPosition { x, y }))
        .map_err(|e| format!("Failed to set position: {}", e))?;
      
      window.set_size(tauri::Size::Physical(tauri::PhysicalSize { width, height }))
        .map_err(|e| format!("Failed to set size: {}", e))?;

      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      toggle_always_on_top,
      set_window_position,
      start_drag,
      get_subs_style
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
