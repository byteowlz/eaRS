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

            #[cfg(target_os = "linux")]
            {
                use tauri::Manager;
                use webkit2gtk::WebViewExt;
                
                let window = app.get_webview_window("main").unwrap();
                window
                    .with_webview(|webview| {
                        let webview = webview.inner().clone();
                        webview.connect_permission_request(|_, request| {
                            use webkit2gtk::PermissionRequestExt;
                            request.allow();
                            true
                        });
                    })
                    .map_err(|e| format!("Failed to set permission handler: {}", e))?;
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
