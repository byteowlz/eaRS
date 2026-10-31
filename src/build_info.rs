//! Read-only compiled capabilities for release/install verification.
//! Does not inspect configuration, access devices or load model/runtime libraries.
pub fn json() -> serde_json::Value {
    serde_json::json!({
        "schema":"ears.build-info/v1",
        "version":env!("CARGO_PKG_VERSION"),
        "source_commit":option_env!("EARS_SOURCE_COMMIT"),
        "os":std::env::consts::OS,"arch":std::env::consts::ARCH,
        "client_vad":cfg!(feature="client-vad"),
        "moonshine":cfg!(feature="moonshine"),
        "parakeet_rs":cfg!(feature="parakeet-rs"),
        "microphone_started":false,"models_loaded":false
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn capability_report_has_no_runtime_side_effect_contract() {
        let value = super::json();
        assert_eq!(value["client_vad"], cfg!(feature = "client-vad"));
        assert_eq!(value["models_loaded"], false);
        assert_eq!(value["microphone_started"], false);
        assert!(value.get("config").is_none());
    }
}
