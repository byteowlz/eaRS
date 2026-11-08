//! List the languages and models available in the Moonshine STT catalog.
//!
//! [`moonshine_rs::get_stt_catalog`] returns the official catalog as JSON:
//! languages, their English names, and the model architectures published for
//! each (with default markers and CDN URLs).
//!
//! Run:
//!
//! ```bash
//! cargo run --example browse_catalog -p moonshine-rs
//! ```

use moonshine_rs::get_stt_catalog;

fn arch_name(arch: i64) -> &'static str {
    match arch {
        0 => "tiny",
        1 => "base",
        2 => "tiny-streaming",
        3 => "base-streaming",
        4 => "small-streaming",
        5 => "medium-streaming",
        _ => "unknown",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog_json = get_stt_catalog()?;
    let catalog: serde_json::Value = serde_json::from_str(&catalog_json)?;

    let languages = catalog["languages"]
        .as_array()
        .ok_or("catalog missing `languages` array")?;

    for lang in languages {
        let code = lang["code"].as_str().unwrap_or("??");
        let name = lang["english_name"].as_str().unwrap_or("Unknown");
        println!("{name} ({code})");

        if let Some(models) = lang["models"].as_array() {
            for model in models {
                let arch = model["model_arch"].as_i64().unwrap_or(-1);
                let default = model["is_default"].as_bool().unwrap_or(false);
                let marker = if default { " [default]" } else { "" };
                println!("    - {}{}", arch_name(arch), marker);
            }
        }
    }

    Ok(())
}
