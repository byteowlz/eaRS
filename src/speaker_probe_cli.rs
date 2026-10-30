//! CLI presentation of the shared, read-only voqx discovery operation.

use anyhow::{anyhow, Result};
use ears::dictation::voqx_discovery::{discover, DiscoveryStatus};
use std::path::Path;

pub async fn run(executable: Option<&Path>, json: bool) -> Result<()> {
    let result = discover(executable).await;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("voqx: {:?}", result.status);
        if let Some(path) = &result.executable {
            println!("executable: {}", path.display());
        }
        if let Some(detail) = &result.detail {
            println!("{detail}");
        }
        println!("Models and recognition profiles have not been checked.");
    }
    if result.status == DiscoveryStatus::CompatibleExecutable {
        Ok(())
    } else {
        Err(anyhow!("no compatible voqx executable selected"))
    }
}
