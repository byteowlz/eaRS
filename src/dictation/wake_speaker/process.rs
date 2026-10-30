//! Bounded subprocess I/O for speaker evidence. Called on the annotation worker,
//! never on the capture or insertion thread. No shell interpolation or reader threads.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};

const MAX_STDOUT: usize = 64 * 1024;
const MAX_STDERR: usize = 64 * 1024;

pub(super) fn run(command: &[String], wav: &Path, timeout: Duration) -> Result<Vec<u8>, String> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err("speaker check requires a blocking worker".into());
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .map_err(|_| "could not initialize speaker command I/O")?;
    runtime.block_on(execute(command, wav, timeout))
}

async fn read_bounded(mut reader: impl AsyncRead + Unpin, limit: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    (&mut reader)
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|_| "could not read speaker command output")?;
    if bytes.len() > limit {
        return Err("speaker command output exceeded limit".into());
    }
    Ok(bytes)
}

async fn execute(command: &[String], wav: &Path, timeout: Duration) -> Result<Vec<u8>, String> {
    let (program, args) = command.split_first().ok_or("empty speaker command")?;
    let mut child = tokio::process::Command::new(program)
        .args(args)
        .arg(wav)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| "could not start speaker command")?;
    let stdout = child.stdout.take().ok_or("missing speaker stdout")?;
    let stderr = child.stderr.take().ok_or("missing speaker stderr")?;
    let result = tokio::time::timeout(timeout, async {
        let (out, _, status) = tokio::try_join!(
            read_bounded(stdout, MAX_STDOUT),
            read_bounded(stderr, MAX_STDERR),
            async {
                child
                    .wait()
                    .await
                    .map_err(|_| "speaker command wait failed".to_string())
            },
        )?;
        if !status.success() {
            // stderr may contain private transcripts or paths; do not copy it into evidence/logs.
            return Err(format!("speaker command exited {status}"));
        }
        Ok(out)
    })
    .await
    .unwrap_or_else(|_| Err("speaker command timed out".into()));
    if result.is_err() {
        // Dropping the cancelled read futures closes both pipes, even if a wrapper
        // left a descendant holding them. Kill/reap our direct child; never wait for EOF.
        let _ = child.kill().await;
    }
    result
}
