// Synthetic native test helper, never an eaRS executable. Only explicit launch
// may execute this fixture. All filesystem writes are under the test temp root.
use std::io::{Read, Write};
use std::time::{Duration, Instant};

#[cfg(unix)]
fn session_id() -> i32 {
    extern "C" {
        fn getsid(pid: i32) -> i32;
    }
    // A read-only process query; the helper uses no external crate or subprocess.
    unsafe { getsid(0) }
}

#[cfg(not(unix))]
fn session_id() -> i32 {
    -1
}

fn main() {
    if let Some(path) = std::env::var_os("EARS_SYNTHETIC_HELPER_START_MARKER") {
        std::fs::write(path, b"helper started").unwrap();
    }
    let args: Vec<_> = std::env::args_os().collect();
    let output = std::path::PathBuf::from(&args[1]);
    let mut stdin = Vec::new();
    std::io::stdin().read_to_end(&mut stdin).unwrap();
    assert!(stdin.is_empty());
    std::io::stdout()
        .write_all(b"helper stdout must not enter launcher JSON\n")
        .unwrap();
    std::io::stderr()
        .write_all(b"helper stderr must not enter launcher output\n")
        .unwrap();
    let literal: Vec<_> = args[2..]
        .iter()
        .map(|arg| arg.as_encoded_bytes().to_vec())
        .collect();
    let record = format!(
        "{}\n{:?}\n{}\n{}\n",
        std::process::id(),
        literal,
        std::env::current_dir().unwrap().display(),
        session_id()
    );
    std::fs::write(&output, record).unwrap();
    let start = Instant::now();
    while !output.with_extension("release").exists() && start.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::fs::write(output.with_extension("done"), b"done").unwrap();
}
