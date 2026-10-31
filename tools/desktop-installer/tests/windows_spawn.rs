#![cfg(windows)]

#[path = "../src/windows_spawn.rs"]
mod windows_spawn;

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Read;
use std::mem::size_of;
use std::os::windows::ffi::OsStringExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::ptr::null_mut;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{
    GetHandleInformation, SetHandleInformation, HANDLE_FLAG_INHERIT, WAIT_TIMEOUT,
};
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
use windows_sys::Win32::System::Console::{
    GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
};
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetProcessHandleCount, OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
};

struct ReleaseOnDrop(PathBuf);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        let _ = fs::write(&self.0, b"release synthetic helper on all test exits");
    }
}

struct MarkerEnvironment(Option<OsString>);
impl Drop for MarkerEnvironment {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => std::env::set_var("EARS_SYNTHETIC_HELPER_START_MARKER", value),
            None => std::env::remove_var("EARS_SYNTHETIC_HELPER_START_MARKER"),
        }
    }
}

fn wait_for(path: &Path) {
    let start = Instant::now();
    while !path.exists() && start.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(path.exists(), "helper did not create {}", path.display());
}

fn std_flags() -> Vec<Option<u32>> {
    [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE]
        .into_iter()
        .map(|id| {
            let mut flags = 0;
            // SAFETY: read-only queries of this test process's standard handles.
            (unsafe { GetHandleInformation(GetStdHandle(id), &mut flags) } != 0).then_some(flags)
        })
        .collect()
}

fn unrelated_inheritable_pipe() -> (OwnedHandle, OwnedHandle) {
    let attributes = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: null_mut(),
        bInheritHandle: 1,
    };
    let (mut read, mut write) = (null_mut(), null_mut());
    // SAFETY: writable output handles and live security attributes.
    assert_ne!(
        unsafe { CreatePipe(&mut read, &mut write, &attributes, 0) },
        0
    );
    // SAFETY: successful CreatePipe transfers two new owned handles.
    let (read, write) = unsafe {
        (
            OwnedHandle::from_raw_handle(read),
            OwnedHandle::from_raw_handle(write),
        )
    };
    // Only this unrelated test pipe's read end is changed, never parent stdio.
    assert_ne!(
        unsafe { SetHandleInformation(read.as_raw_handle(), HANDLE_FLAG_INHERIT, 0) },
        0
    );
    let mut flags = 0;
    assert_ne!(
        unsafe { GetHandleInformation(write.as_raw_handle(), &mut flags) },
        0
    );
    assert_ne!(flags & HANDLE_FLAG_INHERIT, 0);
    (read, write)
}

fn handle_count() -> u32 {
    let mut count = 0;
    // SAFETY: read-only query of this process, not a token or privilege API.
    assert_ne!(
        unsafe { GetProcessHandleCount(GetCurrentProcess(), &mut count) },
        0
    );
    count
}

#[test]
fn allowlist_excludes_unrelated_pipe_preserves_utf16_and_cleans_up_failure_handles() {
    let temp = tempfile::Builder::new()
        .prefix("ears spawn ä ")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let helper = root.join("synthetic helper.exe");
    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let compile = Command::new(compiler)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/launch-helper.rs"))
        .args(["--edition=2021", "-C", "debuginfo=0"])
        .arg("-o")
        .arg(&helper)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let cwd = root.join("working directory");
    fs::create_dir(&cwd).unwrap();
    let output = root.join("probe.record");
    let marker = root.join("existing-environment-marker");
    let _environment = MarkerEnvironment(std::env::var_os("EARS_SYNTHETIC_HELPER_START_MARKER"));
    std::env::set_var("EARS_SYNTHETIC_HELPER_START_MARKER", &marker);
    let literal: Vec<OsString> = vec![
        "".into(),
        "space separated".into(),
        "\"quoted\"".into(),
        r"trailing\".into(),
        r#"slashes\\\"quote"#.into(),
        "\t\r\n".into(),
        "Unicode-ä-\u{10437}".into(),
        OsString::from_wide(&[b'x' as u16, 0xd800, b'y' as u16]),
    ];
    let args: Vec<_> = std::iter::once(output.clone().into_os_string())
        .chain(literal.iter().cloned())
        .collect();
    let before_stdio = std_flags();
    // Create this only after rustc exits, so the compiler cannot inherit it.
    let (read, write) = unrelated_inheritable_pipe();
    let pid = windows_spawn::spawn(&helper, &cwd, &args).unwrap();
    let _release = ReleaseOnDrop(output.with_extension("release"));
    assert_eq!(
        std_flags(),
        before_stdio,
        "spawn mutated parent standard-handle flags"
    );
    // The parent's last writer is dropped; EOF must not wait for helper exit.
    drop(write);
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut file = File::from(read);
        let mut bytes = [0u8; 1];
        let result = file.read(&mut bytes);
        let eof = matches!(result, Ok(0))
            || result
                .as_ref()
                .err()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::BrokenPipe);
        tx.send(eof).unwrap();
    });
    let eof = rx.recv_timeout(Duration::from_secs(5));
    wait_for(&output);
    assert!(marker.exists(), "existing environment was not inherited");
    let record = fs::read_to_string(&output).unwrap();
    let lines: Vec<_> = record.lines().collect();
    assert_eq!(lines[0].parse::<u32>().unwrap(), pid);
    assert_eq!(
        serde_json::from_str::<Vec<Vec<u8>>>(lines[1]).unwrap(),
        literal
            .iter()
            .map(|arg| arg.as_encoded_bytes().to_vec())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        Path::new(lines[2]).canonicalize().unwrap(),
        cwd.canonicalize().unwrap()
    );
    // SAFETY: synchronize-only access to our own known synthetic child PID.
    let raw = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) };
    assert!(!raw.is_null());
    let child = unsafe { OwnedHandle::from_raw_handle(raw) };
    let still_alive = unsafe { WaitForSingleObject(child.as_raw_handle(), 0) } == WAIT_TIMEOUT;
    let done_before_release = output.with_extension("done").exists();
    fs::write(
        output.with_extension("release"),
        b"normal synthetic release",
    )
    .unwrap();
    reader.join().unwrap();
    wait_for(&output.with_extension("done"));
    assert!(
        matches!(eof, Ok(true)),
        "unrelated inheritable writer leaked into child"
    );
    assert!(
        still_alive && !done_before_release,
        "EOF was not observed before helper exit"
    );
    drop(child);
    // Windows lazily initializes its missing-image failure path: the first such
    // failure grows the process handle table even with plain std::Command. Prime
    // it with the standard library, not our code, before asserting exact zero
    // growth across repeated failures in the allowlisted implementation.
    assert!(Command::new(root.join("missing.exe"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_err());
    let before = handle_count();
    for _ in 0..32 {
        assert!(windows_spawn::spawn(&root.join("missing.exe"), &cwd, &[]).is_err());
    }
    assert_eq!(handle_count(), before, "failed spawn leaked handles");
    let nul_arg = OsString::from_wide(&[b'a' as u16, 0, b'b' as u16]);
    assert_eq!(
        windows_spawn::spawn(&helper, &cwd, &[nul_arg])
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
    assert_eq!(
        windows_spawn::spawn(&helper, &cwd, &["x".repeat(32767).into()])
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
}
