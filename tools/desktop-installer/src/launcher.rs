use crate::payload::Manifest;
use anyhow::{ensure, Context, Result};
use clap::ValueEnum;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
#[cfg(not(windows))]
use std::process::{Command, Stdio};

#[cfg(windows)]
#[path = "windows_spawn.rs"]
mod windows_spawn;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Program {
    #[value(name = "ears-ui")]
    Ui,
    #[value(name = "ears")]
    Ears,
    #[value(name = "ears-dictation")]
    Dictation,
}

impl Program {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ui => "ears-ui",
            Self::Ears => "ears",
            Self::Dictation => "ears-dictation",
        }
    }
}

// Pure selection only. Callers must first authenticate and verify the entire
// active payload. Never derive an executable path from a free-form CLI string.
pub fn select(manifest: &Manifest, program: Program) -> Result<PathBuf> {
    ensure!(
        matches!(manifest.os.as_str(), "linux" | "windows" | "macos"),
        "unsupported launch OS"
    );
    let suffix = if manifest.os == "windows" { ".exe" } else { "" };
    let relative = format!("bin/{}{suffix}", program.name());
    let mut candidates = manifest.files.iter().filter(|entry| entry.path == relative);
    let entry = candidates
        .next()
        .context("program is not listed in active manifest")?;
    ensure!(candidates.next().is_none(), "duplicate launch program");
    ensure!(entry.executable, "program is not declared executable");
    Ok(PathBuf::from(relative))
}

// Every desktop launch is detached, including the CLI-named payloads. No shell,
// pipes, terminal inheritance, wait loop, or long-lived launcher process.
pub fn spawn(
    root: &Path,
    manifest: &Manifest,
    program: Program,
    args: &[OsString],
) -> Result<(PathBuf, u32)> {
    ensure!(root.is_absolute(), "launch root must be absolute");
    let path = root.join(select(manifest, program)?);
    #[cfg(windows)]
    let pid =
        windows_spawn::spawn(&path, root, args).context("launching verified active program")?;
    #[cfg(not(windows))]
    let pid = {
        let mut command = Command::new(&path);
        command
            .args(args)
            .current_dir(root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: setsid is the only operation in this post-fork closure and is
            // async-signal-safe; no allocation, locks or inherited Rust state is used.
            unsafe {
                command.pre_exec(|| {
                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        command
            .spawn()
            .context("launching verified active program")?
            .id()
    };
    Ok((path, pid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::{PayloadFile, SCHEMA};

    fn manifest(os: &str) -> Manifest {
        Manifest {
            schema: SCHEMA.into(),
            os: os.into(),
            arch: "synthetic".into(),
            version: "fixture".into(),
            source_commit: "synthetic".into(),
            client_vad: true,
            files: Vec::new(),
        }
    }

    fn file(path: &str, executable: bool) -> PayloadFile {
        PayloadFile {
            path: path.into(),
            sha256: "0".repeat(64),
            executable,
        }
    }

    #[test]
    fn selects_only_exact_platform_bin_paths() {
        for os in ["linux", "windows", "macos"] {
            for program in [Program::Ui, Program::Ears, Program::Dictation] {
                let suffix = if os == "windows" { ".exe" } else { "" };
                let expected = format!("bin/{}{suffix}", program.name());
                let mut m = manifest(os);
                m.files.push(file(&expected, true));
                assert_eq!(select(&m, program).unwrap(), PathBuf::from(expected));
            }
        }
    }

    #[test]
    fn refuses_missing_unlisted_nonexecutable_and_alias_paths() {
        for os in ["linux", "windows"] {
            let correct = if os == "windows" {
                "bin/ears-ui.exe"
            } else {
                "bin/ears-ui"
            };
            for path in [
                "ears-ui",
                "bin/other",
                "bin/EARS-UI",
                "../bin/ears-ui",
                "/bin/ears-ui",
                "bin\\ears-ui",
                "bin/ears-ui.cmd",
                if os == "windows" {
                    "bin/ears-ui"
                } else {
                    "bin/ears-ui.exe"
                },
            ] {
                let mut m = manifest(os);
                m.files.push(file(path, true));
                assert!(select(&m, Program::Ui).is_err(), "{os}: {path}");
            }
            let mut m = manifest(os);
            assert!(select(&m, Program::Ui).is_err());
            m.files.push(file(correct, false));
            assert!(select(&m, Program::Ui).is_err());
            m.files[0].executable = true;
            m.files.push(file(correct, true));
            assert!(select(&m, Program::Ui).is_err());
        }
        assert!(select(&manifest("unknown"), Program::Ui).is_err());
    }

    #[test]
    fn whitelist_has_no_arbitrary_program_or_path_escape() {
        for name in [
            "sh",
            "cmd",
            "powershell",
            "../ears",
            "bin/ears",
            "/bin/ears",
            "ears.exe",
            "ears-ui.exe",
            "EARS",
            "ears-ui;other",
        ] {
            assert!(Program::from_str(name, false).is_err(), "{name}");
        }
        for name in ["ears-ui", "ears", "ears-dictation"] {
            assert!(Program::from_str(name, false).is_ok());
        }
    }
}
