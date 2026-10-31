use super::*;
use std::ffi::OsString;
use std::time::{Duration, Instant};

fn launch(f: &Fixture, program: &str) -> Output {
    f.command()
        .arg("launch")
        .arg(program)
        .arg("--prefix")
        .arg(&f.prefix)
        .output()
        .unwrap()
}

fn ui_path() -> &'static str {
    if cfg!(windows) {
        "bin/ears-ui.exe"
    } else {
        "bin/ears-ui"
    }
}

#[test]
fn launch_parser_requires_whitelist_and_separator_before_forwarded_arguments() {
    let f = Fixture::new();
    for program in [
        "sh",
        "cmd",
        "../ears",
        "/bin/ears",
        "bin/ears",
        "ears-ui.exe",
        "ears-ui;anything",
    ] {
        assert_eq!(launch(&f, program).status.code(), Some(2), "{program}");
    }
    let output = f
        .command()
        .args(["launch", "ears-ui", "not-after-separator"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    failure(output);
    assert!(!f.path("config").exists());
    let output = f.command().args(["launch", "--help"]).output().unwrap();
    assert!(output.status.success());
    assert!(!f.path("config").exists());
}

#[test]
fn launch_refuses_no_active_missing_nonexecutable_corrupt_and_extra_files() {
    let absent = Fixture::new();
    failure(launch(&absent, "ears-ui"));
    assert!(!absent.prefix.exists());

    for mutation in [
        "unlisted-program",
        "not-executable",
        "missing",
        "corrupt-program",
        "corrupt-support",
        "corrupt-manifest",
        "unlisted-extra",
        "no-active",
    ] {
        let f = Fixture::new();
        let (dir, mut manifest) = f.package("v1");
        if mutation != "unlisted-program" {
            fs::write(
                dir.join(ui_path()),
                b"synthetic non-native bytes; must never be attempted",
            )
            .unwrap();
            manifest["files"].as_array_mut().unwrap().push(json!({
                "path": ui_path(),
                "sha256": sha(b"synthetic non-native bytes; must never be attempted"),
                "executable": mutation != "not-executable"
            }));
        }
        let hash = f.write_manifest(&dir, &manifest);
        let installed = success(f.install_output(&dir, &hash));
        let root = PathBuf::from(installed["active"]["path"].as_str().unwrap());
        match mutation {
            "missing" => fs::remove_file(root.join(ui_path())).unwrap(),
            "corrupt-program" => fs::write(root.join(ui_path()), b"corrupt").unwrap(),
            "corrupt-support" => fs::write(root.join("support.dat"), b"corrupt").unwrap(),
            "corrupt-manifest" => {
                let mut bytes = fs::read(root.join("manifest.json")).unwrap();
                bytes.push(b'\n');
                fs::write(root.join("manifest.json"), bytes).unwrap();
            }
            "unlisted-extra" => fs::write(root.join("bin/unlisted"), b"extra").unwrap(),
            "no-active" => {
                let mut receipt: Value = serde_json::from_slice(&f.receipt()).unwrap();
                receipt["active"] = Value::Null;
                fs::write(
                    f.prefix.join("receipt.json"),
                    serde_json::to_vec(&receipt).unwrap(),
                )
                .unwrap();
            }
            _ => {}
        }
        let receipt = f.receipt();
        let result = failure(launch(&f, "ears-ui"));
        let error = result["error"].as_str().unwrap();
        assert!(
            !error.contains("launching verified active program"),
            "attempted spawn for {mutation}: {error}"
        );
        if mutation == "unlisted-program" {
            assert!(error.contains("not listed"));
        }
        if mutation == "not-executable" {
            assert!(error.contains("not declared executable"));
        }
        if mutation == "no-active" {
            assert!(error.contains("no active"));
        }
        assert_eq!(f.receipt(), receipt);
    }
}

#[cfg(unix)]
#[test]
fn launch_refuses_symlink_in_selected_program() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let (dir, mut manifest) = f.package("v1");
    fs::write(dir.join(ui_path()), b"synthetic").unwrap();
    manifest["files"].as_array_mut().unwrap().push(json!({
        "path":ui_path(), "sha256":sha(b"synthetic"), "executable":true
    }));
    let hash = f.write_manifest(&dir, &manifest);
    let installed = success(f.install_output(&dir, &hash));
    let root = PathBuf::from(installed["active"]["path"].as_str().unwrap());
    fs::write(f.path("outside"), b"synthetic").unwrap();
    fs::remove_file(root.join(ui_path())).unwrap();
    symlink(f.path("outside"), root.join(ui_path())).unwrap();
    let result = failure(launch(&f, "ears-ui"));
    assert!(result["error"].as_str().unwrap().contains("symlinks"));
}

fn native_helper(f: &Fixture) -> Option<PathBuf> {
    let path = f.path(if cfg!(windows) {
        "synthetic-helper.exe"
    } else {
        "synthetic-helper"
    });
    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let result = Command::new(compiler)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/launch-helper.rs"))
        .args(["--edition=2021", "-C", "debuginfo=0"])
        .arg("-o")
        .arg(&path)
        .output();
    let result = match result {
        Ok(result) => result,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("native launch execution UNTESTED: native rustc helper compiler unavailable");
            return None;
        }
        Err(error) => panic!("starting native helper compiler: {error}"),
    };
    assert!(
        result.status.success(),
        "native helper compilation failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Some(path)
}

fn wait_for(path: &Path) {
    let start = Instant::now();
    while !path.exists() && start.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(path.exists(), "helper did not create {}", path.display());
}

#[test]
fn native_helper_launch_resolves_rollback_literal_argv_and_detaches_without_pipe_or_lock_inheritance(
) {
    use fs2::FileExt;
    let f = Fixture::new();
    let Some(helper) = native_helper(&f) else {
        return;
    };
    let bytes = fs::read(helper).unwrap();
    let forbidden = f.path("forbidden-implicit-execution");
    let sentinel = f.path("home/.config/ears/config.toml");
    fs::create_dir_all(sentinel.parent().unwrap()).unwrap();
    fs::write(&sentinel, b"keep user data").unwrap();

    let mut installed = Vec::new();
    for version in ["v1", "v2"] {
        let (dir, mut manifest) = f.package(version);
        fs::write(dir.join(ui_path()), &bytes).unwrap();
        manifest["files"].as_array_mut().unwrap().push(json!({
            "path": ui_path(), "sha256":sha(&bytes), "executable":true
        }));
        let hash = f.write_manifest(&dir, &manifest);
        installed.push(success(
            f.command()
                .env("EARS_SYNTHETIC_HELPER_START_MARKER", &forbidden)
                .arg("install")
                .arg("--prefix")
                .arg(&f.prefix)
                .arg("--package")
                .arg(dir)
                .arg("--manifest-sha256")
                .arg(hash)
                .output()
                .unwrap(),
        ));
        assert!(!forbidden.exists(), "install executed the native helper");
    }

    for (index, version) in ["v2", "v1"].into_iter().enumerate() {
        if index == 1 {
            success(
                f.command()
                    .env("EARS_SYNTHETIC_HELPER_START_MARKER", &forbidden)
                    .arg("rollback")
                    .arg("--prefix")
                    .arg(&f.prefix)
                    .output()
                    .unwrap(),
            );
        }
        success(
            f.command()
                .env("EARS_SYNTHETIC_HELPER_START_MARKER", &forbidden)
                .arg("status")
                .arg("--prefix")
                .arg(&f.prefix)
                .output()
                .unwrap(),
        );
        assert!(
            !forbidden.exists(),
            "validation/rollback executed the native helper"
        );
        let receipt = f.receipt();
        let output_file = f.path(&format!("launch-{version}.record"));
        let allowed = f.path(&format!("explicit-{version}"));
        let shell_sentinel = f.path("must-not-be-shell-created");
        let mut literal: Vec<OsString> = vec![
            "space separated argument".into(),
            "".into(),
            "\"quoted\"".into(),
            r"back\slash".into(),
            "--prefix".into(),
            "literal-not-installer-prefix".into(),
            "--help".into(),
            "--".into(),
            "Unicode-ä".into(),
        ];
        literal.push(
            format!(
                "; touch {}; $(touch {}) &",
                shell_sentinel.display(),
                shell_sentinel.display()
            )
            .into(),
        );
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            literal.push(OsString::from_vec(vec![b'x', 0xff]));
        }
        let result = f
            .command()
            .env("EARS_SYNTHETIC_HELPER_START_MARKER", &allowed)
            .arg("--prefix")
            .arg(&f.prefix)
            .args(["launch", "ears-ui", "--"])
            .arg(&output_file)
            .args(&literal)
            .output()
            .unwrap();
        assert!(
            result.stderr.is_empty(),
            "child stderr inherited launcher's capture"
        );
        let result = success(result);
        assert_eq!(result["action"], "launch");
        assert_eq!(result["program"], "ears-ui");
        assert_eq!(result["detached"], true);
        assert_eq!(result["version"], version);
        assert_eq!(
            result["manifest_sha256"],
            installed[1 - index]["active"]["manifest_sha256"]
        );
        wait_for(&output_file);
        assert!(allowed.exists());
        assert!(
            !output_file.with_extension("done").exists(),
            "launcher kept child stdio/session alive"
        );
        let record = fs::read_to_string(&output_file).unwrap();
        let lines: Vec<_> = record.lines().collect();
        assert_eq!(
            lines[0].parse::<u32>().unwrap() as u64,
            result["pid"].as_u64().unwrap()
        );
        assert!(result["pid"].as_u64().unwrap() > 0);
        #[cfg(unix)]
        assert_eq!(
            lines[3].parse::<u32>().unwrap() as u64,
            result["pid"].as_u64().unwrap(),
            "child must lead its own detached session"
        );
        let encoded: Vec<_> = literal
            .iter()
            .map(|arg| arg.as_encoded_bytes().to_vec())
            .collect();
        assert_eq!(lines[1], format!("{encoded:?}"));
        let active_root = PathBuf::from(result["path"].as_str().unwrap())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        assert_eq!(
            Path::new(lines[2]).canonicalize().unwrap(),
            active_root.canonicalize().unwrap()
        );
        assert!(!shell_sentinel.exists());
        // Verify the live detached child did not inherit the installer's lock.
        let lock = fs::File::open(f.prefix.join(".installer.lock")).unwrap();
        lock.try_lock_exclusive().unwrap();
        drop(lock);
        assert_eq!(f.receipt(), receipt);
        assert_eq!(fs::read(&sentinel).unwrap(), b"keep user data");
        fs::write(output_file.with_extension("release"), b"exit normally").unwrap();
        wait_for(&output_file.with_extension("done"));
    }
    eprintln!(
        "native launch execution TESTED: synthetic helper on {}",
        std::env::consts::OS
    );
}
