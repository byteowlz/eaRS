use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

#[path = "support/launch.rs"]
mod launch_tests;

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Fixture {
    root: TempDir,
    prefix: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let prefix = root.path().join("install");
        Self { root, prefix }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.path().canonicalize().unwrap().join(name)
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_ears-desktop-installer"));
        cmd.env_clear()
            .current_dir(self.root.path())
            .env("HOME", self.path("home"))
            .env("USERPROFILE", self.path("home"))
            .env("LOCALAPPDATA", self.path("appdata"))
            .env("XDG_CONFIG_HOME", self.path("config"))
            .env("TMPDIR", self.root.path())
            .env("TMP", self.root.path())
            .env("TEMP", self.root.path());
        #[cfg(windows)]
        if let Some(system_root) = std::env::var_os("SystemRoot") {
            cmd.env("SystemRoot", system_root);
        }
        cmd
    }

    fn package(&self, name: &str) -> (PathBuf, Value) {
        let dir = self.path(name);
        fs::create_dir_all(dir.join("bin")).unwrap();
        // Deliberately not a runnable native executable.
        let bytes = format!("synthetic eaRS payload: {name}\n").into_bytes();
        fs::write(dir.join("bin/ears"), &bytes).unwrap();
        fs::write(dir.join("support.dat"), b"synthetic support bytes").unwrap();
        let manifest = json!({
            "schema": "ears.desktop-payload/v1",
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "version": name,
            "source_commit": "synthetic-not-a-source-build",
            "client_vad": true,
            "files": [
                {"path": "bin/ears", "sha256": sha(&bytes), "executable": true},
                {"path": "support.dat", "sha256": sha(b"synthetic support bytes"), "executable": false}
            ]
        });
        (dir, manifest)
    }

    fn write_manifest(&self, dir: &Path, manifest: &Value) -> String {
        let bytes = serde_json::to_vec_pretty(manifest).unwrap();
        fs::write(dir.join("manifest.json"), &bytes).unwrap();
        sha(&bytes)
    }

    fn install_output(&self, dir: &Path, hash: &str) -> Output {
        self.command()
            .arg("install")
            .arg("--prefix")
            .arg(&self.prefix)
            .arg("--package")
            .arg(dir)
            .arg("--manifest-sha256")
            .arg(hash)
            .output()
            .unwrap()
    }

    fn install(&self, name: &str) -> Value {
        let (dir, manifest) = self.package(name);
        let hash = self.write_manifest(&dir, &manifest);
        success(self.install_output(&dir, &hash))
    }

    fn run(&self, action: &str) -> Output {
        self.command()
            .arg(action)
            .arg("--prefix")
            .arg(&self.prefix)
            .output()
            .unwrap()
    }

    fn receipt(&self) -> Vec<u8> {
        fs::read(self.prefix.join("receipt.json")).unwrap()
    }
}

fn success(output: Output) -> Value {
    assert!(
        output.status.success(),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["ok"], true);
    value
}

fn failure(output: Output) -> Value {
    assert!(
        !output.status.success(),
        "unexpected success: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["ok"], false);
    assert!(value["error"].as_str().unwrap().len() > 3);
    value
}

#[test]
fn install_update_idempotency_rollback_uninstall_preserves_external_data() {
    let f = Fixture::new();
    for sentinel in [
        "home/.config/ears/config.toml",
        "home/.local/share/ears/models/model.bin",
        "home/.local/state/ears/history.jsonl",
    ] {
        let path = f.path(sentinel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"must survive").unwrap();
    }
    let first = f.install("v1");
    assert_eq!(first["changed"], true);
    assert_eq!(first["active"]["version"], "v1");
    assert!(first["previous"].is_null());
    let active_v1 = PathBuf::from(first["active"]["path"].as_str().unwrap());
    assert_eq!(
        fs::read(active_v1.join("bin/ears")).unwrap(),
        b"synthetic eaRS payload: v1\n"
    );
    let receipt = f.receipt();
    // Idempotency is exact manifest identity, not just version equality.
    // Reuse the exact trusted source for this repeat.
    let original_hash = first["active"]["manifest_sha256"].as_str().unwrap();
    let repeated = success(f.install_output(&f.path("v1"), original_hash));
    assert_eq!(repeated["changed"], false);
    assert_eq!(f.receipt(), receipt);
    let second = f.install("v2");
    assert_eq!(second["active"]["version"], "v2");
    assert_eq!(second["previous"]["version"], "v1");
    let v2_receipt = f.receipt();
    success(f.install_output(
        &f.path("v2"),
        second["active"]["manifest_sha256"].as_str().unwrap(),
    ));
    assert_eq!(f.receipt(), v2_receipt);
    let status = success(f.run("status"));
    assert_eq!(status["active"], second["active"]);
    let rolled = success(f.run("rollback"));
    assert_eq!(rolled["active"]["version"], "v1");
    assert_eq!(rolled["previous"]["version"], "v2");
    success(f.run("rollback")); // verified roll-forward is the same swap operation
    f.install("v3");
    assert_eq!(success(f.run("status"))["retained_snapshots"], 3);
    fs::write(f.prefix.join("unrelated.txt"), b"prefix sentinel").unwrap();
    fs::create_dir_all(f.prefix.join("models")).unwrap();
    fs::write(f.prefix.join("models/model.bin"), b"prefix model sentinel").unwrap();
    let uninstalled = success(f.run("uninstall"));
    assert!(uninstalled["active"].is_null());
    assert_eq!(uninstalled["warnings"], json!([]));
    assert_eq!(fs::read_dir(f.prefix.join("payloads")).unwrap().count(), 0);
    assert_eq!(
        fs::read(f.prefix.join("unrelated.txt")).unwrap(),
        b"prefix sentinel"
    );
    assert_eq!(
        fs::read(f.prefix.join("models/model.bin")).unwrap(),
        b"prefix model sentinel"
    );
    for sentinel in [
        "home/.config/ears/config.toml",
        "home/.local/share/ears/models/model.bin",
        "home/.local/state/ears/history.jsonl",
    ] {
        assert_eq!(fs::read(f.path(sentinel)).unwrap(), b"must survive");
    }
    assert_eq!(success(f.run("uninstall"))["changed"], false);
    assert!(success(f.run("status"))["active"].is_null());
    failure(f.run("rollback"));
    assert_eq!(f.install("v4")["active"]["version"], "v4");
}

#[test]
fn malformed_manifests_leave_old_active_and_receipt_unchanged() {
    let f = Fixture::new();
    let active = f.install("good")["active"]["path"]
        .as_str()
        .unwrap()
        .to_owned();
    let before = f.receipt();
    let (dir, original) = f.package("bad");
    let mut cases = Vec::new();
    for path in [
        "../escape",
        "/absolute",
        "C:/absolute",
        "dir\\escape",
        "bin/../ears",
        "bin//ears",
        "bad\nname",
        "NUL.txt",
        "bin/ears.",
        "manifest.json",
    ] {
        let mut m = original.clone();
        m["files"][0]["path"] = json!(path);
        cases.push(m);
    }
    for (key, value) in [
        ("schema", json!("wrong/v1")),
        ("os", json!("not-this-host")),
        ("arch", json!("not-this-arch")),
        ("client_vad", json!(false)),
        ("version", json!("")),
        ("unknown", json!(true)),
    ] {
        let mut m = original.clone();
        m[key] = value;
        cases.push(m);
    }
    let mut duplicate = original.clone();
    duplicate["files"]
        .as_array_mut()
        .unwrap()
        .push(original["files"][0].clone());
    cases.push(duplicate);
    let mut alias = original.clone();
    alias["files"]
        .as_array_mut()
        .unwrap()
        .push(json!({"path":"BIN/EARS", "sha256": sha(b"x"), "executable":true}));
    cases.push(alias);
    let mut dir_alias = original.clone();
    dir_alias["files"]
        .as_array_mut()
        .unwrap()
        .push(json!({"path":"BIN/another", "sha256":sha(b"x"), "executable":false}));
    cases.push(dir_alias);
    let mut collision = original.clone();
    collision["files"][1]["path"] = json!("bin");
    cases.push(collision);
    let mut too_many = original.clone();
    too_many["files"] = json!(vec![original["files"][0].clone(); 1025]);
    cases.push(too_many);
    let mut empty = original.clone();
    empty["files"] = json!([]);
    cases.push(empty);
    let mut corrupt_hash = original.clone();
    corrupt_hash["files"][0]["sha256"] = json!("0".repeat(64));
    cases.push(corrupt_hash);
    for manifest in cases {
        let hash = f.write_manifest(&dir, &manifest);
        failure(f.install_output(&dir, &hash));
        assert_eq!(f.receipt(), before);
        assert_eq!(
            fs::read(Path::new(&active).join("bin/ears")).unwrap(),
            b"synthetic eaRS payload: good\n"
        );
    }
    assert!(!f.path("escape").exists());
    assert_eq!(success(f.run("status"))["active"]["version"], "good");
}

#[test]
fn required_trusted_manifest_hash_is_not_self_asserted() {
    let f = Fixture::new();
    let (dir, manifest) = f.package("v1");
    f.write_manifest(&dir, &manifest);
    failure(f.install_output(&dir, &"0".repeat(64)));
    failure(f.install_output(&dir, "not-a-hash"));
    failure(
        f.command()
            .arg("install")
            .arg("--package")
            .arg(&dir)
            .arg("--prefix")
            .arg(&f.prefix)
            .output()
            .unwrap(),
    );
    assert!(!f.prefix.exists());
}

#[test]
fn corrupt_missing_extra_and_empty_directory_are_rejected() {
    for mutation in [
        "corrupt",
        "missing",
        "extra",
        "directory",
        "manifest-limit",
        "bad-json",
        "payload-limit",
    ]
    .into_iter()
    .filter(|name| cfg!(unix) || *name != "payload-limit")
    {
        // Unix set_len creates a sparse bound-check fixture. Do not allocate a
        // multi-GiB Windows file merely for this test; the stream limit is also
        // covered by a platform-independent unit test.
        let f = Fixture::new();
        let (dir, manifest) = f.package("v1");
        let mut hash = f.write_manifest(&dir, &manifest);
        match mutation {
            "corrupt" => fs::write(dir.join("support.dat"), b"tampered").unwrap(),
            "missing" => fs::remove_file(dir.join("support.dat")).unwrap(),
            "extra" => fs::write(dir.join("extra"), b"unlisted").unwrap(),
            "directory" => fs::create_dir(dir.join("unlisted-empty")).unwrap(),
            "manifest-limit" => {
                let bytes = vec![b' '; 1024 * 1024 + 1];
                fs::write(dir.join("manifest.json"), &bytes).unwrap();
                hash = sha(&bytes);
            }
            "payload-limit" => fs::OpenOptions::new()
                .write(true)
                .open(dir.join("support.dat"))
                .unwrap()
                .set_len(2 * 1024 * 1024 * 1024)
                .unwrap(),
            "bad-json" => {
                fs::write(dir.join("manifest.json"), b"{").unwrap();
                hash = sha(b"{");
            }
            _ => unreachable!(),
        }
        failure(f.install_output(&dir, &hash));
        assert!(!f.prefix.exists(), "{mutation}");
    }
}

#[test]
fn unmanaged_prefix_and_payload_collisions_are_not_adopted() {
    let f = Fixture::new();
    fs::create_dir(&f.prefix).unwrap();
    fs::write(f.prefix.join("sentinel"), b"untouched").unwrap();
    let (dir, manifest) = f.package("v1");
    let hash = f.write_manifest(&dir, &manifest);
    failure(f.install_output(&dir, &hash));
    failure(f.run("uninstall"));
    failure(f.run("status"));
    assert_eq!(fs::read(f.prefix.join("sentinel")).unwrap(), b"untouched");
    assert!(!f.prefix.join("receipt.json").exists());

    let g = Fixture::new();
    g.install("good");
    let receipt = g.receipt();
    fs::create_dir(g.prefix.join("payloads/unmanaged")).unwrap();
    fs::write(g.prefix.join("payloads/unmanaged/sentinel"), b"untouched").unwrap();
    failure(g.run("uninstall"));
    failure(g.run("rollback"));
    assert_eq!(g.receipt(), receipt);
    assert_eq!(
        fs::read(g.prefix.join("payloads/unmanaged/sentinel")).unwrap(),
        b"untouched"
    );
}

#[test]
fn rollback_revalidates_old_payload_and_uninstall_preserves_unknown_nested_files() {
    let f = Fixture::new();
    let v1 = f.install("v1");
    f.install("v2");
    let receipt = f.receipt();
    let old_path = PathBuf::from(v1["active"]["path"].as_str().unwrap());
    fs::write(old_path.join("support.dat"), b"corrupt").unwrap();
    failure(f.run("rollback"));
    failure(f.run("uninstall"));
    assert_eq!(f.receipt(), receipt);
    fs::write(old_path.join("support.dat"), b"synthetic support bytes").unwrap();
    fs::write(old_path.join("bin/sentinel"), b"not managed").unwrap();
    failure(f.run("uninstall"));
    assert_eq!(f.receipt(), receipt);
    assert_eq!(
        fs::read(old_path.join("bin/sentinel")).unwrap(),
        b"not managed"
    );
    fs::remove_file(old_path.join("bin/sentinel")).unwrap();
    success(f.run("rollback"));
    success(f.run("uninstall"));
}

#[test]
fn tampered_receipt_identifiers_cannot_escape_prefix() {
    let f = Fixture::new();
    f.install("v1");
    fs::write(f.path("sentinel"), b"untouched").unwrap();
    let mut receipt: Value = serde_json::from_slice(&f.receipt()).unwrap();
    receipt["active"] = json!("../../sentinel");
    receipt["managed"] = json!(["../../sentinel"]);
    let bad = serde_json::to_vec(&receipt).unwrap();
    fs::write(f.prefix.join("receipt.json"), &bad).unwrap();
    failure(f.run("status"));
    failure(f.run("rollback"));
    failure(f.run("uninstall"));
    assert_eq!(f.receipt(), bad);
    assert_eq!(fs::read(f.path("sentinel")).unwrap(), b"untouched");
}

#[test]
fn absent_operations_do_not_claim_a_prefix() {
    let f = Fixture::new();
    assert!(success(f.run("status"))["active"].is_null());
    assert_eq!(success(f.run("uninstall"))["changed"], false);
    failure(f.run("rollback"));
    assert!(!f.prefix.exists());
    assert!(f.path("config/ears-installer/config.toml").is_file());
    let schema: Value = serde_json::from_slice(
        &fs::read(f.path("config/ears-installer/config.schema.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(schema["properties"]["prefix"]["type"], "string");
}

#[test]
fn help_version_and_parse_errors_do_not_write_configuration() {
    let f = Fixture::new();
    for args in [vec!["--help"], vec!["--version"], vec!["install", "--help"]] {
        let out = f.command().args(args).output().unwrap();
        assert!(out.status.success());
        assert!(!f.path("config").exists());
    }
    failure(f.command().arg("unknown-command").output().unwrap());
    assert!(!f.path("config").exists());
}

#[test]
fn configuration_priority_cli_explicit_env_local_global_default() {
    let f = Fixture::new();
    let global = f.path("config/ears-installer");
    fs::create_dir_all(&global).unwrap();
    let write = |path: &Path, prefix: &str| {
        fs::write(path, format!("prefix = {prefix:?}\n")).unwrap();
    };
    write(&global.join("config.toml"), "global");
    let query = |extra: &[&str], env: Option<&str>| {
        let mut cmd = f.command();
        cmd.arg("status").args(extra);
        if let Some(value) = env {
            cmd.env("EARS_INSTALLER_PREFIX", value);
        }
        let value = success(cmd.output().unwrap());
        let path = PathBuf::from(value["prefix"].as_str().unwrap());
        // macOS /var aliases and Windows extended-length path spellings differ
        // from current_dir(). Compare resolved existing parents, not spellings.
        let normalized = path
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .map(|parent| parent.join(path.file_name().unwrap()))
            .unwrap_or(path);
        normalized.to_str().unwrap().to_owned()
    };
    assert_eq!(query(&[], None), f.path("global").to_str().unwrap());
    write(&f.path("ears-installer.toml"), "local");
    assert_eq!(query(&[], None), f.path("local").to_str().unwrap());
    assert_eq!(
        query(&[], Some("environment")),
        f.path("environment").to_str().unwrap()
    );
    write(&f.path("explicit.toml"), "explicit");
    assert_eq!(
        query(&["--config", "explicit.toml"], Some("environment")),
        f.path("explicit").to_str().unwrap()
    );
    assert_eq!(
        query(
            &["--config", "explicit.toml", "--prefix", "cli"],
            Some("environment")
        ),
        f.path("cli").to_str().unwrap()
    );
    fs::remove_file(f.path("ears-installer.toml")).unwrap();
    fs::write(global.join("config.toml"), "# defaults\n").unwrap();
    #[cfg(target_os = "linux")]
    assert_eq!(
        query(&[], None),
        f.path("home/.local/opt/ears").to_str().unwrap()
    );
    #[cfg(windows)]
    assert_eq!(
        query(&[], None),
        f.path("appdata/Programs/eaRS").to_str().unwrap()
    );
}

#[test]
fn interrupted_uninstall_retries_only_authenticated_remaining_content() {
    for interruption in [
        "file-removed",
        "directories-removed",
        "manifest-removed",
        "snapshot-removed",
        "unknown-extra",
    ] {
        let f = Fixture::new();
        let installed = f.install("v1");
        let path = PathBuf::from(installed["active"]["path"].as_str().unwrap());
        let mut receipt: Value = serde_json::from_slice(&f.receipt()).unwrap();
        // Reproduce disk state after the atomic clear-slots commit and an
        // interruption during cleanup. No production-only crash hook is used.
        receipt["active"] = Value::Null;
        receipt["previous"] = Value::Null;
        fs::write(
            f.prefix.join("receipt.json"),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        fs::remove_file(path.join("bin/ears")).unwrap();
        if interruption != "file-removed" && interruption != "unknown-extra" {
            fs::remove_dir(path.join("bin")).unwrap();
            fs::remove_file(path.join("support.dat")).unwrap();
        }
        if interruption == "manifest-removed" || interruption == "snapshot-removed" {
            fs::remove_file(path.join("manifest.json")).unwrap();
        }
        if interruption == "snapshot-removed" {
            fs::remove_dir(&path).unwrap();
        }
        if interruption == "unknown-extra" {
            fs::write(path.join("unknown"), b"must survive").unwrap();
            let before = f.receipt();
            failure(f.run("uninstall"));
            assert_eq!(f.receipt(), before);
            assert_eq!(fs::read(path.join("unknown")).unwrap(), b"must survive");
            continue;
        }
        let status = success(f.run("status"));
        assert!(status["active"].is_null());
        assert_eq!(status["cleanup_pending"], true);
        failure(f.install_output(
            &f.path("v1"),
            installed["active"]["manifest_sha256"].as_str().unwrap(),
        ));
        let removed = success(f.run("uninstall"));
        assert_eq!(removed["cleanup_complete"], true);
        assert_eq!(removed["cleanup_pending"], false);
        assert!(!path.exists());
    }
}

#[test]
fn concurrent_prefix_lock_prevents_mutation() {
    use fs2::FileExt;
    let f = Fixture::new();
    f.install("v1");
    let receipt = f.receipt();
    let lock = fs::File::open(f.prefix.join(".installer.lock")).unwrap();
    lock.lock_exclusive().unwrap();
    failure(f.run("uninstall"));
    assert_eq!(f.receipt(), receipt);
    drop(lock);
    success(f.run("uninstall"));
}

#[cfg(unix)]
#[test]
fn symlink_sources_destinations_and_special_files_are_rejected() {
    use std::os::unix::fs::symlink;
    for mutation in ["file", "directory", "manifest", "root", "fifo"] {
        let f = Fixture::new();
        let (dir, manifest) = f.package("v1");
        let hash = f.write_manifest(&dir, &manifest);
        fs::write(f.path("outside"), b"synthetic support bytes").unwrap();
        match mutation {
            "file" => {
                fs::remove_file(dir.join("support.dat")).unwrap();
                symlink(f.path("outside"), dir.join("support.dat")).unwrap();
            }
            "directory" => {
                fs::rename(dir.join("bin"), f.path("outside-bin")).unwrap();
                symlink(f.path("outside-bin"), dir.join("bin")).unwrap();
            }
            "manifest" => {
                fs::rename(dir.join("manifest.json"), f.path("outside-manifest")).unwrap();
                symlink(f.path("outside-manifest"), dir.join("manifest.json")).unwrap();
            }
            "root" => {
                fs::rename(&dir, f.path("real-package")).unwrap();
                symlink(f.path("real-package"), &dir).unwrap();
            }
            "fifo" => {
                use std::ffi::CString;
                use std::os::unix::ffi::OsStrExt;
                fs::remove_file(dir.join("support.dat")).unwrap();
                let name = CString::new(dir.join("support.dat").as_os_str().as_bytes()).unwrap();
                assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            }
            _ => unreachable!(),
        }
        failure(f.install_output(&dir, &hash));
        assert!(!f.prefix.exists());
    }
    let f = Fixture::new();
    fs::create_dir(f.path("elsewhere")).unwrap();
    symlink(f.path("elsewhere"), &f.prefix).unwrap();
    let (dir, m) = f.package("v1");
    let hash = f.write_manifest(&dir, &m);
    failure(f.install_output(&dir, &hash));
    assert_eq!(fs::read_dir(f.path("elsewhere")).unwrap().count(), 0);

    let g = Fixture::new();
    g.install("v1");
    let receipt = g.receipt();
    fs::rename(g.prefix.join("payloads"), g.path("elsewhere")).unwrap();
    symlink(g.path("elsewhere"), g.prefix.join("payloads")).unwrap();
    failure(g.run("uninstall"));
    assert_eq!(g.receipt(), receipt);
}

#[cfg(unix)]
#[test]
fn executable_bits_are_preserved_without_suid_or_source_write_bits() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    let (dir, manifest) = f.package("v1");
    fs::set_permissions(dir.join("bin/ears"), fs::Permissions::from_mode(0o6777)).unwrap();
    let hash = f.write_manifest(&dir, &manifest);
    let installed = success(f.install_output(&dir, &hash));
    let path = PathBuf::from(installed["active"]["path"].as_str().unwrap());
    let mode = fs::metadata(path.join("bin/ears"))
        .unwrap()
        .permissions()
        .mode();
    assert_ne!(mode & 0o111, 0);
    assert_eq!(mode & 0o7022, 0);
    let data_mode = fs::metadata(path.join("support.dat"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(data_mode & 0o7111, 0);
    fs::set_permissions(path.join("bin/ears"), fs::Permissions::from_mode(0o4755)).unwrap();
    failure(f.run("status"));
    failure(f.run("uninstall"));
}
