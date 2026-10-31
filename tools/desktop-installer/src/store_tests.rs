use super::*;
use crate::payload::{hash, SCHEMA};

fn source(root: &Path, name: &str) -> (PathBuf, Payload) {
    let path = root.join(name);
    fs::create_dir(&path).unwrap();
    fs::write(path.join("data"), name.as_bytes()).unwrap();
    let manifest = json!({
        "schema": SCHEMA, "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "version": name, "source_commit": "synthetic", "client_vad": true,
        "files": [{"path":"data", "sha256":hash(name.as_bytes()), "executable":false}]
    });
    let bytes = serde_json::to_vec(&manifest).unwrap();
    fs::write(path.join("manifest.json"), &bytes).unwrap();
    let payload = Payload::load(&path, &hash(&bytes)).unwrap();
    (path, payload)
}

#[test]
fn failed_receipt_commit_preserves_active_previous_and_removes_only_new_stage() {
    let root = tempfile::tempdir().unwrap();
    let prefix = root.path().join("prefix");
    let mut store = Store::open(&prefix, true).unwrap().unwrap();
    for name in ["v1", "v2"] {
        let (path, payload) = source(root.path(), name);
        store.install(&path, payload).unwrap();
    }
    let before = fs::read(prefix.join(RECEIPT)).unwrap();
    let (path, payload) = source(root.path(), "v3");
    store.fail_commit = true;
    assert!(store.install(&path, payload).is_err());
    assert_eq!(fs::read(prefix.join(RECEIPT)).unwrap(), before);
    assert_eq!(fs::read_dir(prefix.join("payloads")).unwrap().count(), 2);
    let status = store.status("status", false).unwrap();
    assert_eq!(status["active"]["version"], "v2");
    assert_eq!(status["previous"]["version"], "v1");
    assert!(store.rollback().is_err());
    assert!(store.uninstall().is_err());
    assert_eq!(fs::read(prefix.join(RECEIPT)).unwrap(), before);
    assert_eq!(
        store.status("status", false).unwrap()["active"]["version"],
        "v2"
    );
    store.fail_commit = false;
    store.rollback().unwrap();
    assert_eq!(
        store.status("status", false).unwrap()["active"]["version"],
        "v1"
    );
}

#[test]
fn real_atomic_write_failure_does_not_replace_a_collision() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("receipt.json");
    fs::create_dir(&path).unwrap();
    fs::write(path.join("sentinel"), b"untouched").unwrap();
    assert!(fsutil::atomic_write(&path, b"replacement").is_err());
    assert_eq!(fs::read(path.join("sentinel")).unwrap(), b"untouched");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
}

#[test]
fn atomic_receipt_replacement_is_complete_json() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("receipt.json");
    fsutil::atomic_write(&path, b"{\"version\":1}").unwrap();
    fsutil::atomic_write(&path, b"{\"version\":2}").unwrap();
    let value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(value["version"], 2);
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
}
