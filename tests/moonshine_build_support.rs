#![cfg(feature = "moonshine")]
#[allow(dead_code)]
#[path = "../vendor/moonshine-sys/build_support.rs"]
mod support;

use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use support::{artifact_path, Artifact};

struct Cache(PathBuf);
impl Cache {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ears-ort-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn artifact(&self) -> Artifact {
        Artifact {
            name: "runtime.bin".into(),
            bytes: 7,
            sha256: format!("{:x}", Sha256::digest(b"runtime")),
            url: "https://raw.githubusercontent.com/moonshine-ai/moonshine/pin/runtime.bin".into(),
        }
    }
    fn seed(&self, bytes: &[u8]) -> PathBuf {
        let item = self.artifact();
        let path = self.0.join(&item.sha256).join(&item.name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn offline_cache_hit_is_verified() {
    let cache = Cache::new();
    let expected = cache.seed(b"runtime");
    assert_eq!(
        artifact_path(&cache.0, &cache.artifact(), true).unwrap(),
        expected
    );
}

#[test]
fn same_size_corrupt_cache_fails_without_repair() {
    let cache = Cache::new();
    let path = cache.seed(b"corrupt");
    assert!(artifact_path(&cache.0, &cache.artifact(), false).is_err());
    assert_eq!(fs::read(path).unwrap(), b"corrupt");
}

#[test]
fn missing_offline_dependency_cannot_download() {
    let cache = Cache::new();
    assert!(artifact_path(&cache.0, &cache.artifact(), true).is_err());
    assert_eq!(fs::read_dir(&cache.0).unwrap().count(), 0);
}

#[test]
fn unsafe_names_and_nonpinned_hosts_are_rejected() {
    let cache = Cache::new();
    let mut item = cache.artifact();
    item.name = "../escape".into();
    assert!(artifact_path(&cache.0, &item, false).is_err());
    item = cache.artifact();
    item.url = "https://example.invalid/runtime".into();
    assert!(artifact_path(&cache.0, &item, false).is_err());
}

#[cfg(unix)]
#[test]
fn cache_symlinks_are_not_trusted_even_with_matching_bytes() {
    let cache = Cache::new();
    let item = cache.artifact();
    let target = cache.0.join("target");
    fs::write(&target, b"runtime").unwrap();
    let destination = cache.0.join(&item.sha256).join(&item.name);
    fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(target, destination).unwrap();
    assert!(artifact_path(&cache.0, &item, true).is_err());
}
