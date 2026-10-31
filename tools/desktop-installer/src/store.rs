use crate::{fsutil, launcher, payload::Payload};
use anyhow::{ensure, Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

const RECEIPT: &str = "receipt.json";
const RECEIPT_SCHEMA: &str = "ears.desktop-installer-receipt/v1";
const LOCK: &str = ".installer.lock";
const MAX_SNAPSHOTS: usize = 128;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    active: Option<String>,
    previous: Option<String>,
    managed: Vec<String>,
}

impl Default for Receipt {
    fn default() -> Self {
        Self {
            schema: RECEIPT_SCHEMA.into(),
            active: None,
            previous: None,
            managed: Vec::new(),
        }
    }
}

pub struct Store {
    prefix: PathBuf,
    receipt: Receipt,
    _lock: File,
    #[cfg(test)]
    fail_commit: bool,
}

impl Store {
    pub fn open(prefix: &Path, create: bool) -> Result<Option<Self>> {
        if !fsutil::exists(prefix)? && !create {
            return Ok(None);
        }
        let receipt_path = prefix.join(RECEIPT);
        let new = !fsutil::exists(&receipt_path)?;
        if new {
            if !create {
                fsutil::directory(prefix)?;
                ensure!(
                    fs::read_dir(prefix)?.next().is_none(),
                    "refusing unmanaged prefix"
                );
                return Ok(None);
            }
            fsutil::ensure_new_or_empty(prefix)?;
        } else {
            fsutil::directory(prefix)?;
        }
        let lock = if new {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(prefix.join(LOCK))?
        } else {
            fsutil::open_regular(&prefix.join(LOCK))?
        };
        lock.try_lock_exclusive()
            .context("installer prefix is busy")?;
        let receipt = if new {
            fs::create_dir(prefix.join("payloads"))?;
            let receipt = Receipt::default();
            fsutil::atomic_write(&receipt_path, &serde_json::to_vec_pretty(&receipt)?)?;
            receipt
        } else {
            serde_json::from_slice(&fsutil::read_bounded(&receipt_path, 512 * 1024)?)
                .context("invalid installer receipt")?
        };
        let store = Self {
            prefix: prefix.into(),
            receipt,
            _lock: lock,
            #[cfg(test)]
            fail_commit: false,
        };
        store.validate_receipt()?;
        Ok(Some(store))
    }

    fn validate_receipt(&self) -> Result<()> {
        let receipt = &self.receipt;
        ensure!(receipt.schema == RECEIPT_SCHEMA, "unmanaged receipt schema");
        ensure!(
            receipt.managed.len() <= MAX_SNAPSHOTS,
            "too many managed snapshots"
        );
        let mut ids = BTreeSet::new();
        for id in &receipt.managed {
            ensure!(crate::payload::valid_hash(id), "unsafe receipt identifier");
            ensure!(ids.insert(id), "duplicate receipt identifier");
        }
        for slot in [&receipt.active, &receipt.previous].into_iter().flatten() {
            ensure!(ids.contains(slot), "unmanaged slot in receipt");
        }
        ensure!(
            receipt.previous.is_none() || receipt.active.is_some(),
            "previous without active"
        );
        ensure!(
            receipt.active.is_none() || receipt.active != receipt.previous,
            "duplicate slots"
        );
        fsutil::directory(&self.prefix.join("payloads"))?;
        for entry in fs::read_dir(self.prefix.join("payloads"))? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid snapshot name"))?;
            ensure!(ids.contains(&name), "unmanaged payload collision");
            fsutil::directory(&entry.path())?;
        }
        Ok(())
    }

    fn payload_path(&self, id: &str) -> PathBuf {
        self.prefix.join("payloads").join(id)
    }

    fn verified(&self) -> Result<BTreeMap<String, Payload>> {
        let mut result = BTreeMap::new();
        for id in &self.receipt.managed {
            let path = self.payload_path(id);
            let payload = if self.receipt.active.is_none() {
                // After the atomic uninstall commit, absent listed files are safe
                // to tolerate; any remaining file still needs its trusted hash.
                if !fsutil::exists(&path)? {
                    continue;
                }
                if !fsutil::exists(&path.join("manifest.json"))? {
                    ensure!(
                        fs::read_dir(&path)?.next().is_none(),
                        "unmanaged cleanup content"
                    );
                    continue;
                }
                Payload::load_for_cleanup(&path, id)?
            } else {
                let payload = Payload::load(&path, id)?;
                payload.verify_modes(&path)?;
                payload
            };
            result.insert(id.clone(), payload);
        }
        Ok(result)
    }

    fn commit(&mut self, receipt: Receipt) -> Result<()> {
        #[cfg(test)]
        ensure!(!self.fail_commit, "injected receipt replacement failure");
        fsutil::atomic_write(
            &self.prefix.join(RECEIPT),
            &serde_json::to_vec_pretty(&receipt)?,
        )?;
        self.receipt = receipt;
        Ok(())
    }

    fn describe_slot(&self, id: &Option<String>, payloads: &BTreeMap<String, Payload>) -> Value {
        let Some(id) = id else { return Value::Null };
        let payload = &payloads[id];
        json!({
            "manifest_sha256": id,
            "path": self.payload_path(id),
            "version": payload.manifest.version,
            "source_commit": payload.manifest.source_commit,
            "os": payload.manifest.os,
            "arch": payload.manifest.arch,
            "client_vad": true
        })
    }

    pub fn status(&self, action: &str, changed: bool) -> Result<Value> {
        let payloads = self.verified()?;
        Ok(self.output(&self.receipt, &payloads, action, changed))
    }

    fn output(
        &self,
        receipt: &Receipt,
        payloads: &BTreeMap<String, Payload>,
        action: &str,
        changed: bool,
    ) -> Value {
        json!({
            "ok": true, "action": action, "changed": changed, "prefix": self.prefix,
            "active": self.describe_slot(&receipt.active, payloads),
            "previous": self.describe_slot(&receipt.previous, payloads),
            "retained_snapshots": receipt.managed.len(),
            "cleanup_pending": receipt.active.is_none() && !receipt.managed.is_empty()
        })
    }

    pub fn install(&mut self, package: &Path, payload: Payload) -> Result<Value> {
        ensure!(
            self.receipt.active.is_some() || self.receipt.managed.is_empty(),
            "uninstall cleanup pending; retry uninstall before installing"
        );
        let mut payloads = self.verified()?;
        if self.receipt.active.as_deref() == Some(&payload.hash) {
            return Ok(self.output(&self.receipt, &payloads, "install", false));
        }
        let mut next = self.receipt.clone();
        let destination = self.payload_path(&payload.hash);
        let staged = !next.managed.contains(&payload.hash);
        if staged {
            ensure!(
                next.managed.len() < MAX_SNAPSHOTS,
                "snapshot limit reached; uninstall before reinstalling"
            );
            ensure!(
                !fsutil::exists(&destination)?,
                "unmanaged payload collision"
            );
            let staging = tempfile::Builder::new()
                .prefix(".stage-")
                .tempdir_in(self.prefix.join("payloads"))?;
            payload.copy_to(package, staging.path())?;
            payload.verify_modes(staging.path())?;
            fs::rename(staging.path(), &destination)?;
            next.managed.push(payload.hash.clone());
        }
        next.previous = next.active.take();
        next.active = Some(payload.hash.clone());
        let hash = payload.hash.clone();
        payloads.insert(hash.clone(), payload);
        let output = self.output(&next, &payloads, "install", true);
        if let Err(error) = self.commit(next) {
            // Receipt replacement failed: the old active remains authoritative.
            // Remove only our new, fully validated snapshot, not any old content.
            if staged {
                let _ = payloads[&hash].remove(&destination);
            }
            return Err(error);
        }
        Ok(output)
    }

    pub fn launch(&self, program: launcher::Program, args: &[OsString]) -> Result<Value> {
        let id = self
            .receipt
            .active
            .as_ref()
            .context("no active payload to launch")?;
        let root = self.payload_path(id);
        // Keep the prefix lock through verification and spawn so a cooperating
        // rollback/uninstall cannot change the selected tree mid-launch.
        let payload = Payload::load(&root, id)?;
        payload.verify_modes(&root)?;
        let (path, pid) = launcher::spawn(&root, &payload.manifest, program, args)?;
        Ok(json!({
            "ok": true, "action": "launch", "program": program.name(), "pid": pid,
            "detached": true, "prefix": self.prefix, "path": path,
            "manifest_sha256": id, "version": payload.manifest.version
        }))
    }

    pub fn rollback(&mut self) -> Result<Value> {
        let payloads = self.verified()?;
        ensure!(
            self.receipt.previous.is_some(),
            "no previous payload to roll back to"
        );
        let mut next = self.receipt.clone();
        std::mem::swap(&mut next.active, &mut next.previous);
        let output = self.output(&next, &payloads, "rollback", true);
        self.commit(next)?;
        Ok(output)
    }

    pub fn uninstall(&mut self) -> Result<Value> {
        // Preflight everything before clearing the active slot or removing any file.
        // Unknown files inside a managed snapshot make this fail closed.
        let payloads = self.verified()?;
        let changed = self.receipt.active.is_some() || !self.receipt.managed.is_empty();
        let mut next = self.receipt.clone();
        next.active = None;
        next.previous = None;
        self.commit(next)?;
        let mut warnings = Vec::new();
        let mut retained = Vec::new();
        for id in &self.receipt.managed {
            let path = self.payload_path(id);
            let result = if let Some(payload) = payloads.get(id) {
                payload.remove(&path)
            } else {
                fsutil::exists(&path).and_then(|exists| {
                    if exists {
                        fs::remove_dir(&path).map_err(Into::into)
                    } else {
                        Ok(())
                    }
                })
            };
            if let Err(error) = result {
                warnings.push(format!("cleanup incomplete for {id}: {error}"));
                retained.push(id.clone());
            }
        }
        let mut next = self.receipt.clone();
        next.managed = retained;
        if let Err(error) = self.commit(next) {
            warnings.push(format!("cleanup receipt not compacted: {error}"));
        }
        let mut output = self.output(&self.receipt, &BTreeMap::new(), "uninstall", changed);
        output["cleanup_complete"] = json!(warnings.is_empty());
        output["warnings"] = json!(warnings);
        // Keep the small ownership receipt, payloads directory, and lock. Removing
        // a lock file while another caller holds its inode would break serialization.
        Ok(output)
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;

pub fn absent_status(prefix: &Path, action: &str) -> Value {
    let mut value = json!({"ok": true, "action": action, "changed": false, "prefix": prefix,
        "active": null, "previous": null, "retained_snapshots": 0, "cleanup_pending": false});
    if action == "uninstall" {
        value["cleanup_complete"] = json!(true);
        value["warnings"] = json!([]);
    }
    value
}
