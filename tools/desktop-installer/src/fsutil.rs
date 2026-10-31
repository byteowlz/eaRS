use anyhow::{bail, ensure, Context, Result};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

pub fn metadata(path: &Path) -> Result<Metadata> {
    let meta = fs::symlink_metadata(path).context("reading filesystem entry")?;
    ensure!(!meta.file_type().is_symlink(), "symlinks are forbidden");
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            meta.file_attributes() & 0x400 == 0,
            "reparse points are forbidden"
        );
    }
    ensure!(
        meta.is_file() || meta.is_dir(),
        "special files are forbidden"
    );
    Ok(meta)
}

pub fn directory(path: &Path) -> Result<()> {
    ensure!(metadata(path)?.is_dir(), "expected a directory");
    Ok(())
}

pub fn exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

// No-follow on the final component; tree traversal separately checks every directory.
// The per-user source and destination must not be concurrently mutated by another writer.
pub fn open_regular(path: &Path) -> Result<File> {
    ensure!(metadata(path)?.is_file(), "expected a regular file");
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options.open(path)?;
    ensure!(file.metadata()?.is_file(), "expected a regular file");
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            file.metadata()?.file_attributes() & 0x400 == 0,
            "reparse point"
        );
    }
    Ok(file)
}

pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    open_regular(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "metadata file exceeds size limit"
    );
    Ok(bytes)
}

pub fn create_payload_file(path: &Path, executable: bool) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if executable { 0o755 } else { 0o644 });
    }
    #[cfg(not(unix))]
    let _ = executable;
    Ok(options.open(path)?)
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("missing parent")?;
    if exists(path)? {
        ensure!(metadata(path)?.is_file(), "metadata collision");
    }
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    // tempfile uses atomic replacement on Unix and Windows. Never delete the old
    // receipt first: it is the sole commit point for the logical active/previous slots.
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

pub fn ensure_new_or_empty(path: &Path) -> Result<()> {
    if exists(path)? {
        directory(path)?;
        if fs::read_dir(path)?.next().is_some() {
            bail!("refusing unmanaged prefix");
        }
    } else {
        fs::create_dir_all(path)?;
        directory(path)?;
    }
    Ok(())
}
