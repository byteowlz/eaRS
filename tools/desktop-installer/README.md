# eaRS desktop payload installer

Independent Rust crate for per-user unpacked Linux/Windows payloads.
No root workspace integration, native macOS packaging, archive extraction, process
termination, services, shortcuts, data migration, or signing. Install and validation
never execute payloads; only the explicit, whitelisted launch command can do so.

## CLI

```sh
cargo run --manifest-path tools/desktop-installer/Cargo.toml -- install \
  --package /path/to/unpacked-payload --manifest-sha256 TRUSTED_LOWERCASE_SHA256
# Same binary:
ears-desktop-installer status [--prefix PATH]
ears-desktop-installer rollback [--prefix PATH]
ears-desktop-installer uninstall [--prefix PATH]
ears-desktop-installer --prefix /trusted/ears launch ears-ui -- [literal arguments]
```

Install also accepts `--prefix PATH`; every command accepts `--config FILE`.
Obtain the expected SHA-256 through a trusted release channel, not from the
untrusted package itself. It covers the **exact bytes** of `manifest.json`.
Runtime commands and errors emit one JSON object on stdout; exit codes are 0
(success), 1 (operation rejected), or 2 (CLI syntax). Help/version are plain text.

Defaults: Linux `~/.local/opt/ears`; Windows `LOCALAPPDATA/Programs/eaRS`.
macOS has no default: explicit-prefix runs support synthetic contract tests only;
the native .app/DMG pipeline is outside this tool.

## Explicit launch

`launch PROGRAM -- ARGS...` allows only `ears-ui`, `ears`, `ears-dictation`.
It reads the active receipt under the prefix lock and rechecks the active manifest,
full inventory, hashes, host identity and executable modes. It requires an exact
`bin/PROGRAM` entry (`bin/PROGRAM.exe` on Windows) with `executable: true`.
Missing/no-active/corrupt/unlisted targets refuse; no PATH search or shell is used.
Forwarded arguments must follow `--`; spaces, quotes and option-like values remain
literal argv. Receipt/selection and user config/data are not changed by the launcher.

All three programs detach by default: Unix new session, Windows detached process
and new process group with an explicit NUL-only Win32 handle-inheritance allowlist,
null stdin/stdout/stderr, working directory set to the
selected payload root. No pipes, waiting launcher, or inherited terminal resources.
JSON returns `action: "launch"`, `program`, `pid`, `detached: true`, `path`,
`manifest_sha256`, `version`, `prefix`, `ok: true`. A PID acknowledges spawn,
not application readiness or successful exit; each launch can create another process.
Installer-only first-run config initialization still follows the configuration rules.

Future shortcut callers should invoke a stable, trusted **absolute installer path**
outside snapshot directories with an explicit **trusted absolute --prefix**, not a
snapshot executable. Bare launch
uses normal config/env/local-prefix priority, which can redirect selection from an
untrusted working directory. This tool does not create OS shortcuts or services.

## Manifest

See `schemas/manifest.schema.json`: `schema = ears.desktop-payload/v1`, exact Rust
host OS/architecture names (for example `linux`, `windows`, `x86_64`, `aarch64`),
nonempty `version` and `source_commit`, `client_vad: true`, and `files` entries
with `path`, lowercase `sha256`, and boolean `executable`.
The manifest itself is not listed in `files`.

The runtime validator adds exact inventory and host checks to the JSON Schema:
1–1024 files, at most 2 GiB including manifest bytes, a 1 MiB manifest limit,
portable ASCII relative paths (16 components, 128 bytes/component, 1024 total),
no aliases/duplicates, traversal, absolute/backslash/control/device paths,
symlinks/reparse points, special files, or unlisted files/directories.
Hashes stream through a 64 KiB buffer and are rechecked during staging.

## Transactions and ownership

`receipt.json` atomically selects logical `active`/`previous` slots, each a trusted
manifest hash. Their actual directories are `payloads/<hash>` inside the prefix;
there are deliberately no physical `active`/`previous` links or multi-rename swap.
For inspection, `status.active.path` locates the selected payload. Use `launch`,
not that snapshot path, for stable shortcut targets.
See `schemas/receipt.schema.json`.

An exclusive prefix lock serializes cooperating callers. Each install validates
the source, copies to private same-filesystem staging, verifies the copy, renames
it to its immutable hash directory, then atomically replaces the receipt.
A failed precommit operation keeps the old selection; repeating the same manifest
is a no-op. Rollback verifies installed hashes before atomically swapping slots.

All previously installed snapshots remain receipt-tracked until uninstall
(maximum 128; further updates refuse). This avoids deleting possibly open Windows
executables during an update. Uninstall preflights every managed snapshot, clears
the selection atomically, then removes only listed files and empty directories.
Cleanup failures return `cleanup_complete: false` with warnings; retry uninstall.
Interrupted cleanup tolerates missing listed files only after both slots are clear,
still hashes every remaining file, and refuses unknown extras.

Unmanaged prefixes or payload-directory collisions refuse adoption. Unrelated
top-level prefix entries and all external config/models/history survive.
Unknown files inside a snapshot cause a refusal without deletion. A tiny empty
ownership receipt, payloads directory, and lock remain after uninstall, preserving
serialization and allowing reinstall. No installed directory is recursively deleted.
Unix executable flags create new files as 0755/0644, subject to umask, never copying
suid/sgid/sticky or group/world-write bits. Existing permissions are never changed.

## Configuration

Normal first use creates `config.toml` and `config.schema.json` under
`$XDG_CONFIG_HOME/ears-installer` or `~/.config/ears-installer`. The commented
default resolves the platform prefix at runtime. Help/version do not create files.
Optional TOML setting: `prefix = "/custom/ears"`.
Priority: CLI `--prefix` > explicit `--config` > `EARS_INSTALLER_PREFIX` >
`./ears-installer.toml` > global `config.toml` > platform default.

## Verification and limits

From this directory: `just check`, `just test`, `just fmt`, `just lint`.
Tests use temporary roots and synthetic bytes, not native eaRS executables.
A harmless native Rust helper, when rustc is available, tests explicit launch,
rollback selection, literal argv, PID, detached stdio/session and released locks.
If rustc is absent the test prints UNTESTED; use `cargo test -- --nocapture` to see
execution coverage. Helper compile errors fail the test. These checks do not prove
native eaRS startup, client VAD, signing, desktop registration or .app installation.

Native Windows11 proof on 2026-10-31: 27 synthetic tests pass, including the unchanged
previously failing detached-helper test and a new non-stdio inheritable-handle/
UTF-16/error-cleanup regression. Independent capture EOF moved from waiting for the
helper's10-second exit to about81ms while the helper stayed alive. Null stdio and
console-detach flags alone were insufficient; a narrow STARTUPINFOEXW allowlist fixes
that boundary without mutating parent standard handles or enabling privileges.
On macOS, run this native-helper suite serially with the macOS installer tests:
the fixture name `ears-ui` intentionally triggers their running-app guard.

Requires trusted per-user prefix parents and no concurrent mutation of package or
managed trees by noncooperating writers; it is not a hostile same-user sandbox.
Atomic receipt replacement is not a power-loss durability guarantee. A crash
between staging rename and receipt commit may leave an untracked snapshot/stage;
the next invocation refuses that collision rather than deleting or adopting it.
Remove such leftovers only after independent inspection. Never edit an active
receipt to bypass validation. Distribution, native eaRS execution tests, and crash/
power-loss testing remain parent-owned.

Concrete review concerns (not a complete security audit):
- Tree checks and execution/removal use paths, not retained directory/file handles.
  A noncooperating writer can race parent components or the final executable after
  hashing; the lock only serializes installer callers. Owner/ACL and writable
  ancestor-directory checks are not enforced; executable-mode checks also do not
  reject group/world-write bits added after installation. Writable local receipts
  are ownership state, not publisher signatures or an execution sandbox.
- Directory entries are not fsynced; crash during first-prefix initialization or
  snapshot-rename/receipt-commit can leave fail-closed unmanaged leftovers.
- Config files are parsed even below a CLI override. The local TOML source is not
  no-follow/bounded like payload reads, and first-run config writes are not atomic.
  Untrusted cwd config can redirect a bare launch or cause errors/blocking reads.
- Uninstall clears slots before cleanup. Late I/O/open-file errors mean partial
  removal, explicit warnings and retry, not atomic physical deletion. After launch
  the child does not hold the installer lock; a later uninstall may remove its
  runtime files on Unix or fail to delete open Windows files.
- Inherited environment/dynamic-loader behavior and application behavior are not
  sandboxed. A returned PID is not health/readiness proof; retries are not idempotent.
