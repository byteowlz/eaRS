# Per-user macOS installer

Runtime dependencies: stock macOS Bash/tools, including `plutil`, `codesign`,
`spctl`, `ditto`, `shasum` and `lockf` with descriptor locking. No package-manager
runtime dependency. This script never executes the supplied app, signs anything,
changes quarantine/TCC/login items, kills apps or migrates user data. Quit eaRS first.

## CLI

```sh
bash scripts/release/install-macos.sh --app /release/eaRS.app \
  --app-cdhash TRUSTED_40_HEX \
  --prefix "$HOME/Applications" --state-dir "$HOME/.local/state/ears/install"

bash scripts/release/install-macos.sh --rollback  [--prefix PATH] [--state-dir PATH]
bash scripts/release/install-macos.sh --uninstall [--prefix PATH] [--state-dir PATH]
bash scripts/release/install-macos.sh --recover   [--prefix PATH] [--state-dir PATH]
```

Obtain the CDHash from an authenticated release channel (the parent pipeline's
`artifact.json.app_cdhash`), not by inspecting an untrusted download to decide what
to trust. It is 40 hex characters, normalized to lowercase, for the host-selected
macOS CodeDirectory. Signature verification and Gatekeeper assessment must also pass.
There is no bypass flag or inferred-pin fallback. Rollback/uninstall use recorded
pins and reverify both current app and referenced previous backup.

Duplicate options, conflicting actions, missing values and install-only arguments
on other actions refuse before writes. Help must be used alone. Prefix/state must
be absolute, non-root, disjoint paths, without dot traversal, control characters,
quotes or backslashes; source may be relative but must resolve outside prefix/state.
Final directory symlinks refuse. Normal operations may create missing prefix/state
directories, but help and parser/path errors do not.

Defaults remain `~/Applications` and
`${XDG_STATE_HOME:-$HOME/.local/state}/ears/install`.
Use the **same prefix/state pair** for all operations and recovery. Do not run old
and new installers concurrently or edit state to force adoption.

## Ownership and retained data

`state/receipt.json` is strict `ears.install/v2`, with string fields in this order:
`schema`, `path`, `current_cdhash`, `previous_path`, `previous_cdhash`.
It must have the installer's exact canonical one-line JSON encoding, including LF:
unknown/duplicate fields, wrong types, reformatted/legacy receipts, path mismatches,
symlinks, hardlinks, excessive size, wrong owner or group/world-write bits refuse.
An empty current pin means uninstalled; previous path/hash are both present or empty.
Backup paths are restricted to `prefix/.ears-install.<six-alnum>/old.app`.
State and prefix directories must be owned and not group/world writable.

An absent destination never permits overwriting an unknown receipt. An existing
app without a matching receipt is not adopted. Same bundle ID or a valid signature
alone does not establish ownership: the CDHash must match the recorded trusted pin.
The old pre-public v1 receipts and legacy state/lock entries deliberately fail closed;
there is no silent migration or cleanup.

Updates and uninstall move the former current app into a fresh, retained
`.ears-install.XXXXXX/old.app` under the prefix. Rollback copies the verified previous
app through a new transaction, retaining earlier backups. Uninstall removes the
managed `eaRS.app` destination by retaining it as a backup; it does not reclaim its
bytes. Older backups, uncommitted/rejected app copies, unused metadata temps and
unknown extras are never recursively deleted. External configs/models/history are
untouched. Persistent prefix/state lock files also remain.

## Interruption protocol

Both prefix and state have persistent descriptor locks, acquired in that order.
The kernel releases them after process death; files are never unlinked/truncated.

The staged next receipt is created in the **state directory**, so its replacement
of receipt.json is a same-filesystem atomic rename even across prefix/state volumes.
App staging/backup moves are within the prefix. Before any app rename, an immutable
`ears.install-journal/v1` journal is atomically published in state. It binds old/new
pins, transaction/receipt paths, previous backup and old/next receipt SHA-256 values.

On ordinary errors or INT/TERM/HUP, the exit trap invokes recovery, not recursive
cleanup. After SIGKILL, rerun the same command or `--recover` with the same paths:
- Old receipt digest: verify the saved old app, retain an uncommitted new app as
  rejected.app if needed, then restore the old selection.
- New receipt digest: verify the committed app/backup and finish the commit.
- Unknown receipt/journal/content or collisions: refuse, retaining every app copy
  and the journal for independent inspection.

Only a validated, completed journal is removed. A nonzero exit can still follow
a completed receipt commit; inspect/recover rather than assuming no change.
This is process-interruption recovery, **not a power-loss durability guarantee**:
there are no fsync tests or directory-fsync promises. Errors before journal
publication can leave unreferenced staging/metadata, but not remove the old app.

## Tests and limits

```sh
just --justfile scripts/release/install-macos.just test
# Or:
/bin/bash -n scripts/release/install-macos.sh
bun test scripts/release/install-macos.test.ts
```

Bun/just are development-only. Tests use temporary roots and synthetic metadata.
They cover parser no-write behavior, unsafe paths, strict receipt refusals, unknown
destination/journal preservation, prepared-WAL cancellation, actual descriptor locks,
unsigned same-ID refusal using stock codesign, and test-driver boundary matching.

Signature-sensitive tests explicitly skip unless these fixture inputs are supplied:
`MACOS_INSTALLER_TEST_APP`, `MACOS_INSTALLER_TEST_APP_CDHASH`, and, for distinct
updates, `MACOS_INSTALLER_TEST_APP_B`, `MACOS_INSTALLER_TEST_APP_B_CDHASH`.
Both apps must really pass codesign/spctl with bundle ID dev.byteowlz.ears-ui.
Trusted pins must be supplied independently. Tests never launch or sign the apps.
They exercise lifecycle, wrong pins, same-ID substitutions and ordinary-error/
SIGTERM/SIGKILL cuts around app and receipt renames. Test-only Bash DEBUG drivers
inject interruption; production has no verification override or fault flag.

This is not a hostile same-user sandbox: path-based checks/moves remain vulnerable
to noncooperating mutations, ancestors/ACLs are not comprehensively pinned, and
local writable state is not publisher authentication. Changing state-dir after an
interrupted install can strand recovery; always reuse the original pair. Retained
backups can consume unbounded disk space; garbage collection is intentionally absent.
On 2026-10-31 an ARM64 macOS26.6.2 VM ran all **13 signed-fixture tests**:
0 failures,240 assertions, with exact host/guest script and test hashes matched.
Fixtures were independently notarized app revisions `d468142f` and `deb0ca24`.
This includes corruption/substitution refusal, data retention and the documented
rename interruption cases. A separate real idle-setup app check proved running-app
refusal (exit4) without changing its receipt or user-data sentinels. No test granted
capture permissions or loaded models. Hash-bound raw evidence remains private.

Distinct volumes, physical power loss, minimum-macOS14 GUI, other Gatekeeper/network
conditions and the newly added canonical build pipeline remain unqualified.
These results do not establish live speech/input/accessibility acceptance.
