# Desktop installation contract

The companion and CLI are one versioned desktop payload. Installers must not rely
on a Rust checkout, a sibling repository, Homebrew paths or an operator's shell
initialization. The production ASR server is a separate deployment.

## Safety and compatibility

- Normal companion builds use the accepted parametric renderer; experimental
  overlays require an explicit developer feature. Existing serialized indicator
  values are retained rather than resetting the user's configuration.
- `ears --build-info`, `ears-dictation --build-info` and `ears-ui --build-info` report compiled capabilities as
  JSON without configuration reads, microphone access or model/runtime loading.
  Packages require client-VAD capability, matching architecture and source commit.
- Finder and desktop shortcuts use the same home-based `.config/ears` convention
  as shell launches, unless `XDG_CONFIG_HOME` overrides it. Existing native-OS
  configuration directories remain readable when no preferred directory exists;
  this is compatibility, not an implicit data migration.
- Fresh companion startup creates default configuration, opens existing Settings
  and stays in setup mode. It does not start the engine, microphone or wake listener.
  Closing/reopening does not finish setup: the pending marker persists until Apply.
  `ears-ui --setup` opens setup deliberately on an existing installation too.
- Synthetic previews never start the engine, including when their controls are
  exercised. Models/voices/private configuration are not copied into release assets.
- Updating or uninstalling must preserve settings, models, identities and history.
  Running processes must be closed explicitly; installers do not kill unrelated apps.
  Package verification precedes replacement; failed updates retain the previous app.

## Repeatable source builds

Build prerequisites: Rust1.96+, Bun1.4.4, Git, native C/C++ tools/SDK and the listed
Linux dependencies; `just` is a command wrapper. End-user installs need none of
Rust, Bun, a checkout or sibling repositories.

Use `just desktop-build <full-40-character-commit> /absolute/new/output [1..4 jobs]`.
The builder archives that commit, checks the accepted renderer hash, builds only
`ears`, `ears-dictation` and `ears-ui` with client-VAD (plus the standalone installer
on Linux/Windows), and records binary/source hashes, capabilities and tool versions.
It never builds from dirty/shared Cargo edits or installs/starts the application.
The output directory must be new. This is a repeatable procedure, **not** a claim
of bit-identical builds: toolchains, SDKs, system packages and signing timestamps
still matter. A capability string alone is not source attestation.

`just desktop-policy-test` runs packaging policy and synthetic installer tests.
Signed-app and platform-native tests must report UNTESTED when their fixtures are
absent; a successful compile or skipped test is not installation qualification.

Fetch a checksum-pinned public runtime with
`just desktop-runtime macos_arm64|linux_x64|windows_x64 /absolute/cache`.
Pins/profiles are in `packaging/desktop.toml`; mismatch or incomplete downloads fail.
Models, voices, private configuration and recordings are never release inputs.

`.github/workflows/desktop-candidates.yml` is manual, read-only and bounded to two
parallel native builds. It uploads **unsigned/unqualified build candidates**, not
a public release. It does not import signing credentials, create tags, dispatch tap
updates or grant OS permissions. The existing tag workflow is CLI/server-only;
it does not release this desktop application. Do not publish desktop assets until
the exact artifact/revision passes the qualification matrix below.

## macOS signing and notarization

`scripts/release/macos.ts` assembles a normal application bundle from a release
binary directory and qualified upstream ONNX Runtime, signs nested code with
Developer ID/hardened runtime, submits to Apple with a locally stored notary profile,
staples the app and DMG and writes an external artifact checksum record. Signing
identities and credentials are never checked into the repository.

The package contains no private models, configuration or recordings. Licensing
notices are resources in the bundle, not on-screen credits. No launch agents,
security-setting changes, automatic permission grants or capture occur at install.

`scripts/release/install-macos.sh` supplies a non-interactive per-user install,
rollback and uninstall equivalent to dragging the verified app into Applications.
It refuses unmanaged replacements, keeps a previous version and preserves all data.
An application CDHash pin must come from trusted release metadata, **not** by querying
an untrusted supplied app. Notarization alone does not authorize any app using our
bundle identifier. Rollback/uninstall must also match the managed receipt.

```sh
just desktop-macos-package /build/bin /cache/runtime.tgz FULL_COMMIT /new/package LOCAL_KEYCHAIN_PROFILE
# Verify DMG SHA256 against a trusted release channel before mounting it.
just desktop-macos-install /mounted/eaRS.app TRUSTED_APP_CDHASH
just desktop-macos-rollback
just desktop-macos-uninstall
```

Developer ID Application credentials and the notary profile stay in Keychain.
No Developer ID Installer certificate is needed for this app/DMG workflow.
Minimum macOS14 derives from actual Mach-O load commands, including the bundled
runtime. This minimum declaration does not establish GUI acceptance on every OS14+.
Normal downloaded-app confirmation remains intact; never strip quarantine or grant
TCC automatically. Runtime qualification uses the installed, signed program:

```sh
~/Applications/eaRS.app/Contents/MacOS/ears --runtime-check ~/Applications/eaRS.app/Contents/Frameworks/libonnxruntime.dylib
```

This loads the library only: no inference session/model/microphone. Model selection,
VAD enablement and permissions remain explicit setup. Font/runtime licences ship as
resource text files, not on-screen credits.

## Linux and Windows candidates

`bun scripts/release/portable.ts --bin-dir /build/bin --ort-archive /cache/runtime --source-commit FULL_COMMIT --out /new/package`
produces a native-only candidate with a strict hash manifest and all three clients.
Windows additionally requires `--vc-runtime-dir` and `--vc-license` containing
operator-approved redistributable VC143 CRT files/licence. The CRT is app-local for
both the clients and installer; a preinstalled development VM runtime is not proof
of prerequisite packaging. No CRT download, elevation or security-policy change is
implicit. Linux targets the Ubuntu22.04/glibc2.35 baseline; Windows targets Windows11.
Both still need real desktop/package qualification. Intel macOS/other CPU profiles
are not silently advertised as tested. CI and the optional Ubuntu22.04 Rust builder
share `packaging/linux/build-deps.txt`; the builder context is `packaging/linux`,
not a dirty home/checkout. It supplies Rust/system libraries, not Bun or a desktop
session. Use the documented Bun/Git build prerequisites for the orchestration CLI.
Floating distro packages mean dependency versions must still be recorded for QA.

The standalone `tools/desktop-installer` has its own workspace/lock/schema and emits
JSON for install, status, explicit whitelisted launch, rollback and uninstall:

```sh
/path/to/ears-desktop-installer --prefix /absolute/per-user/ears install --package /extracted/payload --manifest-sha256 TRUSTED_SHA256
/path/to/ears-desktop-installer --prefix /absolute/per-user/ears status
/path/to/ears-desktop-installer --prefix /absolute/per-user/ears launch ears-ui -- --setup
/path/to/ears-desktop-installer --prefix /absolute/per-user/ears rollback
/path/to/ears-desktop-installer --prefix /absolute/per-user/ears uninstall
```

On Windows use the native `.exe`. Verify the **whole archive/installer** before
executing it; manifest integrity does not authenticate the installer itself. Keep
that executable at a stable absolute path outside managed payload snapshots and
pin `--prefix` in launch invocations. OS shortcut registration is not implemented.
Installation/status/rollback/uninstall never execute payloads. Launch PID means
spawned, not healthy or exactly once; check runtime/GUI readiness separately. These
are trusted per-user directories, not a sandbox against same-user hostile writers.
Uninstall's late file-sharing failures are explicit and retryable, not hidden.
See `tools/desktop-installer/README.md` for limits and configuration priority.

## Qualification matrix (required for each exact artifact)

Record immutable source and artifact hashes, OS/toolchain, inputs and results for:

1. Strict integrity/signature/notarization/architecture/runtime prerequisite checks.
2. Fresh per-user install without shell initialization, capture or private inputs.
3. Normal first-open confirmation; visible Settings with complete control labels.
4. Setup stays idle across relaunch until explicit Apply; permission state unchanged.
5. Repeat install, version upgrade and verified rollback, preserving byte-hashed
   settings/model/history/identity sentinels. No running-app replacement.
6. Reject corrupt payloads, unmanaged collisions, malformed receipts and unsafe paths;
   interruption/failure recovery must not delete the only managed app copy.
7. Uninstall/retry retains user data and refuses deleting unknown content.
8. Actual desktop render/scale/accessibility/input and live wake/speech/trailing text
   are **separate acceptance**, never inferred from compiler or synthetic checks.

Private logs/screenshots stay local; publish only synthetic/paraphrased results.
Native platform findings and blockers are tracked in trx-hkxd. Do not infer
release readiness from this contract or from compilation: actual signed-artifact,
fresh-user/upgrade/rollback/uninstall, GUI and runtime evidence is required, with
permission/device and X11/Wayland limitations reported separately.
