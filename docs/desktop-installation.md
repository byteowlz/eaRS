# Desktop installation contract

The companion and CLI are one versioned desktop payload. Installers must not rely
on a Rust checkout, a sibling repository, Homebrew paths or an operator's shell
initialization. The production ASR server is a separate deployment.

## Safety and compatibility

- Normal companion builds use the accepted parametric renderer; experimental
  overlays require an explicit developer feature. Existing serialized indicator
  values are retained rather than resetting the user's configuration.
- `ears --build-info` and `ears-ui --build-info` report compiled capabilities as
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

## macOS candidate workflow (under qualification)

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

Native Windows/Linux packaging and VM results are tracked in trx-hkxd. Do not infer
release readiness from this contract or from compilation: actual signed-artifact,
fresh-user/upgrade/rollback/uninstall, GUI and runtime evidence is required, with
permission/device and X11/Wayland limitations reported separately.
