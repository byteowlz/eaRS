# Changelog

All notable changes to eaRS are documented in this file.

## [Unreleased]

### Removed

- **engines**: remove the legacy prkt-based `parakeet` engine, vendored prkt crate, legacy model/device/chunk flags and `parakeet-*` acceleration features. Nemotron (`parakeet-rs`) and its model/language flags are unchanged; use `apple,parakeet-rs` or `nvidia,parakeet-rs` rather than the removed feature aliases. Installers, benchmark clients and older app selectors now request the supported engine.

### Fixes

- **wake speaker annotation**: invoke the documented voqx command with only the WAV positional argument, keep the expected identity local, and parse complete multiline JSON. Named matches require a confident status and matching identity; missing evidence or low confidence stays unknown. Bound both output streams and the full call deadline without blocking capture/insertion; errors are logged without raw provider text. Filtering remains unimplemented. Includes an opt-in actual-CLI contract test using an isolated generated-signal registry.
- **ui**: activate the app before opening Settings from the notch to avoid a macOS GPUI focus-callback deadlock.
- **ui**: add outward notch throats and move the indicator toward the outer wing; remove macOS popup chrome from floating overlays and keep long settings hints from clipping controls. Reduced Motion is still honored by default, with an explicit per-app animation override.
- **ui**: attach the macOS notch panel to the screen edge and keep its active width stable as partial transcripts change. A synthetic `--preview-placement` fixture exercises the real screen anchor without starting dictation.
- **dictation**: on macOS, listen to physical hotkey events without rdev's HIToolbox text translation, which can trap on the hotkey thread. Input Monitoring remains required.
- **installation**: activate the exact ONNX Runtime Rust binding pin for `parakeet-rs` and use `--locked` in installers. Fresh `cargo install` no longer selects incompatible `ort` rc.15 (`Session::builder` E0061) on macOS or Linux.

### Features

- **ui**: optional Oqtopus vector indicator using a locally supplied pack and a shared motion runtime also used by procedural Ears bars. Includes a no-dictation pack check and preview; existing indicators and reduced-motion preferences remain available. Legacy Oqtopus packs remain supported.
- **engines**: optional native `moonshine` engine with a public German Tiny base cached under `HF_HOME/moonshine`, verified against pinned SHA-256 hashes, and `--moonshine-model` / `[moonshine].model_dir` for private native ORT fine-tune bundles. Explicit local paths never download or silently fall back. Tentative lines are previews; finalized phrases are safe for typing. Included by `install-all` on Apple Silicon and supported Linux hosts.
- **dictation**: opt-in wake phrases (`[dictation.wake]`, off by default). eaRS stays the only microphone owner and, while dictation is paused, streams 16 kHz audio to any detector process speaking `wake-stream/1` (binary PCM frames in, JSON-line detections out; reference implementation `wake listen --manifest <release>` in the `wake` repository). Phrases map to actions in config (`start_dictation` or `ignore`), so models are swapped by changing the command, not eaRS. The reference detector is `waqy listen` from the [waqy](https://github.com/byteowlz/waqy) repository (Apache-2.0); any program speaking `wake-stream/1` works. A detection does what the toggle hotkey does and nothing more; an unavailable detector is reported while dictation keeps working. See `docs/wake.md`.
- **dictation**: wake follow-ups: pre-roll after a wake (`preroll_ms`, default 1.5 s) so the first words after the phrase are kept, with the wake phrase stripped once (`strip_words`); configurable whole-utterance `stop_phrases` (e.g. "oqto stopp") that pause capture without typing, held back like the `eaRS` prefix and punctuation-insensitive; a local wake log of detections and near misses (no audio) plus one console line per wake.
- **dictation**: opt-in local audio history (`[dictation.audio_history]`, off by default): one private WAV plus metadata per committed utterance, excluded apps and macOS Secure Event Input pause it, size/age limits evict oldest first, and `ears dictation history path|list|export|purge` manage it. Export writes a manifest with session-disjoint split hints; nothing is marked training-eligible.
- **speaker gate**: add bounded completed-audio ASR/evidence joining and single-use, scoped instruction dispatch with synthetic regression proof. The optional prepared-voqx adapter is evidence-only; enabled legacy WebSocket consumers fail closed because exact audio/final binding is unsupported. Ordinary disabled dictation is unchanged. See `docs/speaker-gate.md`; no live-filter or robot-readiness claim.
- **dictation**: add opt-in client-side deterministic command-aware and command-only modes with exact reserved `eaRS` controls, VAD-bounded literal-text commands, fail-closed TOML registry and word-by-word provisional island preview. Normal dictation remains streaming; no speculative keystroke retraction. See `docs/voice-commands.md` for current adapter limits.
- **evaluation**: run pinned local Kev-0.5B offline against synthetic completed utterances with a reproducible shadow runner; results in `docs/kev-shadow.md`. Kev is not in the production command path.
- **ui**: add `ears-ui` (apps/ears-ui), a GPUI companion UI: floating
  dictation island (non-activating, all-spaces panel with live partials,
  morph animation, island/notch mode, drag + click-through), tray/menu-bar
  presence with launch-at-login, and a settings window (backend switcher,
  dictionaries with live test, context-profile editor, overlay preferences).
  Supersedes the deprecated Tauri `apps/simple_gui`.
- **dictation**: extract the headless engine from the `ears-dictation`
  binary into `ears::dictation` (library) with typed commands/events so
  front ends can embed it; the binary is now a thin wrapper.
- **dictation**: runtime-switchable hotkey mode and escape-cancels
  (`SetHotkeyMode` / `SetEscapeCancels` commands).
- **profiles**: context profiles (`ears profile add|list|remove|test`)
  matching the frontmost application (bundle id / binary / window-title
  regex) to dictionary sets, language and insertion mode.
- **dictation**: insertion modes `clipboard` and `send_as_prompt` alongside
  cursor insertion; `PromptReady` event for agent front ends.
- **dictation**: backend identity + latency events; optional live text
  observation for UIs.

## [0.6.0] - 2026-09-07

### Features

- **server**: add transcribe.cpp GGUF streaming engine
- **server**: auto-download transcribe.cpp models by slug
- **server**: engine-agnostic boundary VAD emits Speech events at audio ingress
- **server**: additive `Speech { active }` VAD boundary message
- **server**: per-connection `SetBoundaryVad` command to opt out of Speech events
- **server**: `--no-boundary-vad` flag to disable the ingress boundary VAD
- **cli**: externalize model catalog to JSON with auto-refresh
- **cli**: add `ears models list` and fzf-based `models pull`
- **dictation**: flush held words on speech boundary, shorten phrase hold to 700 ms
- **dictation**: support optional Opus transport
- **dictation**: configurable boundary VAD hangover
- **protocol**: optional Opus audio transport via `setcodec` command
- **parakeet-rs**: auto-download missing Nemotron model files from Hugging Face

### Bug Fixes

- **build**: support CUDA 13.3 toolkits and resolve cblas link failure on Arch
- **cuda**: propagate NCCL when static ggml-cuda detects it
- **dictation**: replace phrases across stream events
- **dictation**: fix single-letter replacement delay
- **transcribe-cpp**: stream interim text and restart at utterance boundaries
- **transcribe-cpp**: resolve short language codes to model locales
- **parakeet-rs**: retain short utterances across VAD boundaries
- **parakeet-rs**: gate model input during long silence to protect utterance onsets
- **parakeet-rs**: resolve hf_hub symlinks before placing model files

### Documentation

- Updated config example

### Removed

- sherpa-onnx engine
- whisper post-processing feature

## [0.5.0] - 2026-07-11

### Features

- **sherpa**: add multilingual streaming engine for CPU
- **parakeet**: add streaming dedup, buffer trim fix, test harness, and hub dataset
- **dictation**: add dictionary replacement with automatic word replacement
- **dictation**: add dictionary `add`/`remove` commands
- **dictation**: add hotkey modes (toggle/push_to_talk/hybrid) and `start_paused`
- **dictation**: add punctuation spacing and session language switching
- **dictation**: buffer live dictation replacements
- **server**: add restart with start arguments
- **debug**: add WebSocket ingress telemetry for stalls
- **debug**: add opt-in Kyutai backend telemetry log
- **config**: add parakeet-rs defaults

### Bug Fixes

- **kyutai**: recover from silent dictation stalls and capture server logs
- **dictation**: stop via SIGINT first so stop hooks run
- **dictation**: fix dictation stop shutdown and allow hooks to run
- **dictation**: enforce strict dictation startup checks
- **parakeet**: force-advance buffer after 3 consecutive empty chunks
- **parakeet-rs**: flush trailing word after speaking pause
- **parakeet-rs**: buffer partial SentencePiece word deltas
- **parakeet-rs**: dynamically load ORT beside eaRS
- **sherpa**: macOS rpath and dylib install
- **sherpa**: include in install-all and embed `$ORIGIN` rpath
- **sherpa**: expose CLI flags via `ears server start`
- **security**: update dependencies to resolve known vulnerabilities
- **build**: use git dependency for parakeet-rs instead of local path
- **build**: install Opus dependency on macOS
- **release**: set `CMAKE_POLICY_VERSION_MINIMUM` for Opus build on CMake 4
- **release**: pin Darwin builds to macos-14 runner
- **release**: drop Linux prebuilt targets to unblock release
- **release**: vendor OpenSSL on Linux for cross-compiled builds
- **release**: install libdbus-1-dev for Linux builds
- **release**: remove unsupported AUR installer from dist config

### Documentation

- Document dictionary replacement status

[0.6.0]: https://github.com/byteowlz/eaRS/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/byteowlz/eaRS/compare/v0.4.6...v0.5.0
