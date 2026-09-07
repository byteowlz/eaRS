# Changelog

All notable changes to eaRS are documented in this file.

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
