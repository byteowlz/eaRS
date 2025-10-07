# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **ears-auto**: Removed VAD auto-timeout (runs indefinitely until manually stopped or paused)
- **ears-auto**: Dictation now types each word immediately on arrival (not batched at Final)
- **ears-ctl**: Added connection timeout (2s) and response timeout (800ms) to prevent hangs
- **ears-ctl**: Toggle command now falls back gracefully if status query times out
- **ears-auto**: Moved hotkey listener to dedicated OS thread for improved macOS compatibility
- **WebSocket**: Unified schema to tagged format `{ "type": "...", ... }` for all messages
- Migrated from cargo-dist to semantic-release + GoReleaser workflow
- Added automated semantic versioning based on conventional commits

### Fixed
- ears-ctl no longer hangs indefinitely when server is unreachable
- Hotkeys now work reliably on macOS with proper thread handling
- Dictation types immediately instead of batching text until session ends

### Documentation
- Updated README.md with macOS permission requirements (Accessibility + Input Monitoring)
- Updated WEBSOCKET.md with correct tagged message schema and Status example
- Added troubleshooting section covering common issues (hangs, hotkeys, dictation)
- Clarified VAD timeout behavior in config and docs