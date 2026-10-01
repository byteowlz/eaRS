# Browser recognition SDK

The standalone implementation and deployment contract live in
[`packages/browser`](../packages/browser/README.md). It recognizes English/German
locally with Moonshine CPU/WASM or streams to an explicitly selected eaRS server.
It does not depend on Foxline, capture microphones, inject text or execute commands.
Tracked as `trx-2kmg`; Oqto integration is `oqto-d0pc` with ADR 0051 there.

## Verified runtime/model pairing

A real isolated Chromium probe found that npm's `@moonshine-ai/moonshine-wasm`
0.1.5 binary does not match the same-version GitHub release. The npm binary rejects
the August split-frontends with `Invalid argument`; its catalog also lacks German.
Its WASM SHA-256 is
`d0e783f661f8be68f7af48d859d3bbc7a6381a980b35b4128fc36344218a6682`.

The SDK's stage script pins the working GitHub v0.1.5 artifact:

- Archive SHA-256:
  `c515bf7691e12048f70a92cc82b3b0894c16c3773ffcacb7d48944fb150e4837`.
- WASM SHA-256:
  `22fca4a5b2dc50fe36dc68e4d25fab73b0250b71f744d9fea511c3a308b2420a`.
- English Tiny streaming: `quantized_26_08_21`; German Tiny streaming:
  `quantized_26_08_24`.

This is a load-time compatibility finding, not an ASR accuracy benchmark. Never
substitute a same-version binary without testing the actual language/model pair.
No native Rust engine/default was changed.

## Proof

`bun install`, `just check`, `just test` (14 tests), and `just build` pass inside
`packages/browser`. The SDK build is approximately 15 KB, excluding the staged
13 MB WASM and model downloads. Tests cover explicit language confirmation,
replacement previews, boundary/cumulative finals, finish timeout, queue overflow,
resampling chunk invariance, cancellation during preparation and late-event fences.

The reproducible `just browser-smoke <english.wav> <german.wav>` harness uses
`agent-browser` and external public/consented WAV files. It paces audio into the
real off-thread runtime, checks unique non-empty completions and installs hard
WebSocket/non-GET-fetch guards in the page and STT worker. Page request capture
also showed GET-only traffic. Runtime binaries, fixtures and browser state are
not checked into the repository.

On this Apple Silicon host, a Chromium run on 2026-10-01 produced:

| Fixture | Duration | Completed recognition |
| --- | ---: | --- |
| LibriSpeech `1272-128104-0001` | 4.815 s | Nor is Mr. Crawter's manner less interesting than his matter. |
| FLEURS German `1738` | 11.16 s | Er griff auch alles an, was ins Wasser kam. Selbst ein großer Dinosaurier wie der T-Rex war ihm nicht gewachsen. |

English emitted five preview revisions and one completion; German emitted eight
previews and two completions with distinct opaque decimal IDs. Preparation was
approximately 4.3 seconds for each language in that run; cache/network conditions
are not controlled, so these are smoke-test timings, not cold-start benchmarks.
The English proper noun is imperfect; no general accuracy claim is implied.

Cross-origin isolation/HTTPS is required for the pinned threaded release. The
SDK has no server fallback and no bundled single-thread fallback. Physical
microphones, Safari/mobile runtime performance, mixed-language speech, offline
cache retention and a real remote eaRS server session remain unverified. Remote
behavior currently has source-contract review and fixture regression tests only.
