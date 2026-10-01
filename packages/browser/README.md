# eaRS browser recognition

Standalone recognition for callers that own their microphone and UI. No Foxline,
Brain, desktop injection, command execution or implicit provider fallback.

```ts
import { Recognizer } from "@byteowlz/ears-browser";

const recognizer = new Recognizer();
const unsubscribe = recognizer.subscribe(event => {
  // Replace a preview by event.id; only completed events are committed text.
  console.log(event);
});
await recognizer.prepare({
  provider: "moonshine", language: "de", sampleRate: 48000,
  runtimeBaseUrl: "/speech/moonshine-0.1.5/",
});
recognizer.feed(monoFloat32Pcm);
await recognizer.finish(); // drains inference, then releases the worker
recognizer.dispose();
unsubscribe();
```

`prepare` selects English or German explicitly and starts one recognition session.
`feed` accepts finite normalized mono Float32 PCM at the declared 8–96 kHz rate.
`finish` drains once; `cancel` drops provisional text and fences late results.
Changing a provider/language requires a new preparation. Failures require an
explicit retry. Events are state, byte progress, replacement preview, completed,
optional speech boundary and error. IDs are opaque strings scoped to preparation;
never parse them as numbers. Only completed events may populate committed text.
Synthesis and conversational turn authority are outside this package.

## Local runtime deployment

The runtime is pinned to Moonshine's GitHub **v0.1.5 WASM release**, including its
archive and WASM SHA-256 digests. npm's same-version package embeds an older core
which rejects these split-frontends and lacks German catalog support. Do not
substitute it. Stage the verified ESM/WASM/worker tree on the application origin:

```js
import { stageMoonshineRuntime } from "@byteowlz/ears-browser/stage-runtime";
await stageMoonshineRuntime("public/speech/moonshine-0.1.5");
```

Asset staging downloads approximately 6.6 MB once from GitHub and verifies its
SHA-256 before extraction. The verified archive is cached under
`$XDG_CACHE_HOME/ears-browser` (or `~/.cache/ears-browser`) for offline rebuilds.
A changed release artifact fails verification rather than silently changing code.

The published runtime is threaded CPU/WASM (not WebGPU). HTTPS/localhost, Worker,
WebAssembly SIMD and **cross-origin isolation** are required. Send these headers
on the application and runtime assets:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Do not enable isolation blindly in an application that embeds external content:
it affects cross-origin frames/resources and opener relationships. Check those
integrations or deploy a compatible single-thread Moonshine build separately;
the published package does not ship that fallback. Without isolation preparation
fails explicitly; there is no audio-upload fallback. CSP must permit the hosted
module/pthread workers, WASM compilation, the capture worklet (a Blob in Oqto),
and model downloads. The runtime modules are served intact, not rebundled.

Pinned streaming Tiny model directories: English `quantized_26_08_21`, German
`quantized_26_08_24`. The eight files total approximately 45 MB English / 32 MB
German, plus approximately 13 MB runtime WASM. German Small is deliberately not
the default. Models download via GET from `download.moonshine.ai`, with upstream
Cache API caching (`moonshine-models-v1`). Microphone audio stays in a local
worker. Use `modelBaseUrl` for an equivalent self-hosted model directory (absolute
URL, trailing slash). Runtime caching uses normal HTTP caching; model downloads
are optional external network traffic, not speech uploads. Warm-cache offline
operation depends on the host caching the runtime and on browser storage retention.

## Explicit remote recognition

```ts
await recognizer.prepare({
  provider: "ears", language: "en", sampleRate: 48000,
  url: "wss://example.test/recognize",
});
```

This is explicit consent to stream audio to that endpoint. The adapter requests
PCM, boundary VAD and language confirmation before accepting audio. It resamples
to mono 24 kHz little-endian float32. The local path resamples to 16 kHz. The
streaming area-average filter preserves fractional bins across chunks; it is an
ASR-oriented filter, not studio-quality reconstruction.

`Interim` replaces the preview; `Word` builds an append-only preview when no
interim is present. `Speech(active:false)` commits a flushed segment. `Final`
finishes the stream; in boundary mode its cumulative word list is indexed after
already completed words so a teardown summary does not insert them again.
`Pause` and client-side silence timers do not commit previews. Older servers
without boundaries only complete on `Final`. Unsupported language, queue overflow,
disconnection or missing final are errors, never silently dropped audio or a
local/server switch. Authentication is supplied by the caller's endpoint URL;
do not log credential-bearing URLs.

## Development

Use Bun: `bun install`, then `just check`, `just test`, `just build`.
For real CPU/WASM streaming proof, install `agent-browser` and run
`just browser-smoke /path/to/english.wav /path/to/german.wav`. The harness
paces public/consented fixtures, checks non-empty exactly-once completions,
and forbids WebSockets/non-GET fetches in the page and STT worker. Audio
fixtures, browser state and runtime binaries stay outside git.
The package exports TypeScript source for Bun/Vite consumers; `dist/index.js` is
an independently usable browser ESM build. Keep generated/runtime assets and
private evaluation audio outside git. Oqto vendors this small source package for
checkout-independent builds; sync it explicitly after SDK updates, never fork it.
