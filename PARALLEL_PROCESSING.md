# kyutai parallel processing

Here’s a tight, practical path to **parallel (multi-user) streaming STT in pure Rust with Candle** using Kyutai’s DSM stack.

# What to use

- **Rust server (moshi-server)** — production-grade WebSocket server that batches multiple concurrent audio streams on GPU/CPU and runs the Kyutai STT models via **Candle**. This is the reference path to parallel streams. citeturn4search0
- It’s part of the DSM release repo but the **server code lives in `kyutai-labs/moshi`**; install the crate and run it with the provided TOML configs. citeturn4search0turn4search2
- The Rust crates behind the server depend on Candle (`candle-core`, `candle-nn`, `candle-transformers`), i.e. **pure Rust/Candle** inference. citeturn3search3turn3search8

# Why this gets you parallel streams

- Kyutai’s STT is **streaming** and **batchable**: the server aggregates chunks from many clients into a single batch per step, so one model instance serves many users. (E.g., **64 simultaneous** streams at \~3× real-time on an L40S; **hundreds on an H100**.) citeturn4search0turn1search1

# Minimal setup (server side)

1. **Install the server (with CUDA if you have NVIDIA):**

   ```bash
   cargo install --features cuda moshi-server
   ```

   citeturn4search0
2. **Pick a config** (from the DSM repo) matching your model:
   - `configs/config-stt-en_fr-hf.toml` (1B EN/FR, low latency, semantic VAD)
   - `configs/config-stt-en-hf.toml` (2.6B EN, higher accuracy) citeturn4search0
3. **Run the worker:**

   ```bash
   moshi-server worker --config configs/config-stt-en_fr-hf.toml
   ```

   (Adjust the **batch size** in the config to fit your GPU VRAM and target concurrency.) citeturn4search0

> Tip: the server speaks **WS/WSS** and is optimized for real-time (e.g., TCP\_NODELAY). Use TLS certs in production for WSS. citeturn3search6

# Client side (your EARS CLI or a TS app)

- Open one **WebSocket** per user/session and **stream PCM chunks** (or your current capture format).
- The server **handles scheduling and batching**; you just keep sending frames and read partial results back over the same socket. (Kyutai’s example clients demonstrate this protocol.) citeturn4search0

# Building blocks if you want to roll your own Rust service

If you prefer embedding the model directly in your EARS server instead of running `moshi-server`, Kyutai’s Rust crates expose streaming abstractions on top of Candle:

- Use **`moshi_db::streaming::{StreamTensor, StreamMask, StreamingModule}`** to keep **per-stream state** (caches) while batching different users in the **batch dimension** for each forward pass. citeturn3search7
- A typical architecture:
  1. **WS ingress (Axum/Tokio)** → one task per client pushes audio frames into an **MPSC channel**.
  2. **Batcher task** ticks every *Δt* (e.g., 20–40 ms), drains up to *N* ready frames across clients, packs them into a `Tensor` shaped `[B, T, …]`, and calls the Candle model once.
  3. **Demux** the batched outputs by stream ID; push partial transcripts back over each client’s WS.
- This reproduces what the reference server does, but within your own binary.

# Model choices & capacity planning

- **`kyutai/stt-1b-en_fr`** (\~1B) → lower latency, semantic VAD; fits more streams per GPU.
- **`kyutai/stt-2.6b-en`** (\~2.6B) → higher accuracy; fewer parallel streams unless you have more VRAM.
- Scale concurrency by tuning **batch size** in the TOML; Kyutai’s public numbers: **\~64 streams on L40S @ 3× RTF**, **\~400 streams on H100**. citeturn4search0turn1search1

# Quick way to prove it works (no code)

- Start the server (above), then run their ready-made clients that stream from mic/file; spin up multiple clients at once to see the batching kick in. (See `scripts/stt_from_mic_rust_server.py` and `scripts/stt_from_file_rust_server.py` in the DSM repo.) citeturn4search0
