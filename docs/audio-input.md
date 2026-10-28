# Microphone input conversion

Capture uses one stream-owned converter (`src/audio_input.rs`) to produce mono
24 kHz PCM for both recognition and retained recordings. It supports F32/I16/U16
at 8–192 kHz, averages arbitrary interleaved channels, preserves partial channel
frames, and uses the shared windowed-sinc resampler in bounded-ratio stages.
Fixed internal quanta keep output independent of callback partitioning. Native
24 kHz bypasses filtering; other rates retain at most 128 target samples (<6 ms)
of lookahead/buffering at stream shutdown, not at every utterance boundary.

The former callback-local linear interpolator lost samples at noninteger ratios,
aliased high frequencies, mishandled more than two channels, and could panic on
split stereo frames. Nine synthetic tests check rate/phase/partition invariance,
channel/sample formats, bounded lag, passband preservation and alias rejection:

```bash
cargo test --locked --lib audio::tests
```

This is not a gain, denoising or normalization feature. Quiet input remains quiet;
equal channel mixing is not speaker-layout-aware. Native microphone quality,
realtime scheduling and device selection require separate live verification.
Capture-disconnect shutdown/logging remains tracked separately in `trx-ntjy`.
