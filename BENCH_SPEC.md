# eaRS Benchmark Specification

## Overview

Add a `bench` subcommand to eaRS for standardized STT (Speech-to-Text) benchmarking as part of the OwlBench suite.

**Issue**: `owlbench-cgh.1` - Add bench subcommand to eaRS (STT benchmarking)

## Command Interface

```bash
ears bench [OPTIONS]
ears bench --output json > results.json
ears bench --engine kyutai --runs 5
ears bench --engine parakeet --runs 5
```

## Required Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `rtf` | Real-Time Factor (audio_duration / processing_time) | ratio |
| `latency_first_word_ms` | Time to first transcribed word | ms |
| `latency_p50_ms` | 50th percentile latency | ms |
| `latency_p99_ms` | 99th percentile latency | ms |
| `throughput_audio_sec` | Audio seconds processed per wall-clock second | sec/sec |
| `memory_peak_mb` | Peak memory usage during inference | MB |
| `wer` | Word Error Rate (optional, requires reference transcript) | % |

## Test Audio Files

Use bundled test audio or download standard test set:
- Short utterance (~3s): Quick latency test
- Medium utterance (~15s): Typical use case  
- Long utterance (~60s): Sustained throughput test

Consider using LibriSpeech test-clean subset for reproducibility.

## JSON Output Schema

```json
{
  "benchmark": "ears-stt",
  "version": "0.1.0",
  "timestamp": "2025-12-11T16:00:00Z",
  "hardware": {
    "cpu": "Apple M3 Max",
    "gpu": "Apple M3 Max (Metal)",
    "memory_gb": 64
  },
  "config": {
    "engine": "kyutai",
    "model": "kyutai/stt-1b-en_fr-candle",
    "device": "metal",
    "runs": 5
  },
  "results": {
    "rtf": 0.15,
    "latency_first_word_ms": 120,
    "latency_p50_ms": 85,
    "latency_p99_ms": 210,
    "throughput_audio_sec": 6.67,
    "memory_peak_mb": 1850
  }
}
```

## Implementation Notes

1. **Warm-up run**: Discard first run to exclude model loading time
2. **Multiple runs**: Default 5 runs, report median and p99
3. **Engine comparison**: Support both Kyutai and Parakeet engines
4. **Server mode**: Can benchmark via server or direct inference
5. **Hardware detection**: Reuse existing ONNX Runtime device detection

## Integration with OwlBench

Results will be collected by `owlbench collect` command and aggregated into composite Audio Score.

```bash
# OwlBench will call:
ears bench --output json > /tmp/ears_bench.json
```
