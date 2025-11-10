# Parakeet WebSocket Server Implementation Plan

## Overview

This document outlines the plan for implementing a drop-in replacement for the current eaRS WebSocket server using NVIDIA's Parakeet TDT 0.6B v3 model. The goal is to maintain API compatibility while leveraging Parakeet's multilingual capabilities and improved accuracy.

## Model Comparison

### Current: Kyutai STT 1B
- Model size: ~1B parameters
- Architecture: Custom Kyutai architecture with Mimi audio tokenizer
- Sample rate: 24kHz
- Output: Real-time streaming with word-level timestamps
- Languages: English, French (limited multilingual)
- Latency: Low (streaming architecture)

### Proposed: NVIDIA Parakeet TDT 0.6B v3
- Model size: 600M parameters
- Architecture: FastConformer-TDT (Transducer with Token-and-Duration Transducer)
- Sample rate: 16kHz
- Output: Real-time streaming with word/segment timestamps
- Languages: 25 European languages with automatic language detection
- Latency: Very low (designed for high-throughput)
- License: CC-BY-4.0 (permissive, commercial use allowed)

## Key Differences & Challenges

### 1. Audio Format
**Challenge**: Parakeet expects 16kHz mono audio while current system uses 24kHz
**Solution**: 
- Add resampling layer in audio processing pipeline
- Use linear interpolation or proper resampling filter (librosa/rubato)
- Resample from 24kHz → 16kHz before feeding to Parakeet

### 2. Model Integration
**Challenge**: Parakeet native format uses NeMo (Python), current stack is pure Rust
**Solutions**:

- **Option A (Recommended - ONNX)**: Use ONNX Runtime in pure Rust
  - ✅ Pre-converted ONNX model available: `istupakov/parakeet-tdt-0.6b-v3-onnx`
  - ✅ Works on ALL hardware: CPU, CUDA, Metal, DirectML, etc.
  - ✅ Use `onnxruntime-rs` crate (v0.0.18+) for inference  
  - ✅ Pure Rust solution - no Python runtime needed
  - ✅ Simpler deployment and distribution
  - ✅ Lower latency (no IPC overhead)
  - Reference implementation: `onnx-asr` Python package shows architecture
  - Pros: Cross-platform, efficient, pure Rust
  - Cons: Need to implement preprocessing/decoding logic

- **Option B**: Create Python microservice using NeMo
  - Use native NeMo model for inference
  - Expose HTTP/gRPC API for inference
  - Current Rust server calls Python service
  - Pros: Official implementation, easier to maintain
  - Cons: Deployment complexity, IPC overhead, Python dependency
  
- **Option C**: Use PyO3 bindings
  - Embed Python interpreter in Rust binary
  - Direct NeMo model calls from Rust
  - Pros: Single binary deployment
  - Cons: Complex build, GIL contention, large binary size

### 3. Streaming vs Batching
**Challenge**: Current system streams 1920-sample chunks (80ms @ 24kHz), Parakeet designed for chunked inference
**Solution**:
- Adapt chunked streaming approach from NeMo's `speech_to_text_streaming_infer_rnnt.py`
- Configure chunk parameters:
  - `chunk_secs`: 2.0 seconds (processing chunk size)
  - `right_context_secs`: 2.0 seconds (lookahead context)
  - `left_context_secs`: 10.0 seconds (lookback context)
- Buffer incoming audio and process in overlapping windows

### 4. Language Detection
**Challenge**: Current system requires manual language priming, Parakeet auto-detects
**Solution**:
- Remove language priming requirement in server
- Expose detected language in WebSocket messages
- Add optional language hint command for forced language mode

### 5. Output Format Compatibility
**Challenge**: Maintain existing WebSocket message format
**Solution**: Map Parakeet outputs to current schema
```json
// Current format (maintain this)
{"type": "word", "word": "hello", "start_time": 1.23, "end_time": 1.45}
{"type": "final", "text": "complete text", "words": [...]}

// Add new optional fields
{"type": "language_detected", "lang": "en"}
{"type": "segment", "text": "segment text", "start_time": 0.0, "end_time": 5.0}
```

## Implementation Architecture

### Recommended Architecture: Pure Rust with ONNX

```
┌─────────────────────────────────────────────────────┐
│  eaRS WebSocket Server (Pure Rust)                  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Audio Processing Pipeline                    │  │
│  │  - Capture @ 24kHz                            │  │
│  │  - Resample 24kHz → 16kHz (rubato)            │  │
│  │  - Buffer & chunk audio                       │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Parakeet ONNX Inference                      │  │
│  │  - Load: istupakov/parakeet-tdt-0.6b-v3-onnx │  │
│  │  - Preprocessor: Log-mel spectrogram          │  │
│  │  - Encoder: FastConformer (ONNX Runtime)      │  │
│  │  - Decoder: TDT greedy search                 │  │
│  │  - Works on: CPU/CUDA/Metal/DirectML          │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  WebSocket Protocol                           │  │
│  │  - Message routing & serialization            │  │
│  │  - Client connection management               │  │
│  │  - Word/segment timestamp emission            │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Key Advantages**:
- ✅ **Cross-platform**: Runs on x86, ARM, with or without GPU
- ✅ **Zero Python**: No runtime dependencies beyond ONNX Runtime
- ✅ **Single binary**: Easy distribution via cargo
- ✅ **Lower latency**: No IPC/serialization overhead
- ✅ **Better resource usage**: Direct memory access to model

### Alternative: Python Microservice Architecture

```
┌─────────────────────────────────────────────────┐
│  eaRS WebSocket Server (Rust)                   │
│  - Audio capture @ 24kHz                        │
│  - Resample 24kHz → 16kHz                       │
│  - WebSocket message routing                    │
│  - Client connection management                 │
└─────────────────┬───────────────────────────────┘
                  │ HTTP/gRPC
┌─────────────────▼───────────────────────────────┐
│  Parakeet Inference Service (Python + NeMo)     │
│  - Model: nvidia/parakeet-tdt-0.6b-v3           │
│  - Streaming inference pipeline                 │
│  - GPU acceleration (CUDA/Metal)                │
│  - Returns: word timestamps + detected language │
└─────────────────────────────────────────────────┘
```

**Use this if**: You need official NeMo features or don't want to reimplement preprocessing

## Implementation Phases

### Phase 1: Research & ONNX Prototyping (2-3 days)
**Goal**: Validate ONNX approach and identify implementation requirements

1. **ONNX Model Analysis**
   - Download `istupakov/parakeet-tdt-0.6b-v3-onnx` model
   - Examine model structure with ONNX tools
   - Study `onnx-asr` Python reference implementation
   - Identify required preprocessing steps (log-mel spectrogram)
   - Understand TDT decoder algorithm from reference code

2. **Rust ONNX Integration Test**
   - Set up `onnxruntime-rs` crate (check latest version)
   - Load Parakeet ONNX model in Rust
   - Test basic inference with sample audio
   - Verify output tensor shapes and types
   - Benchmark CPU vs GPU inference latency

3. **Audio Pipeline Implementation**
   - Implement 24kHz → 16kHz resampling using `rubato`
   - Port log-mel spectrogram preprocessing from `onnx-asr`
   - Validate feature extraction matches reference
   - Test with reference audio samples

4. **Decoder Prototype**
   - Implement TDT greedy search decoder in Rust
   - Port tokenizer/vocabulary loading
   - Test end-to-end: audio → text
   - Validate WER against `onnx-asr` Python results

### Phase 2: Core ONNX Implementation (5-7 days)

#### 2.1 Parakeet Module Structure
**Files to create**:
- `src/parakeet/mod.rs` - Module entry point and public API
- `src/parakeet/model.rs` - ONNX model loading and management
- `src/parakeet/preprocessor.rs` - Log-mel spectrogram feature extraction
- `src/parakeet/decoder.rs` - TDT greedy search decoder
- `src/parakeet/tokenizer.rs` - Vocabulary and tokenization
- `src/parakeet/resampler.rs` - Audio resampling utilities

**Key features**:
- Load ONNX model from Hugging Face or local path
- Configurable execution provider (CPU, CUDA, Metal, DirectML)
- Streaming inference with chunked audio processing
- Return word-level and segment-level timestamps
- Automatic language detection (model output)
- Thread-safe model sharing for concurrent sessions

#### 2.2 Preprocessor Implementation
**Reference**: Port from `onnx-asr/src/onnx_asr/preprocessor.py`

```rust
pub struct LogMelPreprocessor {
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    // FFT window, mel filterbank, etc.
}

impl LogMelPreprocessor {
    pub fn process(&self, waveform: &[f32]) -> Array2<f32> {
        // 1. Apply FFT windowing
        // 2. Compute power spectrogram
        // 3. Apply mel filterbank
        // 4. Log scale
        // 5. Normalize features
    }
}
```

#### 2.3 TDT Decoder Implementation
**Reference**: Port from `onnx-asr/src/onnx_asr/models/nemo/tdt_decoder.py`

```rust
pub struct TDTDecoder {
    model: Session, // ONNX Runtime session
    vocab: Vec<String>,
    blank_id: usize,
}

impl TDTDecoder {
    pub fn decode_greedy(&mut self, features: Array2<f32>) -> Vec<WordTimestamp> {
        // 1. Run encoder forward pass
        // 2. Greedy token-duration decoding
        // 3. Map tokens to words
        // 4. Calculate timestamps from durations
    }
}
```

#### 2.4 Server Integration
**Files to modify**:
- `src/server.rs` - Add Parakeet backend option
- `src/lib.rs` - Add Parakeet model enum variant
- `src/main.rs` - Add CLI flag for backend selection

**Key changes**:
1. Add `--backend parakeet|kyutai` flag to server start command
2. Modify `Model` enum to support both backends:
   ```rust
   pub enum ModelBackend {
       Kyutai(KyutaiModel),
       Parakeet(ParakeetModel),
   }
   ```
3. Implement unified `transcribe_live` interface
4. Handle resampling in audio capture pipeline
5. Maintain WebSocket protocol compatibility

#### 2.5 Configuration Updates
**Files to modify**:
- `config/config.toml` - Add Parakeet section
- `src/config.rs` - Add Parakeet configuration structs

```toml
[parakeet]
enabled = false
model_path = "istupakov/parakeet-tdt-0.6b-v3-onnx"
device = "cuda"  # or "cpu", "metal", "directml"
num_threads = 4  # CPU threads for inference
chunk_length = 2.0  # seconds
```

### Phase 3: Feature Parity & Testing (5-7 days)

#### 3.1 Feature Implementation
- [ ] Word-level timestamps
- [ ] Segment-level timestamps
- [ ] Language detection and reporting
- [ ] VAD integration (if needed)
- [ ] Session management (concurrent clients)
- [ ] Long audio support (up to 24 minutes)
- [ ] Error handling and recovery
- [ ] Graceful degradation on service failure

#### 3.2 WebSocket Protocol Extensions
Add new message types while maintaining compatibility:
```json
// New messages
{"type": "language_detected", "language": "en", "confidence": 0.95}
{"type": "segment", "text": "...", "start_time": 0.0, "end_time": 5.0}

// Enhanced word messages
{"type": "word", "word": "hello", "start_time": 1.23, "end_time": 1.45, "confidence": 0.98}
```

#### 3.3 Testing Strategy
1. **Unit Tests**
   - Resampling accuracy
   - Audio buffer management
   - Message format conversion

2. **Integration Tests**
   - End-to-end streaming with real audio
   - Multiple concurrent clients
   - Language switching mid-stream
   - Error recovery scenarios

3. **Performance Tests**
   - Latency measurements (end-to-end)
   - Throughput (concurrent sessions)
   - Memory usage under load
   - GPU utilization

4. **Accuracy Tests**
   - Compare WER against Kyutai on reference audio
   - Test multilingual capabilities
   - Validate timestamps accuracy
   - Test with various audio conditions (noise, accents)

### Phase 4: Optimization & Production Readiness (3-5 days)

#### 4.1 Performance Optimization
- Profile inference latency bottlenecks
- Optimize audio buffering and resampling
- Tune chunk sizes and context windows
- Implement connection pooling for HTTP client
- Add request batching if applicable

#### 4.2 Deployment & Monitoring
- Docker containerization for Python service
- Health checks and liveness probes
- Metrics collection (Prometheus-compatible)
- Logging and tracing integration
- Resource limits and auto-scaling

#### 4.3 Documentation
- Update `WEBSOCKET.md` with Parakeet-specific details
- Add `PARAKEET.md` deployment guide
- Document configuration options
- Provide migration guide from Kyutai
- Add troubleshooting section

## Dependency Changes

### New Rust Dependencies
```toml
[dependencies]
# ONNX Runtime for inference
onnxruntime = "0.0.18"  # Supports CPU, CUDA, Metal, DirectML

# Audio resampling
rubato = "0.15"  # High-quality resampling library

# FFT for spectrogram
rustfft = "6.2"  # Fast Fourier Transform

# Linear algebra for preprocessing
ndarray = "0.16"  # N-dimensional arrays

# Optional: Download models from Hugging Face
hf-hub = "0.4"  # Already in dependencies

[features]
default = []
parakeet = []
parakeet-cuda = ["onnxruntime/cuda"]
parakeet-metal = ["onnxruntime/metal"]  # If available in onnxruntime-rs
```

### Removed Dependencies
- No Python runtime needed
- No HTTP client needed (reqwest not required)
- No gRPC infrastructure needed

## Benchmarks & Targets

### Latency Targets
- Word emission: < 200ms from speech end
- End-to-end (audio → WebSocket): < 300ms
- Service cold start: < 10s
- First word latency: < 500ms

### Accuracy Targets
- English WER: < 5% (vs Kyutai baseline)
- Multilingual WER: < 12% average across 25 languages
- Timestamp accuracy: ± 50ms

### Resource Targets
- GPU memory: < 4GB per session
- CPU usage: < 50% on 4-core system
- Concurrent sessions: 5+ on single GPU

## Migration Strategy

### Deployment Options

#### Option 1: Side-by-side Deployment
- Run Parakeet server on different port
- Allow users to choose backend via flag
- Gradual migration based on user feedback

#### Option 2: Feature Flag
- Single server with runtime backend selection
- Default to Kyutai, opt-in to Parakeet
- A/B testing capabilities

#### Option 3: Full Replacement
- Direct replacement of Kyutai backend
- Comprehensive testing before rollout
- Fallback mechanism to Kyutai on issues

### Rollback Plan
- Keep Kyutai code path intact for 2-3 releases
- Document known limitations of Parakeet approach
- Provide clear upgrade/downgrade instructions

## Risks & Mitigations

### Technical Risks

1. **Latency Overhead from Python Service**
   - Risk: Network/serialization adds 50-100ms
   - Mitigation: Use local Unix socket or shared memory
   - Fallback: Embed Python interpreter with PyO3

2. **Model Compatibility Issues**
   - Risk: NeMo version changes break compatibility
   - Mitigation: Pin NeMo version, containerize service
   - Fallback: Export to ONNX format

3. **Memory Pressure with Multiple Sessions**
   - Risk: GPU OOM with concurrent streams
   - Mitigation: Implement queue with configurable max sessions
   - Fallback: Dynamic model unloading/loading

4. **Audio Quality Degradation from Resampling**
   - Risk: 24kHz→16kHz loses high frequencies
   - Mitigation: Use high-quality resampler (rubato)
   - Testing: Validate WER doesn't degrade

### Operational Risks

1. **Deployment Complexity**
   - Risk: Two-process architecture harder to deploy
   - Mitigation: Provide Docker Compose setup
   - Fallback: Single container with supervisord

2. **Debugging Difficulty**
   - Risk: Errors span Rust/Python boundary
   - Mitigation: Comprehensive logging and tracing
   - Tooling: Request ID propagation

## Success Metrics

### Performance Metrics
- ✓ Latency within 10% of Kyutai baseline
- ✓ Support 5+ concurrent sessions on single GPU
- ✓ < 5s cold start time
- ✓ Word timestamp accuracy ± 50ms

### Quality Metrics
- ✓ WER improvement of 10%+ on multilingual audio
- ✓ Automatic language detection accuracy > 95%
- ✓ Punctuation and capitalization included
- ✓ No regressions on English-only audio

### Developer Metrics
- ✓ API remains backward compatible
- ✓ Configuration migration path documented
- ✓ Deployment time < 10 minutes from scratch
- ✓ Clear error messages and debugging guides

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: ONNX Research & Prototyping | 2-3 days | Working ONNX inference in Rust |
| Phase 2: Core ONNX Implementation | 5-7 days | Functional Parakeet backend |
| Phase 3: Feature Parity & Testing | 4-6 days | Production-ready implementation |
| Phase 4: Optimization & Docs | 2-4 days | Optimized, documented system |
| **Total** | **13-20 days** | Full Parakeet integration (pure Rust) |

**Note**: Timeline is shorter than original estimate due to:
- No Python/Rust IPC complexity
- Pre-converted ONNX model available
- Reference implementation (`onnx-asr`) to port from
- No microservice deployment complexity

## Next Steps

1. **Immediate Actions** (Day 1)
   - Add `onnxruntime`, `rubato`, `rustfft`, `ndarray` to Cargo.toml
   - Download ONNX model: `istupakov/parakeet-tdt-0.6b-v3-onnx`
   - Study `onnx-asr` preprocessor implementation
   - Create `src/parakeet/` module structure

2. **Week 1 Goals** (Days 2-7)
   - Working ONNX inference (load model, run encoder)
   - Port log-mel preprocessor to Rust
   - Implement basic TDT decoder
   - End-to-end test: audio file → text output
   - Validate WER against reference implementation

3. **Week 2 Goals** (Days 8-14)
   - Integrate with WebSocket server
   - Streaming audio processing
   - Word-level timestamp generation
   - Concurrent session support
   - Integration tests passing

4. **Week 3 Goals** (Days 15-20)
   - Performance optimization
   - GPU acceleration testing (CUDA/Metal)
   - Comprehensive benchmarks
   - Documentation and examples
   - Ready for production use

## References

### Parakeet Model Resources
- [Parakeet TDT 0.6B v3 Model Card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Parakeet ONNX Model](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- [FastConformer Paper](https://arxiv.org/abs/2305.05084)
- [TDT Decoder Paper](https://arxiv.org/abs/2304.06795)

### ONNX & Implementation References
- [onnx-asr GitHub Repository](https://github.com/istupakov/onnx-asr) - **Primary reference implementation**
- [onnx-asr Preprocessor](https://github.com/istupakov/onnx-asr/blob/main/src/onnx_asr/preprocessor.py)
- [onnx-asr TDT Decoder](https://github.com/istupakov/onnx-asr/blob/main/src/onnx_asr/models/nemo/tdt_decoder.py)
- [onnxruntime-rs Documentation](https://docs.rs/onnxruntime/latest/onnxruntime/)

### Rust Libraries
- [rubato Documentation](https://docs.rs/rubato/) - Audio resampling
- [rustfft Documentation](https://docs.rs/rustfft/) - Fast Fourier Transform
- [ndarray Documentation](https://docs.rs/ndarray/) - N-dimensional arrays

## Key Benefits of ONNX Approach

### Technical Benefits
1. **Hardware Flexibility**
   - ✅ Works on CPU (x86, ARM - even Raspberry Pi)
   - ✅ Works on NVIDIA GPUs (CUDA)
   - ✅ Works on Apple Silicon (Metal)
   - ✅ Works on Windows GPUs (DirectML)
   - ✅ No hardware lock-in

2. **Deployment Simplicity**
   - ✅ Single Rust binary (no Python runtime)
   - ✅ Smaller binary size (~50MB vs 500MB+ with Python)
   - ✅ No complex dependency management
   - ✅ Easy cross-compilation
   - ✅ Works in embedded/IoT environments

3. **Performance**
   - ✅ Lower latency (no IPC overhead)
   - ✅ Better memory efficiency (direct model access)
   - ✅ Benchmarked by `onnx-asr` author: competitive with native
   - ✅ Suitable for real-time streaming

4. **Multilingual Support**
   - ✅ 25 European languages automatic detection
   - ✅ Better accuracy than current Kyutai model
   - ✅ Punctuation and capitalization included
   - ✅ WER improvements: 10%+ on multilingual content

### Potential Challenges
1. **Implementation Effort**
   - Need to port preprocessing from Python
   - Need to port TDT decoder logic
   - Mitigated by: clear reference implementation available

2. **Maintenance**
   - Need to track upstream model changes
   - Mitigated by: ONNX format is stable

3. **Feature Completeness**
   - May miss some NeMo-specific features
   - Mitigated by: core ASR functionality is fully supported

## Questions for Stakeholders

1. **Priority**: Is multilingual support (25 languages) important?
2. **Timeline**: 2-3 weeks for ONNX implementation acceptable?
3. **Hardware**: Should we prioritize CPU-only support for wider deployment?
4. **Migration**: Gradual rollout with feature flag, or full replacement?
5. **Testing**: Access to multilingual test audio available?
6. **Accuracy**: Is 10% WER improvement on multilingual worth the effort?
