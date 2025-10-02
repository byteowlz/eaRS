# Whisper Integration Planning

## Overview

This document outlines the integration of Whisper models for optional high-accuracy transcription enhancement in eaRS, specifically targeting German and Japanese languages while maintaining real-time performance.

## Architecture

### Hybrid Transcription Flow
1. **Primary**: Kyutai STT provides real-time transcription
2. **Secondary**: Whisper processes complete sentences for accuracy enhancement
3. **Output**: Live display with real-time updates and retroactive corrections

```
Audio Input → Kyutai (real-time) → Live Display
     ↓              ↓
Audio Buffer → Sentence Detection → Whisper → Correction → Update Display
```

## Configuration

### Config File (`config.toml`)
```toml
[whisper]
enabled = false
default_model = "large-v3-turbo"
model_format = "gguf"  # or "safetensors"
quantization = "Q4_K_M"  # Q4_K_M, Q5_K_M, Q8_0, f16, f32
languages = ["de", "ja", "it"]  # Only enhance these languages (ISO 639-1)
confidence_threshold = 0.7
storage_dir = "default"  # Use HF cache or custom path

[whisper.sentence_detection]
min_duration = 1.0  # Minimum sentence duration in seconds
max_duration = 30.0  # Maximum sentence duration in seconds
vad_pause_threshold = 0.8  # VAD confidence for pause detection
silence_duration = 0.5  # Minimum silence for sentence boundary
punctuation_markers = [".","!","?","。","！","？"]
```

### Storage Strategy
- **Default**: Use HuggingFace cache (separate from Kyutai models)
- **Custom**: User-defined directory via config
- **Models**: Downloaded from `ggml-org/whisper.cpp` or `openai/whisper-*`

## Command Line Interface

### New Flags
- `-w, --whisper` - Enable Whisper enhancement
- `-wm, --whisper-model <MODEL>` - Override default model
- `-wq, --whisper-quantization <QUANT>` - Override quantization level
- `-wl, --whisper-languages <LANGS>` - Comma-separated language list

### Examples
```bash
# Enable Whisper with defaults
ears --live --whisper -l de

# Custom model and quantization
ears --live --whisper --whisper-model large-v3 --whisper-quantization Q5_K_M

# Multiple languages
ears --live --whisper --whisper-languages de,ja,es,it
```

## Build Features

- `whisper`: Enables optional Whisper integration (adds related CLI flags).
- `metal`: Enables GPU acceleration via Metal for both Kyutai (Candle) and Whisper.
- `cuda`: Enables GPU acceleration via CUDA for both Kyutai (Candle) and Whisper.
- `cudnn`: Optional; enhances Candle’s CUDA backend if cuDNN is available.

Notes
- CLI flags `--whisper`, `--whisper-model`, `--whisper-quantization`, `--whisper-languages` are only available when built with the `whisper` feature.
- GPU for Whisper is selected at compile time through features; there is no separate runtime switch.

### Build/Run Examples
```bash
# CPU + Whisper
cargo run --features whisper -- --live --whisper

# macOS Metal + Whisper (GPU for both Kyutai and Whisper)
cargo run --features "metal whisper" -- --live --whisper

# Linux CUDA + Whisper (GPU for both Kyutai and Whisper)
cargo run --features "cuda whisper" -- --live --whisper

# CUDA with cuDNN (optional)
cargo run --features "cuda cudnn whisper" -- --live --whisper
```

## Sentence Boundary Detection

### Multi-Factor Approach
1. **VAD-based**: Monitor silence duration > threshold
2. **Punctuation**: Detect sentence-ending punctuation in text
3. **Duration**: Enforce min/max sentence lengths
4. **Word gaps**: Large gaps between word timestamps

### Implementation Strategy
```rust
struct SentenceDetector {
    current_sentence: SentenceBuffer,
    silence_start: Option<f64>,
    last_word_time: f64,
    config: SentenceConfig,
}

impl SentenceDetector {
    fn process_word(&mut self, word: &WordTimestamp) -> Option<SentenceBuffer> {
        // Check duration gaps, VAD silence, punctuation
        // Return complete sentence when boundary detected
    }
}
```

## Audio Buffer Management

### Circular Buffer Design
```rust
struct AudioBuffer {
    samples: VecDeque<f32>,
    timestamps: VecDeque<f64>,
    max_duration: f64,  // 30 seconds rolling window
    sample_rate: u32,   // 24kHz
}
```

### Sentence Extraction
- Maintain rolling buffer of audio samples with timestamps
- Extract precise audio segments when sentence boundaries detected
- Handle overlapping segments for context

## Model Loading & Caching

### Download Progress
```rust
use indicatif::{ProgressBar, ProgressStyle};

async fn download_whisper_model(model: &str) -> Result<PathBuf> {
    let pb = ProgressBar::new(0);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes:>7}/{total_bytes:7} {msg}")
        .progress_chars("##-"));
    
    // Download with progress updates
}
```

### Model Selection Priority
1. Command-line override (`--whisper-model`)
2. Config file setting (`whisper.default_model`)
3. Built-in default (`large-v3-turbo`)

## Terminal Display Enhancement

### Live Update Strategy
```
Current transcription: "Hello, how are you doing today?"
                      ↑ Kyutai real-time output

Enhanced transcription: "Hello, how are you doing today?"
                       ↑ Whisper-corrected (different color)
```

### Color Coding
- **White/Default**: Kyutai real-time transcription
- **Green**: Whisper-confirmed (no changes)
- **Yellow**: Whisper-corrected (while processing)
- **Cyan**: Whisper-corrected (final)

### Anti-Flicker Implementation
```rust
struct DisplayManager {
    current_lines: Vec<String>,
    whisper_status: HashMap<usize, WhisperStatus>, // line -> status
}

enum WhisperStatus {
    Pending,
    Processing,
    Corrected(String),
    Confirmed,
}
```

## WebSocket Integration

### Extended Message Types
```json
{
  "WhisperProcessing": {
    "sentence_id": "uuid",
    "original_text": "...",
    "start_time": 1.23,
    "end_time": 4.56
  }
}

{
  "WhisperComplete": {
    "sentence_id": "uuid",
    "original_text": "...",
    "corrected_text": "...",
    "confidence": 0.95,
    "changed": true
  }
}
```

## Performance Considerations

### Threading Strategy
- **Main Thread**: Kyutai processing + display
- **Audio Buffer Thread**: Continuous audio capture
- **Whisper Thread Pool**: Async sentence processing
- **Display Thread**: Non-blocking terminal updates

### Memory Management
- Limit audio buffer to 30-second rolling window
- Process sentences asynchronously without blocking
- Release processed audio segments immediately

### Latency Targets
- **Real-time**: Kyutai continues uninterrupted
- **Enhancement**: Whisper results within 2-5 seconds
- **Display**: Smooth updates without flicker

## Error Handling

### Graceful Degradation
- If Whisper fails to load: Continue with Kyutai-only
- If Whisper processing errors: Skip enhancement for that sentence
- If model download fails: Provide clear error message and continue

### User Feedback
```
Loading Whisper model (large-v3-turbo)... [████████████████] 100%
Whisper enhancement enabled for: German, Japanese
Starting transcription with Kyutai + Whisper enhancement...
```

## Testing Strategy

### Unit Tests
- Sentence boundary detection accuracy
- Audio buffer extraction precision
- Model loading and caching

### Integration Tests
- Full pipeline with sample German/Japanese audio
- WebSocket message flow validation
- Terminal display consistency

### Performance Tests
- Memory usage with long transcriptions
- Latency measurements for sentence processing
- Concurrent processing stress tests

## Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETED
- [x] Config structure and parsing
- [x] Model downloading with progress
- [x] Basic Whisper integration
- [x] Sentence boundary detection
- [x] Audio buffer management
- [x] CLI flags and options

### Phase 2: Display Enhancement
- [ ] Terminal color management
- [ ] Anti-flicker display updates
- [ ] WebSocket message extensions

### Phase 3: Optimization
- [ ] Threading optimization
- [ ] Memory management tuning
- [ ] Performance monitoring

### Phase 4: Testing & Polish
- [ ] Comprehensive testing suite
- [ ] Documentation updates
- [ ] Error handling refinement

## Future Enhancements

### Potential Improvements
- **Adaptive thresholds**: Learn optimal sentence boundaries per user
- **Language detection**: Auto-enable Whisper for detected languages
- **Custom models**: Support user-provided Whisper models
- **Streaming Whisper**: Real-time Whisper processing for ultra-low latency
