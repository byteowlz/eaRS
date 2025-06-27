// Basic usage example for the eaRS library

use anyhow::Result;
use ears::{Model, TranscriptionOptions};

fn main() -> Result<()> {
    // Create transcription options
    let options = TranscriptionOptions {
        timestamps: true,
        vad: false,
        save_audio: Some("recorded_audio.wav".to_string()),
    };

    // Load the model
    let mut model = Model::load_from_hf("kyutai/stt-1b-en_fr-candle", false, options)?;

    // Transcribe an audio file
    let result = model.transcribe_file("path/to/your/audio.wav", Some("saved_audio.wav"))?;

    // Print the transcription
    println!("Transcription: {}", result.text);

    // Print word-level timestamps
    for word in result.words {
        if let Some(end_time) = word.end_time {
            println!("[{:5.2}-{:5.2}] {}", word.start_time, end_time, word.word);
        } else {
            println!("[{:5.2}-     ] {}", word.start_time, word.word);
        }
    }

    Ok(())
}