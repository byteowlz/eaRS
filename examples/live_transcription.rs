// Live transcription example for the eaRS library

use anyhow::Result;
use crossbeam_channel::unbounded;
use ears::{Model, TranscriptionOptions, audio};
use std::thread;

fn main() -> Result<()> {
    // Create transcription options
    let options = TranscriptionOptions {
        timestamps: false,
        vad: true,
        save_audio: Some("live_recording.wav".to_string()),
    };

    // Load the model
    let mut model = Model::load_from_hf("kyutai/stt-1b-en_fr-candle", false, options)?;

    // Set up audio capture
    let (audio_tx, audio_rx) = unbounded();

    // Start audio capture in a separate thread
    let audio_handle = thread::spawn(move || {
        if let Err(e) = audio::start_audio_capture(audio_tx) {
            eprintln!("Audio capture error: {}", e);
        }
    });

    println!("Starting live transcription. Press Ctrl+C to stop.");
    println!("Speak into your microphone...");

    // Run live transcription
    let result = model.transcribe_live(audio_rx, Some("live_session.wav"))?;

    println!("\nFinal transcription: {}", result.text);

    audio_handle.join().unwrap();
    Ok(())
}