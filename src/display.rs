use colored::*;
use std::collections::HashMap;
use std::io::{self, Write};

use crate::WordTimestamp;
use crate::whisper::{WhisperMessage, WhisperStatus};

#[derive(Debug, Clone)]
pub struct DisplayManager {
    current_lines: Vec<String>,
    whisper_status: HashMap<String, WhisperStatus>, // sentence_id -> status
    sentence_map: HashMap<usize, String>,           // line_index -> sentence_id
    current_line: usize,
    live_text: String,
}

impl DisplayManager {
    pub fn new() -> Self {
        Self {
            current_lines: Vec::new(),
            whisper_status: HashMap::new(),
            sentence_map: HashMap::new(),
            current_line: 0,
            live_text: String::new(),
        }
    }

    pub fn add_live_word(&mut self, word: &str) {
        if !self.live_text.is_empty() {
            self.live_text.push(' ');
        }
        self.live_text.push_str(word);
        self.display_live_text();
    }

    pub fn complete_sentence(&mut self, sentence_id: String, text: String) {
        // Move live text to a completed line
        if !self.live_text.is_empty() {
            let line_index = self.current_lines.len();
            self.current_lines.push(text.clone());
            self.sentence_map.insert(line_index, sentence_id.clone());
            self.whisper_status
                .insert(sentence_id, WhisperStatus::Pending);
            self.live_text.clear();
            self.current_line += 1;
            self.redraw();
        }
    }

    pub fn handle_whisper_message(&mut self, message: WhisperMessage) {
        match message {
            WhisperMessage::Processing { sentence_id, .. } => {
                self.whisper_status
                    .insert(sentence_id, WhisperStatus::Processing);
                self.redraw();
            }
            WhisperMessage::Complete {
                sentence_id,
                corrected_text,
                changed,
                ..
            } => {
                if changed {
                    self.whisper_status.insert(
                        sentence_id.clone(),
                        WhisperStatus::Corrected(corrected_text.clone()),
                    );
                    // Update the line with corrected text
                    for (&line_index, line_sentence_id) in &self.sentence_map {
                        if line_sentence_id == &sentence_id {
                            if let Some(line) = self.current_lines.get_mut(line_index) {
                                *line = corrected_text;
                                break;
                            }
                        }
                    }
                } else {
                    self.whisper_status
                        .insert(sentence_id, WhisperStatus::Confirmed);
                }
                self.redraw();
            }
        }
    }

    fn display_live_text(&self) {
        if atty::is(atty::Stream::Stdout) {
            print!("\r{}", " ".repeat(100)); // Clear line
            print!("\r{}", self.live_text.white());
            io::stdout().flush().ok();
        }
    }

    fn redraw(&self) {
        if !atty::is(atty::Stream::Stdout) {
            return;
        }

        // Move cursor up to the beginning of our display area
        if !self.current_lines.is_empty() {
            print!("\r\x1b[{}A", self.current_lines.len());
        }

        // Redraw all completed lines
        for (i, line) in self.current_lines.iter().enumerate() {
            let sentence_id = self.sentence_map.get(&i);
            let status = sentence_id
                .and_then(|id| self.whisper_status.get(id))
                .unwrap_or(&WhisperStatus::Confirmed);

            let colored_line = match status {
                WhisperStatus::Pending => line.white(),
                WhisperStatus::Processing => line.yellow(),
                WhisperStatus::Corrected(_) => line.cyan(),
                WhisperStatus::Confirmed => line.green(),
            };

            print!("\r{}{}\n", " ".repeat(120), ""); // Clear line
            print!("\r{}\n", colored_line);
        }

        // Display current live text
        if !self.live_text.is_empty() {
            print!("\r{}", " ".repeat(120)); // Clear line
            print!("\r{}", self.live_text.white());
        }

        io::stdout().flush().ok();
    }

    pub fn finish(&mut self) {
        if !self.live_text.is_empty() {
            println!("\n{}", self.live_text);
            self.live_text.clear();
        }

        // Final display of all lines with their final status
        for (i, line) in self.current_lines.iter().enumerate() {
            let sentence_id = self.sentence_map.get(&i);
            let status = sentence_id
                .and_then(|id| self.whisper_status.get(id))
                .unwrap_or(&WhisperStatus::Confirmed);

            let colored_line = match status {
                WhisperStatus::Corrected(_) => line.cyan(),
                WhisperStatus::Confirmed => line.green(),
                _ => line.white(),
            };

            println!("{}", colored_line);
        }
    }

    pub fn print_timestamps(&self, words: &[WordTimestamp]) {
        for word in words {
            let _sentence_id = ""; // TODO: Map words to sentences for coloring
            let status = WhisperStatus::Confirmed; // Default for timestamp mode

            let colored_word = match status {
                WhisperStatus::Processing => word.word.yellow(),
                WhisperStatus::Corrected(_) => word.word.cyan(),
                WhisperStatus::Confirmed => word.word.green(),
                _ => word.word.white(),
            };

            if let Some(end_time) = word.end_time {
                println!(
                    "[{:5.2}-{:5.2}] {}",
                    word.start_time, end_time, colored_word
                );
            } else {
                println!("[{:5.2}-     ] {}", word.start_time, colored_word);
            }
        }
    }

    pub fn clear_screen(&self) {
        if atty::is(atty::Stream::Stdout) {
            print!("\x1b[2J\x1b[H"); // Clear screen and move cursor to top
            io::stdout().flush().ok();
        }
    }

    pub fn show_whisper_loading(&self, model: &str) {
        if atty::is(atty::Stream::Stdout) {
            println!(
                "{}",
                format!("Loading Whisper model ({})...", model).yellow()
            );
            println!(
                "{}",
                "Whisper enhancement will be available shortly.".cyan()
            );
            println!("{}", "-".repeat(50).bright_black());
        }
    }

    pub fn show_whisper_ready(&self, languages: &[String]) {
        if atty::is(atty::Stream::Stdout) {
            println!("{}", "Whisper enhancement enabled".green());
            println!(
                "{}",
                format!("Enhanced languages: {}", languages.join(", ")).cyan()
            );
            println!("{}", "-".repeat(50).bright_black());
        }
    }
}

impl Default for DisplayManager {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for non-interactive display
pub fn print_whisper_status(message: &WhisperMessage) {
    match message {
        WhisperMessage::Processing {
            sentence_id,
            original_text,
            ..
        } => {
            if !atty::is(atty::Stream::Stdout) {
                eprintln!(
                    "Processing sentence {}: {}",
                    sentence_id.chars().take(8).collect::<String>(),
                    original_text
                );
            }
        }
        WhisperMessage::Complete {
            sentence_id,
            original_text,
            corrected_text,
            changed,
            confidence,
            ..
        } => {
            if !atty::is(atty::Stream::Stdout) {
                let id_short = sentence_id.chars().take(8).collect::<String>();
                if *changed {
                    eprintln!(
                        "Corrected {}: {} -> {} (confidence: {:.2})",
                        id_short, original_text, corrected_text, confidence
                    );
                } else {
                    eprintln!(
                        "Confirmed {}: {} (confidence: {:.2})",
                        id_short, original_text, confidence
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_manager_creation() {
        let dm = DisplayManager::new();
        assert!(dm.current_lines.is_empty());
        assert!(dm.whisper_status.is_empty());
    }

    #[test]
    fn test_live_word_addition() {
        let mut dm = DisplayManager::new();
        dm.add_live_word("hello");
        assert_eq!(dm.live_text, "hello");

        dm.add_live_word("world");
        assert_eq!(dm.live_text, "hello world");
    }
}
