//! Exercise the exact transcribe.cpp revision used by eaRS without starting a server.
use std::path::PathBuf;
use transcribe_cpp::Model;

fn main() {
    let paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!("usage: loader-probe MODEL.gguf [MODEL.gguf ...]");
        std::process::exit(2);
    }
    for path in paths {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        match Model::load(&path) {
            Ok(model) => println!(
                "{name}: loaded arch={} supports_streaming={}",
                model.arch(),
                model.capabilities().supports_streaming
            ),
            Err(error) => println!("{name}: rejected: {error}"),
        }
    }
}
