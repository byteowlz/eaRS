use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3, ArrayViewD};
use ort::session::{Session, builder::{SessionBuilder, GraphOptimizationLevel}};
use ort::value::Tensor;
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "rocm")]
use ort::execution_providers::ROCmExecutionProvider;
use rustfft::{num_complex::Complex, FftPlanner};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "rocm")]
    ROCm,
}

const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;  // Parakeet uses 128 mel bins, not 80
const MEL_FMIN: f32 = 0.0;
const MEL_FMAX: f32 = 8000.0;

pub struct ParakeetModel {
    preprocessor: Session,
    encoder: Session,
    decoder_joint: Session,
    vocab: Vec<String>,
    blank_id: usize,
}

impl ParakeetModel {
    pub fn load_from_hf(repo: &str, model_dir: Option<&Path>, device: Device) -> Result<Self> {
        let api = if let Some(dir) = model_dir {
            hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(dir.to_path_buf())
                .build()?
        } else {
            hf_hub::api::sync::Api::new()?
        };

        let repo = api.model(repo.to_string());
        
        eprintln!("Downloading model files...");
        let preprocessor_path = repo.get("nemo128.onnx")?;
        let encoder_path = repo.get("encoder-model.onnx")?;
        let _encoder_data_path = repo.get("encoder-model.onnx.data")?; // External weights
        let decoder_joint_path = repo.get("decoder_joint-model.onnx")?;
        let vocab_path = repo.get("vocab.txt")?;

        eprintln!("Loading ONNX models with device: {:?}", device);
        
        // Initialize the environment with execution providers based on device
        match device {
            Device::Cpu => {
                eprintln!("Using CPU execution");
                ort::init().commit()?;
            }
            #[cfg(feature = "cuda")]
            Device::Cuda => {
                eprintln!("Using CUDA execution");
                ort::init()
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .commit()?;
            }
            #[cfg(feature = "coreml")]
            Device::CoreML => {
                eprintln!("Using CoreML execution");
                ort::init()
                    .with_execution_providers([CoreMLExecutionProvider::default().build()])
                    .commit()?;
            }
            #[cfg(feature = "directml")]
            Device::DirectML => {
                eprintln!("Using DirectML execution");
                ort::init()
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])
                    .commit()?;
            }
            #[cfg(feature = "rocm")]
            Device::ROCm => {
                eprintln!("Using ROCm execution for AMD GPU");
                ort::init()
                    .with_execution_providers([ROCmExecutionProvider::default().build()])
                    .commit()?;
            }
        }

        let preprocessor = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&preprocessor_path)?;

        let encoder = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&encoder_path)?;

        let decoder_joint = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&decoder_joint_path)?;

        eprintln!("Loading vocabulary...");
        let vocab = Self::load_vocab(&vocab_path)?;
        let blank_id = vocab.len() - 1;

        eprintln!("Model loaded successfully!");
        eprintln!("  Vocabulary size: {}", vocab.len());
        eprintln!("  Blank ID: {}", blank_id);

        Ok(Self {
            preprocessor,
            encoder,
            decoder_joint,
            vocab,
            blank_id,
        })
    }

    fn load_vocab(path: &Path) -> Result<Vec<String>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut vocab = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if !parts.is_empty() {
                vocab.push(parts[0].to_string());
            }
        }

        Ok(vocab)
    }

    pub fn transcribe(&mut self, waveform: &[f32], sample_rate: usize) -> Result<String> {
        let resampled = if sample_rate != SAMPLE_RATE {
            resample_audio(waveform, sample_rate, SAMPLE_RATE)?
        } else {
            waveform.to_vec()
        };

        let (features, features_lens) = self.preprocess(&resampled)?;
        
        let transcript = self.decode_greedy(&features, &features_lens)?;

        Ok(transcript)
    }
    
    fn preprocess(&mut self, waveform: &[f32]) -> Result<(Array3<f32>, Array1<i64>)> {
        let waveforms = Array2::from_shape_vec((1, waveform.len()), waveform.to_vec())?;
        let waveforms_len = Array1::from_vec(vec![waveform.len() as i64]);
        
        let outputs = self.preprocessor.run(ort::inputs![
            Tensor::from_array(waveforms)?,
            Tensor::from_array(waveforms_len)?
        ])?;
        
        let features: ArrayViewD<f32> = outputs[0].try_extract_array()?;
        let features_lens: ArrayViewD<i64> = outputs[1].try_extract_array()?;
        
        let features = features.to_owned().into_dimensionality()?;
        let features_lens = features_lens.to_owned().into_dimensionality()?;
        
        Ok((features, features_lens))
    }

    fn decode_greedy(&mut self, features: &Array3<f32>, features_lens: &Array1<i64>) -> Result<String> {
        let encoder_outputs = self.encoder.run(ort::inputs![
            Tensor::from_array(features.clone())?,
            Tensor::from_array(features_lens.clone())?
        ])?;

        let encoder_output_dyn: ArrayViewD<f32> = encoder_outputs[0].try_extract_array()?;
        let encoded_len_i64: ArrayViewD<i64> = encoder_outputs[1].try_extract_array()?;
        
        let encoder_output_dyn = encoder_output_dyn.to_owned();
        let encoded_len_i64 = encoded_len_i64.to_owned();
        let encoded_frames = encoded_len_i64[[0]] as usize;
        
        if encoder_output_dyn.shape().len() != 3 {
            return Err(anyhow!("Unexpected encoder output shape: {:?}", encoder_output_dyn.shape()));
        }
        
        let encoder_output_3d: Array3<f32> = encoder_output_dyn.into_dimensionality()?;
        let encoder_permuted = encoder_output_3d.permuted_axes([0, 2, 1]);
        let encoder_2d = encoder_permuted.index_axis_move(ndarray::Axis(0), 0);
        
        let mut tokens: Vec<i32> = Vec::new();
        
        const RNN_HIDDEN_SIZE: usize = 640;
        const MAX_TOKENS_PER_STEP: usize = 10;
        let mut state_1 = Array3::<f32>::zeros((2, 1, RNN_HIDDEN_SIZE));
        let mut state_2 = Array3::<f32>::zeros((2, 1, RNN_HIDDEN_SIZE));
        
        let mut t = 0;
        let mut emitted_tokens = 0;
        
        while t < encoded_frames {
            let encoder_step_1d = encoder_2d.slice(s![t, ..]).to_owned();
            let encoder_step = encoder_step_1d
                .insert_axis(ndarray::Axis(0))
                .insert_axis(ndarray::Axis(2));
            
            let target_token = tokens.last().copied().unwrap_or(self.blank_id as i32);
            let targets = Array2::from_shape_vec((1, 1), vec![target_token])?;
            let target_length = Array1::from_vec(vec![1i32]);
            
            let outputs = self.decoder_joint.run(ort::inputs![
                Tensor::from_array(encoder_step)?,
                Tensor::from_array(targets)?,
                Tensor::from_array(target_length)?,
                Tensor::from_array(state_1.clone())?,
                Tensor::from_array(state_2.clone())?
            ])?;
            
            let logits: ArrayViewD<f32> = outputs[0].try_extract_array()?;
            
            let new_state_1: ArrayViewD<f32> = outputs[2].try_extract_array()?;
            let new_state_2: ArrayViewD<f32> = outputs[3].try_extract_array()?;
            
            let new_state_1 = new_state_1.to_owned().into_dimensionality()?;
            let new_state_2 = new_state_2.to_owned().into_dimensionality()?;
            
            let logit_slice = logits.as_slice().ok_or_else(|| {
                anyhow!("Failed to convert logits to slice")
            })?;
            
            let total_logits = logit_slice.len();
            let vocab_size = self.vocab.len();
            
            let vocab_logits = if total_logits > vocab_size {
                &logit_slice[..vocab_size]
            } else {
                logit_slice
            };
            
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            
            for (i, &val) in vocab_logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            if max_idx != self.blank_id {
                state_1 = new_state_1;
                state_2 = new_state_2;
                tokens.push(max_idx as i32);
                emitted_tokens += 1;
            }
            
            if max_idx == self.blank_id || emitted_tokens >= MAX_TOKENS_PER_STEP {
                t += 1;
                emitted_tokens = 0;
            }
        }
        
        // Drop encoder_outputs to release the mutable borrow
        drop(encoder_outputs);

        let text = self.tokens_to_text(&tokens);
        Ok(text)
    }

    fn tokens_to_text(&self, tokens: &[i32]) -> String {
        let mut text = String::new();
        for &token_id in tokens {
            let idx = token_id as usize;
            if idx < self.vocab.len() {
                let token = &self.vocab[idx];
                if token == "▁" {
                    text.push(' ');
                } else if token.starts_with('▁') {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&token[3..]);
                } else {
                    text.push_str(token);
                }
            }
        }
        text.trim().to_string()
    }
}

pub fn resample_audio(input: &[f32], in_rate: usize, out_rate: usize) -> Result<Vec<f32>> {
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        out_rate as f64 / in_rate as f64,
        2.0,
        params,
        input.len(),
        1,
    )?;

    let waves_in = vec![input.to_vec()];
    let waves_out = resampler.process(&waves_in, None)?;

    Ok(waves_out[0].clone())
}

pub fn extract_log_mel_features(waveform: &[f32]) -> Result<Array2<f32>> {
    let n_frames = (waveform.len() - N_FFT) / HOP_LENGTH + 1;
    
    let mut spectrogram = Array2::<f32>::zeros((n_frames, N_FFT / 2 + 1));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);
    
    let window = hann_window(N_FFT);

    for i in 0..n_frames {
        let start = i * HOP_LENGTH;
        let end = start + N_FFT;
        
        if end > waveform.len() {
            break;
        }

        let mut frame: Vec<Complex<f32>> = waveform[start..end]
            .iter()
            .zip(window.iter())
            .map(|(&x, &w)| Complex::new(x * w, 0.0))
            .collect();

        fft.process(&mut frame);

        for j in 0..(N_FFT / 2 + 1) {
            spectrogram[[i, j]] = frame[j].norm().powi(2);
        }
    }

    let mel_basis = create_mel_filterbank(N_FFT / 2 + 1, N_MELS, SAMPLE_RATE, MEL_FMIN, MEL_FMAX);
    
    let mut mel_spec = Array2::<f32>::zeros((n_frames, N_MELS));
    for i in 0..n_frames {
        for j in 0..N_MELS {
            let mut sum = 0.0;
            for k in 0..(N_FFT / 2 + 1) {
                sum += spectrogram[[i, k]] * mel_basis[[j, k]];
            }
            mel_spec[[i, j]] = sum;
        }
    }

    for i in 0..n_frames {
        for j in 0..N_MELS {
            mel_spec[[i, j]] = (mel_spec[[i, j]].max(1e-10)).ln();
        }
    }

    let mean = mel_spec.mean().unwrap_or(0.0);
    let std = mel_spec.std(0.0).max(1e-10);
    mel_spec = (mel_spec - mean) / std;

    Ok(mel_spec)
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let cos_val = (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos();
            0.5 * (1.0 - cos_val)
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn create_mel_filterbank(n_fft: usize, n_mels: usize, sr: usize, fmin: f32, fmax: f32) -> Array2<f32> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft - 1) as f32 * hz / (sr as f32 / 2.0)).floor() as usize)
        .collect();

    let mut filterbank = Array2::<f32>::zeros((n_mels, n_fft));

    for i in 0..n_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for j in left..center {
            filterbank[[i, j]] = (j - left) as f32 / (center - left) as f32;
        }
        
        for j in center..right {
            filterbank[[i, j]] = (right - j) as f32 / (right - center) as f32;
        }
    }

    filterbank
}

pub fn load_wav_file(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max_val = 2_i32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
    };

    let mono_samples = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };

    Ok((mono_samples, spec.sample_rate))
}
