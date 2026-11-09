//! Legacy rustpotter backend (DTW references and trained tiny models).
//! Kept feature-gated for the pilot record; the product backend is `detector`.
#![allow(deprecated)]

use rustpotter::{
    Rustpotter, RustpotterConfig, SampleFormat, VADMode, WakewordLoad, WakewordModel,
    WakewordModelTrain, WakewordModelTrainOptions, WakewordRef, WakewordRefBuildFromFiles,
    WakewordSave,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

pub use rustpotter::ModelType;

/// MFCC coefficients per frame used for both artifact kinds.
pub const MFCC_SIZE: u16 = 16;

/// Kind of wake artifact; both are files produced by rustpotter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactKind {
    /// Averaged MFCC templates from a few positive takes, matched with DTW. No training.
    Reference,
    /// Small classifier trained on labelled positives and negatives.
    Model,
}

/// Build a reference artifact from positive takes (any WAV rate; rustpotter resamples).
pub fn build_reference(name: &str, takes: &[PathBuf], out: &Path) -> Result<()> {
    if takes.len() < 2 {
        bail!("a reference needs at least two takes, got {}", takes.len());
    }
    let samples = takes
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let reference =
        WakewordRef::new_from_sample_files(name.to_owned(), None, None, samples, MFCC_SIZE)
            .map_err(|e| anyhow!(e))
            .context("building wake reference")?;
    reference
        .save_to_file(&out.to_string_lossy())
        .map_err(|e| anyhow!(e))
}

/// Train a model artifact. Each directory holds `.wav` files whose names start with a
/// `[label]` prefix; `[none]` is the conventional negative label.
pub fn train_model(
    train_dir: &Path,
    test_dir: &Path,
    model_type: ModelType,
    epochs: usize,
    out: &Path,
) -> Result<()> {
    let options =
        WakewordModelTrainOptions::new(model_type, 0.017, epochs, (epochs / 10).max(1), MFCC_SIZE);
    let model = WakewordModel::train_from_dirs(
        options,
        train_dir.to_string_lossy().into_owned(),
        test_dir.to_string_lossy().into_owned(),
        None,
    )
    .context("training wake model")?;
    model
        .save_to_file(&out.to_string_lossy())
        .map_err(|e| anyhow!(e))
}

/// Detector thresholds; defaults follow rustpotter's own defaults.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    pub threshold: f32,
    pub avg_threshold: f32,
    pub min_scores: usize,
    /// Voice-activity gate: 0 off, 1 easy, 2 medium, 3 hard. Silent frames are skipped.
    pub vad: u8,
    /// Normalize input gain towards the artifact's recorded loudness.
    pub gain_normalizer: bool,
}

impl Default for Thresholds {
    fn default() -> Self {
        let detector = RustpotterConfig::default().detector;
        Self {
            threshold: detector.threshold,
            avg_threshold: detector.avg_threshold,
            min_scores: detector.min_scores,
            vad: 0,
            gain_normalizer: false,
        }
    }
}

/// One detection inside a stream or file.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub name: String,
    pub score: f32,
    pub avg_score: f32,
    /// Stream position (seconds) at which the detection was emitted.
    pub at_seconds: f64,
}

/// Streaming detector over mono f32 PCM at a fixed input sample rate.
pub struct Spotter {
    inner: Rustpotter,
    frame: usize,
    pending: Vec<f32>,
    sample_rate: usize,
    consumed: usize,
}

impl Spotter {
    /// Load one artifact (reference or model, detected from its contents).
    pub fn new(artifact: &Path, sample_rate: usize, thresholds: Thresholds) -> Result<Self> {
        let mut config = RustpotterConfig::default();
        config.fmt.sample_rate = sample_rate;
        config.fmt.sample_format = SampleFormat::F32;
        config.fmt.channels = 1;
        config.detector.threshold = thresholds.threshold;
        config.detector.avg_threshold = thresholds.avg_threshold;
        config.detector.min_scores = thresholds.min_scores;
        config.detector.vad_mode = match thresholds.vad {
            0 => None,
            1 => Some(VADMode::Easy),
            2 => Some(VADMode::Medium),
            _ => Some(VADMode::Hard),
        };
        config.filters.gain_normalizer.enabled = thresholds.gain_normalizer;
        let mut inner = Rustpotter::new(&config).map_err(|e| anyhow!(e))?;
        let key = artifact
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("wake")
            .to_owned();
        inner
            .add_wakeword_from_file(&key, &artifact.to_string_lossy())
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("loading {}", artifact.display()))?;
        let frame = inner.get_samples_per_frame();
        Ok(Self {
            inner,
            frame,
            pending: Vec::new(),
            sample_rate,
            consumed: 0,
        })
    }

    /// Feed PCM; returns detections completed by this chunk.
    pub fn push(&mut self, pcm: &[f32]) -> Vec<Hit> {
        self.pending.extend_from_slice(pcm);
        let mut hits = Vec::new();
        let mut offset = 0;
        while self.pending.len() - offset >= self.frame {
            let chunk = self.pending[offset..offset + self.frame].to_vec();
            offset += self.frame;
            self.consumed += self.frame;
            if let Some(d) = self.inner.process_samples(chunk) {
                hits.push(Hit {
                    name: d.name,
                    score: d.score,
                    avg_score: d.avg_score,
                    at_seconds: self.consumed as f64 / self.sample_rate as f64,
                });
            }
        }
        self.pending.drain(..offset);
        hits
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.pending.clear();
        self.consumed = 0;
    }
}

/// Read a WAV file as mono f32 PCM plus its sample rate.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, usize)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate as usize))
}

/// Scan a whole file with silence padding so a phrase at either edge can complete.
pub fn scan_file(artifact: &Path, wav: &Path, thresholds: Thresholds) -> Result<Vec<Hit>> {
    let (pcm, rate) = read_wav(wav)?;
    let mut spotter = Spotter::new(artifact, rate, thresholds)?;
    let pad = vec![0.0f32; rate];
    let mut hits = spotter.push(&pad);
    hits.extend(spotter.push(&pcm));
    hits.extend(spotter.push(&pad));
    hits.extend(spotter.push(&pad));
    Ok(hits)
}

/// Whether a loaded artifact is a reference or a trained model.
pub fn artifact_kind(path: &Path) -> Result<ArtifactKind> {
    if WakewordRef::load_from_file(&path.to_string_lossy()).is_ok() {
        Ok(ArtifactKind::Reference)
    } else if WakewordModel::load_from_file(&path.to_string_lossy()).is_ok() {
        Ok(ArtifactKind::Model)
    } else {
        bail!("{} is not a rustpotter reference or model", path.display())
    }
}
