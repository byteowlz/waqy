use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "rustpotter")]
use serde::Serialize;
#[cfg(feature = "rustpotter")]
use waqy::{ModelType, artifact_kind, build_reference, scan_file, train_model};

/// Oqto wake-phrase artifacts and offline evaluation. Detections are evidence only.
#[derive(Parser)]
#[command(name = "waqy", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Kind {
    Tiny,
    Small,
    Medium,
    Large,
}

#[derive(Subcommand)]
enum Command {
    /// Build a reference (DTW templates) from a few positive takes. No training.
    #[cfg(feature = "rustpotter")]
    Build {
        #[arg(long)]
        name: String,
        #[arg(long)]
        out: PathBuf,
        takes: Vec<PathBuf>,
    },
    /// Train a classifier from `[label]`-prefixed WAVs in train/test directories.
    #[cfg(feature = "rustpotter")]
    Train {
        #[arg(long)]
        train: PathBuf,
        #[arg(long)]
        test: PathBuf,
        #[arg(long, value_enum, default_value = "small")]
        kind: Kind,
        #[arg(long, default_value_t = 1000)]
        epochs: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run a manifest-driven detector over the wake-stream/1 protocol on stdin/stdout.
    Listen {
        /// Release directory with manifest.json (front end + one head per phrase).
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        threshold: Option<f32>,
        /// Also report near misses (peaks at or above this score that stay below the threshold).
        #[arg(long)]
        report_above: Option<f32>,
        /// Input dither standard deviation (default about -70 dBFS; 0 disables).
        #[arg(long, default_value_t = waqy::detector::DEFAULT_DITHER)]
        dither: f32,
        /// ONNX Runtime library (else ORT_DYLIB_PATH or a platform default).
        #[arg(long)]
        ort_dylib: Option<PathBuf>,
    },
    /// Validate a release folder: manifest, protocol expectations, per-file hashes, phrases.
    Check {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Measure CPU cost of a release: milliseconds per 80 ms frame, one thread.
    Bench {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = 300)]
        frames: usize,
        #[arg(long)]
        ort_dylib: Option<PathBuf>,
    },
    /// Stream 16 kHz WAV files through a manifest detector; one JSON line per detection.
    DetectFile {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        threshold: Option<f32>,
        #[arg(long)]
        ort_dylib: Option<PathBuf>,
        /// Also print every frame's scores (parity checks).
        #[arg(long)]
        scores: bool,
        /// Input dither standard deviation (0 for parity checks against the Python path).
        #[arg(long, default_value_t = waqy::detector::DEFAULT_DITHER)]
        dither: f32,
        files: Vec<PathBuf>,
    },
    /// Scan WAV files and print one JSON line per file (legacy rustpotter backend).
    #[cfg(feature = "rustpotter")]
    Scan {
        #[arg(long)]
        model: PathBuf,
        #[command(flatten)]
        thresholds: ThresholdArgs,
        files: Vec<PathBuf>,
    },
}

#[cfg(feature = "rustpotter")]
#[derive(clap::Args)]
struct ThresholdArgs {
    #[arg(long)]
    threshold: Option<f32>,
    #[arg(long)]
    avg_threshold: Option<f32>,
    #[arg(long)]
    min_scores: Option<usize>,
    /// Voice-activity gate: 0 off, 1 easy, 2 medium, 3 hard.
    #[arg(long, default_value_t = 0)]
    vad: u8,
    #[arg(long)]
    gain_normalizer: bool,
}

#[cfg(feature = "rustpotter")]
impl ThresholdArgs {
    fn resolve(&self) -> Thresholds {
        let d = Thresholds::default();
        Thresholds {
            threshold: self.threshold.unwrap_or(d.threshold),
            avg_threshold: self.avg_threshold.unwrap_or(d.avg_threshold),
            min_scores: self.min_scores.unwrap_or(d.min_scores),
            vad: self.vad,
            gain_normalizer: self.gain_normalizer,
        }
    }
}

#[cfg(feature = "rustpotter")]
#[derive(Serialize)]
struct ScanLine {
    file: String,
    detected: bool,
    best_score: Option<f32>,
    hits: Vec<waqy::Hit>,
    error: Option<String>,
}

/// Read a WAV file as mono f32 plus its sample rate (detect-file input).
fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, usize)> {
    let mut reader = hound::WavReader::open(path).map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<std::result::Result<_, _>>()?
        }
    };
    let mono = interleaved.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
    Ok((mono, spec.sample_rate as usize))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Listen {
            manifest,
            threshold,
            report_above,
            dither,
            ort_dylib,
        } => {
            waqy::detector::init_runtime(ort_dylib.as_deref())?;
            let mut detector = waqy::detector::EmbeddingHeads::load(&manifest, threshold)?
                .report_above(report_above)
                .with_dither(dither);
            let release = detector.release().to_string();
            eprintln!(
                "wake listen: release {release}, threshold {}, phrases {:?}",
                detector.threshold(),
                waqy::detector::WakeDetector::phrases(&detector)
            );
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            return waqy::protocol::serve(
                &mut detector,
                "embedding-heads",
                &release,
                &mut stdin.lock(),
                &mut stdout.lock(),
            );
        }
        Command::Check { manifest } => {
            // No ORT needed: the check is about the folder, not about running it.
            let parsed = wake_manifest(&manifest)?;
            println!("{}", parsed);
            Ok(())
        }
        Command::Bench {
            manifest,
            frames,
            ort_dylib,
        } => {
            waqy::detector::init_runtime(ort_dylib.as_deref())?;
            let mut detector = waqy::detector::EmbeddingHeads::load(&manifest, None)?;
            let mut audio = vec![0.0f32; 1280];
            let mut seed = 0x9E37_79B9_7F4A_7C15u64;
            for x in audio.iter_mut() {
                seed ^= seed >> 12;
                seed ^= seed << 25;
                seed ^= seed >> 27;
                *x = ((seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / 8_388_608.0 - 0.5)
                    * 0.05;
            }
            // warm up (session init, first allocations), then measure
            for _ in 0..20 {
                waqy::detector::WakeDetector::push(&mut detector, &audio)?;
            }
            let started = std::time::Instant::now();
            for _ in 0..frames {
                waqy::detector::WakeDetector::push(&mut detector, &audio)?;
            }
            let ms = started.elapsed().as_secs_f64() * 1000.0 / frames as f64;
            println!(
                "{}",
                serde_json::json!({
                    "release": detector.release(),
                    "phrases": waqy::detector::WakeDetector::phrases(&detector),
                    "frames": frames,
                    "ms_per_80ms": (ms * 1000.0).round() / 1000.0,
                    "percent_of_one_core": ((ms / 80.0) * 100.0).round() / 100.0,
                })
            );
            Ok(())
        }
        Command::DetectFile {
            manifest,
            threshold,
            ort_dylib,
            scores,
            dither,
            files,
        } => {
            waqy::detector::init_runtime(ort_dylib.as_deref())?;
            let mut detector =
                waqy::detector::EmbeddingHeads::load(&manifest, threshold)?.with_dither(dither);
            for file in files {
                let (pcm, rate) = read_wav(&file)?;
                anyhow::ensure!(
                    rate == 16000,
                    "{} is {rate} Hz; detect-file needs 16 kHz",
                    file.display()
                );
                waqy::detector::WakeDetector::reset(&mut detector);
                let (found, all) = detector.push_scores(&pcm)?;
                if scores {
                    println!("{}", serde_json::json!({"file": file, "scores": all}));
                }
                for d in found {
                    println!(
                        "{}",
                        serde_json::json!({"file": file, "phrase": d.phrase, "score": d.score, "at_sample": d.at_sample})
                    );
                }
            }
            Ok(())
        }
        #[cfg(feature = "rustpotter")]
        other => run_legacy(Cli { command: other }),
    }
}

/// Validate a release folder without loading ONNX Runtime.
fn wake_manifest(dir: &std::path::Path) -> Result<serde_json::Value> {
    let manifest = waqy::detector::Manifest::load(dir)?;
    let mut files = serde_json::Map::new();
    for key in ["mel", "embedding"]
        .iter()
        .map(|k| k.to_string())
        .chain(manifest.classifiers.iter().cloned())
    {
        let path = manifest.verified_path(dir, &key)?;
        files.insert(key, serde_json::json!({"file": path.file_name().unwrap_or_default().to_string_lossy(), "bytes": std::fs::metadata(&path)?.len()}));
    }
    Ok(serde_json::json!({
        "ok": true,
        "release": manifest.release,
        "threshold": manifest.threshold,
        "phrases": manifest.classifiers,
        "files": files,
        "authority": manifest.authority,
    }))
}

#[cfg(feature = "rustpotter")]
fn run_legacy(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Build { name, out, takes } => {
            build_reference(&name, &takes, &out)?;
            eprintln!(
                "wrote reference {} from {} takes",
                out.display(),
                takes.len()
            );
        }
        Command::Train {
            train,
            test,
            kind,
            epochs,
            out,
        } => {
            let model_type = match kind {
                Kind::Tiny => ModelType::Tiny,
                Kind::Small => ModelType::Small,
                Kind::Medium => ModelType::Medium,
                Kind::Large => ModelType::Large,
            };
            train_model(&train, &test, model_type, epochs, &out)?;
            eprintln!("wrote model {}", out.display());
        }
        Command::Listen { .. }
        | Command::DetectFile { .. }
        | Command::Check { .. }
        | Command::Bench { .. } => unreachable!("handled in main"),
        Command::Scan {
            model,
            thresholds,
            files,
        } => {
            eprintln!("artifact kind: {:?}", artifact_kind(&model)?);
            let thresholds = thresholds.resolve();
            for file in files {
                let line = match scan_file(&model, &file, thresholds) {
                    Ok(hits) => ScanLine {
                        file: file.display().to_string(),
                        detected: !hits.is_empty(),
                        best_score: hits.iter().map(|h| h.score).reduce(f32::max),
                        hits,
                        error: None,
                    },
                    Err(e) => ScanLine {
                        file: file.display().to_string(),
                        detected: false,
                        best_score: None,
                        hits: vec![],
                        error: Some(format!("{e:#}")),
                    },
                };
                println!("{}", serde_json::to_string(&line)?);
            }
        }
    }
    Ok(())
}
