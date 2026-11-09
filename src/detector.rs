//! Streaming wake detectors behind one interface.
//!
//! A detector consumes 16 kHz mono PCM and reports detections as evidence
//! ({phrase, score, sample position}). It never decides what a detection means:
//! consumers (eaRS, Foxline, Oqto) map phrases to actions.
//!
//! `EmbeddingHeads` loads any release directory described by a `manifest.json`
//! (front end + one classifier head per phrase + threshold), so the model is data,
//! not code. Its streaming layout is identical to `web/waqy-engine.mjs`:
//! per 80 ms frame, 8 new log-mel frames from the last 1760 samples, floored at the
//! loudest frame of the trailing 2 s minus 80 dB, scaled x/10+2; one 96-dim
//! embedding from the last 76 mel frames; each head scores the last 16 embeddings.

use anyhow::{Context, Result, anyhow, bail};
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME: usize = 1280;
const MEL_CONTEXT: usize = 480;
const MEL_WINDOW: usize = 76;
const MEL_BINS: usize = 32;
const EMB_WINDOW: usize = 16;
const EMB_DIM: usize = 96;
const FLOOR_FRAMES: usize = 200;
const TOP_DB: f32 = 80.0;
/// Inaudible white noise (about -70 dBFS) added to every input sample. Exact digital
/// silence before speech made the heads fire on look-alikes (Oktave/Oktopus: up to half
/// of look-alike sentences); with this dither they stay at 0/240 and recall is unchanged.
pub const DEFAULT_DITHER: f32 = 3e-4;

/// One detection. `at_sample` counts samples since the detector started (or was reset),
/// including samples dropped while playback was active.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub phrase: String,
    pub score: f32,
    pub at_sample: u64,
}

/// The interface every backend implements.
pub trait WakeDetector: Send {
    /// Phrases this detector can report.
    fn phrases(&self) -> Vec<String>;
    /// Feed 16 kHz mono samples in [-1, 1]; returns detections completed by this audio.
    fn push(&mut self, pcm: &[f32]) -> Result<Vec<Detection>>;
    /// Half-duplex gate: while the host plays audio, the microphone hears it too.
    fn set_playback(&mut self, active: bool);
    /// Forget all audio context (e.g. after a device change).
    fn reset(&mut self);
    /// Near misses since the last call: score peaks that rose above the report floor
    /// but never crossed the threshold. Diagnostic only; never acted upon.
    fn drain_candidates(&mut self) -> Vec<Detection> {
        Vec::new()
    }
}

/// `manifest.json` written by `training/release.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub release: String,
    pub threshold: f32,
    pub classifiers: Vec<String>,
    pub files: BTreeMap<String, ManifestFile>,
    #[serde(default)]
    pub authority: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ManifestFile {
    pub file: String,
    pub sha256: String,
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("manifest.json"))
            .with_context(|| format!("reading {}/manifest.json", dir.display()))?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        for key in ["mel", "embedding"] {
            if !manifest.files.contains_key(key) {
                bail!("manifest has no '{key}' file");
            }
        }
        for phrase in &manifest.classifiers {
            if !manifest.files.contains_key(phrase) {
                bail!("manifest lists classifier '{phrase}' without a file");
            }
        }
        Ok(manifest)
    }

    /// Path of a listed file after checking its sha256 (a release is immutable).
    pub fn verified_path(&self, dir: &Path, key: &str) -> Result<PathBuf> {
        use sha2::{Digest, Sha256};
        let entry = self
            .files
            .get(key)
            .ok_or_else(|| anyhow!("no file '{key}' in manifest"))?;
        if entry.file.contains('/') || entry.file.contains("..") {
            bail!("manifest file name '{}' must be a plain name", entry.file);
        }
        let path = dir.join(&entry.file);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if digest != entry.sha256 {
            bail!("checksum mismatch for {}", entry.file);
        }
        Ok(path)
    }
}

/// Load ONNX Runtime once (dynamic library; ORT_DYLIB_PATH or a platform default).
pub fn init_runtime(explicit: Option<&Path>) -> Result<()> {
    use std::sync::OnceLock;
    static DONE: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    DONE.get_or_init(|| {
        let candidates: Vec<PathBuf> = match explicit
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
        {
            Some(path) => vec![path],
            None => [
                "/opt/homebrew/lib/libonnxruntime.dylib",
                "/usr/local/lib/libonnxruntime.dylib",
                "/usr/lib/libonnxruntime.so",
                "/usr/local/lib/libonnxruntime.so",
                "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
            ]
            .iter()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .collect(),
        };
        let path = candidates
            .first()
            .ok_or("ONNX Runtime not found; set ORT_DYLIB_PATH")?;
        ort::init_from(path)
            .map_err(|e| e.to_string())?
            .with_name("wake")
            .commit();
        Ok(())
    })
    .clone()
    .map_err(|e| anyhow!(e))
}

fn session(path: &Path) -> Result<Session> {
    Session::builder()?
        .with_intra_threads(1)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}

fn first_output(
    session: &mut Session,
    name: &str,
    shape: Vec<i64>,
    data: Vec<f32>,
) -> Result<Vec<f32>> {
    let input = Tensor::from_array((shape, data))?;
    let outputs = session.run(ort::inputs![name => input])?;
    let (_, values) = outputs[0].try_extract_tensor::<f32>()?;
    Ok(values.to_vec())
}

/// Manifest-driven detector: shared front end plus one head per phrase.
pub struct EmbeddingHeads {
    release: String,
    threshold: f32,
    refractory_frames: u64,
    mel: Session,
    embedding: Session,
    heads: Vec<(String, Session)>,
    // streaming state
    pending: Vec<f32>,
    audio: Vec<f32>,
    peaks: VecDeque<f32>,
    mel_frames: VecDeque<[f32; MEL_BINS]>,
    embeddings: VecDeque<Vec<f32>>,
    frame_index: u64,
    paused_until: Option<u64>, // Some(u64::MAX) while playing
    cooldown: Vec<u64>,
    report_above: Option<f32>,
    dither: f32,
    rng: u64,
    runs: Vec<Option<(f32, u64, bool)>>, // per head: (peak, at_sample, crossed) while above the floor
    candidates: Vec<Detection>,
}

impl EmbeddingHeads {
    pub fn load(dir: &Path, threshold: Option<f32>) -> Result<Self> {
        let manifest = Manifest::load(dir)?;
        let mel = session(&manifest.verified_path(dir, "mel")?)?;
        let embedding = session(&manifest.verified_path(dir, "embedding")?)?;
        let mut heads = Vec::new();
        for phrase in &manifest.classifiers {
            heads.push((
                phrase.clone(),
                session(&manifest.verified_path(dir, phrase)?)?,
            ));
        }
        let mut detector = Self {
            release: manifest.release.clone(),
            threshold: threshold.unwrap_or(manifest.threshold),
            refractory_frames: 25, // 2 s
            mel,
            embedding,
            cooldown: vec![0; heads.len()],
            report_above: None,
            dither: DEFAULT_DITHER,
            rng: 0x9E37_79B9_7F4A_7C15,
            runs: vec![None; heads.len()],
            candidates: Vec::new(),
            heads,
            pending: Vec::new(),
            audio: Vec::new(),
            peaks: VecDeque::new(),
            mel_frames: VecDeque::new(),
            embeddings: VecDeque::new(),
            frame_index: 0,
            paused_until: None,
        };
        detector.reset();
        Ok(detector)
    }

    /// Also report near misses: peaks at or above `floor` that stay below the threshold.
    pub fn report_above(mut self, floor: Option<f32>) -> Self {
        self.report_above = floor;
        self
    }

    /// Dither standard deviation (0 disables it; parity tests against the Python path use 0).
    pub fn with_dither(mut self, std: f32) -> Self {
        self.dither = std;
        self
    }

    fn gaussian(&mut self) -> f32 {
        // xorshift64* + Box-Muller: deterministic, no dependency
        let mut next = || {
            self.rng ^= self.rng >> 12;
            self.rng ^= self.rng << 25;
            self.rng ^= self.rng >> 27;
            ((self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5)
                / (1u64 << 53) as f64
        };
        let (u, v) = (next(), next());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }

    pub fn release(&self) -> &str {
        &self.release
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    fn clear_state(&mut self) {
        self.pending.clear();
        self.audio = vec![0.0; FRAME + MEL_CONTEXT];
        self.peaks.clear();
        self.mel_frames.clear();
        self.embeddings.clear();
        self.cooldown.iter_mut().for_each(|c| *c = 0);
        self.runs.iter_mut().for_each(|r| *r = None);
    }

    fn paused(&self) -> bool {
        matches!(self.paused_until, Some(until) if self.frame_index < until)
    }

    /// One 80 ms frame. Returns per-phrase scores once 16 embeddings exist.
    fn step(&mut self, frame: &[f32]) -> Result<Option<Vec<(usize, f32)>>> {
        if self.paused() {
            self.frame_index += 1;
            return Ok(None);
        }
        self.audio.copy_within(FRAME.., 0);
        self.audio[MEL_CONTEXT..].copy_from_slice(frame);
        self.frame_index += 1;

        let n = self.audio.len() as i64;
        let mel = first_output(&mut self.mel, "input", vec![1, n], self.audio.clone())?;
        for row in mel.chunks_exact(MEL_BINS) {
            let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            self.peaks.push_back(peak);
            if self.peaks.len() > FLOOR_FRAMES {
                self.peaks.pop_front();
            }
            let floor = self.peaks.iter().copied().fold(f32::NEG_INFINITY, f32::max) - TOP_DB;
            let mut out = [0.0f32; MEL_BINS];
            for (o, &x) in out.iter_mut().zip(row) {
                *o = x.max(floor) / 10.0 + 2.0;
            }
            self.mel_frames.push_back(out);
        }
        while self.mel_frames.len() > MEL_WINDOW {
            self.mel_frames.pop_front();
        }
        if self.mel_frames.len() < MEL_WINDOW {
            return Ok(None);
        }
        let window: Vec<f32> = self
            .mel_frames
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        let emb = first_output(
            &mut self.embedding,
            "input_1",
            vec![1, MEL_WINDOW as i64, MEL_BINS as i64, 1],
            window,
        )?;
        self.embeddings.push_back(emb[..EMB_DIM].to_vec());
        if self.embeddings.len() > EMB_WINDOW {
            self.embeddings.pop_front();
        }
        if self.embeddings.len() < EMB_WINDOW {
            return Ok(None);
        }
        let feats: Vec<f32> = self
            .embeddings
            .iter()
            .flat_map(|e| e.iter().copied())
            .collect();
        let mut scores = Vec::with_capacity(self.heads.len());
        for (i, (_, head)) in self.heads.iter_mut().enumerate() {
            let out = first_output(
                head,
                "embeddings",
                vec![1, EMB_WINDOW as i64, EMB_DIM as i64],
                feats.clone(),
            )?;
            scores.push((i, out[0]));
        }
        Ok(Some(scores))
    }

    /// Like `push`, but also returns every frame's scores (for parity tests and logging).
    pub fn push_scores(&mut self, pcm: &[f32]) -> Result<(Vec<Detection>, Vec<Vec<f32>>)> {
        if self.dither > 0.0 {
            for &x in pcm {
                let n = self.gaussian() * self.dither;
                self.pending.push(x + n);
            }
        } else {
            self.pending.extend_from_slice(pcm);
        }
        let mut detections = Vec::new();
        let mut all = Vec::new();
        let mut offset = 0;
        while self.pending.len() - offset >= FRAME {
            let frame: Vec<f32> = self.pending[offset..offset + FRAME].to_vec();
            offset += FRAME;
            if let Some(scores) = self.step(&frame)? {
                all.push(scores.iter().map(|&(_, s)| s).collect());
                let at_sample = self.frame_index * FRAME as u64;
                for (i, score) in scores {
                    let mut crossed = false;
                    if self.cooldown[i] > 0 {
                        self.cooldown[i] -= 1;
                    } else if score >= self.threshold {
                        self.cooldown[i] = self.refractory_frames;
                        crossed = true;
                        detections.push(Detection {
                            phrase: self.heads[i].0.clone(),
                            score,
                            at_sample,
                        });
                    }
                    if let Some(floor) = self.report_above {
                        if score >= floor || crossed {
                            let run = self.runs[i].get_or_insert((score, at_sample, false));
                            if score > run.0 {
                                (run.0, run.1) = (score, at_sample);
                            }
                            run.2 |= crossed || score >= self.threshold;
                        } else if let Some((peak, at, false)) = self.runs[i].take() {
                            self.candidates.push(Detection {
                                phrase: self.heads[i].0.clone(),
                                score: peak,
                                at_sample: at,
                            });
                        }
                    }
                }
            }
        }
        self.pending.drain(..offset);
        Ok((detections, all))
    }
}

impl WakeDetector for EmbeddingHeads {
    fn phrases(&self) -> Vec<String> {
        self.heads.iter().map(|(p, _)| p.clone()).collect()
    }

    fn push(&mut self, pcm: &[f32]) -> Result<Vec<Detection>> {
        Ok(self.push_scores(pcm)?.0)
    }

    fn set_playback(&mut self, active: bool) {
        if active {
            self.paused_until = Some(u64::MAX);
        } else if self.paused_until == Some(u64::MAX) {
            self.clear_state();
            self.paused_until = Some(self.frame_index + 350 / 80);
        }
    }

    fn drain_candidates(&mut self) -> Vec<Detection> {
        std::mem::take(&mut self.candidates)
    }

    fn reset(&mut self) {
        self.frame_index = 0;
        self.paused_until = None;
        self.clear_state();
    }
}
