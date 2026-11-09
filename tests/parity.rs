//! Rust detector vs the Python reference path (training/parity_ref.py).
//! Run with WAKE_PARITY_JSON=<ref.json> WAKE_MANIFEST=<release dir> WAKE_PARITY_HEAD=<phrase>.
use waqy::detector::{EmbeddingHeads, init_runtime};

#[test]
fn scores_match_python_reference() {
    let (Ok(reference), Ok(manifest)) = (
        std::env::var("WAKE_PARITY_JSON"),
        std::env::var("WAKE_MANIFEST"),
    ) else {
        eprintln!("skipped: set WAKE_PARITY_JSON and WAKE_MANIFEST");
        return;
    };
    let head = std::env::var("WAKE_PARITY_HEAD").unwrap_or_else(|_| "oqto".into());
    init_runtime(None).unwrap();
    let mut detector = EmbeddingHeads::load(std::path::Path::new(&manifest), None)
        .unwrap()
        .with_dither(0.0);
    let index = waqy::detector::WakeDetector::phrases(&detector)
        .iter()
        .position(|p| *p == head)
        .expect("head in manifest");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let audio: Vec<f32> = case["audio"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected: Vec<f32> = case["scores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        waqy::detector::WakeDetector::reset(&mut detector);
        let (_, frames) = detector.push_scores(&audio).unwrap();
        let got: Vec<f32> = frames.iter().map(|f| f[index]).collect();
        assert_eq!(got.len(), expected.len(), "{}: frame count", case["wav"]);
        let worst = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.01, "{}: max |diff| {worst}", case["wav"]);
    }
}
