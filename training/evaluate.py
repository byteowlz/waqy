"""Streaming evaluation of an exported wake classifier.

Scores are computed the way the runtime sees audio: one decision per 80 ms embedding frame,
each over the last 16 frames (1.28 s). A detection is a threshold crossing followed by a
2 s refractory period, so false accepts are counted as events, not windows.

  general FA/h   held-out general audio (validation_set_features.npy + .segments.npy)
  recall         held-out-voice synthetic positives (positive_test raw clips), per language
  adversarial    held-out-voice confusable phrases (negative_test raw clips), per language
  owner          optional: wake-collect-url-1 takes, by phrase id (evidence only)

Usage: .venv/bin/python evaluate.py <root> <model> [--onnx PATH] [--owner DIR]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from build_data import RATE, Featurizer, to16k

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
REFRACTORY = 25  # frames, 2 s


class Scorer:
    def __init__(self, onnx_path: Path):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def stream(self, frames: np.ndarray) -> np.ndarray:
        """Per-frame scores for a contiguous (n,96) frame sequence (n-15 decisions)."""
        if len(frames) < 16:
            return np.zeros(0, np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(frames.astype(np.float32), (16, 96))[:, 0]
        out = [self.session.run(None, {"embeddings": windows[i:i + 4096]})[0][:, 0] for i in range(0, len(windows), 4096)]
        return np.concatenate(out)


def events(scores: np.ndarray, threshold: float) -> int:
    count, i = 0, 0
    above = np.flatnonzero(scores >= threshold)
    for idx in above:
        if idx >= i:
            count += 1
            i = idx + REFRACTORY
    return count


def clip_frames(feat: Featurizer, pcm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Embed a short clip with 1.5 s lead-in and 0.6 s tail of faint noise, as in a live stream."""
    lead = rng.normal(0, 3e-4, int(1.5 * RATE) + int(rng.integers(0, 1280))).astype(np.float32)  # random 80 ms phase
    tail = rng.normal(0, 3e-4, int(0.6 * RATE)).astype(np.float32)
    return feat.frames(np.concatenate([lead, pcm.astype(np.float32), tail]))


def max_scores(scorer: Scorer, feat: Featurizer, paths: list[Path]) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = []
    for path in paths:
        pcm, rate = sf.read(str(path), dtype="float32")
        s = scorer.stream(clip_frames(feat, to16k(pcm, rate), rng))
        out.append(float(s.max()) if len(s) else 0.0)
    return np.array(out)


def family(source: str) -> str:
    if source.startswith("musan:"):
        parts = source.split("/")
        return "musan-" + parts[1] if len(parts) > 2 else "musan"
    if source.startswith("voxpopuli:"):
        return "voxpopuli-de"
    return source.split(":")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("model")
    ap.add_argument("--onnx")
    ap.add_argument("--owner")
    ap.add_argument("--out")
    ap.add_argument("--owner-sessions", help="comma-separated sessionId prefixes to score (others are not opened)")
    ap.add_argument("--clips", help="model dir holding synth-manifest.jsonl and raw test clips (default: the model's own)")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out_dir = root / "output" / args.model
    clip_dir = root / "output" / (args.clips or args.model)
    onnx_path = Path(args.onnx) if args.onnx else out_dir / f"{args.model}.onnx"
    scorer, feat = Scorer(onnx_path), Featurizer()
    report: dict = {"model": args.model, "onnx": str(onnx_path), "thresholds": THRESHOLDS,
                    "decision": "per 80 ms frame over last 16 frames; event = crossing + 2 s refractory"}

    frames = np.load(root / "data/features/validation_set_features.npy")
    lengths = np.load(root / "data/features/validation_set_features.segments.npy")
    owners = []  # family per segment, e.g. musan-speech, voxpopuli-de, fleurs-en-test
    for entry in json.loads((root / "data/features/validation_set_features.segment-sources.json").read_text()):
        owners += [family(entry["source"])] * entry["segments"]
    assert len(owners) == len(lengths)
    hours = frames.shape[0] * 0.08 / 3600
    counts, fam_counts, fam_frames = defaultdict(int), defaultdict(lambda: defaultdict(int)), defaultdict(int)
    worst = []
    start = 0
    for n, fam in zip(lengths, owners):
        s = scorer.stream(frames[start:start + n])
        start += n
        fam_frames[fam] += int(n)
        if len(s):
            worst.append((float(s.max()), fam, int(start - n + 15 + int(s.argmax()))))
        for t in THRESHOLDS:
            e = events(s, t)
            counts[t] += e
            fam_counts[fam][t] += e
    report["general"] = {"hours": round(hours, 3), "segments": int(len(lengths)),
                         "fa_per_hour": {str(t): round(counts[t] / hours, 3) for t in THRESHOLDS},
                         "fa_events": {str(t): counts[t] for t in THRESHOLDS},
                         "by_family": {f: {"hours": round(fam_frames[f] * 0.08 / 3600, 3),
                                           "fa_per_hour": {str(t): round(fam_counts[f][t] / (fam_frames[f] * 0.08 / 3600), 3) for t in THRESHOLDS}}
                                       for f in sorted(fam_frames)},
                         "top_scores": [{"score": round(sc, 4), "family": f, "frame": fr} for sc, f, fr in sorted(worst, reverse=True)[:10]]}

    manifest = [json.loads(l) for l in (clip_dir / "synth-manifest.jsonl").read_text().splitlines()]
    for kind in ("positive", "negative"):
        rows = [r for r in manifest if r["split"] == "test" and r["kind"] == kind
                and (clip_dir / f"{kind}_test" / r["file"]).exists()]
        result = {}
        for lang in ("de", "en"):
            picked = [r for r in rows if r["lang"] == lang]
            scores = max_scores(scorer, feat, [clip_dir / f"{kind}_test" / r["file"] for r in picked])
            rate = {str(t): round(float((scores >= t).mean()), 4) if len(scores) else None for t in THRESHOLDS}
            result[lang] = {"n": len(picked), ("recall" if kind == "positive" else "false_accept_rate"): rate}
            if kind == "negative":
                by_text = defaultdict(list)
                for r, sc in zip(picked, scores):
                    by_text[r["text"]].append(sc)
                result[lang]["worst_phrases_at_0.5"] = sorted(
                    ((t, round(float(np.mean(np.array(v) >= 0.5)), 3), len(v)) for t, v in by_text.items()),
                    key=lambda x: -x[1])[:8]
        report["synthetic_" + ("positives" if kind == "positive" else "adversarial")] = result

    if args.owner:
        owner = Path(args.owner).expanduser()
        rows = []
        wanted = args.owner_sessions.split(",") if args.owner_sessions else None
        for meta_path in sorted(owner.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text())
            if wanted and not any(meta["sessionId"].startswith(w) for w in wanted):
                continue
            rows.append((meta, meta_path.parent / "audio.wav"))
        if rows:
            scores = max_scores(scorer, feat, [p for _, p in rows])
            by_phrase = defaultdict(list)
            for (meta, _), sc in zip(rows, scores):
                by_phrase[f"{meta['language']}/{meta['label']}/{meta['phrase']}"].append(sc)
            report["owner_takes"] = {k: {"n": len(v), "max_scores": [round(float(x), 3) for x in v],
                                         "hit_rate": {str(t): round(float(np.mean(np.array(v) >= t)), 3) for t in THRESHOLDS}}
                                     for k, v in sorted(by_phrase.items())}
            report["owner_sessions"] = sorted({m["sessionId"][:8] for m, _ in rows})
        else:
            report["owner_takes"] = "none recorded"

    out = Path(args.out) if args.out else out_dir / "streaming-eval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
