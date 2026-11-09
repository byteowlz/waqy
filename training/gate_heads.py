"""Runtime gate for retrained heads: the real detector, the owner's held-out sittings.

    uv run --no-project python gate_heads.py --candidates <dir with <head>/<head>.onnx>
        --holdout <mix>/holdout.json --store <owner recordings> [--baseline <release dir>]
        [--waqy target/release/waqy] [--out gate.json]

Every held-out take is padded with 1 s of silence on each side (the detector needs context) and
streamed through `waqy detect-file --scores` once per pack, so candidate and baseline see
identical audio through the shipped runtime, not a Python re-implementation. Per head:

- wake recall: held-out owner wake takes whose clip maximum reaches the threshold. Addressed
  heads (hey_x) count only addressed takes; bare and command heads count every wake take.
- look-alike fires: held-out owner look-alike takes that reach the threshold.
- other-word fires: the owner's takes of every other word (wake phrases and look-alikes) that
  reach the threshold. With owner takes weighted up, the failure to catch is a head that learns
  the owner's voice instead of the word; any fire here is a false trigger in the owner's voice.

A candidate passes when it is no worse than the baseline on both (or, without a baseline head,
catches at least half of the held-out wake takes and fires on no more than a fifth of the
look-alikes). Counts are small; the report gives them, not just rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = Path(__file__).parent / "frontend"
from addresses import PREFIXES as ADDRESS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_pack(directory: Path, heads: dict[str, Path], threshold: float) -> Path:
    directory.mkdir(parents=True)
    files = {}
    for key, name in (("mel", "melspectrogram_raw.onnx"), ("embedding", "embedding_model.onnx")):
        shutil.copy2(FRONTEND / name, directory / name)
        files[key] = {"file": name, "sha256": sha256(directory / name)}
    for head, path in heads.items():
        shutil.copy2(path, directory / f"{head}.onnx")
        files[head] = {"file": f"{head}.onnx", "sha256": sha256(directory / f"{head}.onnx")}
    (directory / "manifest.json").write_text(json.dumps({"release": directory.name, "threshold": threshold,
                                                         "classifiers": sorted(heads), "files": files}))
    return directory


def padded(src: Path, dst: Path) -> Path:
    with wave.open(str(src)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        frames = w.readframes(w.getnframes())
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
        out.writeframes(b"\0\0" * 16000 + frames + b"\0\0" * 16000)
    return dst


def clip_max(waqy: str, pack: Path, wavs: list[Path]) -> dict[str, dict[str, float]]:
    out = subprocess.run([waqy, "detect-file", "--manifest", str(pack), "--scores", *map(str, wavs)],
                         capture_output=True, text=True, check=True).stdout
    phrases = json.loads((pack / "manifest.json").read_text())["classifiers"]
    result = {}
    for line in out.splitlines():
        row = json.loads(line)
        if "scores" not in row:
            continue
        frames = row["scores"]
        result[Path(row["file"]).stem] = {p: max(f[i] for f in frames) for i, p in enumerate(phrases)}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--waqy", default=str(ROOT / "target/release/waqy"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-other", type=int, default=400, help="cap on other-word owner takes scored")
    ap.add_argument("--others-eval", type=Path, help="owner_negative_eval.json: only these takes may serve as other-word negatives (the rest trained)")
    args = ap.parse_args()
    holdout = json.loads(args.holdout.read_text())
    head_words = {(h.split("_", 1)[1] if h.startswith(("hey_", "he_")) else h) for h in
                  (p.parent.name for p in args.candidates.glob("*/*.onnx") if p.stem == p.parent.name)}
    allowed = set(json.loads(args.others_eval.read_text())) if args.others_eval else None
    others = []
    for d in sorted(args.store.iterdir()):
        if allowed is not None and d.name not in allowed:
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("word") and meta["word"] not in head_words:
            others.append({"id": d.name, "word": meta["word"], "label": meta["label"], "phrase": meta["phrase"], "other": True})
    step = max(1, len(others) // args.max_other)
    others = others[::step][:args.max_other]
    candidates = {p.parent.name: p for p in sorted(args.candidates.glob("*/*.onnx")) if p.stem == p.parent.name}
    baseline = {}
    if args.baseline:
        manifest = json.loads((args.baseline / "manifest.json").read_text())
        baseline = {h: args.baseline / manifest["files"][h]["file"] for h in manifest["classifiers"] if h in candidates}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "wav").mkdir()
        wavs = [padded(args.store / t["id"] / "audio.wav", tmp / "wav" / f"{t['id']}.wav") for t in holdout + others]
        cand = clip_max(args.waqy, make_pack(tmp / "candidate", candidates, args.threshold), wavs)
        base = clip_max(args.waqy, make_pack(tmp / "baseline", baseline, args.threshold), wavs) if baseline else {}
    report = {}
    for head in sorted(candidates):
        addressed = head.startswith(("hey_", "he_"))
        word = head.split("_", 1)[1] if addressed else head
        mine = [t for t in holdout if t["word"] == word]
        wake = [t for t in mine if t["label"] == "positive" and (not addressed or t["phrase"].lower().startswith(ADDRESS))]
        look = [t for t in mine if t["label"] == "negative"]

        def score(table):
            hits = sum(table[t["id"]][head] >= args.threshold for t in wake)
            fires = sum(table[t["id"]][head] >= args.threshold for t in look)
            other = [t for t in others if table[t["id"]][head] >= args.threshold]
            return {"wake_hits": hits, "wake_n": len(wake), "lookalike_fires": fires, "lookalike_n": len(look),
                    "other_fires": len(other), "other_n": len(others),
                    "other_fired_on": sorted({f"{t['word']}:{t['phrase']}" for t in other})[:12],
                    "wake_scores": [round(table[t["id"]][head], 3) for t in wake]}

        c = score(cand)
        # No held-out evidence is not a failure: without wake takes or look-alikes for this head the
        # gate cannot say anything, and calling that "fail" hides the difference from a regression.
        row = {"candidate": c, "evidence": "none" if c["wake_n"] == 0 and c["lookalike_n"] == 0
               and c["other_n"] == 0 else ("weak" if c["wake_n"] == 0 else "ok")}
        if head in baseline:
            b = score(base)
            row["baseline"] = b
            row["passes"] = (c["wake_hits"] >= b["wake_hits"] and c["lookalike_fires"] <= b["lookalike_fires"]
                             and c["other_fires"] <= b["other_fires"] and c["wake_n"] > 0)
        else:
            row["passes"] = (c["wake_n"] > 0 and c["wake_hits"] * 2 >= c["wake_n"] and c["lookalike_fires"] * 5 <= max(1, c["lookalike_n"])
                             and c["other_fires"] * 50 <= max(1, c["other_n"]))
        report[head] = row
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    for head, row in report.items():
        c = row["candidate"]
        line = f"{head:12} new: wake {c['wake_hits']}/{c['wake_n']} look {c['lookalike_fires']}/{c['lookalike_n']} other-word {c['other_fires']}/{c['other_n']}"
        if "baseline" in row:
            b = row["baseline"]
            line += f" | old: wake {b['wake_hits']}/{b['wake_n']} look {b['lookalike_fires']}/{b['lookalike_n']} other-word {b['other_fires']}/{b['other_n']}"
        verdict = "PASS" if row["passes"] else "needs review"
        print(f"{line}  [{row['evidence']}] {verdict}")


if __name__ == "__main__":
    main()
