"""Package a browser detector release: front end + classifiers + manifest with sha256 and provenance.

Usage: .venv/bin/python release.py <root> <release-name> <threshold> <model> [<model> ...]
Writes <root>/../wake-models/<release-name>/ (refuses to overwrite an existing release).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

FRONTEND = Path(__file__).parent / "frontend"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root, name, threshold, models = Path(sys.argv[1]).expanduser(), sys.argv[2], float(sys.argv[3]), sys.argv[4:]
    out = root.parent / "wake-models" / name
    if out.exists():
        raise SystemExit(f"{out} exists; releases are immutable, pick a new name")
    out.mkdir(parents=True)
    files = {}
    for key, src in (("mel", FRONTEND / "melspectrogram_raw.onnx"), ("embedding", FRONTEND / "embedding_model.onnx")):
        shutil.copy2(src, out / src.name)
        files[key] = {"file": src.name, "sha256": sha(out / src.name)}
    evals = {}
    for model in models:
        src = root / "output" / model / f"{model}.onnx"
        target = out / f"{model.split('-')[0]}.onnx"
        shutil.copy2(src, target)
        files[model.split("-")[0]] = {"file": target.name, "sha256": sha(target), "trained_as": model}
        report = root / "output" / model / "streaming-eval.json"
        if report.exists():
            evals[model] = json.loads(report.read_text())
    note = (Path(__file__).parent / "release-note.txt")
    manifest = {
        "release": name,
        "training_data": note.read_text().strip() if note.exists() else "synthetic voices only (no owner recordings)",
        "threshold": threshold,
        "classifiers": [m.split("-")[0] for m in models],
        "files": files,
        "description": (
            "Oqto wake detector (livekit-wakeword method). Front end: formula-built log-mel graph (no learned "
            "weights) and Google speech_embedding (Apache-2.0, verified layer-by-layer). General negatives: MUSAN "
            "(PD/CC, no NC), VoxPopuli (CC0), FLEURS (CC-BY-4.0), Thorsten (CC0). Positives: see training_data."),
        "authority": "evidence only; never authentication, speaker identity or permission to act",
        "evaluation": {m: {k: v for k, v in e.items() if k in ("general", "synthetic_positives", "synthetic_adversarial", "owner_takes")}
                       for m, e in evals.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
