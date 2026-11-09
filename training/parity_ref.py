"""Reference scores for the browser-engine parity test.

Runs the Python batch path (whole-stream mel + stride-8 embeddings, as in training and
evaluate.py) on a wav with the evaluator's lead-in/tail padding, and writes the padded audio
plus per-frame scores so web/parity.test.mjs can stream the same samples through WakeEngine.

Usage: .venv/bin/python parity_ref.py <classifier.onnx> <out.json> <wav>...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from build_data import RATE, Featurizer, to16k
from evaluate import Scorer


def main():
    scorer, feat = Scorer(Path(sys.argv[1])), Featurizer()
    cases = []
    rng = np.random.default_rng(0)
    for wav in sys.argv[3:]:
        pcm, rate = sf.read(wav, dtype="float32")
        pcm = to16k(pcm, rate)
        lead = rng.normal(0, 3e-4, int(1.5 * RATE) + int(rng.integers(0, 1280))).astype(np.float32)  # random 80 ms phase
        tail = rng.normal(0, 3e-4, int(0.6 * RATE)).astype(np.float32)
        audio = np.concatenate([lead, pcm, tail]).astype(np.float32)
        audio = audio[: len(audio) // 1280 * 1280]
        scores = scorer.stream(feat.frames(audio))
        cases.append({"wav": Path(wav).name, "audio": [round(float(x), 7) for x in audio],
                      "scores": [round(float(s), 6) for s in scores]})
    Path(sys.argv[2]).write_text(json.dumps({"classifier": str(sys.argv[1]), "cases": cases}))


if __name__ == "__main__":
    main()
