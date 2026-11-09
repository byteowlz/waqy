"""Acoustic stress test of frozen held-out takes through the shipped Rust detector.

    uv run --no-project --with numpy --with soundfile python training/gate_rooms.py \
        --packs candidate=/path/to/release,baseline=/path/to/release \
        --holdout mix/holdout.json --store /path/to/recordings \
        --rirs /path/to/rirs --backgrounds /path/to/backgrounds \
        --others-eval mix/owner_negative_eval.json --out /path/to/report.json

Reports wake recall and false fires separately for every head and condition at a fixed
threshold. Addressed heads only count addressed positives; bare heads count both forms,
as in gate_heads.py. Missing scores/audio are errors, not silent misses/exclusions.

Quiet conditions attenuate the entire waveform without normalising it back. Reverb is
peak-matched to isolate its effect; the combined quiet/reverb/noise condition adds a
separate -12 dB attenuation. This is synthetic stress testing, NOT real far-field validation
or unseen-room generalisation: supplied RIRs may be the same bank used during training.
No thresholds are fitted and no generated audio is retained.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from gate_heads import ADDRESS, ROOT, clip_max

RATE = 16000
CONDITIONS = ("clean", "quiet-12dB", "quiet-24dB", "room", "noise-10dB", "room+noise", "quiet+room+noise")


def read_audio(path: Path) -> np.ndarray:
    pcm, rate = sf.read(path, dtype="float32")
    if rate != RATE or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError(f"Expected finite, nonempty mono 16 kHz audio: {path}")
    return pcm


def convolve(pcm: np.ndarray, rir: np.ndarray) -> np.ndarray:
    # Retain the reverberant tail; FFT avoids a long direct convolution per take.
    n = len(pcm) + len(rir) - 1
    size = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(pcm, size) * np.fft.rfft(rir, size), size)[:n]
    peak = np.abs(out).max()
    return out / peak * np.abs(pcm).max() if peak else out


def add_noise(pcm: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.tile(noise, len(pcm) // len(noise) + 1)[:len(pcm)]
    p_sig, p_noise = np.mean(pcm ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    return pcm + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))


def variants(pcm: np.ndarray, rir: np.ndarray, noise: np.ndarray) -> dict[str, np.ndarray]:
    room = convolve(pcm, rir)
    noisy_room = add_noise(room, noise, 10)
    return dict(zip(CONDITIONS, (pcm, pcm * 10 ** (-12 / 20), pcm * 10 ** (-24 / 20),
                                 room, add_noise(pcm, noise, 10), noisy_room,
                                 noisy_room * 10 ** (-12 / 20))))


def select_cases(holdout: list[dict], store: Path, allowed: set[str], words: set[str], cap: int) -> list[dict]:
    cases = {t["id"]: t for t in holdout}
    others = []
    for take_id in sorted(allowed - cases.keys()):
        meta = json.loads((store / take_id / "meta.json").read_text())
        if meta.get("word") and meta["word"] not in words:
            others.append({"id": take_id, "word": meta["word"], "label": meta["label"],
                           "phrase": meta["phrase"], "other": True})
    # Evenly spread a bounded subset over the frozen eligible IDs, rather than taking its prefix.
    if others and cap:
        indices = np.linspace(0, len(others) - 1, min(cap, len(others)), dtype=int)
        cases.update((others[i]["id"], others[i]) for i in indices)
    return list(cases.values())


def score_head(head: str, takes: list[dict], scores: dict, threshold: float) -> dict:
    addressed = head.startswith(("hey_", "he_"))
    word = head.split("_", 1)[1] if addressed else head
    groups = {"wake": [], "lookalike": [], "other": []}
    for take in takes:
        if take.get("other"):
            groups["other"].append(take)
        elif take["word"] == word:
            if take["label"] == "negative":
                groups["lookalike"].append(take)
            elif take["label"] == "positive" and (not addressed or take["phrase"].lower().startswith(ADDRESS)):
                groups["wake"].append(take)
    result = {}
    for group, rows in groups.items():
        fired = [t["id"] for t in rows if scores[t["id"]][head] >= threshold]
        result[group] = {"hits": len(fired), "n": len(rows), "fired_ids": fired}
    return result


def evaluate(pack: Path, takes: list[dict], wavs: dict, waqy: str, threshold: float) -> dict:
    heads = json.loads((pack / "manifest.json").read_text())["classifiers"]
    report = {}
    for condition, paths in wavs.items():
        scores = clip_max(waqy, pack, paths)
        if set(scores) != {t["id"] for t in takes}:
            raise ValueError(f"Incomplete detector output: {pack.name}/{condition}")
        report[condition] = {h: score_head(h, takes, scores, threshold) for h in heads}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packs", required=True, help="name=path,name=path")
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--rirs", type=Path, required=True)
    ap.add_argument("--backgrounds", type=Path, required=True)
    ap.add_argument("--others-eval", type=Path, help="Frozen IDs excluded from training as other-word negatives")
    ap.add_argument("--max-other", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--waqy", default=str(ROOT / "target/release/waqy"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.max_other < 0 or not 0 <= args.threshold <= 1:
        ap.error("max-other must be nonnegative; threshold must be in [0, 1]")
    packs = {name: Path(path) for name, path in (item.split("=", 1) for item in args.packs.split(","))}
    heads = {h for p in packs.values() for h in json.loads((p / "manifest.json").read_text())["classifiers"]}
    words = {h.split("_", 1)[1] if h.startswith(("hey_", "he_")) else h for h in heads}
    allowed = set(json.loads(args.others_eval.read_text())) if args.others_eval else set()
    takes = select_cases(json.loads(args.holdout.read_text()), args.store, allowed, words, args.max_other)
    rirs = [read_audio(p) for p in sorted(args.rirs.glob("*.wav"))]
    backgrounds = [read_audio(p) for p in sorted(args.backgrounds.glob("*.wav"))]
    if not takes or not rirs or not backgrounds:
        ap.error("Need held-out takes and nonempty RIR/background banks")
    rng = np.random.default_rng(7)
    report = {"threshold": args.threshold, "seed": 7, "take_count": len(takes), "takes": takes,
              "limitation": "Synthetic stress; RIR bank may overlap training; not real far-field qualification",
              "packs": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wavs = {c: [] for c in CONDITIONS}
        for condition in CONDITIONS:
            (tmp / condition).mkdir()
        for take in takes:
            pcm = read_audio(args.store / take["id"] / "audio.wav")
            rir, noise = rirs[int(rng.integers(len(rirs)))], backgrounds[int(rng.integers(len(backgrounds)))]
            for condition, signal in variants(pcm, rir, noise).items():
                path = tmp / condition / f"{take['id']}.wav"
                # Float WAV avoids silently clipping mixes that exceed full scale; raw audio is untouched.
                sf.write(path, np.pad(signal, (RATE, RATE)), RATE, subtype="FLOAT")
                wavs[condition].append(path)
        report["complete"] = False
        for name, pack in packs.items():
            report["packs"][name] = {}
            for condition, paths in wavs.items():
                report["packs"][name].update(evaluate(pack, takes, {condition: paths}, args.waqy, args.threshold))
                checkpoint = args.out.with_suffix(".tmp")
                checkpoint.write_text(json.dumps(report, indent=1))
                checkpoint.replace(args.out)
                print(f"Scored {name}/{condition}: {len(takes)} takes", flush=True)
    report["complete"] = True
    checkpoint = args.out.with_suffix(".tmp")
    checkpoint.write_text(json.dumps(report, indent=1))
    checkpoint.replace(args.out)
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
