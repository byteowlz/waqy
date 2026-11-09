"""Prepare owner takes as training audio: trim silence, then level-normalize.

Owner takes from the browser are often extremely quiet (median peak 0.02 = -34 dBFS on the first
batch), because the microphone captures at low gain and no AGC is requested. At that level the word
sits near the noise floor and its embedding does not match any training audio, so a retrain cannot
help until the audio is levelled. This writes a prepared copy; the originals are never modified.

    python prepare_owner_takes.py <store> <out> [--peak 0.7] [--keep-ms 120] [--words hex,elissa]

Emits <out>/meta.jsonl (one row per prepared take, with measured levels) and <out>/wav/<id>.wav.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16_000
ID = __import__("re").compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")


def db(value: float) -> float | None:
    return 20 * np.log10(value) if value > 0 else None


def prepare(pcm: np.ndarray, peak_target: float, keep: int, trim: bool = True) -> tuple[np.ndarray, dict]:
    mono = pcm.mean(axis=1) if pcm.ndim > 1 else pcm
    if not trim:
        # Level only. Trimming cut words off (first retrain); the detector expects context.
        current = float(np.abs(mono).max()) if len(mono) else 0.0
        gain = float(min(peak_target / current, 40.0)) if current > 0 else 1.0
        return mono * gain, {"trimmed": False, "gain_db": round(float(20 * np.log10(gain)), 1)}
    frame = 320
    energy = np.sqrt((mono[: len(mono) // frame * frame].reshape(-1, frame) ** 2).mean(axis=1))
    if not len(energy):
        return mono, {"trimmed": False}
    voiced = np.flatnonzero(energy > max(energy.max() * 0.05, 1e-4))
    start = max(0, int(voiced[0]) * frame - keep) if len(voiced) else 0
    end = min(len(mono), (int(voiced[-1]) + 1) * frame + keep) if len(voiced) else len(mono)
    trimmed = mono[start:end]
    current = float(np.abs(trimmed).max()) if len(trimmed) else 0.0
    gain = (peak_target / current) if current > 0 else 1.0
    gain = float(min(gain, 40.0))  # never amplify more than +32 dB; beyond that it is only noise
    return trimmed * gain, {"trimmed": True, "start": start, "end": end, "gain_db": round(float(20 * np.log10(gain)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--peak", type=float, default=0.7)
    ap.add_argument("--keep-ms", type=int, default=120)
    ap.add_argument("--words", default="")
    ap.add_argument("--no-trim", action="store_true", help="level only, keep the full take (use for training)")
    args = ap.parse_args()
    words = {w.strip() for w in args.words.split(",") if w.strip()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wav").mkdir(exist_ok=True)
    rows = []
    for entry in sorted(args.store.iterdir()):
        if not ID.match(entry.name):
            continue
        meta_path, audio_path = entry / "meta.json", entry / "audio.wav"
        if not meta_path.exists() or not audio_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if words and meta.get("word") not in words:
            continue
        pcm, rate = sf.read(str(audio_path), dtype="float32")
        if rate != RATE:
            raise SystemExit(f"{audio_path} is {rate} Hz, expected {RATE}")
        before_peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
        prepared, info = prepare(pcm, args.peak, int(args.keep_ms * RATE / 1000), trim=not args.no_trim)
        target = args.out / "wav" / f"{entry.name}.wav"
        sf.write(str(target), prepared.astype(np.float32), RATE, subtype="PCM_16")
        after_peak = float(np.abs(prepared).max()) if len(prepared) else 0.0
        rows.append({"id": entry.name, "word": meta.get("word"), "label": meta.get("label"),
                     "form": meta.get("form"), "language": meta.get("language"),
                     "session": (meta.get("sessionId") or "")[:8], "at": meta.get("receivedAt"),
                     "phrase": meta.get("phrase"),
                     "peak_before_dbfs": None if not before_peak else round(float(db(before_peak)), 1),
                     "peak_after_dbfs": None if not after_peak else round(float(db(after_peak)), 1),
                     "seconds_before": round(len(pcm) / RATE, 3),
                     "seconds_after": round(len(prepared) / RATE, 3), **info})
    (args.out / "meta.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    loud = [r for r in rows if r.get("peak_after_dbfs") is not None]
    print(json.dumps({"prepared": len(rows), "out": str(args.out),
                      "median_gain_db": float(np.median([r["gain_db"] for r in rows if "gain_db" in r])) if rows else None,
                      "median_peak_after_dbfs": float(np.median([r["peak_after_dbfs"] for r in loud])) if loud else None}))


if __name__ == "__main__":
    main()
