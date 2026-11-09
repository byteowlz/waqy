"""Write rustpotter train/test dirs of `[label]`-prefixed 16 kHz mono WAVs.

Fixed before training (no test data enters train):
- positives: pilot crops with status `name-matched`; train-candidate voices -> train,
  held-out voices -> test. Labels `[oqto]` and `[hey_oqto]` (DE and EN together).
- negatives `[none]`, cut into 1.6 s windows (hop 0.8 s), never containing a wake phrase:
  train: FLEURS de_de replay clips (train split, disjoint from the FLEURS test set used
  for false-accept scoring), pilot train-voice confusable carriers, nonspeech-1 train;
  test: pilot held-out confusable carriers, nonspeech-1 validation.
Optional: owner takes from the wake recording store, split by recording session so a
whole session is held out (pass --owner). Owner takes stay human-review-pending.

Usage: uv run --no-project --with numpy --with soundfile --with soxr python \
         scripts/make_train_set.py <out_dir> [--owner]
"""
import argparse
import json
import pathlib

import numpy as np
import soundfile as sf
import soxr

HOME = pathlib.Path.home()
PILOT = HOME / ".local/share/foxline/wake-corpus/20261021-pilot-1/attempt-2"
FLEURS = HOME / ".local/share/foxline/voice-corpus/fleurs-de-1"
NONSPEECH = HOME / ".local/share/foxline/voice-corpus/nonspeech-1"
OWNER = HOME / ".local/share/foxline/wake-collect-url-1"
RATE, WIN, HOP = 16000, int(1.6 * 16000), int(0.8 * 16000)


def load(path):
    pcm, rate = sf.read(str(path), dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    return soxr.resample(pcm, rate, RATE) if rate != RATE else pcm


def write(directory, label, stem, pcm):
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(str(directory / f"[{label}]{stem}.wav"), pcm, RATE, subtype="PCM_16")


def windows(pcm):
    if len(pcm) <= WIN:
        yield np.pad(pcm, (0, WIN - len(pcm)))
        return
    for start in range(0, len(pcm) - WIN + 1, HOP):
        yield pcm[start:start + WIN]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--owner", action="store_true")
    parser.add_argument("--fleurs-clips", type=int, default=120)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    counts = {}

    def put(split, label, stem, pcm):
        write(out / split, label, stem, pcm)
        counts[f"{split}/{label}"] = counts.get(f"{split}/{label}", 0) + 1

    for r in json.loads((PILOT / "dataset-manifest.json").read_text())["records"]:
        split = "train" if r["split"] == "train-candidate" else "test"
        if r["label"] == "none":
            for i, w in enumerate(windows(load(PILOT / r["source_wav"]))):
                put(split, "none", f"pilot-{r['id']}-{i}", w)
        elif r["status"] == "name-matched":
            put(split, r["label"].replace(" ", "_"), f"pilot-{r['id']}", load(PILOT / r["wav"]))

    replay = json.loads((FLEURS / "manifests/replay-cased.json").read_text())["replay"][: args.fleurs_clips]
    for r in replay:
        for i, w in enumerate(windows(load(r["wav"]))):
            put("train", "none", f"fleurs-{r['id']}-{i}", w)

    for line in (NONSPEECH / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        split = {"train": "train", "validation": "test"}.get(r["role"])
        if split:
            for i, w in enumerate(windows(load(r["wav"]))):
                put(split, "none", f"nonspeech-{r['id']}-{i}", w)

    if args.owner:
        takes = [json.loads((d / "meta.json").read_text()) | {"dir": d} for d in OWNER.iterdir() if (d / "meta.json").exists()]
        sessions = sorted({t["sessionId"] for t in takes})
        held = sessions[-1] if len(sessions) > 1 else None
        for t in takes:
            split = "test" if t["sessionId"] == held else "train"
            pcm = load(t["dir"] / "audio.wav")
            if t["label"] == "positive":
                label = "hey_oqto" if t["phrase"].startswith("hey") else "oqto"
                put(split, label, f"owner-{t['id']}", pcm)
            else:
                for i, w in enumerate(windows(pcm)):
                    put(split, "none", f"owner-{t['id']}-{i}", w)
        counts["owner_sessions"] = len(sessions)
        counts["owner_heldout_session"] = held

    (out / "counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
