"""Evaluate wake references on the synthetic spqx pilot plus real non-wake speech.

Splits are fixed here, before any score is read:
- reference takes: pilot `train-candidate` crops with status `name-matched`, per language+phrase
- positive test: pilot `held-out-voice` crops (name-matched) AND their full raw carrier sentences
- negative test: all pilot `none` sentences (full confusable carriers, never truncated),
  FLEURS de_de test clips (real speech, no wake phrase), nonspeech-1 test clips
Thresholds: rustpotter defaults, then a sweep reported over every set; the operating point
is chosen on TRAIN-voice carriers/negatives only and applied unchanged to held-out voices.
Synthetic voices are not the owner's voice; this proves the tool, not product accuracy.

Usage: uv run --no-project python scripts/eval_pilot.py <out_dir>
"""
import json
import pathlib
import subprocess
import sys

HOME = pathlib.Path.home()
PILOT = HOME / ".local/share/foxline/wake-corpus/20261021-pilot-1/attempt-2"
FLEURS = HOME / ".local/share/foxline/voice-corpus/fleurs-de-1"
NONSPEECH = HOME / ".local/share/foxline/voice-corpus/nonspeech-1"
WAKE = pathlib.Path(__file__).resolve().parents[1] / "target/release/wake"
LANG = {"german": "de", "english": "en"}
PHRASES = {"oqto": "oqto", "hey oqto": "hey-oqto"}


def run(args):
    return subprocess.run([str(WAKE), *map(str, args)], check=True, capture_output=True, text=True).stdout


def scan(model, files, extra=()):
    rows = []
    for i in range(0, len(files), 50):
        out = run(["scan", "--model", model, *extra, *files[i:i + 50]])
        rows.extend(json.loads(line) for line in out.splitlines())
    return rows


def main(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    records = json.loads((PILOT / "dataset-manifest.json").read_text())["records"]
    fleurs = [r["wav"] for r in json.loads((FLEURS / "manifests/cased-eval.json").read_text())["rows"]]
    fleurs_seconds = sum(r["duration_seconds"] for r in json.loads((FLEURS / "manifests/cased-eval.json").read_text())["rows"])
    nonspeech = [json.loads(l) for l in (NONSPEECH / "manifest.jsonl").read_text().splitlines()]
    nonspeech = [r["wav"] for r in nonspeech if r["role"] == "test"]
    negatives = {split: [str(PILOT / r["source_wav"]) for r in records if r["label"] == "none" and (r["split"] == "train-candidate") == (split == "train")]
                 for split in ("train", "held-out")}
    plan = {"pilot": str(PILOT), "fleurs_test_clips": len(fleurs), "fleurs_test_seconds": fleurs_seconds,
            "nonspeech_test_clips": len(nonspeech), "negatives": {k: len(v) for k, v in negatives.items()}}
    results = {"plan": plan, "references": {}}
    for language, lang in LANG.items():
        for phrase, slug in PHRASES.items():
            mine = [r for r in records if r["language"] == language and r["label"] == phrase and r["status"] == "name-matched"]
            train = [r for r in mine if r["split"] == "train-candidate"]
            held = [r for r in mine if r["split"] == "held-out-voice"]
            name = f"{lang}-{slug}"
            ref = out / f"{name}.rpw"
            run(["build", "--name", name, "--out", ref, *[PILOT / r["wav"] for r in train]])
            sets = {
                "train_carriers": [str(PILOT / r["source_wav"]) for r in train],
                "heldout_crops": [str(PILOT / r["wav"]) for r in held],
                "heldout_carriers": [str(PILOT / r["source_wav"]) for r in held],
                "negatives_train": negatives["train"],
                "negatives_heldout": negatives["held-out"],
                "fleurs_de_test": fleurs,
                "nonspeech_test": nonspeech,
            }
            scored = {k: scan(ref, v, ["--threshold", "0.3", "--avg-threshold", "0", "--min-scores", "1"]) for k, v in sets.items()}
            (out / f"{name}.scores.json").write_text(json.dumps(scored, indent=1))
            results["references"][name] = {"takes": len(train), "bytes": ref.stat().st_size,
                                           "best_scores": {k: [r["best_score"] for r in v] for k, v in scored.items()},
                                           "errors": [r["error"] for v in scored.values() for r in v if r["error"]]}
            print(name, "takes", len(train), "bytes", ref.stat().st_size, flush=True)
    (out / "raw-results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
