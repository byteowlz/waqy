"""Scan a wake artifact over fixed held-out sets and report detections.

Sets (all outside training, see make_train_set.py):
- heldout_wake: pilot held-out-voice full carrier sentences that contain a wake phrase
- heldout_confusables: pilot held-out-voice confusable sentences (Oktober/Oktopus/...)
- fleurs_de_test: 60 FLEURS de_de test clips (real speech, no wake phrase), false accepts/hour
- nonspeech_test: nonspeech-1 test clips (silence/noise)
- owner_heldout: owner takes from the held-out recording session, if any exist

Usage: uv run --no-project python scripts/eval_model.py <artifact> <out.json> [--threshold T ...]
"""
import json
import pathlib
import subprocess
import sys

HOME = pathlib.Path.home()
PILOT = HOME / ".local/share/foxline/wake-corpus/20261021-pilot-1/attempt-2"
FLEURS = HOME / ".local/share/foxline/voice-corpus/fleurs-de-1"
NONSPEECH = HOME / ".local/share/foxline/voice-corpus/nonspeech-1"
OWNER = HOME / ".local/share/foxline/wake-collect-url-1"
WAKE = pathlib.Path(__file__).resolve().parents[1] / "target/release/wake"


def scan(artifact, files, extra):
    rows = []
    for i in range(0, len(files), 50):
        out = subprocess.run([str(WAKE), "scan", "--model", str(artifact), *extra, *map(str, files[i:i + 50])],
                             check=True, capture_output=True, text=True).stdout
        rows.extend(json.loads(line) for line in out.splitlines())
    return rows


def main(artifact, out_path, extra):
    records = json.loads((PILOT / "dataset-manifest.json").read_text())["records"]
    held = [r for r in records if r["split"] == "held-out-voice"]
    fleurs_rows = json.loads((FLEURS / "manifests/cased-eval.json").read_text())["rows"]
    nonspeech = [json.loads(l) for l in (NONSPEECH / "manifest.jsonl").read_text().splitlines()]
    sets = {
        "heldout_wake": sorted({str(PILOT / r["source_wav"]) for r in held if r["label"] != "none" and r["status"] == "name-matched"}),
        "heldout_confusables": [str(PILOT / r["source_wav"]) for r in held if r["label"] == "none"],
        "fleurs_de_test": [r["wav"] for r in fleurs_rows],
        "nonspeech_test": [r["wav"] for r in nonspeech if r["role"] == "test"],
    }
    takes = [json.loads((d / "meta.json").read_text()) | {"dir": str(d)} for d in OWNER.iterdir() if (d / "meta.json").exists()] if OWNER.exists() else []
    sessions = sorted({t["sessionId"] for t in takes})
    if len(sessions) > 1:
        heldout = [t for t in takes if t["sessionId"] == sessions[-1]]
        sets["owner_heldout_wake"] = [t["dir"] + "/audio.wav" for t in heldout if t["label"] == "positive"]
        sets["owner_heldout_confusables"] = [t["dir"] + "/audio.wav" for t in heldout if t["label"] != "positive"]
    report = {"artifact": str(artifact), "thresholds_args": extra, "sets": {}}
    for name, files in sets.items():
        rows = scan(artifact, files, extra) if files else []
        hits = sum(r["detected"] for r in rows)
        entry = {"files": len(rows), "detected": hits, "errors": sum(bool(r["error"]) for r in rows),
                 "best_scores": sorted((round(r["best_score"], 3) for r in rows if r["best_score"] is not None), reverse=True)[:10]}
        if name == "fleurs_de_test":
            seconds = sum(r["duration_seconds"] for r in fleurs_rows)
            entry["false_accepts_per_hour"] = round(sum(len(r["hits"]) for r in rows) * 3600 / seconds, 1)
            entry["seconds"] = seconds
        report["sets"][name] = entry
        print(f"{name:26s} {hits}/{len(rows)} detected  top={entry['best_scores'][:5]}" + (f"  FA/h={entry['false_accepts_per_hour']}" if "false_accepts_per_hour" in entry else ""))
    pathlib.Path(out_path).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
