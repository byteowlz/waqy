"""Review a recorded live detector session: transcribe it and line up spoken wake words with detections.

Input: a session directory written by the /detect page (chunk-*.wav, chunks.jsonl, scores.jsonl,
marks.jsonl, meta.json). Output (in <session>/review-<n>/): session.wav, Whisper word-timestamp
transcripts (German and English pass), review.json, review.md and short WAV clips for every miss,
false trigger and look-alike word, so each case can be listened to.

  hit            a wake word in the transcript with a detection within [start - 0.3 s, end + 1.5 s]
  miss           a wake word in the transcript without such a detection
  false trigger  a detection with no wake word in [-2.5 s, +0.5 s] around it
  look-alike     Oktober/October/Oktave/octave/Oktopus/octopus/Doktor/Otto, with scores near it

The transcript is evidence, not truth: Whisper bends "Oqto" toward real words (Otto, Oktober), so
"Otto" is listed as a possible wake word and every case comes with a clip for listening.

Usage: .venv/bin/python live_review.py <session-dir> [--model ggml-large-v3-turbo-q8_0.bin] [--threads 8]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
FRAME_SECONDS = 0.08
WHISPER = "/opt/homebrew/bin/whisper-cli"
WAKE = re.compile(r"^(?:hey|hei|hi|he)?[,.\s]*(?:oqto|okto|octo|okdo|oqdo|akto|acto|ockto|oktoh|octoh)[!?.,]*$")
POSSIBLE = re.compile(r"^(?:otto|auto|okta|octa|opto|oktor)[!?.,]*$")
LOOKALIKE = re.compile(r"(oktober|october|oktav|octav|oktop|octop|doktor|doctor)")
HEY = re.compile(r"^(hey|hei|hi|he|hej)[!?.,]*$")


def load_session(session: Path) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    chunks = [json.loads(l) for l in (session / "chunks.jsonl").read_text().splitlines()]
    chunks.sort(key=lambda c: c["seq"])
    parts = []
    for c in chunks:
        pcm, rate = sf.read(str(session / c["file"]), dtype="float32")
        assert rate == RATE
        parts.append(pcm)
    scores = [json.loads(l) for l in (session / "scores.jsonl").read_text().splitlines()] if (session / "scores.jsonl").exists() else []
    marks = [json.loads(l) for l in (session / "marks.jsonl").read_text().splitlines()] if (session / "marks.jsonl").exists() else []
    return np.concatenate(parts), scores, marks, json.loads((session / "meta.json").read_text())


def transcribe(wav: Path, lang: str, model: Path, threads: int) -> list[dict]:
    """Word-level segments [{w, t0, t1}] from whisper.cpp (-ml 1 -sow), CPU only."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "out"
        subprocess.run([WHISPER, "-ng", "-t", str(threads), "-m", str(model), "-l", lang, "-ml", "1", "-sow", "-oj", "-of", str(prefix),
                        "--prompt", "Hey Oqto. Oqto." if lang == "en" else "Hey Oqto. Oqto, mach mal.", "-np", str(wav)],
                       check=True, capture_output=True, text=True)
        data = json.loads(prefix.with_suffix(".json").read_text())
    words = []
    for seg in data.get("transcription", []):
        text = seg["text"].strip()
        if text:
            words.append({"w": text, "t0": seg["offsets"]["from"] / 1000, "t1": seg["offsets"]["to"] / 1000})
    return words


def norm(word: str) -> str:
    return re.sub(r"[^\wäöüß]", "", word.lower())


def wake_candidates(words: list[dict], lang: str) -> list[dict]:
    out = []
    for i, w in enumerate(words):
        n = norm(w["w"])
        kind = "wake" if WAKE.match(n) else "possible" if POSSIBLE.match(n) else None
        if not kind:
            continue
        hey = i > 0 and HEY.match(norm(words[i - 1]["w"])) and w["t0"] - words[i - 1]["t1"] < 0.6
        out.append({"word": w["w"], "t0": words[i - 1]["t0"] if hey else w["t0"], "t1": w["t1"], "hey": bool(hey), "kind": kind, "lang": lang})
    return out


def merge(candidates: list[dict]) -> list[dict]:
    """Same spoken word found by both language passes: keep one (prefer 'wake', then the hey form)."""
    merged = []
    for c in sorted(candidates, key=lambda c: c["t1"]):
        if merged and abs(merged[-1]["t1"] - c["t1"]) < 0.6:
            keep = merged[-1]
            if (c["kind"] == "wake" and keep["kind"] != "wake") or (c["hey"] and not keep["hey"]):
                merged[-1] = c | {"also": keep["word"]}
            else:
                keep.setdefault("also", c["word"])
            continue
        merged.append(dict(c))
    return merged


def context(words: list[dict], t0: float, t1: float) -> str:
    return " ".join(w["w"] for w in words if w["t1"] >= t0 and w["t0"] <= t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--model", default=str(Path.home() / "models/Whisper/ggml-large-v3-turbo-q8_0.bin"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    session = Path(args.session).expanduser()
    n = 1
    while (session / f"review-{n}").exists():
        n += 1
    out = session / f"review-{n}"
    (out / "clips").mkdir(parents=True)

    pcm, scores, marks, meta = load_session(session)
    wav = out / "session.wav"
    sf.write(str(wav), pcm, RATE, subtype="PCM_16")
    duration = len(pcm) / RATE
    words = {lang: transcribe(wav, lang, Path(args.model), args.threads) for lang in ("de", "en")}
    for lang, ws in words.items():
        (out / f"words-{lang}.json").write_text(json.dumps(ws, ensure_ascii=False, indent=0))

    models = sorted({k for row in scores for k in row["s"]})
    t = np.array([(row["f"] + 1) * FRAME_SECONDS for row in scores])
    series = {m: np.array([row["s"].get(m, 0.0) for row in scores]) for m in models}
    detections = [{"t": (row["f"] + 1) * FRAME_SECONDS, "model": m, "score": row["s"][m]} for row in scores for m in row["d"]]

    def peak(m, a, b):
        sel = (t >= a) & (t <= b)
        return round(float(series[m][sel].max()), 3) if sel.any() else None

    def clip(name, a, b):
        a, b = max(0.0, a), min(duration, b)
        sf.write(str(out / "clips" / name), pcm[int(a * RATE):int(b * RATE)], RATE, subtype="PCM_16")
        return f"clips/{name}"

    candidates = merge(wake_candidates(words["de"], "de") + wake_candidates(words["en"], "en"))
    # The Oqto-biased prompt can turn "Oktober" into "Oqto" in one pass; if either pass heard a
    # look-alike at the same time, the word is a conflict to listen to, not a counted wake word.
    for c in candidates:
        clash = [w["w"] for lang in ("de", "en") for w in words[lang]
                 if LOOKALIKE.search(norm(w["w"])) and c["t0"] - 0.15 <= (w["t0"] + w["t1"]) / 2 <= c["t1"] + 0.15]
        if clash:
            c["kind"], c["heard_as_lookalike"] = "conflict", clash[0]
    spoken = []
    for i, c in enumerate(candidates):
        hits = [d for d in detections if c["t0"] - 0.3 <= d["t"] <= c["t1"] + 1.5]
        row = c | {"detected_by": sorted({d["model"] for d in hits}), "peaks": {m: peak(m, c["t0"] - 0.3, c["t1"] + 1.5) for m in models},
                   "context_de": context(words["de"], c["t0"] - 2.5, c["t1"] + 2.5), "context_en": context(words["en"], c["t0"] - 2.5, c["t1"] + 2.5)}
        row["result"] = "hit" if hits else "miss"
        if not hits or c["kind"] != "wake":
            row["clip"] = clip(f"{row['result']}-{i:03d}-{c['t0']:.1f}s.wav", c["t0"] - 2.0, c["t1"] + 1.0)
        spoken.append(row)

    false = []
    for i, d in enumerate(detections):
        near = [c for c in candidates if c["t0"] - 0.5 <= d["t"] <= c["t1"] + 2.5]
        if near:
            continue
        false.append(d | {"context_de": context(words["de"], d["t"] - 2.5, d["t"] + 0.5), "context_en": context(words["en"], d["t"] - 2.5, d["t"] + 0.5),
                          "clip": clip(f"false-{i:03d}-{d['model']}-{d['t']:.1f}s.wav", d["t"] - 2.5, d["t"] + 0.8)})

    lookalikes = []
    for lang in ("de", "en"):
        for w in words[lang]:
            if any(abs(l["t0"] - w["t0"]) < 0.6 for l in lookalikes):
                continue  # same word heard by the other pass
            if LOOKALIKE.search(norm(w["w"])):
                fired = [d for d in detections if w["t0"] - 0.3 <= d["t"] <= w["t1"] + 1.5]
                lookalikes.append({"word": w["w"], "lang": lang, "t0": w["t0"], "fired": sorted({d["model"] for d in fired}),
                                   "peaks": {m: peak(m, w["t0"] - 0.3, w["t1"] + 1.5) for m in models}})

    marked = []
    for m in marks:
        at = m["atSample"] / RATE
        if m["kind"] == "missed":
            marked.append(m | {"t": at, "context_de": context(words["de"], at - 4, at), "context_en": context(words["en"], at - 4, at),
                               "peaks": {k: peak(k, at - 4, at) for k in models}, "clip": clip(f"marked-missed-{at:.1f}s.wav", at - 4, at + 0.3)})
        else:
            marked.append(m | {"t": at, "context_de": context(words["de"], at - 2.5, at + 0.5)})

    chunk_thresholds = sorted({c.get("threshold") for c in map(json.loads, (session / "chunks.jsonl").read_text().splitlines()) if c.get("threshold") is not None})
    summary = {
        "session": meta["sessionId"], "release": meta.get("release"), "threshold": chunk_thresholds or meta.get("threshold"),
        "lowest_detection_score": round(min((d["score"] for d in detections), default=float("nan")), 3), "minutes": round(duration / 60, 2),
        "asr": f"whisper.cpp {Path(args.model).name}, CPU, word timestamps (-ml 1 -sow), de and en passes, prompt biased to 'Oqto'",
        "spoken_wake_words": len([s for s in spoken if s["kind"] == "wake"]),
        "hits": len([s for s in spoken if s["kind"] == "wake" and s["result"] == "hit"]),
        "misses": len([s for s in spoken if s["kind"] == "wake" and s["result"] == "miss"]),
        "possible_wake_words": len([s for s in spoken if s["kind"] == "possible"]),
        "conflicts_with_lookalike": len([s for s in spoken if s["kind"] == "conflict"]),
        "detections": len(detections), "false_triggers": len(false),
        "false_triggers_per_hour": round(len(false) / max(duration / 3600, 1e-9), 1),
        "lookalikes_spoken": len(lookalikes), "lookalikes_fired": len([l for l in lookalikes if l["fired"]]),
        "marks": {k: sum(m["kind"] == k for m in marks) for k in ("missed", "false-trigger", "correct")},
    }
    report = {"summary": summary, "spoken": spoken, "false_triggers": false, "lookalikes": lookalikes, "marks": marked}
    (out / "review.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    lines = [f"# Live session {summary['session'][:8]} ({summary['minutes']} min, release {summary['release']}, threshold {summary['threshold']}, lowest detection {summary['lowest_detection_score']})", "",
             f"Wake words heard by Whisper: {summary['spoken_wake_words']}; detected {summary['hits']}, missed {summary['misses']}. "
             f"Possible (e.g. 'Otto'): {summary['possible_wake_words']}; conflicting with a look-alike: {summary['conflicts_with_lookalike']}. Detections: {summary['detections']}, false triggers: {summary['false_triggers']} "
             f"({summary['false_triggers_per_hour']}/h). Look-alike words: {summary['lookalikes_spoken']}, fired on {summary['lookalikes_fired']}.", "",
             "## Spoken wake words", "", "| t (s) | heard | result | peaks | context |", "|---|---|---|---|---|"]
    for s in spoken:
        ctx = s["context_de"] if s["lang"] == "de" else s["context_en"]
        note = {"possible": " (possible)", "conflict": f" (other pass: {s.get('heard_as_lookalike')})"}.get(s["kind"], "")
        lines.append(f"| {s['t0']:.1f} | {s['word']}{note} | {s['result'] if s['kind'] == 'wake' else 'listen: ' + s['result']} {','.join(s['detected_by'])} | {s['peaks']} | {ctx} | ")
    lines += ["", "## False triggers", "", "| t (s) | model | score | context (de / en) |", "|---|---|---|---|"]
    lines += [f"| {f['t']:.1f} | {f['model']} | {f['score']:.2f} | {f['context_de']} / {f['context_en']} |" for f in false]
    lines += ["", "## Look-alike words", "", "| t (s) | word | fired | peaks |", "|---|---|---|---|"]
    lines += [f"| {l['t0']:.1f} | {l['word']} | {','.join(l['fired']) or '-'} | {l['peaks']} |" for l in lookalikes]
    lines += ["", "## Owner marks", ""] + [f"- {m['t']:.1f} s {m['kind']}: {m.get('context_de', '')} {m.get('peaks', '')}" for m in marked]
    (out / "review.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
