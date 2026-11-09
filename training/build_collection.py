"""Generate the owner collection protocol from the word-pack specs.

The recording page must collect exactly what training needs: the natural addressed
and bare forms per language, plus the confusables that misfire. Hand-copying these
lists is how a collector drifts from the training data.

Usage:
    uv run --no-project python training/build_collection.py [--out web/app/wordpack.json]
"""
import argparse
import json
import re
from pathlib import Path
from addresses import ADDRESS, ADDRESS_WORDS, is_addressed

ROOT = Path(__file__).resolve().parent
# Command words (dictation triggers) are collected in the same sitting but tagged
# separately so they never enter a wake head as positives.
COMMANDS = {"dictate", "diktiere", "dicta", "stopp"}


def tokens(text: str) -> list[str]:
    return re.sub(r"[.,!?;:\"'()\-]+", " ", text.lower()).split()


def wake_words(spec: dict, lang: str) -> list[list[str]]:
    """This language's wake forms without their address, as token sequences."""
    out = []
    for text in (spec.get("forms") or {}).get(lang, []):
        bare = tokens(ADDRESS.sub("", " ".join(tokens(text))))
        if bare and bare not in out:
            out.append(bare)
    return out


def homophones(spec: dict) -> set[str]:
    """accept_stems are ASR screening stems (any language); the non-address ones mark
    spellings that sound like the wake word, so one of them can never be a look-alike."""
    return {" ".join(tokens(s)) for s in spec.get("accept_stems", []) if " ".join(tokens(s)) not in ADDRESS_WORDS}


def lookalike_problem(text: str, spec: dict, lang: str) -> str | None:
    words = tokens(text)
    for wake in wake_words(spec, lang):
        if any(words[i:i + len(wake)] == wake for i in range(len(words) - len(wake) + 1)):
            return "contains the wake word"
    if " ".join(words) in homophones(spec):
        return "is an accepted spelling (same sound)"
    return None


def phrase(text: str, kind: str, form: str, note: str = "") -> dict:
    return {"line": text, "spoken": text, "kind": kind, "form": form, "note": note}


def word_entry(path: Path) -> dict | None:
    spec = json.loads(path.read_text())
    name = spec.get("name")
    if not name or name in COMMANDS:
        return None
    languages = {}
    for lang, forms in (spec.get("forms") or {}).items():
        if lang not in ("de", "en"):
            continue
        positives, seen = [], set()
        for text in forms:
            key = text.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            bare = not is_addressed(key)
            positives.append(phrase(text, "positive", "bare" if bare else "addressed"))
        bad = {text: why for text in (spec.get("confusables") or {}).get(lang, []) if (why := lookalike_problem(text, spec, lang))}
        if bad:
            raise SystemExit(f"{path.name} [{lang}]: these look-alikes sound like the wake word, remove them: {bad}")
        negatives = [
            phrase(text, "negative", "confusable")
            for text in (spec.get("confusables") or {}).get(lang, [])
            if text.strip() and text.strip().lower() not in seen
        ]
        languages[lang] = {"positives": positives, "negatives": negatives}
    if not languages:
        return None
    # Accepted spellings sound like the wake word, so the look-alike editor must refuse
    # them as negatives (a homophone is a positive, never a hard negative).
    # One word can be one language only (e.g. "Hey Jupp" is German); languages without forms are
    # simply absent, so the recorder never asks for a phrase nobody would say.
    accept = sorted({s.strip().lower() for s in spec.get("accept_stems") or [] if s.strip()})
    return {"name": name, "display": spec.get("display") or name.title(), "languages": languages, "accept": accept}


def build(private_words: Path | None = None) -> dict:
    """private_words: a roster outside git (personal names). Its entries are marked private, so
    pack/curate/publish never ship them and the committed wordpack.json never contains them."""
    words = [entry for path in sorted(ROOT.glob("words/*.json")) if (entry := word_entry(path))]
    if private_words:
        for path in sorted(private_words.glob("*.json")):
            if (entry := word_entry(path)):
                words.append({**entry, "private": True})
    priority = ["oqto", "hex", "elissa", "colonel", "opencode", "otacon", "pi"]
    words.sort(key=lambda w: (priority.index(w["name"]) if w["name"] in priority else len(priority), w["name"]))
    return {
        "version": 2,
        "tracker": "waqy trx-wv17, trx-sg05.4",
        "purpose": (
            "Owner recordings for wake heads. The deployed heads were trained only on designed "
            "synthetic voices, so the owner's own voice is out of distribution: hex does not "
            "trigger, elissa/pi/otacon are hit or miss. Positives teach the owner's voice; "
            "confusables are hard negatives. Saving is not training consent; review first."
        ),
        "words": words,
        "protocol": {
            "capture": "one Record press runs every take for the selected word and session: 0.8 s cue, 2.6 s window, 0.4 s break.",
            "sample_rate": "mono 16 kHz PCM16, echoCancellation/noiseSuppression/autoGainControl off",
            "take_max_seconds": 4,
            "save": "private on this host as an owner-wake-candidate with word, form, language, style, session and taken number; unreviewed, trainingEligible false",
            "coverage": "the page shows how many accepted takes exist per word and session so gaps are visible",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "web/app/wordpack.json")
    parser.add_argument("--private-words", type=Path, help="private roster dir (never committed)")
    parser.add_argument("--private-out", type=Path, help="full pack incl. private words, outside git (WAQY_WORDPACK)")
    args = parser.parse_args()
    data = build()
    if args.private_words and args.private_out:
        full = build(args.private_words)
        args.private_out.parent.mkdir(parents=True, exist_ok=True)
        args.private_out.write_text(json.dumps(full, indent=1, ensure_ascii=False) + "\n")
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    counts = {w["name"]: {lang: len(v["positives"]) + len(v["negatives"]) for lang, v in w["languages"].items()} for w in data["words"]}
    print(json.dumps({"out": str(args.out), "words": len(data["words"]), "takes_per_word_language": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
