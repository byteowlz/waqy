"""Find look-alikes that SOUND like their wake word, using espeak-ng's IPA.

    uv run --no-project python training/audit_phonemes.py [wordpack.json] [--overrides overrides.json]

Spelling is not sound: "pie" is "pi", German "Hecks" is "Hex", "Kernel" is "Colonel", German
"Jarwis" is "Jarvis" (w = v) and German "Kid" is "Kitt" (final devoicing). The build guard in
build_collection.py compares tokens, so it cannot see any of these; this reports them.

Read-only. Prints two lists:
  SAME SOUND      the look-alike is the wake word: move it to accept_stems, never keep it as a negative
  CONTAINS        the look-alike contains the wake sound: review by ear (Hexe, Oktober, running)

espeak-ng misreads some loanwords (German "Aktion" as /aktsjo:n/, English "Elisa" as /ɪlɪsə/), so a
hit is a question, not a verdict. The owner's ear decides.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

VOICE = {"de": "de", "en": "en-us"}
from addresses import ADDRESS, ADDRESS_WORDS as ADDRESS_STEMS


@lru_cache(maxsize=None)
def ipa(text: str, lang: str) -> str:
    out = subprocess.run(["espeak-ng", "-q", "-v", VOICE.get(lang, lang), "--ipa", text],
                         capture_output=True, text=True, check=False).stdout
    return re.sub(r"[ˈˌ\s‿-]", "", out.strip())   # stress marks and spacing do not change the sound


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wordpack", nargs="?", default=str(Path(__file__).parent.parent / "web/app/wordpack.json"))
    ap.add_argument("--overrides", help="look-alike overrides file, so removed phrases are not judged")
    args = ap.parse_args()
    pack = json.loads(Path(args.wordpack).expanduser().read_text())
    overrides = json.loads(Path(args.overrides).expanduser().read_text()) if args.overrides else {"removed": [], "added": {}}
    removed = set(overrides.get("removed", []))
    same, contains = [], []
    for word in pack["words"]:
        for lang in ("de", "en"):
            lists = word["languages"].get(lang)
            if not lists:
                continue
            wake = {}
            for phrase in lists["positives"]:
                bare = ADDRESS.sub("", phrase["line"]).strip(" ?!.")
                wake[ipa(bare, lang)] = bare
            for stem in word.get("accept") or []:
                if stem.strip("?").lower() not in ADDRESS_STEMS:
                    wake.setdefault(ipa(stem.strip("?"), lang), stem)
            items = [p["line"] for p in lists["negatives"] if f"{word['name']}|{lang}|{p['line']}" not in removed]
            items += overrides.get("added", {}).get(f"{word['name']}|{lang}", [])
            for line in items:
                sound = ipa(line, lang)
                if sound in wake:
                    same.append((word["name"], lang, line, sound, wake[sound]))
                    continue
                hit = next(((k, v) for k, v in wake.items() if len(k) >= 3 and k in sound), None)
                if hit:
                    contains.append((word["name"], lang, line, sound, hit[1], hit[0]))
    print("SAME SOUND as a wake form (move to accept_stems, never a negative):")
    for name, lang, line, sound, wakeform in same:
        print(f'  {name:16} {lang}  "{line}"  /{sound}/  = wake "{wakeform}"')
    print("CONTAINS the wake sound (review by ear):")
    for name, lang, line, sound, wakeform, form in contains:
        print(f'  {name:16} {lang}  "{line}"  /{sound}/  contains "{wakeform}" /{form}/')
    print(json.dumps({"same_sound": len(same), "contains": len(contains)}))


if __name__ == "__main__":
    main()
