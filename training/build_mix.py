"""Build one word-pack training set from synthetic banks plus the owner's collected takes.

    python build_mix.py <out> --words hex,pi --banks <bank>[,<bank>] --owner <prepared-owner-dir>
                        [--owner-share 0.3]

Synthetic banks hold clips.jsonl + wav/ (spqx VoiceDesign). Owner takes come from
prepare_owner_takes.py --no-trim (levelled, full length: trimming cut words off and made the
first owner retrain worse than the shipped heads).

Rules, each learned the hard way:
- Synthetic clips are re-checked against today's specs: a wake clip must still contain a wake
  spelling, a look-alike clip must still be a listed look-alike. Old banks contain look-alikes
  that were later found to be homophones or to contain the wake word (trx-h8tw).
- Owner positives are weighted to about --owner-share of each word's positives. At 2-4% the
  synthetic voices drowned the owner out. Addressed takes ("Hey Elissa") get their own weight
  against the synthetic addressed clips, or the few of them are diluted by the bare ones.
- The owner's takes of every OTHER word are negatives. Without them owner voice correlated with
  "positive": the first weighted pilot's bare elissa head fired on 38 of 400 owner takes of other
  words (pack1: 1). A hash-chosen 30% of those takes never train; the gate scores them
  (owner_negative_eval.json).
- Held-out owner takes never train. A word with takes from two or more sittings holds out its
  newest sitting entirely (a sitting shares mic, room and mood, so a random split leaks). A word
  with one sitting holds out its newest quarter per language and label, and says so.

Writes <out>/clips.jsonl, <out>/wav/ (symlinks and copies), <out>/holdout.json, <out>/mix.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import build_collection as bc

WORDS_DIR = Path(__file__).parent / "words"


def tokens(text: str) -> list[str]:
    return re.sub(r"[.,!?;:\"'()\-]+", " ", text.lower()).split()


def wake_spellings(spec: dict, lang: str) -> list[list[str]]:
    spellings = bc.wake_words(spec, lang)
    for stem in spec.get("accept_stems", []):
        t = tokens(stem)
        if t and " ".join(t) not in bc.ADDRESS_WORDS and t not in spellings:
            spellings.append(t)
    return spellings


def contains(text: str, spellings: list[list[str]]) -> bool:
    words = tokens(text)
    return any(words[i:i + len(s)] == s for s in spellings for i in range(len(words) - len(s) + 1))


def synthetic_rows(bank: Path, words: set[str], specs: dict[str, dict], dropped: collections.Counter) -> list[tuple[dict, Path]]:
    out = []
    for line in (bank / "clips.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["word"] not in words:
            continue
        spec, lang = specs[r["word"]], r["lang"]
        if lang not in (spec.get("forms") or {}):
            dropped["language no longer in spec"] += 1
            continue
        if r["control"] == "wake":
            if not contains(r["text"], wake_spellings(spec, lang)):
                dropped[f"wake clip without a current wake spelling ({r['word']})"] += 1
                continue
        else:
            listed = {t.lower() for t in (spec.get("confusables") or {}).get(lang, [])}
            if r["text"].lower() not in listed:
                dropped[f"look-alike no longer listed ({r['word']}: {r['text']})"] += 1
                continue
        out.append((r, bank / "wav" / f"{r['id']}.wav"))
    return out


def owner_split(rows: list[dict]) -> tuple[list[dict], list[dict], str]:
    """Hold out whole sittings, newest first, until at least a quarter of the word's wake takes
    (and at least 3) are held out. A newest sitting with one take would otherwise be the whole
    test set."""
    positives = [r for r in rows if r["label"] == "positive"]
    sessions = sorted({r["session"] for r in positives}, key=lambda s: max(r["at"] for r in rows if r["session"] == s), reverse=True)
    need = max(3, math.ceil(len(positives) / 4))
    held, count = [], 0
    for s in sessions[:-1]:                     # never hold out every sitting
        held.append(s)
        count += sum(1 for r in positives if r["session"] == s)
        if count >= need:
            return ([r for r in rows if r["session"] not in held], [r for r in rows if r["session"] in held],
                    f"newest sittings {','.join(held)} ({count} wake takes)")
    train, hold = [], []
    for key in sorted({(r["language"], r["label"]) for r in rows}):
        group = sorted([r for r in rows if (r["language"], r["label"]) == key], key=lambda r: r["at"])
        n = len(group) // 4 if len(group) >= 4 else 0
        train += group[:len(group) - n]
        hold += group[len(group) - n:]
    return train, hold, "newest quarter per language/label (not enough separate sittings: weaker holdout)"


def is_eval(take_id: str) -> bool:
    """Stable 30% of the owner's takes kept out of training as cross-word gate negatives."""
    return int(hashlib.sha1(take_id.encode()).hexdigest(), 16) % 10 < 3


def addressed(text: str) -> bool:
    from addresses import is_addressed
    return is_addressed(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--words", required=True)
    ap.add_argument("--banks", required=True)
    ap.add_argument("--owner", type=Path, required=True)
    ap.add_argument("--owner-share", type=float, default=0.3)
    ap.add_argument("--legacy-word", default="oqto", help="word that the untagged pre-word-pack takes belong to")
    ap.add_argument("--legacy-train", default="", help="sessions (8-char prefixes) of untagged takes with training consent")
    ap.add_argument("--legacy-test", default="", help="sessions of untagged takes used only as held-out test material")
    args = ap.parse_args()
    words = [w for w in args.words.split(",") if w]
    specs = {w: json.loads((WORDS_DIR / f"{w}.json").read_text()) for w in words}
    out = args.out
    if out.exists():
        raise SystemExit(f"{out} exists; a mix is immutable, use a new name")
    (out / "wav").mkdir(parents=True)

    dropped = collections.Counter()
    synth = [row for bank in args.banks.split(",") for row in synthetic_rows(Path(bank).expanduser(), set(words), specs, dropped)]
    owner_all = [json.loads(l) for l in (args.owner / "meta.jsonl").read_text().splitlines()]
    owner_meta = [m for m in owner_all if m.get("word") in words]

    clips, holdout, report = [], [], {}
    for r, wav in synth:
        (out / "wav" / f"{r['id']}.wav").symlink_to(wav.resolve())
        clips.append(r)
    for word in words:
        mine = [m for m in owner_meta if m["word"] == word]
        listed = {lang: ({p for p in specs[word]["forms"].get(lang, [])}, {c for c in (specs[word].get("confusables") or {}).get(lang, [])}) for lang in specs[word].get("forms", {})}
        # Only takes whose phrase is in today's lists (cleaned-up phrases are already in the trash).
        mine = [m for m in mine if m["language"] in listed and m["phrase"] in listed[m["language"]][0 if m["label"] == "positive" else 1]]
        train, hold, policy = owner_split(mine) if mine else ([], [], "no owner takes")
        if word == args.legacy_word:
            # Pre-word-pack sittings: no word tag, phrases spelled the old way ("hey ok-to").
            # Only consented sessions train; the others are held-out test material only.
            legacy_train = {s for s in args.legacy_train.split(",") if s}
            legacy_test = {s for s in args.legacy_test.split(",") if s}
            for m in owner_all:
                if m.get("word") or m.get("label") not in ("positive", "negative"):
                    continue
                session = (m.get("session") or "")[:8]
                if session in legacy_train:
                    train.append(m)
                elif session in legacy_test:
                    hold.append(m)
            policy += f"; legacy train {sorted(legacy_train)}, legacy test {sorted(legacy_test)}"
        synth = [r for r in clips if r["word"] == word and r["control"] == "wake" and not r.get("owner")]
        ratio = args.owner_share / (1 - args.owner_share)
        def weight_for(n_owner: int, n_synth: int) -> int:
            return 1 if not n_owner else max(1, min(30, math.ceil(ratio * n_synth / n_owner)))
        owner_wake = [m for m in train if m["label"] == "positive"]
        w_all = weight_for(len(owner_wake), len(synth))
        w_addr = weight_for(sum(addressed(m["phrase"]) for m in owner_wake), sum(addressed(r["text"]) for r in synth))
        for m in train:
            shutil.copy2(args.owner / "wav" / f"{m['id']}.wav", out / "wav" / f"{m['id']}.wav")
            positive = m["label"] == "positive"
            clips.append({"id": m["id"], "word": word, "lang": m["language"], "kind": "wake" if positive else "context",
                          "control": "wake" if positive else "lookalike", "text": m["phrase"], "voice": "owner",
                          "owner": True, "weight": w_addr if positive and addressed(m["phrase"]) else w_all, "session": m["session"]})
        # The owner's takes of every other word: same voice, wrong word.
        negatives = [m for m in owner_all if m.get("word") != word and not (m.get("word") is None and word == args.legacy_word)]
        n_neg = 0
        for m in negatives:
            if is_eval(m["id"]):
                continue
            neg_id = f"{m['id']}--not-{word}"
            dst = out / "wav" / f"{neg_id}.wav"
            if not dst.exists():
                dst.symlink_to((args.owner / "wav" / f"{m['id']}.wav").resolve())
            clips.append({"id": neg_id, "word": word, "lang": m.get("language") or "de", "kind": "context", "control": "lookalike",
                          "text": m.get("phrase") or "", "voice": "owner", "owner": True, "weight": 1, "session": m.get("session"),
                          "other_word": m.get("word")})
            n_neg += 1
        weight = {"bare": w_all, "addressed": w_addr, "other_word_negatives": n_neg}
        holdout += [{"id": m["id"], "word": word, "language": m["language"], "label": m["label"], "phrase": m["phrase"],
                     "session": m["session"]} for m in hold]
        report[word] = {"synthetic_wake": len(synth), "owner_train": len(train), "owner_wake_train": len(owner_wake),
                        "owner_holdout": len(hold), "owner_weight": weight, "holdout_policy": policy}
    (out / "clips.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clips))
    (out / "holdout.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=1))
    (out / "owner_negative_eval.json").write_text(json.dumps(sorted(m["id"] for m in owner_all if is_eval(m["id"])), indent=1))
    (out / "mix.json").write_text(json.dumps({"words": words, "banks": args.banks.split(","), "owner_share": args.owner_share,
                                              "per_word": report, "dropped_synthetic": dict(dropped)}, ensure_ascii=False, indent=1))
    print(json.dumps({"clips": len(clips), "holdout": len(holdout), "per_word": report,
                      "dropped_synthetic": sum(dropped.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
