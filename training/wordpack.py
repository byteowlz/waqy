"""Word packs: build training text from word specs, then screen, featurize, train and pack.

    python wordpack.py texts <spec-dir> <out.json> [--per-word 480]

Each words/<name>.json gives the spoken forms, confusables and in-context templates for one
word. This stage only produces the synthesis inventory; `synth_spqx.py` speaks it with the
designed-voice bank, then realign_bank/decide_timings screen it and the trainer makes one
head per word (hey_<name> for the trigger, <name> bare for evidence and per-user tuning).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from addresses import is_addressed

TRIGGER_SHARE = 0.55  # of the wake clips, the rest are bare forms
# Owner priority: these four first, then the rest.
PRIORITY = ["elissa", "pi", "opencode", "hex", "ginee"]
LANGS = ("de", "en", "es")
# Used when a spec has no in-context templates: mix natural addressed and bare sentences.
DEFAULT_TEMPLATES = {
    "de": ["{A} {W}.", "{A} {W}, bitte.", "{W}.", "{W}!", "{W}?", "Ich habe {W} gesagt.", "Sag mal {W}.", "{A} {W}, hörst du mich?"],
    "en": ["{A} {W}.", "{A} {W}, please.", "{W}.", "{W}!", "{W}?", "I said {W}.", "Say {W}.", "{A} {W}, can you hear me?"],
    "es": ["{A} {W}.", "{A} {W}, por favor.", "{W}.", "{W}?", "He dicho {W}.", "Di {W}.", "{A} {W}, ¿me oyes?"],
}
LOOKALIKE_SHARE = 0.18


def build(spec_dir: Path, per_word: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    def order(path: Path) -> tuple[int, str]:
        name = json.loads(path.read_text()).get("name", path.stem)
        return (PRIORITY.index(name) if name in PRIORITY else len(PRIORITY), name)

    for spec_path in sorted(spec_dir.glob("*.json"), key=order):
        spec = json.loads(spec_path.read_text())
        if "forms" not in spec:  # legacy spec shape (stopp), not a word pack entry
            continue
        name = spec["name"]
        for lang in LANGS:
            if lang not in spec["forms"]:
                continue
            forms = spec["forms"][lang]
            templates = spec.get("templates", {}).get(lang) or DEFAULT_TEMPLATES[lang]
            if spec.get("kind") == "command":
                # A command word is never preceded by an address.
                templates = [t for t in DEFAULT_TEMPLATES[lang] if "{A}" not in t]
            addressed_only = spec.get("train_bare", True) is False
            if addressed_only:
                # No bare head (the word alone is too common), so every wake clip is addressed.
                templates = [t for t in templates if "{A}" in t] or ["{A} {W}."]
            hey_forms = [f for f in forms if is_addressed(f)]
            # Spellings of the word itself. An addressed-only word has no bare forms, so strip the
            # address instead of reusing "Hey Jupp" (which produced "Hey Hey Jupp").
            bare_forms = [f for f in forms if f not in hey_forms] or sorted({f.split(" ", 1)[-1] for f in hey_forms})
            primary_address = spec.get("address_primary", {}).get(lang, "Hey")
            alternatives = spec.get("address_variants", {}).get(lang, [])
            address = [primary_address] * 3 + [a for a in alternatives if a != primary_address]
            wake_n = int(per_word * (1 - LOOKALIKE_SHARE))
            for i in range(wake_n):
                # alternate trigger-shaped and bare sentences, so both classes exist
                # Spoken spellings matter: a spec may list phonetic variants (ginee/genie/dschini).
                # spoken_primary wins over the display name: a word can be written one way and said another
                primary = spec.get("spoken_primary", {}).get(lang, spec.get("display", spec["name"]))
                display = primary
                spellings = [primary] + [f for f in bare_forms if f.rstrip("?!") != primary]
                spellings = list(dict.fromkeys(s.rstrip("?!") for s in spellings))
                # Forms carry their own punctuation ("Stop!", "Hex?"); templates add the sentence's.
                spoken = rng.choices(spellings, weights=[7] + [1] * (len(spellings) - 1))[0].rstrip("?!")
                templated = [t for t in templates if "{A}" in t]
                bare_templates = [t for t in templates if "{A}" not in t]
                if spec.get("kind") == "command":
                    text = rng.choice(bare_templates or ["{W}"]).replace("{W}", spoken)
                elif addressed_only or i % 2 == 0:
                    text = rng.choice(templated or ["{W}"]).replace("{A}", rng.choice(address)).replace("{W}", spoken)
                else:
                    text = rng.choice(bare_templates or ["{W}"]).replace("{W}", spoken)
                out.append({"id": f"{name}-{lang}-wake-{i:04d}", "word": name, "lang": lang, "kind": "wake",
                            "control": "wake", "text": text})
            confusables = spec.get("confusables", {}).get(lang) or []
            for i in range(int(per_word * LOOKALIKE_SHARE) if confusables else 0):
                text = confusables[i % len(confusables)]
                out.append({"id": f"{name}-{lang}-look-{i:04d}", "word": name, "lang": lang, "kind": "context",
                            "control": "lookalike", "text": text})
    return out


def _arg(name: str, default=None):
    argv = sys.argv
    return argv[argv.index(name) + 1] if name in argv else default


if __name__ == "__main__":
    spec_dir = Path(sys.argv[2]).expanduser()
    per_word = int(_arg("--per-word", 480))
    rows = build(spec_dir, per_word)
    Path(sys.argv[3]).write_text(json.dumps(rows, ensure_ascii=False))
    wake = sum(1 for r in rows if r["control"] == "wake")
    print(json.dumps({"clips": len(rows), "wake": wake, "lookalike": len(rows) - wake,
                      "words": len({r["word"] for r in rows}), "by_lang": {l: sum(1 for r in rows if r["lang"] == l) for l in ("de", "en")}}, indent=1))
