"""Wake words in context: synthetic sentences with "Okto" inside, and the owner's own recordings.

The first models only ever saw an isolated phrase followed by silence, so a wake word inside a
sentence ("hast du Okto schon ausprobiert", "hey Okto, mach das Licht an") scored ~0.01. This
builds (N,16,96) windows on the browser engine's 80 ms grid:

  positive  windows ending 0.08 s before to 0.35 s after the wake word ends
  negative  windows ending before the wake word starts, or > 1.4 s after it ended (the 1.28 s
            window no longer contains it), and every window over look-alike or plain speech

Subcommands:
  synth <root> <n_per_lang>    Piper sentences (clean voices, speaker-disjoint split), word end
                               times from whisper-server word timestamps; unheard sentences dropped
  owner <root>                 owner recordings covered by TRAINING-CONSENT-20261026.json; the
                               held-out sitting is refused by session id, never opened
  assemble <root> <model> <base_model>
                               writes output/<model>/ training arrays: base synthetic + context +
                               owner (augmented), shuffled; test arrays stay synthetic only
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from build_data import (RATE, WHISPER_SERVERS, Featurizer, held_out_speaker, heard_wake, to16k, trim,
                        voice_plan, LENGTH, NOISE, NOISE_W)

HOME = Path.home()
OWNER_TAKES = HOME / ".local/share/foxline/wake-collect-url-1"
OWNER_LIVE = HOME / ".local/share/foxline/wake-live-1"
HELD_OUT_SESSIONS = ("d19ae2f1",)  # second sitting: final test only
CONSENTED = {"e9f2af53", "d82c43e3", "6176d748"}
POS_BEFORE, POS_AFTER, NEG_AFTER = 0.08, 0.35, 1.4

CMDS = {
    "de": ["mach das Licht an", "wie wird das Wetter morgen", "starte einen Timer für zehn Minuten", "schreib Anna eine Nachricht",
           "was steht heute im Kalender", "spiel etwas Musik", "fass die Mail zusammen", "wie spät ist es", "öffne das Projekt",
           "erinnere mich morgen an den Termin"],
    "en": ["turn on the lights", "what's the weather tomorrow", "set a timer for ten minutes", "send Anna a message",
           "what's on my calendar today", "play some music", "summarize this email", "what time is it", "open the project",
           "remind me about the meeting tomorrow"],
}
# {W} is the wake word as spoken in that language; "hey" templates count for both models.
TEMPLATES = {
    "de": [("{W}, {C}.", "bare"), ("Hey {W}, {C}.", "hey"), ("He {W}, {C}.", "hey"), ("Okay, hey {W}, {C}.", "hey"),
           ("Hast du {W} schon ausprobiert?", "bare"), ("Ich habe {W} gestern installiert.", "bare"),
           ("Kannst du {W} mal fragen, wie spät es ist?", "bare"), ("Mit {W} geht das viel schneller.", "bare"),
           ("Sag {W}, dass ich später komme.", "bare"), ("Frag doch einfach {W}.", "bare"), ("Das macht jetzt {W}.", "bare"),
           ("Ich würde dir {W} empfehlen.", "bare"), ("Also {W}, {C}.", "bare"), ("Na gut, {W}, {C}.", "bare")],
    "en": [("{W}, {C}.", "bare"), ("Hey {W}, {C}.", "hey"), ("Okay, hey {W}, {C}.", "hey"), ("Have you tried {W} yet?", "bare"),
           ("I installed {W} yesterday.", "bare"), ("Can you ask {W} what time it is?", "bare"), ("With {W} this is much faster.", "bare"),
           ("Tell {W} I'll be late.", "bare"), ("Just ask {W}.", "bare"), ("I love {W}, it's great.", "bare"),
           ("I'd recommend {W} to you.", "bare"), ("So {W}, {C}.", "bare"), ("Alright, {W}, {C}.", "bare")],
}
WORD = {"de": "Okto", "en": "Octo"}
LOOKALIKE = re.compile(r"(oktober|october|oktav|octav|oktop|octop|doktor|doctor)")

def norm(word: str) -> str:
    return re.sub(r"[^\wäöüß]", "", word.lower())


def whisper_words(server: str, path: Path, lang: str) -> list[dict]:
    """Merged word timestamps from whisper-server verbose_json (tokens without a leading space join the previous word)."""
    import urllib.request
    boundary = "----ctx" + str(abs(hash(str(path))))[:12]
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in (("language", lang), ("response_format", "verbose_json"), ("temperature", "0"))]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + path.read_bytes() + b"\r\n" + f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(server, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read())
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            text = w["word"]
            if not text.strip() or not any(ch.isalnum() for ch in text):
                continue
            if words and not text.startswith(" "):
                words[-1]["w"] += text
                words[-1]["t1"] = w["end"]
            else:
                words.append({"w": text.strip(), "t0": w["start"], "t1": w["end"]})
    return words


# ---------------------------------------------------------------- synthetic sentences
def synth(root: Path, n_per_lang: int, seed: int = 20261027) -> None:
    from piper import PiperVoice, SynthesisConfig

    out = root / "context"
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    voices, speakers_by = {}, {}

    def voice(name):
        if name not in voices:
            v = PiperVoice.load(str(root / "voices" / f"{name}.onnx"))
            voices[name] = v
            names = sorted(v.config.speaker_id_map or {})
            speakers_by[name] = {"held-out": [v.config.speaker_id_map[n] for n in names if held_out_speaker(n)],
                                 "train": [v.config.speaker_id_map[n] for n in names if not held_out_speaker(n)],
                                 "all": list(range(v.config.num_speakers))}
        return voices[name]

    german = voice("de_DE-thorsten-medium")
    rows = []
    for lang in ("de", "en"):
        for i in range(n_per_lang):
            split = "test" if i % 6 == 0 else "train"
            plan = voice_plan(split, lang)
            name, selector, _ = rng.choices(plan, weights=[w for *_, w in plan])[0]
            v = voice(name)
            speaker = rng.choice(speakers_by[name][selector]) if selector else None
            template, form = rng.choice(TEMPLATES[lang])
            text = template.format(W=WORD[lang], C=rng.choice(CMDS[lang]))
            cfg = SynthesisConfig(speaker_id=speaker, length_scale=rng.choice(LENGTH), noise_scale=rng.choice(NOISE), noise_w_scale=rng.choice(NOISE_W))
            if lang == "de" and not name.startswith("de_"):
                pcm = v.phoneme_ids_to_audio([x for s in german.phonemize(text) for x in v.phonemes_to_ids(s)], cfg)
            else:
                pcm = np.concatenate([c.audio_float_array for c in v.synthesize(text, syn_config=cfg)])
            pcm = trim(to16k(np.asarray(pcm, np.float32), v.config.sample_rate))
            if len(pcm) / RATE > 7.5:
                continue  # babble
            key = f"{lang}-{i:05d}"
            sf.write(str(out / "wav" / f"{key}.wav"), pcm, RATE, subtype="PCM_16")
            rows.append({"key": key, "lang": lang, "split": split, "voice": name, "speaker": speaker, "text": text, "form": form})
    print(f"synthesized {len(rows)} sentences; aligning", flush=True)

    def align(item):
        i, row = item
        words = whisper_words(WHISPER_SERVERS[i % len(WHISPER_SERVERS)], out / "wav" / f"{row['key']}.wav", row["lang"])
        hits = [w for w in words if heard_wake(w["w"])]
        return row | {"words": words, "wake": hits[0] if len(hits) == 1 else None}

    started = time.time()
    with ThreadPoolExecutor(2 * len(WHISPER_SERVERS)) as pool:
        aligned = list(pool.map(align, enumerate(rows)))
    with (out / "manifest.jsonl").open("w") as f:
        for row in aligned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept = sum(r["wake"] is not None for r in aligned)
    print(f"aligned in {time.time() - started:.0f}s: wake word heard once in {kept}/{len(aligned)}", flush=True)


# ---------------------------------------------------------------- windows
def windows_around(frames: np.ndarray, lead: float, events: list[tuple[float, float, str]]) -> dict[str, list[np.ndarray]]:
    """Label each 16-frame window by its decision time (end) relative to events (t0, t1, tag).

    tag 'wake' -> positive in [t1-POS_BEFORE, t1+POS_AFTER]; ambiguous until t1+NEG_AFTER.
    tag 'neg'  -> negative while the window overlaps the word (t0 .. t1+NEG_AFTER).
    Windows touching no event are 'plain' negatives.
    """
    out = {"pos": [], "neg": [], "plain": []}
    for k in range(15, len(frames)):
        t = (k + 1) * 0.08 - lead
        tag = "plain"
        for t0, t1, kind in events:
            if kind == "wake":
                if t1 - POS_BEFORE <= t <= t1 + POS_AFTER:
                    tag = "pos"
                    break
                if t0 - 0.1 <= t <= t1 + NEG_AFTER:
                    tag = "skip"
            elif t0 <= t <= t1 + NEG_AFTER and tag == "plain":
                tag = "neg"
        if tag != "skip":
            out[tag].append(frames[k - 15:k + 1])
    return out


class Augmenter:
    """Gain, background noise at 5-25 dB SNR, simulated room echo (same clean sources as training)."""

    def __init__(self, root: Path, seed: int):
        self.rng = np.random.default_rng(seed)
        self.bg = sorted((root / "data/backgrounds").glob("*.wav"))
        self.rirs = sorted((root / "data/rirs").glob("*.wav"))

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        rng = self.rng
        x = pcm.astype(np.float32)
        if rng.random() < 0.4:
            rir, _ = sf.read(str(self.rirs[rng.integers(len(self.rirs))]), dtype="float32")
            x = np.convolve(x, rir)[: len(x)].astype(np.float32)
        noise, _ = sf.read(str(self.bg[rng.integers(len(self.bg))]), dtype="float32")
        if noise.ndim > 1:
            noise = noise[:, 0]
        noise = np.tile(noise, int(np.ceil(len(x) / max(len(noise), 1))) + 1)
        start = rng.integers(0, max(1, len(noise) - len(x)))
        noise = noise[start:start + len(x)]
        snr = rng.uniform(5, 25)
        p_sig, p_noise = np.mean(x ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
        x = x + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr / 10)))
        x *= 10 ** (rng.uniform(-6, 6) / 20)
        peak = np.abs(x).max()
        return x / peak * 0.99 if peak > 0.99 else x


def featurize(feat: Featurizer, aug: Augmenter | None, pcm: np.ndarray, lead: float, events, copies: int) -> dict[str, list]:
    pad = np.random.default_rng(len(pcm)).normal(0, 3e-4, int(lead * RATE)).astype(np.float32)
    out = {"pos": [], "neg": [], "plain": []}
    for c in range(copies + 1):
        x = pcm if c == 0 or aug is None else aug(pcm)
        w = windows_around(feat.frames(np.concatenate([pad, x])), lead, events)
        for k in out:
            out[k] += w[k]
    return out


def save(path: Path, rows: list[np.ndarray]) -> int:
    arr = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 16, 96), np.float32)
    np.save(path, arr)
    return len(arr)


def context_features(root: Path) -> None:
    out = root / "context"
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    feat, aug = Featurizer(), Augmenter(root, 3)
    sets: dict[str, list] = {}
    counts = {}
    for row in rows:
        if row["wake"] is None:
            continue
        pcm, _ = sf.read(str(out / "wav" / f"{row['key']}.wav"), dtype="float32")
        w = row["wake"]
        is_hey = row["form"] == "hey"
        res = featurize(feat, aug if row["split"] == "train" else None, pcm, 1.0, [(w["t0"], w["t1"], "wake")], 2 if row["split"] == "train" else 0)
        for model in ("hey_oqto", "oqto"):
            positive = model == "oqto" or is_hey
            key = f"{model}-{row['split']}"
            sets.setdefault(key + "-pos", []).extend(res["pos"] if positive else [])
            sets.setdefault(key + "-neg", []).extend(res["plain"] + ([] if positive else res["pos"]))
    for key, data in sets.items():
        counts[key] = save(out / f"{key}.npy", data)
    (out / "features-summary.json").write_text(json.dumps({"counts": counts, "positive_window": [-POS_BEFORE, POS_AFTER],
                                                           "negative_after": NEG_AFTER}, indent=1))
    print(json.dumps(counts, indent=1))


# ---------------------------------------------------------------- owner recordings
def owner(root: Path, copies: int = 12) -> None:
    consent = json.loads((OWNER_TAKES / "TRAINING-CONSENT-20261026.json").read_text())
    assert any("d19ae2f1" in s for s in consent["held_out_never_trained"])
    out = root / "owner"
    out.mkdir(parents=True, exist_ok=True)
    feat, aug = Featurizer(), Augmenter(root, 11)
    sets: dict[str, list] = {}
    used = {"takes": 0, "live_sessions": []}

    def add(model, res, positive):
        sets.setdefault(f"{model}-pos", []).extend(res["pos"] if positive else [])
        sets.setdefault(f"{model}-neg", []).extend(res["neg"] + res["plain"] + ([] if positive else res["pos"]))

    from build_data import speech_span
    for meta_path in sorted(OWNER_TAKES.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        session = meta["sessionId"][:8]
        if session in HELD_OUT_SESSIONS or session not in CONSENTED:
            continue
        pcm, rate = sf.read(str(meta_path.parent / "audio.wav"), dtype="float32")
        if len(pcm) / rate < 1.0:
            continue  # truncated by the old 60 s session cap
        span = speech_span(pcm)
        if span is None:
            continue
        t0, t1 = span[0] / RATE, span[1] / RATE
        used["takes"] += 1
        wake = meta["label"] == "positive"
        res = featurize(feat, aug, pcm, 1.0, [(t0, t1, "wake" if wake else "neg")], copies)
        is_hey = meta["phrase"].lower().startswith("hey")
        add("oqto", res, wake)
        add("hey_oqto", res, wake and is_hey)

    for session_dir in sorted(OWNER_LIVE.glob("*-*-*-*-*")):
        if session_dir.name[:8] not in CONSENTED:
            continue
        reviews = sorted(session_dir.glob("review-*/review.json"), key=lambda p: int(p.parent.name.split("-")[1]))
        if not reviews:
            continue
        review = json.loads(reviews[-1].read_text())
        pcm, _ = sf.read(str(reviews[-1].parent / "session.wav"), dtype="float32")
        events_oqto, events_hey = [], []
        for s in review["spoken"]:
            if s["kind"] == "wake":
                events_oqto.append((s["t0"], s["t1"], "wake"))
                events_hey.append((s["t0"], s["t1"], "wake" if s["hey"] else "neg"))
            else:  # conflicts with a look-alike (owner marked these as wrong) and 'possible' words
                events_oqto.append((s["t0"], s["t1"], "neg"))
                events_hey.append((s["t0"], s["t1"], "neg"))
        for l in review["lookalikes"]:
            events_oqto.append((l["t0"], l["t0"] + 0.6, "neg"))
            events_hey.append((l["t0"], l["t0"] + 0.6, "neg"))
        for model, events in (("oqto", events_oqto), ("hey_oqto", events_hey)):
            res = featurize(feat, aug, pcm, 1.0, events, copies)
            add(model, res, True)
        used["live_sessions"].append(session_dir.name[:8])

    counts = {key: save(out / f"{key}.npy", data) for key, data in sets.items()}
    (out / "summary.json").write_text(json.dumps({"counts": counts, "used": used, "copies": copies,
                                                  "held_out_never_opened": HELD_OUT_SESSIONS}, indent=1))
    print(json.dumps({"counts": counts, "used": used}, indent=1))


# ---------------------------------------------------------------- external bank (spqx designed voices)
def bank(root: Path, bank_dir: Path, copies: int = 1, models: tuple[str, ...] = ("hey_oqto", "oqto")) -> None:
    """Features from an aligned spqx voice bank. Test = voices whose name hash falls in the held-out set."""
    out = Path(os.environ.get("WAKE_BANK_OUT", root / "context-spqx"))
    out.mkdir(parents=True, exist_ok=True)
    # clean.jsonl (decide_timings.py) carries word-level spans screened on the current files;
    # aligned.jsonl is the older, unscreened pass kept for comparison.
    source = bank_dir / "clean.jsonl" if (bank_dir / "clean.jsonl").exists() else bank_dir / "aligned.jsonl"
    rows = [json.loads(l) for l in source.read_text().splitlines()]
    print(f"bank source: {source.name}", flush=True)
    feat, aug = Featurizer(), Augmenter(root, 17)
    sets: dict[str, list] = {}
    counts = {"clips": 0, "dropped_no_wake": 0}
    # the generator assigned a voice per clip as pool[index % len(pool)]; recover it when the
    # clips file carries no per-clip voice, so the train/test split stays speaker-disjoint
    pools: dict[str, list] = {}
    voices_file = bank_dir / "voices.json"
    if voices_file.exists():
        for v in json.loads(voices_file.read_text()):
            pools.setdefault(v["lang"], []).append(v["voice"])

    def split_of(voice):
        return "test" if held_out_speaker(voice) else "train"

    for row in rows:
        if "wake" in row and row["control"] == "wake" and row["wake"] is None:
            counts["dropped_no_wake"] += 1
            continue
        pcm, _ = sf.read(str(bank_dir / "wav" / f"{row['id']}.wav"), dtype="float32")
        voice = row.get("voice") or "unknown"
        if pools.get(row["lang"]) and voice in ("unknown", "stoppbank"):
            pool = pools[row["lang"]]
            voice = pool[int(row["id"].split("-")[-1]) % len(pool)]
        # the shared corpus ships its own voice-disjoint split; honour it when present
        split = "test" if row.get("consumer_split") == "eval" else split_of(voice)
        events = []
        if row["control"] == "wake":
            span = row.get("wake") or {"t0": row["t0"], "t1": row["t1"]}
            events.append((span["t0"], span["t1"], "wake"))
        else:
            # a look-alike sentence: every window that contains the look-alike word is a negative for both heads
            if row.get("words"):
                for w in row["words"]:
                    if LOOKALIKE.search(norm(w["w"])):
                        events.append((w["t0"], w["t1"], "neg"))
            else:  # clean.jsonl look-alike rows carry the located word span directly
                events.append((row["t0"], row["t1"], "neg"))
        res = featurize(feat, aug if split == "train" else None, pcm, 1.0, events, copies if split == "train" else 0)
        is_hey = any(norm(w["w"]) in ("hey", "hei", "hi", "hej") for w in (row.get("words") or [])) or row["text"].lower().lstrip().startswith(("hey", "he ", "okay, hey"))
        for model in models:
            # a "hey_" head only counts the hey form as positive; any other head (oqto, stopp, ...) counts every form
            positive = (not model.startswith("hey_")) or is_hey
            key = f"{model}-{split}"
            sets.setdefault(key + "-pos", []).extend(res["pos"] if positive else [])
            sets.setdefault(key + "-neg", []).extend(res["neg"] + res["plain"] + ([] if positive else res["pos"]))
        counts["clips"] += 1
    for key, data in sorted(sets.items()):
        counts[key] = save(out / f"{key}.npy", data)
    (out / "summary.json").write_text(json.dumps(counts | {"bank": str(bank_dir), "models": list(models),
                                                           "held_out_rule": "sha256(voice) % 7 == 0"}, indent=1))
    print(json.dumps(counts, indent=1))


# ---------------------------------------------------------------- assemble
def assemble(root: Path, model: str, base: str) -> None:
    kind = model.split("-")[0]
    src, dst = root / "output" / base, root / "output" / model
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)

    def load(path):
        return np.load(path).astype(np.float32) if path.exists() else np.zeros((0, 16, 96), np.float32)

    syn_pos, syn_neg = load(src / "positive_features_train.npy"), load(src / "negative_features_train.npy")
    ctx_pos, ctx_neg = load(root / "context" / f"{kind}-train-pos.npy"), load(root / "context" / f"{kind}-train-neg.npy")
    if (root / "context-spqx" / f"{kind}-train-pos.npy").exists():  # spqx designed-voice bank
        ctx_pos = np.concatenate([ctx_pos, load(root / "context-spqx" / f"{kind}-train-pos.npy")])
        ctx_neg = np.concatenate([ctx_neg, load(root / "context-spqx" / f"{kind}-train-neg.npy")])
    own_pos, own_neg = load(root / "owner" / f"{kind}-pos.npy"), load(root / "owner" / f"{kind}-neg.npy")
    pos = np.concatenate([syn_pos, ctx_pos, own_pos])
    neg = np.concatenate([syn_neg, ctx_neg, own_neg])
    np.save(dst / "positive_features_train.npy", pos[rng.permutation(len(pos))])
    np.save(dst / "negative_features_train.npy", neg[rng.permutation(len(neg))])
    for name in ("background_noise_features_train.npy", "background_noise_features_test.npy", "negative_features_test.npy"):
        np.save(dst / name, load(src / name))
    test_pos = np.concatenate([load(src / "positive_features_test.npy"), load(root / "context" / f"{kind}-test-pos.npy")])
    if (root / "context-spqx" / f"{kind}-test-pos.npy").exists():
        test_pos = np.concatenate([test_pos, load(root / "context-spqx" / f"{kind}-test-pos.npy")])
    # spqx held-out-voice windows are kept separately for evaluation only, never trained on
    for name in ("pos", "neg"):
        src_spqx = root / "context-spqx" / f"{kind}-test-{name}.npy"
        if src_spqx.exists():
            np.save(root / "context-spqx-eval" / f"{kind}-test-{name}.npy" if (root / "context-spqx-eval").mkdir(parents=True, exist_ok=True) is None else None,
                    load(src_spqx))
    np.save(dst / "positive_features_test.npy", test_pos)
    summary = {"model": model, "base": base, "synthetic_context_sources": "piper + spqx designed voices (train-voice windows)",
               "train_pos": {"synthetic": len(syn_pos), "context": len(ctx_pos), "owner": len(own_pos)},
               "train_neg": {"synthetic_adversarial": len(syn_neg), "context": len(ctx_neg), "owner": len(own_neg)},
               "test_pos": len(test_pos), "owner_test_data": "none (second sitting is held out)"}
    (dst / "assemble-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    root = Path(sys.argv[2]).expanduser()
    command = sys.argv[1]
    if command == "synth":
        synth(root, int(sys.argv[3]))
        context_features(root)
    elif command == "context-features":
        context_features(root)
    elif command == "bank":
        bank(root, Path(sys.argv[3]).expanduser(), int(sys.argv[4]) if len(sys.argv) > 4 else 1,
             tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else ("hey_oqto", "oqto"))
    elif command == "owner":
        owner(root)
    elif command == "assemble":
        assemble(root, sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
