"""Turn a synthesized word-pack inventory into one head per word (and train them).

  python wordpack_train.py <synth-dir> <out-dir> --words elissa,pi [--train] [--steps 20000]

Per clip: whisper-server confirms the clip says the word and gives its timing. Positives are
windows ending just after the word; negatives are windows from elsewhere in the same clip and
from look-alike sentences, augmented on the training side (echo, noise, gain). Each word gets
two heads: `hey_<word>` (the trigger form) and `<word>` (bare, kept for evidence and per-user
tuning; the runtime gates bare heads so they cannot fire inside continuous speech).

`--train` writes the feature arrays, runs pipeline.py per head (small model, CUDA if available)
and exports; otherwise it only reports what it would build.
"""
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pathlib

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from build_data import WHISPER_SERVERS, Featurizer, held_out_speaker  # noqa: E402

# Screening ASR: Parakeet (parakeet.cpp CLI, word timestamps) when PARAKEET_CLI is set, else whisper-server.
PARAKEET_CLI = __import__("os").environ.get("PARAKEET_CLI")
PARAKEET_MODEL = __import__("os").environ.get("PARAKEET_MODEL")


PARAKEET_URL = __import__("os").environ.get("PARAKEET_URL")


def bench_transcripts(clips: list[Path], cli: str, model: str, workers: int = 8) -> dict[str, str]:
    """Transcribe many clips in one model load with `parakeet-cli bench --json` (~0.3 s per clip)."""
    import json as _json
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for c in clips:
            f.write(str(c) + "\n")
        manifest = f.name
    out = subprocess.run([cli, "bench", "--model", model, "--manifest", manifest, "--threads", str(workers),
                          "--concurrency", str(max(1, workers // 4)), "--json", "/tmp/bench.json"],
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-300:] or "parakeet bench failed")
    data = _json.loads(pathlib.Path("/tmp/bench.json").read_text())
    return {pathlib.Path(f["path"]).stem: f.get("text", "") for f in data.get("files", [])}


def energy_span_words(path: Path) -> list[dict]:
    """A short clip is one utterance: treat its speech span as the word (no ASR timing needed)."""
    import soundfile as sf
    pcm, rate = sf.read(str(path), dtype="float32")
    if rate != 16000:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from build_data import to16k
        pcm = to16k(pcm, rate)
    frame = 320
    e = __import__("numpy").sqrt((pcm[: len(pcm) // frame * frame].reshape(-1, frame) ** 2).mean(axis=1))
    on = __import__("numpy").flatnonzero(e > max(e.max() * 0.05, 1e-4))
    if not len(on):
        return []
    return [{"w": "utterance", "t0": float(on[0] * frame / 16000), "t1": float((on[-1] + 1) * frame / 16000)}]


def parakeet_server_words(path, lang):
    """Text + segment times from parakeet-server; word times estimated by character share.

    The server does not expose word timestamps, and one CLI call per clip pays a model load, so
    the fast path screens with the server and apportions each segment's duration across its words
    by character count. `validate_parakeet_timing` measures that estimate against CLI word times.
    """
    import json as _json
    import urllib.request
    boundary = "----pk" + str(abs(hash(str(path))))[:10]
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{lang}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            f'Content-Type: audio/wav\r\n\r\n').encode() + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(PARAKEET_URL, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = _json.loads(r.read())
    words = []
    for seg in data.get("segments", []):
        toks = seg["text"].split()
        if not toks:
            continue
        span = max(0.05, float(seg["end"]) - float(seg["start"]))
        total = sum(len(t) for t in toks)
        t = float(seg["start"])
        for tok in toks:
            d = span * len(tok) / total
            words.append({"w": tok, "t0": t, "t1": t + d})
            t += d
    return words


def parakeet_words(path, lang):
    import json as _json
    import subprocess
    lang_map = {"de": "de", "en": "en", "es": "es"}
    # Two threads per process: 8 workers with default threads oversubscribed the 8-core box and
    # took ~22 s per clip; the timing pass is the slowest stage of a word-pack run.
    out = subprocess.run([PARAKEET_CLI, "transcribe", "--model", PARAKEET_MODEL, "--input", str(path),
                          "--lang", lang_map.get(lang, lang), "--timestamps", "--json", "--threads", "2"],
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-200:] or "parakeet failed")
    data = _json.loads(out.stdout.strip().splitlines()[-1])
    return [{"w": w["w"], "t0": float(w["start"]), "t1": float(w["end"])} for w in data.get("words", [])]
from context_data import Augmenter, featurize, whisper_words  # noqa: E402
from addresses import is_addressed  # noqa: E402

ACCEPT: dict[str, list[str]] = {}
# Resolved from the output root at run time: this tree has been renamed more than once, and a
# hard-coded data root silently disables augmentation (empty background list).
GENERAL_NAME = "clean_general_negatives.npy"
BACKGROUND_HEAD = "oqto-pub1"


def data_root(out: Path) -> Path:
    """The training data root that holds data/ and output/ next to the word-pack output dir."""
    return out.parent


def stems(name: str, extra: list[str] | None = None) -> list[str]:
    """Accepted spoken spellings; a spec may list variants (e.g. ginee/genie/dschini)."""
    if extra:
        return [e.lower() for e in extra]
    return [b.lower() for b in name.replace("-", " ").split() if b.lower() not in ("hey", "he")]


def heard_word(tokens: list[str], name: str, extra: list[str] | None = None) -> int | None:
    want = stems(name, extra)
    for i, tok in enumerate(tokens):
        t = tok.lower().strip(".,!?¿¡")
        for w in want:
            if t and (w in t or difflib.SequenceMatcher(None, t, w).ratio() >= 0.72):
                return i
    return None


SPECS: dict[str, dict] = {}


def lookalike_says_wake(heard: str, word: str, lang: str) -> bool:
    """A look-alike clip whose transcript contains the wake word would teach the head to ignore
    its own word. Same rule as the collection generator (build_collection.lookalike_problem):
    contains a wake form as whole words, or equals a same-sound spelling. Not fuzzy, so near-misses
    like Hexe/Hex stay."""
    from build_collection import lookalike_problem
    spec = SPECS.get(word)
    return bool(spec and lang in (spec.get("forms") or {}) and lookalike_problem(heard, spec, lang))


def load_aligned(paths: list[Path]) -> dict[str, dict]:
    """Earlier screening results by clip id. Synthetic clips are immutable, so their transcript and
    word timing never change; only the rules applied to them do (re-checked in align_all)."""
    cache = {}
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("heard") is not None or r.get("drop"):
                cache[r["id"]] = r
    return cache


def align_all(rows: list[dict], synth: Path, workers: int, cache: dict[str, dict] | None = None) -> list[dict]:
    cache = cache or {}
    reused = []
    fresh = []
    for r in rows:
        old = cache.get(r["id"])
        if old is None or r.get("owner"):
            fresh.append(r)
            continue
        merged = r | {k: old.get(k) for k in ("t0", "t1", "heard", "timing", "drop")}
        if merged["control"] != "wake" and not merged["drop"] and lookalike_says_wake(merged.get("heard") or "", r["word"], r["lang"]):
            merged["drop"] = "lookalike_says_wake"
        reused.append(merged)
    if reused:
        print(f"reusing screening for {len(reused)} clips, screening {len(fresh)}", flush=True)
    return reused + _align_fresh(fresh, synth, workers)


def _align_fresh(rows: list[dict], synth: Path, workers: int) -> list[dict]:
    # One batched pass gives every clip's transcript; only clips whose word sits inside a longer
    # sentence need a per-clip timing pass, and short clips use their own speech span.
    texts: dict[str, str] = {}
    synthetic = [synth / "wav" / f"{r['id']}.wav" for r in rows if not r.get("owner")]
    if PARAKEET_CLI and PARAKEET_MODEL and synthetic:   # owner takes skip ASR; bench refuses an empty list
        texts = bench_transcripts(synthetic, PARAKEET_CLI, PARAKEET_MODEL, workers)

    def one(item):
        i, r = item
        wav = synth / "wav" / f"{r['id']}.wav"
        if not wav.exists():
            return r | {"drop": "missing"}
        if r.get("owner"):
            # The owner's own label is ground truth: ASR screening rejected natural owner
            # pronunciations ("Hey Alyssa") before. A take is one utterance; use its speech span.
            span = energy_span_words(wav)
            if not span:
                return r | {"drop": "silent"}
            return r | {"t0": span[0]["t0"], "t1": span[0]["t1"], "heard": "(owner label)", "drop": None, "timing": "owner-energy"}
        heard = texts.get(r["id"])
        if heard is None:
            tokens = parakeet_words(wav, r["lang"]) if PARAKEET_CLI else whisper_words(
                WHISPER_SERVERS[i % len(WHISPER_SERVERS)], wav, r["lang"])
            heard = " ".join(w["w"] for w in tokens)
        if r["control"] == "wake":
            if not heard_word(heard.split(), r["word"], ACCEPT.get(r["word"])):
                return r | {"drop": "word_not_heard", "heard": heard}
            if len(heard.split()) <= 2:  # the clip is essentially the word: no timing pass needed
                span = energy_span_words(wav)
                if not span:
                    return r | {"drop": "silent", "heard": heard}
                return r | {"t0": span[0]["t0"], "t1": span[0]["t1"], "heard": heard, "drop": None, "timing": "energy"}
            tokens = parakeet_words(wav, r["lang"]) if PARAKEET_CLI else whisper_words(
                WHISPER_SERVERS[i % len(WHISPER_SERVERS)], wav, r["lang"])
            idx = heard_word([w["w"] for w in tokens], r["word"], ACCEPT.get(r["word"]))
            if idx is None:
                return r | {"drop": "word_not_heard_timed", "heard": heard}
            return r | {"t0": tokens[idx]["t0"], "t1": tokens[idx]["t1"], "heard": heard, "drop": None, "timing": "asr"}
        if not heard.split():
            return r | {"drop": "no_words"}
        if lookalike_says_wake(heard, r["word"], r["lang"]):
            return r | {"drop": "lookalike_says_wake", "heard": heard}
        return r | {"t0": 0.0, "t1": max(0.3, len(heard.split()) * 0.25), "heard": heard, "drop": None, "timing": "approx"}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, enumerate(rows)))


def heads_for(word: str, spec: dict) -> tuple[list[str], list[str]]:
    """Addresses and head names for one word. train_bare false means addressed-only: the bare
    word is too common on its own (Jupp is also how people say yes) to be a head."""
    if spec.get("kind") == "command":
        return [], [word]   # a command is never addressed: one bare head
    addresses = [a.lower() for a in spec.get("address", {}).get("en", ["Hey"])]
    bare = spec.get("train_bare", True) is not False
    return addresses, [f"{a}_{word}" for a in addresses] + ([word] if bare else [])


def build_word(word: str, aligned: list[dict], synth: Path, out: Path, copies: int, addresses=("hey",), models=None) -> dict:
    base = data_root(out)
    feat, aug = Featurizer(), Augmenter(base, 31)
    models = models or [f"{a}_{word}" for a in addresses] + [word]
    data = {(m, s): {"pos": [], "neg": []} for m in models for s in ("train", "test")}
    kept = {"wake": 0, "lookalike": 0}
    for r in aligned:
        if r.get("drop"):
            continue
        pcm, _ = sf.read(synth / "wav" / f"{r['id']}.wav", dtype="float32")
        # Owner takes always train (their held-out sitting is excluded upstream and scored by the
        # runtime gate); synthetic voices split by speaker.
        split = "train" if r.get("owner") else ("test" if held_out_speaker(str(r.get("voice", ""))) else "train")
        control = "wake" if r["control"] == "wake" else "neg"
        weight = max(1, int(r.get("weight", 1)))
        res = featurize(feat, aug if split == "train" else None, pcm, 1.0, [(r["t0"], r["t1"], control)],
                        copies * weight if split == "train" else 0)
        kept["wake" if control == "wake" else "lookalike"] += 1
        addressed = is_addressed(r["text"])
        for model in models:
            positive = addressed if model.rsplit("_", 1)[0] in addresses else True
            bucket = data[(model, split)]
            bucket["pos"].extend(res["pos"] if positive else [])
            bucket["neg"].extend(res["neg"] + res["plain"] + ([] if positive else res["pos"]))
    for model in models:
        d = out / "output" / model
        d.mkdir(parents=True, exist_ok=True)
        rng = __import__("numpy").random.default_rng(abs(hash(model)) % 2**32)
        for split in ("train", "test"):
            for kind in ("pos", "neg"):
                rows = data[(model, split)][kind]
                arr = __import__("numpy").stack(rows).astype("float32") if rows else __import__("numpy").zeros((0, 16, 96), "float32")
                if split == "train" and len(arr):
                    arr = arr[rng.permutation(len(arr))]
                name = {"pos": "positive_features_" + split, "neg": "negative_features_" + split}[kind]
                __import__("numpy").save(d / f"{name}.npy", arr)
        for n in ("background_noise_features_train.npy", "background_noise_features_test.npy"):
            src = base / "output" / BACKGROUND_HEAD / n
            if src.exists():
                __import__("numpy").save(d / n, __import__("numpy").load(src))
    return {"word": word, "kept": kept,
            **{f"{m}-{s}-{k}": len(data[(m, s)][k]) for m in models for s in ("train", "test") for k in ("pos", "neg")}}


def train_head(root: Path, model: str, steps: int, words: list[str]) -> dict:
    """The pack1 recipe (ADR 0007, CHANGELOG 2026-10-29), same as run_train_heads.sh:
    cross-word negatives for addressed heads, max_negative_weight 1 and balanced batches (the
    default ramp collapsed small sets to constant-negative), small model, output_dir pinned to
    this root, export from the explicit checkpoint (livekit's run_export exported random weights
    when the directories differed), then the collapse gate on the head's own held-out positives."""
    here = Path(__file__).parent
    py = sys.executable
    env = {**__import__("os").environ, "WAKE_DEVICE": __import__("os").environ.get("WAKE_DEVICE", "cuda"), "WAKE_GPU_GB": "1.2",
           "WAKE_BATCH_JSON": '{"positive": 200, "adversarial_negative": 200, "ACAV100M_sample": 200, "background_noise": 50}',
           "WAQY_ADV_WORDS": ",".join(words)}
    head_dir = root / "output" / model
    if not (head_dir / "positive_features_train.npy").exists():
        return {"model": model, "error": "no features"}
    if model.startswith("hey_") or model.startswith("he_"):
        r = subprocess.run([py, str(here / "adv_negative.py"), model.split("_", 1)[1], str(root)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            return {"model": model, "stage": "adversarial", "error": r.stderr[-400:]}
    subprocess.run([py, str(here / "pipeline.py"), str(root), model, "config"], check=True, capture_output=True, env=env)
    cfg = root / "configs" / f"{model}.yaml"
    import re as _re
    text = cfg.read_text()
    for key, value in (("output_dir", str(root / "output")), ("steps", str(steps)), ("max_negative_weight", "1"), ("model_size", "small")):
        text = _re.sub(rf"^{key}: .*$", f"{key}: {value}", text, flags=_re.M)
    cfg.write_text(text)
    log = root / "logs" / f"{model}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w") as handle:
        r = subprocess.run([py, str(here / "pipeline.py"), str(root), model, "train"], stdout=handle, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        return {"model": model, "stage": "train", "error": log.read_text()[-400:]}
    checkpoints = sorted(head_dir.rglob(f"{model}.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return {"model": model, "stage": "export", "error": "no checkpoint"}
    onnx = head_dir / f"{model}.onnx"
    r = subprocess.run([py, str(here / "export_head.py"), str(cfg), str(checkpoints[-1]), str(onnx)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return {"model": model, "stage": "export", "error": r.stderr[-400:]}
    # An empty test split (no held-out synthetic voice for this word) is not a collapse: verify on
    # training positives instead and say so. Deleting those exports lost 13 trained heads once.
    test = head_dir / "positive_features_test.npy"
    import numpy as _np
    verified_on = "test" if test.exists() and len(_np.load(test, mmap_mode="r")) else "train"
    sample = test if verified_on == "test" else head_dir / "positive_features_train.npy"
    r = subprocess.run([py, str(here / "verify_head.py"), str(onnx), str(sample)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        onnx.unlink(missing_ok=True)
        return {"model": model, "collapsed": True, "verify": r.stdout.strip(), "verified_on": verified_on}
    return {"model": model, "trained": True, "verify": r.stdout.strip(), "verified_on": verified_on, "checkpoint": str(checkpoints[-1])}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("synth")
    ap.add_argument("out")
    ap.add_argument("--words", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--reuse-aligned", default="", help="glob of earlier aligned-*.jsonl to reuse screening from")
    args = ap.parse_args()
    synth = Path(args.synth).expanduser()
    root = Path(args.out).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (synth / "clips.jsonl").read_text().splitlines()]
    words = args.words.split(",")
    for w in words:
        spec_path = Path(__file__).parent / "words" / f"{w}.json"
        if spec_path.exists():
            SPECS[w] = json.loads(spec_path.read_text())
            ACCEPT[w] = SPECS[w].get("accept_stems")
    rows = [r for r in rows if r["word"] in words]
    aligned_path = root / f"aligned-{args.words.replace(',', '_')}.jsonl"  # per word set: a shared cache emptied the other words
    if aligned_path.exists():
        aligned = [json.loads(l) for l in aligned_path.read_text().splitlines()]
        print(f"reusing {len(aligned)} aligned clips", flush=True)
    else:
        import glob as _glob
        cache = load_aligned([Path(p) for p in _glob.glob(str(Path(args.reuse_aligned).expanduser()))]) if args.reuse_aligned else {}
        aligned = align_all(rows, synth, args.workers, cache)
        aligned_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in aligned) + "\n")
    report, heads = {}, {}
    for word in words:
        spec = json.loads((Path(__file__).parent / "words" / f"{word}.json").read_text())
        addresses, heads[word] = heads_for(word, spec)
        report[word] = build_word(word, [r for r in aligned if r["word"] == word], synth, root, args.copies, addresses, heads[word])
        print(json.dumps(report[word]), flush=True)
    (root / "wordpack-report.json").write_text(json.dumps(report, indent=1))
    if args.train:
        # Per-word heads: the address list used to leak from the last word into every word.
        for word in words:
            for model in heads[word]:
                print(json.dumps(train_head(root, model, args.steps, words)), flush=True)


if __name__ == "__main__":
    main()
