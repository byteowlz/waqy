"""Speak a word-pack inventory with spqx designed voices (Qwen3-TTS VoiceDesign, Apache-2.0).

    python synth_spqx.py <texts.json> <voice-bank-dir> <out-dir> [--workers 3]

The voice bank is a folder with voices.json and one reference WAV per voice; each clip is
spoken by ICL from one of those references, so every voice is fictional and described in text.
`--bank-lang de|en|es` forces a reference language regardless of the text language, which is how
the accent experiments are generated (e.g. English sentences with German references). Runs
wherever spqx and a GPU live; this stage does not need Whisper.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path(os.path.expanduser(os.environ.get("SPQX_DESIGN_DIR", "~/.local/share/spqx/native-voice-design-217edd91")))
BIN = D / "source/target/release/spqx"
ENV = dict(os.environ, LD_LIBRARY_PATH=str(D / "runtime"), CUDA_VISIBLE_DEVICES=os.environ.get("SPQX_GPU", "0"), OMP_NUM_THREADS="2")


def run(args, timeout=900):
    r = subprocess.run(args, env=ENV, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def base_config(out: Path) -> Path:
    cfg = out / "base.toml"
    cfg.write_text(f'''[model]
path = "{D}/models/base.gguf"
[engine]
backend = "ggml"
device = "cuda:{os.environ.get('SPQX_GPU', '0')}"
library = "{D}/runtime/libqwen.so"
codec = "{D}/models/qwen-tokenizer-12hz-F32.gguf"
''')
    return cfg


def worker(items, bank, wavdir, cfg, bank_lang=None):
    done = []
    for t in items:
        target = wavdir / f"{t['id']}.wav"
        if target.exists():
            done.append(t)
            continue
        want = bank_lang or t["lang"]  # cross-lingual: force the reference language (accent experiments)
        pool = [v for v in bank if v["lang"] == want] or [v for v in bank if v["lang"] == t["lang"]]
        if not pool:
            continue
        voice = pool[hash(t["id"]) % len(pool)]
        code, _, err = run([str(BIN), "--config", str(cfg), "say", t["text"], "--ref-audio", voice["ref"],
                            "--ref-text", voice["ref_text"], "--out", str(target), "--no-play", "--quiet"])
        if code != 0 or not target.exists():
            print(json.dumps({"fail": t["id"], "err": err[-200:]}), flush=True)
            continue
        done.append(t | {"voice": voice["voice"]})
    return done


if __name__ == "__main__":
    texts = json.loads(Path(sys.argv[1]).read_text())
    bank = json.loads((Path(sys.argv[2]) / "voices.json").read_text())
    out = Path(sys.argv[3]).expanduser()
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    bank_lang = sys.argv[sys.argv.index("--bank-lang") + 1] if "--bank-lang" in sys.argv else None
    wavdir = out / "wav"
    wavdir.mkdir(parents=True, exist_ok=True)
    cfg = base_config(out)
    started = time.time()
    chunks = [texts[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda c: worker(c, bank, wavdir, cfg, bank_lang), chunks))
    rows = [t for group in results for t in group]
    (out / "clips.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(json.dumps({"clips": len(rows), "of": len(texts), "bank_lang": bank_lang, "minutes": round((time.time() - started) / 60, 1)}))
