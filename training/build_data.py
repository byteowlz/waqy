"""Build clean-licence training data for the Oqto wake models (livekit-wakeword layout).

Every input is open: Piper voices trained on CC0/CC-BY data (pinned rhasspy/piper-voices
revision), MUSAN noise/speech (public domain) and music (CC, no NC), FLEURS (CC-BY-4.0),
Thorsten (CC0), procedural nonspeech, simulated room impulse responses. No ACAV100M,
DiPCo or other NC/unknown-licence material is used.

Subcommands (each resumable, each writes a provenance JSON next to its output):
  synth <model>      positives + adversarial negatives as 16 kHz clip_%06d.wav, speaker-disjoint splits
  backgrounds        MUSAN noise/free-sound (public domain) + nonspeech-1 train clips -> data/backgrounds
  rirs               simulated shoebox room impulse responses -> data/rirs
  negatives          streamed general negatives -> data/features/clean_general_negatives.npy (N,16,96)
  validation         held-out false-positive set -> data/features/validation_set_features.npy

Usage: .venv/bin/python build_data.py <root> <subcommand> [args]
  root = private work dir, e.g. ~/.local/share/foxline/wake-train-1
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

HOME = Path.home()
FOX = HOME / ".local/share/foxline"
MUSAN_REPO, MUSAN_REV = "FluidInference/musan", "3edcfdf89b56dbe6a395ff29f9c29489e03d1321"
RATE = 16000

# Spoken forms. German espeak reads "Hey" as /haI/, so German keeps several spellings.
PHRASES = {
    "hey_oqto": {
        "de": ["He Okto", "Hej Okto", "Hey Okto", "He, Okto!", "Hey, Okto?", "He Okto?"],
        "en": ["Hey Octo", "Hey, Octo!", "Hey Octo?", "Hey, Octo."],
    },
    "oqto": {
        "de": ["Okto", "Okto!", "Okto?", "Okto."],
        "en": ["Octo", "Octo!", "Octo?", "Octo."],
    },
}
ADVERSARIAL = {
    "de": ["Oktober", "Im Oktober", "Oktopus", "Otto", "He Otto", "Hey Otto", "Lotto", "Motto",
           "Doktor", "Oktave", "Okay", "Hey", "He", "Ok", "Hallo", "Achtung", "Echo", "Oktoberfest",
           "Hey Oktopus", "Hallo Otto", "Okta", "Obst", "Ecke", "Hey Doktor"],
    "en": ["October", "octopus", "Otto", "hey Otto", "auto", "hey auto", "photo", "lotto", "motto",
           "doctor", "octave", "okay", "hey", "hello", "act two", "hey Oscar", "echo", "hey octopus",
           "hey doctor", "oak tree", "actor", "hey photo"],
}
# The "hey" model must not fire on the bare word; the bare model may fire on "hey oqto".
EXTRA_NEGATIVE = {"hey_oqto": {"de": ["Okto", "Okto!"], "en": ["Octo", "Octo!"]}, "oqto": {"de": [], "en": []}}

# Voices: (piper voice, speaker selector, weight). German text is phonemized by espeak "de" (via the
# Thorsten voice) and may be spoken by English multi-speaker models ("cross-lingual"): their phoneme
# sets are identical, so 1,900+ voices say /heː ˈɔktoː/. MLS German and Kerstin were dropped after
# ASR checks showed unintelligible output. Held-out speakers are chosen by speaker NAME, so the two
# LibriTTS models (same corpus) never put one person on both sides.
NATIVE_DE = [("de_DE-thorsten-medium", None, 1.0), ("de_DE-thorsten-high", None, 1.0), ("de_DE-thorsten_emotional-medium", "all", 1.5)]
MULTI = [("en_US-libritts_r-medium", 2.0), ("en_US-libritts-high", 2.0), ("en_GB-vctk-medium", 1.0)]


def held_out_speaker(name: str) -> bool:
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % 7 == 0


def voice_plan(split: str, lang: str):
    """[(voice, 'held-out'|'train'|None, weight)]; None = single-speaker or all speakers of that voice."""
    side = "held-out" if split == "test" else "train"
    multi = [(name, side, w) for name, w in MULTI]
    if lang == "en":
        return multi
    if split == "test":
        return multi  # no unseen native German voice exists; owner takes are the native German test
    native_weight = sum(w for *_, w in NATIVE_DE)
    scale = 0.35 / native_weight * sum(w for *_, w in multi) / 0.65
    return [(n, sel, w * scale) for n, sel, w in NATIVE_DE] + multi


LENGTH = [0.8, 0.9, 1.0, 1.0, 1.15, 1.3]
NOISE = [0.5, 0.667, 0.8]
NOISE_W = [0.6, 0.8, 1.0]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def speech_span(pcm: np.ndarray, frame: int = 320) -> tuple[int, int] | None:
    """First/last sample of frames above 5% of the peak frame energy."""
    if len(pcm) < frame:
        return None
    energy = np.sqrt(np.mean(pcm[: len(pcm) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    on = np.where(energy > max(energy.max() * 0.05, 1e-4))[0]
    return (int(on[0] * frame), int((on[-1] + 1) * frame)) if len(on) else None


def trim(pcm: np.ndarray, pad: float = 0.15) -> np.ndarray:
    span = speech_span(pcm)
    if span is None:
        return pcm
    p = int(pad * RATE)
    return pcm[max(0, span[0] - p): span[1] + p]


def max_speech_seconds(text: str) -> float:
    """Generous duration ceiling for a correctly spoken short phrase (rejects TTS babble)."""
    words = len(text.replace(",", " ").split())
    return 0.75 + 0.55 * words


def to16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    pcm = pcm.astype(np.float32)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return soxr.resample(pcm, rate, RATE) if rate != RATE else pcm


# ---------------------------------------------------------------- synth
def synth(root: Path, model: str, n_train: int, n_test: int, seed: int = 20261026) -> None:
    from piper import PiperVoice, SynthesisConfig

    phrases_key = model.split("-")[0]  # e.g. "hey_oqto-smoke" uses the hey_oqto phrase set
    rng = random.Random(seed)
    out = root / "output" / model
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

    german = voice("de_DE-thorsten-medium")  # espeak "de" phonemizer for cross-lingual German
    out.mkdir(parents=True, exist_ok=True)
    manifest = (out / "synth-manifest.jsonl").open("a")
    rejected = {True: 0, False: 0}
    for split, count in (("train", n_train), ("test", n_test)):
        for kind, phrase_sets in (("positive", PHRASES[phrases_key]),
                                  ("negative", {lang: ADVERSARIAL[lang] + EXTRA_NEGATIVE[phrases_key][lang] for lang in ("de", "en")})):
            directory = out / f"{kind}_{split}"
            directory.mkdir(parents=True, exist_ok=True)
            existing = len(list(directory.glob("clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav")))
            started = time.time()
            for i in range(existing, count):
                lang = "de" if i % 2 == 0 else "en"
                plan = voice_plan(split, lang)
                for attempt in range(40):
                    name, selector, _ = rng.choices(plan, weights=[w for *_, w in plan])[0]
                    v = voice(name)
                    speaker = rng.choice(speakers_by[name][selector]) if selector else None
                    text = rng.choice(phrase_sets[lang])
                    cfg = SynthesisConfig(speaker_id=speaker, length_scale=rng.choice(LENGTH),
                                          noise_scale=rng.choice(NOISE), noise_w_scale=rng.choice(NOISE_W))
                    cross = lang == "de" and not name.startswith("de_")
                    if cross:
                        ids = [x for sentence in german.phonemize(text) for x in v.phonemes_to_ids(sentence)]
                        pcm = v.phoneme_ids_to_audio(ids, cfg)
                    else:
                        pcm = np.concatenate([c.audio_float_array for c in v.synthesize(text, syn_config=cfg)])
                    pcm = trim(to16k(np.asarray(pcm, dtype=np.float32), v.config.sample_rate))
                    span = speech_span(pcm)
                    ok = span is not None and (span[1] - span[0]) / RATE <= max_speech_seconds(text) and len(pcm) / RATE <= 1.95
                    rejected[ok] += 1
                    if ok:
                        break
                else:
                    raise RuntimeError(f"no acceptable clip for {kind}_{split} #{i} after 40 attempts")
                path = directory / f"clip_{i:06d}.wav"
                sf.write(str(path), pcm, RATE, subtype="PCM_16")
                manifest.write(json.dumps({"split": split, "kind": kind, "file": path.name, "voice": name, "speaker": speaker,
                                           "lang": lang, "cross_lingual": cross, "text": text, "seconds": round(len(pcm) / RATE, 3),
                                           "attempts": attempt + 1}, ensure_ascii=False) + "\n")
                manifest.flush()
                if (i + 1) % 1000 == 0:
                    print(f"{model} {kind}_{split} {i + 1}/{count} ({time.time() - started:.0f}s)", flush=True)
    manifest.close()
    write_json(out / "synth-provenance.json", {
        "model": model, "seed": seed, "n_train": n_train, "n_test": n_test, "phrases": PHRASES[phrases_key],
        "adversarial": ADVERSARIAL, "extra_negative": EXTRA_NEGATIVE[phrases_key],
        "voices": {"native_de": NATIVE_DE, "multi_speaker": MULTI, "german_share_native": 0.35,
                   "held_out_rule": "sha256(speaker name) % 7 == 0, same rule for every multi-speaker voice"},
        "piper_voices_revision": (root / "voices/REVISION").read_text().strip(),
        "voice_licences": "Thorsten, Thorsten-emotional CC0; LibriTTS-R, LibriTTS, VCTK CC-BY-4.0 (MODEL_CARD files in voices/)",
        "generator": "piper-tts 1.3.0 (espeak-ng phonemizer); generation tool only, not shipped",
        "quality_gate": "silence-trimmed to 0.15 s padding; rejected if speech span > 0.75 + 0.55*words s or clip > 1.95 s; up to 40 retries; then ASR gate",
        "accepted_this_run": rejected[True], "rejected_this_run": rejected[False],
    })


# ---------------------------------------------------------------- backgrounds + rirs
def musan_files(prefix: str) -> list[str]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(MUSAN_REPO, repo_type="dataset", revision=MUSAN_REV)
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".wav"))


def fetch(path: str, tmp: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(MUSAN_REPO, path, repo_type="dataset", revision=MUSAN_REV, local_dir=str(tmp)))


def backgrounds(root: Path) -> None:
    out = root / "data/backgrounds"
    out.mkdir(parents=True, exist_ok=True)
    files = musan_files("noise/free-sound/")
    written = 0
    for path in files:
        target = out / ("musan-" + Path(path).name)
        if target.exists():
            continue
        local = fetch(path, root / "tmp")
        pcm, rate = sf.read(str(local), dtype="float32")
        sf.write(str(target), to16k(pcm, rate), RATE, subtype="PCM_16")
        local.unlink()
        written += 1
    nonspeech = [json.loads(l) for l in (FOX / "voice-corpus/nonspeech-1/manifest.jsonl").read_text().splitlines()]
    for r in nonspeech:
        if r["role"] == "train":
            pcm, rate = sf.read(r["wav"], dtype="float32")
            sf.write(str(out / f"nonspeech-{r['id']}.wav"), to16k(pcm, rate), RATE, subtype="PCM_16")
    write_json(root / "data/backgrounds-provenance.json", {
        "musan": {"repo": MUSAN_REPO, "revision": MUSAN_REV, "subset": "noise/free-sound", "files": len(files),
                  "licence": "Public Domain (noise/free-sound/LICENSE)"},
        "nonspeech-1": "procedural silence/noise, train role only", "written_this_run": written})


def rirs(root: Path, n: int = 300, seed: int = 7) -> None:
    import pyroomacoustics as pra
    out = root / "data/rirs"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        target = out / f"sim-{i:04d}.wav"
        if target.exists():
            continue
        dims = rng.uniform([3, 3, 2.4], [12, 10, 4.0])
        rt60 = rng.uniform(0.15, 0.9)
        try:
            absorption, max_order = pra.inverse_sabine(rt60, dims)
        except ValueError:
            continue  # this room cannot reach the drawn RT60; skip it rather than force a value
        room = pra.ShoeBox(dims, fs=RATE, materials=pra.Material(absorption), max_order=min(max_order, 40))
        room.add_source(rng.uniform([0.5, 0.5, 1.0], dims - [0.5, 0.5, 0.5]))
        room.add_microphone(rng.uniform([0.5, 0.5, 0.8], dims - [0.5, 0.5, 0.5]))
        room.compute_rir()
        rir = np.asarray(room.rir[0][0], dtype=np.float32)
        rir = rir / (np.max(np.abs(rir)) + 1e-9)
        sf.write(str(target), rir[: RATE], RATE, subtype="FLOAT")
    write_json(root / "data/rirs-provenance.json", {"generator": "pyroomacoustics shoebox image method (MIT)", "count": n,
                                                     "seed": seed, "rt60_range_s": [0.15, 0.9]})


# ---------------------------------------------------------------- features
FRONTEND = Path(__file__).parent / "frontend"
FLOOR_FRAMES = 200  # the 80 dB floor is relative to the loudest frame in the trailing 2 s


def trailing_floor(db: np.ndarray) -> np.ndarray:
    """Apply max(x, trailing-2s peak - 80) per frame to raw (T,32) log-mel, as the runtimes do."""
    peak = db.max(axis=1)
    view = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(FLOOR_FRAMES - 1, -np.inf), peak]), FLOOR_FRAMES)
    return np.maximum(db, view.max(axis=1)[:, None] - 80.0)


def ort_options():
    """Bounded intra-op threads so concurrent data jobs do not oversubscribe the CPU (WAKE_ORT_THREADS, default 4)."""
    import os
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get("WAKE_ORT_THREADS", "4"))
    options.inter_op_num_threads = 1
    return options


class StreamingMel:
    """Raw log-mel (formula-built graph) + trailing floor + x/10+2; same values the browser engine sees."""

    def __init__(self):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(FRONTEND / "melspectrogram_raw.onnx"), sess_options=ort_options(),
                                            providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = np.atleast_2d(audio).astype(np.float32)
        out = [trailing_floor(self.session.run(None, {"input": a[None, :]})[0][0, 0]) / 10.0 + 2.0 for a in audio]
        return np.stack(out)


class Featurizer:
    def __init__(self):
        from livekit.wakeword.models.feature_extractor import SpeechEmbedding
        self.mel = StreamingMel()
        self.emb = SpeechEmbedding(onnx_path=str(FRONTEND / "embedding_model.onnx"), sess_options=ort_options())

    def frames(self, pcm: np.ndarray) -> np.ndarray:
        """Contiguous (n,96) embedding frames, one per 80 ms, on exactly the browser engine's grid.

        The engine feeds 1280-sample frames with 480 samples of context, starting from zeros, and
        takes one 76-frame embedding window ending at the last mel frame of every 80 ms step.
        Prepending 480 zeros and dropping the first 4 mel frames puts stride-8 windows on that grid.
        Long audio is processed in 60 s pieces (1 s overlap), so seams are approximate.
        """
        out = []
        chunk = RATE * 60
        for start in range(0, len(pcm), chunk):
            piece = pcm[start:start + chunk + RATE]
            if len(piece) < RATE * 2:
                continue
            mel = self.mel(np.concatenate([np.zeros(480, np.float32), piece.astype(np.float32)])[np.newaxis, :])[:, 4:]
            emb = self.emb.extract_embeddings(mel)[0]
            out.append(emb[: 60 * 1000 // 80])
        return np.concatenate(out).astype(np.float16) if out else np.zeros((0, 96), np.float16)

    def windows(self, pcm: np.ndarray) -> np.ndarray:
        """Non-overlapping (n,16,96) feature windows (1.28 s each) of long audio."""
        f = self.frames(pcm)
        return f[: len(f) // 16 * 16].reshape(-1, 16, 96)


def general_sources(held_out: bool):
    """Yield (name, licence, pcm) for general non-wake audio; held_out picks the validation side."""
    # MUSAN speech/music and VoxPopuli arrive through `stream` (the HF MUSAN mirror only carries noise).
    fleurs = FOX / "voice-corpus/fleurs-de-1"
    if held_out:
        for r in json.loads((fleurs / "manifests/cased-eval.json").read_text())["rows"]:
            yield f"fleurs-de-test:{Path(r['wav']).name}", "CC-BY-4.0", Path(r["wav"])
        spike = FOX / "eg2-speech-spike-1/data/en_us_test.tar.gz"
        if spike.exists():
            yield "fleurs-en-test", "CC-BY-4.0", spike
        for line in (FOX / "voice-corpus/nonspeech-1/manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["role"] == "test":
                yield f"nonspeech:{r['id']}", "procedural", Path(r["wav"])
    else:
        yield "fleurs-de-train", "CC-BY-4.0", fleurs / "inputs/train.tar.gz"
        spike = FOX / "eg2-speech-spike-1/data/en_us_train.tar.gz"
        if spike.exists():
            yield "fleurs-en-train", "CC-BY-4.0", spike
        for line in (FOX / "voice-corpus/intake-2/public/proposals.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["role"] == "replay":
                yield f"thorsten:{r['id']}", "CC0", Path(r["wav"])


TRUNCATED: list[dict] = []


def pcm_items(item, root: Path):
    """Expand one source into (pcm) arrays: a MUSAN path, a wav path or a FLEURS tar archive."""
    if isinstance(item, str):
        local = fetch(item, root / "tmp")
        pcm, rate = sf.read(str(local), dtype="float32")
        local.unlink()
        yield to16k(pcm, rate)
    elif str(item).endswith(".tar.gz"):
        try:
            with tarfile.open(item, "r|gz") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(".wav"):
                        pcm, rate = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                        yield to16k(pcm, rate)
        except (tarfile.ReadError, EOFError) as error:  # truncated archive: keep the clips read so far
            TRUNCATED.append({"archive": str(item), "error": str(error)})
            print(f"warning: {item} truncated ({error}); using the clips before the cut", flush=True)
    else:
        pcm, rate = sf.read(str(item), dtype="float32")
        yield to16k(pcm, rate)


def build_features(root: Path, held_out: bool, out_name: str, max_hours: float) -> None:
    out = root / "data/features"
    out.mkdir(parents=True, exist_ok=True)
    part_dir = root / "tmp" / (out_name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in part_dir.glob("*.npy")}
    feat = Featurizer()
    seconds, sources, started = 0.0, [], time.time()
    for name, licence, item in general_sources(held_out):
        key = hashlib.sha256(name.encode()).hexdigest()[:16]
        if key in done:
            seconds += float(np.load(part_dir / f"{key}.npy", mmap_mode="r").shape[0]) * (0.08 if held_out else 1.28)
            meta = part_dir / f"{key}.meta.json"
            if meta.exists():
                TRUNCATED.extend(json.loads(meta.read_text()).get("truncated", []))
            sources.append({"source": name, "licence": licence, "cached": True})
            continue
        before = len(TRUNCATED)
        if seconds / 3600 >= max_hours:
            break
        if held_out:  # contiguous frames, so the evaluator can stream them; one part per source
            arrays = [feat.frames(pcm) for pcm in pcm_items(item, root)]
            arrays = [a for a in arrays if len(a) >= 16]
            data = np.concatenate(arrays) if arrays else np.zeros((0, 96), np.float16)
            np.save(part_dir / f"{key}.npy", data)
            np.save(part_dir / f"{key}.lengths.npy", np.array([len(a) for a in arrays], np.int64))
            seconds += data.shape[0] * 0.08
        else:
            arrays = [feat.windows(pcm) for pcm in pcm_items(item, root)]
            arrays = [a for a in arrays if len(a)]
            data = np.concatenate(arrays) if arrays else np.zeros((0, 16, 96), np.float16)
            np.save(part_dir / f"{key}.npy", data)
            seconds += data.shape[0] * 1.28
        if len(TRUNCATED) > before:  # remember per part, so cached reuse keeps the note
            write_json(part_dir / f"{key}.meta.json", {"truncated": TRUNCATED[before:]})
        sources.append({"source": name, "licence": licence, "windows": int(data.shape[0])})
        if len(sources) % 50 == 0:
            print(f"{out_name}: {len(sources)} sources, {seconds / 3600:.2f} h, {time.time() - started:.0f}s", flush=True)
    streamed = part_dir / "streamed-sources.jsonl"
    if streamed.exists():
        for line in streamed.read_text().splitlines():
            row = json.loads(line)
            sources.append({"source": row["key"], "licence": "see musan/voxpopuli provenance", "rows": row["rows"], "streamed": True})
    keys = sorted(p.name[:-4] for p in part_dir.glob("*.npy") if not p.name.endswith(".lengths.npy"))
    parts = [np.load(part_dir / f"{k}.npy") for k in keys]
    shape = (0, 96) if held_out else (0, 16, 96)
    total = np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.zeros(shape, np.float16)
    np.save(out / out_name, total)
    if held_out:  # segment lengths in frames, in file order, so streams never cross a recording boundary
        per_part = [np.load(part_dir / f"{k}.lengths.npy") for k in keys]
        lengths = np.concatenate(per_part)
        assert lengths.sum() == total.shape[0]
        np.save(out / (Path(out_name).stem + ".segments.npy"), lengths)
        names = {hashlib.sha256(src["source"].encode()).hexdigest()[:16]: src["source"] for src in sources if not src.get("streamed")}
        if streamed.exists():
            names.update({json.loads(l)["part"]: json.loads(l)["key"] for l in streamed.read_text().splitlines()})
        write_json(out / (Path(out_name).stem + ".segment-sources.json"),
                   [{"source": names.get(k, "unknown:" + k), "segments": int(len(n))} for k, n in zip(keys, per_part)])
    hours = total.shape[0] * (0.08 if held_out else 1.28) / 3600  # includes streamed parts
    write_json(out / (Path(out_name).stem + ".provenance.json"), {
        "held_out": held_out, "rows": int(total.shape[0]), "hours": round(hours, 3),
        "format": ("(N,96) float16 contiguous 80 ms embedding frames (segments in .segments.npy)" if held_out else
                   "(N,16,96) float16, non-overlapping 1.28 s windows") + " from livekit-wakeword frozen mel+embedding",
        "sources": sources, "no_nc_data": True, "truncated_archives": TRUNCATED,
        "split_rule": "MUSAN: file index % 10 == 0 is held out; FLEURS test/nonspeech test held out; FLEURS train and Thorsten replay are training side"})
    print(f"wrote {out / out_name} {total.shape} {hours:.2f} h", flush=True)


# ---------------------------------------------------------------- streamed large sources
MUSAN_URL = "https://openslr.elda.org/resources/17/musan.tar.gz"  # OpenSLR 17, 11,086,114,085 bytes
VOXPOPULI_REPO, VOXPOPULI_REV = "facebook/voxpopuli", "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
VOXPOPULI_SHARDS = {"train": ["de/train-00000-of-00016.parquet", "de/train-00001-of-00016.parquet"],
                    "held_out": ["de/test-00000-of-00001.parquet"]}


def save_part(root: Path, held_out: bool, key: str, frames: list[np.ndarray]) -> int:
    """Store one source's features in the train or held-out part directory (format as build_features)."""
    name = "validation_set_features.npy" if held_out else "clean_general_negatives.npy"
    part_dir = root / "tmp" / (name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    if held_out:
        frames = [f for f in frames if len(f) >= 16]
        data = np.concatenate(frames) if frames else np.zeros((0, 96), np.float16)
        np.save(part_dir / f"{digest}.lengths.npy", np.array([len(f) for f in frames], np.int64))
    else:
        windows = [f[: len(f) // 16 * 16].reshape(-1, 16, 96) for f in frames if len(f) >= 16]
        data = np.concatenate(windows) if windows else np.zeros((0, 16, 96), np.float16)
    np.save(part_dir / f"{digest}.npy", data)
    with (part_dir / "streamed-sources.jsonl").open("a") as log:
        log.write(json.dumps({"key": key, "part": digest, "rows": int(data.shape[0])}) + "\n")
    return int(data.shape[0])


def done_keys(root: Path) -> set[str]:
    keys = set()
    for name in ("validation_set_features.npy", "clean_general_negatives.npy"):
        log = root / "tmp" / (name + ".parts") / "streamed-sources.jsonl"
        if log.exists():
            keys |= {json.loads(l)["key"] for l in log.read_text().splitlines()}
    return keys


def stream_musan(root: Path) -> None:
    """Stream OpenSLR 17 once; speech/ and music/ files: index % 10 == 0 held out, rest training."""
    import urllib.request
    feat, done, counters = Featurizer(), done_keys(root), {}
    with urllib.request.urlopen(MUSAN_URL) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            name = member.name
            if not (member.isfile() and name.endswith(".wav")) or "/noise/" in name:
                continue
            group = "/".join(name.split("/")[1:3])  # e.g. speech/librivox, music/jamendo
            index = counters.get(group, 0)
            counters[group] = index + 1
            key = f"musan:{name}"
            if key in done:
                continue
            pcm, rate = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
            save_part(root, index % 10 == 0, key, [feat.frames(to16k(pcm, rate))])
    write_json(root / "data/musan-stream-provenance.json", {
        "url": MUSAN_URL, "bytes": 11086114085, "groups": counters,
        "licence": "MUSAN: speech/librivox and speech/us-gov Public Domain; music Creative Commons without NC (per-file LICENSE in archive)",
        "split_rule": "per group, file index % 10 == 0 held out (tar order)"})


def stream_voxpopuli(root: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    feat, done = Featurizer(), done_keys(root)
    for side, shards in VOXPOPULI_SHARDS.items():
        for shard in shards:
            local = Path(hf_hub_download(VOXPOPULI_REPO, shard, repo_type="dataset", revision=VOXPOPULI_REV,
                                         local_dir=str(root / "tmp/voxpopuli")))
            table = pq.ParquetFile(str(local))
            for g in range(table.num_row_groups):
                key = f"voxpopuli:{shard}:rg{g}"
                if key in done:
                    continue
                rows = table.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
                frames = []
                for row in rows:
                    pcm, rate = sf.read(io.BytesIO(row["bytes"]), dtype="float32")
                    frames.append(feat.frames(to16k(pcm, rate)) if len(pcm) >= rate * 2 else np.zeros((0, 96), np.float16))
                save_part(root, side == "held_out", key, frames)
                print(f"{key}: {len(rows)} utterances", flush=True)
            local.unlink()  # raw shard is not kept; features + provenance are
    write_json(root / "data/voxpopuli-provenance.json", {
        "repo": VOXPOPULI_REPO, "revision": VOXPOPULI_REV, "shards": VOXPOPULI_SHARDS, "licence": "CC0-1.0 (dataset card); European Parliament legal notice for raw data",
        "note": "utterances shorter than 2 s skipped"})


# ---------------------------------------------------------------- ASR gate
# whisper-server -ng -t 3 -m ggml-small.bin instances; WAKE_WHISPER_PORTS="8094,8095" selects others
WHISPER_SERVERS = [f"http://127.0.0.1:{port}/inference" for port in
                   __import__("os").environ.get("WAKE_WHISPER_PORTS", "8090,8091,8092,8093").split(",")]
WHISPER_MODEL = HOME / "models/Whisper/ggml-small.bin"
WAKE_HEARD = ("okto", "octo", "akto", "acto", "oktö")
NOT_WAKE = ("oktor", "octor", "oktober", "october", "octop", "oktop", "octav", "oktav")


WAKE_PATTERN = re.compile(r"[aeiouyäö][hr]?(?:ck|c|k)t[aeiouyäö]")


def heard_wake(text: str) -> bool:
    """Heard as the wake word: a vowel + (c)k + t + vowel cluster (Okto, Octo, Akta, Ockter, Arcto) in a
    word that is not a known confusable (Oktober, Doktor, Octopus, Oktave)."""
    words = re.findall(r"[\wäöüß]+", text.lower())
    return any(WAKE_PATTERN.search(w) for w in words if not any(bad in w for bad in NOT_WAKE))


def transcribe(server: str, path: Path, lang: str) -> str:
    import urllib.request
    boundary = "----wakegate" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    parts = []
    for key, value in (("language", lang), ("response_format", "text"), ("temperature", "0")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(server, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return " ".join(response.read().decode().split())


def asr_gate(root: Path, model: str, workers: int = 8) -> None:
    """Whisper-check positives: keep only clips heard as the wake word (see heard_wake)."""
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    out = root / "output" / model
    lock = (out / "asr-gate.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"asr gate {model}: another gate process holds the lock; leaving it to finish", flush=True)
        return
    rows = [json.loads(l) for l in (out / "synth-manifest.jsonl").read_text().splitlines()]
    gate_file = out / "asr-gate.jsonl"
    seen = {json.loads(l)["file_key"] for l in gate_file.read_text().splitlines()} if gate_file.exists() else set()
    # Positives only: several adversarial negatives are wake-like on purpose (bare "Okto" for the hey model).
    todo = [r for r in rows if r["kind"] == "positive" and f"{r['kind']}_{r['split']}/{r['file']}" not in seen]

    def check(item):
        index, r = item
        key = f"{r['kind']}_{r['split']}/{r['file']}"
        if not (out / key).exists():
            return None
        heard = transcribe(WHISPER_SERVERS[index % len(WHISPER_SERVERS)], out / key, r["lang"])
        wake = heard_wake(heard)
        return {"file_key": key, "kind": r["kind"], "lang": r["lang"], "text": r["text"], "heard": heard,
                "accepted": wake if r["kind"] == "positive" else not wake}

    rejected = out / "rejected"
    started = time.time()
    with ThreadPoolExecutor(workers) as pool, gate_file.open("a") as log:
        for i, result in enumerate(pool.map(check, enumerate(todo))):
            if result is None:
                continue
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            if not result["accepted"]:
                target = rejected / result["file_key"]
                target.parent.mkdir(parents=True, exist_ok=True)
                (out / result["file_key"]).rename(target)
            if (i + 1) % 1000 == 0:
                print(f"asr gate {model}: {i + 1}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)
    results = [json.loads(l) for l in gate_file.read_text().splitlines()]
    summary = {}
    for kind in ("positive",):
        for lang in ("de", "en"):
            sel = [r for r in results if r["kind"] == kind and r["lang"] == lang]
            summary[f"{kind}_{lang}"] = {"checked": len(sel), "accepted": sum(r["accepted"] for r in sel)}
    write_json(out / "asr-gate-summary.json", {
        "counts": summary, "asr": f"whisper.cpp server, {WHISPER_MODEL.name}, CPU, language = clip language, temperature 0",
        "rule": f"positive accepted if a heard word matches {WAKE_PATTERN.pattern} and contains none of {NOT_WAKE}; negative accepted otherwise",
        "scope": "positives only; negatives are not gated (some are wake-like by design)",
        "stability": "full 30 s context; rerun of 96 clips reproduced 94/96 decisions. Reduced --audio-ctx 128/256/512 agreed on only 76-80/96, so it is not used",
        "note": "Whisper is biased toward real words (Oktober/Otto) on 2-syllable synthetic phrases, so the gate also rejects some correct clips"})


def main():
    root = Path(sys.argv[1]).expanduser()
    command = sys.argv[2]
    if command == "synth":
        synth(root, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif command == "stream":
        {"musan": stream_musan, "voxpopuli": stream_voxpopuli}[sys.argv[3]](root)
    elif command == "asrgate":
        asr_gate(root, sys.argv[3])
    elif command == "backgrounds":
        backgrounds(root)
    elif command == "rirs":
        rirs(root)
    elif command == "negatives":
        build_features(root, False, "clean_general_negatives.npy", float(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif command == "validation":
        build_features(root, True, "validation_set_features.npy", float(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
