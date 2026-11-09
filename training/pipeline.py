"""Drive livekit-wakeword stages on our clean data instead of its NC-licensed defaults.

Differences from `livekit-wakeword run`:
  - clips come from build_data.py (Piper voices with CC0/CC-BY training data), not `generate`;
  - the general-negative slot (upstream key ACAV100M_sample, CC BY-NC-SA 4.0 features) is fed
    from data/features/clean_general_negatives.npy (public-domain/CC-BY/CC0 speech and music).

Usage: .venv/bin/python pipeline.py <root> <model> <stage> [...]
  stages: config | backgrounds | augment (augment + extract) | extract | train | export
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

CONFIGS = {
    "stopp": {"target_phrases": ["stopp"], "steps": 40000},
    "hey_oqto": {"target_phrases": ["hey oqto"], "steps": 60000},
    "oqto": {"target_phrases": ["oqto"], "steps": 60000},
}


def config_path(root: Path, model: str) -> Path:
    return root / "configs" / f"{model}.yaml"


def write_config(root: Path, model: str) -> None:
    # Named recipes for the production heads; anything else (a word-pack head such as hey_elissa)
    # derives its own target phrase and defaults.
    key = model.split("-")[0]
    spec = CONFIGS.get(key) or {"target_phrases": [model.replace("_", " ")], "steps": 20000}
    cfg = {
        "model_name": model,
        "target_phrases": spec["target_phrases"],
        "n_background_samples": 2000,
        "n_background_samples_val": 500,
        "data_dir": str(root / "data"),
        "output_dir": str(root / "output"),
        "augmentation": {"clip_duration": 2.0, "batch_size": 16, "rounds": 2,
                         "background_paths": [str(root / "data/backgrounds")], "rir_paths": [str(root / "data/rirs")]},
        "model": {"model_type": "conv_attention", "model_size": "small"},
        "steps": spec["steps"],
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
        "max_negative_weight": 3000,
        "target_fp_per_hour": 0.2,
        "batch_n_per_class": json.loads(os.environ["WAKE_BATCH_JSON"]) if os.environ.get("WAKE_BATCH_JSON") else
                              {"positive": 50, "adversarial_negative": 50, "ACAV100M_sample": 1024, "background_noise": 50},
    }
    path = config_path(root, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(path)


def load(root: Path, model: str):
    from livekit.wakeword.config import WakeWordConfig
    return WakeWordConfig(**yaml.safe_load(config_path(root, model).read_text()))


def clean_negatives(root: Path) -> Path:
    import os
    path = Path(os.environ.get("WAKE_NEGATIVES", root / "data/features/clean_general_negatives.npy"))  # override: smoke tests only
    if not path.exists():
        raise SystemExit(f"missing {path}; run build_data.py negatives first")
    return path


SPLITS = [("positive_train", "positive_features_train.npy"), ("positive_test", "positive_features_test.npy"),
          ("negative_train", "negative_features_train.npy"), ("negative_test", "negative_features_test.npy"),
          ("background_train", "background_noise_features_train.npy"),
          ("background_test", "background_noise_features_test.npy")]


def extract(config) -> None:
    """Features for augmented 2 s clips through build_data.Featurizer (the browser engine's exact front end).

    Replaces livekit's run_extraction, whose per-clip 80 dB floor and grid differ slightly from
    the streaming runtime. Output: (N,16,96) float32, the last 16 embeddings of each clip.
    """
    import re

    import numpy as np
    import soundfile as sf

    from build_data import Featurizer

    feat = Featurizer()
    pattern = re.compile(r"^clip_\d{6}_r\d+\.wav$")
    out_dir = config.model_output_dir
    for split, name in SPLITS:
        clip_dir = out_dir / split
        wavs = sorted(p for p in clip_dir.glob("*.wav") if pattern.match(p.name)) if clip_dir.exists() else []
        rows = []
        for i, wav in enumerate(wavs):
            audio, rate = sf.read(str(wav), dtype="float32")
            assert rate == 16000, wav
            frames = feat.frames(audio if audio.ndim == 1 else audio[:, 0])
            if len(frames) < 16:
                frames = np.concatenate([np.zeros((16 - len(frames), 96), np.float16), frames])
            rows.append(frames[-16:].astype(np.float32))
            if (i + 1) % 5000 == 0:
                print(f"extract {split}: {i + 1}/{len(wavs)}", flush=True)
        data = np.stack(rows) if rows else np.zeros((0, 16, 96), np.float32)
        np.save(out_dir / name, data)
        print(f"{name}: {data.shape}", flush=True)


def float32_batches(loader):
    """Feature files may be float16 (the general-negative set is) while others are float32; a batch
    that mixes them reaches the conv half-precision and fails. Cast every batch once, here."""
    for features, labels in loader:
        if hasattr(features, "float"):
            features = features.float()
        yield features, labels


def train(root: Path, model: str) -> None:
    from livekit.wakeword.data.dataset import create_dataloader
    from livekit.wakeword.training import trainer as trainer_mod

    config = load(root, model)
    negatives = clean_negatives(root)

    def build_dataloader(self):
        out = self.config.model_output_dir
        files = {"positive": out / "positive_features_train.npy",
                 "adversarial_negative": out / "negative_features_train.npy",
                 "ACAV100M_sample": negatives}  # slot name is upstream's; content is our clean set
        bg = out / "background_noise_features_train.npy"
        if bg.exists():
            files["background_noise"] = bg
        loader = create_dataloader(data_files=files, n_per_class=self.config.batch_n_per_class,
                                   label_funcs={k: (lambda _: 1) if k == "positive" else (lambda _: 0) for k in files})
        return float32_batches(loader)

    trainer_mod.WakeWordTrainer._build_dataloader = build_dataloader
    upstream = (config.data_path / "features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    if upstream.exists():
        raise SystemExit(f"refusing to train: NC file present at {upstream}")
    import os
    if os.environ.get("WAKE_DEVICE", "cpu") == "cpu":  # small head; CPU keeps training off shared GPU capacity
        import torch
        torch.set_num_threads(int(os.environ.get("WAKE_THREADS", "8")))
        from livekit.wakeword import utils
        utils.get_device = lambda: torch.device("cpu")
        trainer_mod.get_device = utils.get_device
    elif os.environ.get("WAKE_DEVICE") == "cuda":  # stay inside the dpty lease: WAKE_GPU_GB (default 1.4 per run)
        import torch
        gb = float(os.environ.get("WAKE_GPU_GB", "1.4"))
        torch.cuda.set_per_process_memory_fraction(gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from livekit.wakeword.training.trainer import WakeWordTrainer, run_train
    init = os.environ.get("WAKE_INIT")  # warm start: continue from an existing head instead of scratch
    if init:
        import torch
        trainer = WakeWordTrainer(config)
        state = torch.load(init, map_location="cpu", weights_only=True)
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        print(f"warm start from {init}: missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
        trainer.train()
        trainer.save(config.model_output_dir / f"{config.model_name}.pt")
    else:
        run_train(config)


def main():
    root, model, stage = Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3]
    if stage == "config":
        write_config(root, model)
    elif stage == "backgrounds":
        from livekit.wakeword.data.generate import _generate_background_clips
        config = load(root, model)
        _generate_background_clips(config, "background_train", config.n_background_samples)
        _generate_background_clips(config, "background_test", config.n_background_samples_val)
    elif stage == "augment":
        from livekit.wakeword.data.augment import run_augment
        config = load(root, model)
        run_augment(config)
        extract(config)
    elif stage == "extract":
        extract(load(root, model))
    elif stage == "train":
        train(root, model)
    elif stage == "export":
        from livekit.wakeword.export.onnx import run_export
        print(run_export(load(root, model)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
