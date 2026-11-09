"""Export a trained head to ONNX from an explicit checkpoint and to an explicit path.

`run_export(config)` derives both from `config.model_output_dir`, which is correct in principle but
was silently exporting a different graph whenever training and export disagreed about the directory
(measured: 12 heads exported as identical 156 KB graphs predicting a constant 0.002). Naming the
checkpoint and the destination removes the ambiguity.

Usage: export_head.py <config.yaml> <checkpoint.pt> <out.onnx>
"""
import sys
from pathlib import Path

from livekit.wakeword.config import load_config
from livekit.wakeword.export.onnx import export_onnx

config_path, checkpoint, out_path = (Path(p) for p in sys.argv[1:4])
if not checkpoint.exists():
    raise SystemExit(f'no checkpoint at {checkpoint}')
config = load_config(config_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
print(export_onnx(config, checkpoint, out_path))
