"""Check that the shipped embedding_model.onnx carries Google's speech_embedding/1 weights.

For every ONNX Conv (BN folded by tf2onnx) find the TF conv kernel of the same shape whose
BatchNorm-folded weights and bias reproduce it. Prints max relative error per layer.
Usage: python -I compare_embedding.py <variables prefix> <embedding_model.onnx>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).parent))
from read_tf_bundle import read_bundle  # noqa: E402


def main():
    tf = read_bundle(Path(sys.argv[1]))
    model = onnx.load(sys.argv[2])
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    kernels = {k: v for k, v in tf.items() if k.endswith("weights") and v.ndim == 4}
    print(f"TF conv kernels: {len(kernels)}; ONNX Conv nodes: {sum(n.op_type == 'Conv' for n in model.graph.node)}")
    for k, v in kernels.items():
        print("  tf", k, v.shape)
    bns = {}
    for k in tf:
        if k.endswith("/gamma"):
            base = k[: -len("/gamma")]
            bns[base] = tuple(tf[f"{base}/{p}"] for p in ("gamma", "beta", "moving_mean", "moving_variance"))
    used, worst = set(), 0.0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        w = inits.get(node.input[1])
        b = inits.get(node.input[2]) if len(node.input) > 2 else None
        if w is None:
            print("  onnx", node.name, "weight not an initializer")
            continue
        best = None
        for kname, kern in kernels.items():
            k_oihw = kern.transpose(3, 2, 0, 1)
            if k_oihw.shape != w.shape:
                continue
            for bname, (g, beta, mean, var) in bns.items():
                if g.shape[0] != w.shape[0]:
                    continue
                for eps in (1e-3, 1e-5):
                    scale = g / np.sqrt(var + eps)
                    fw = k_oihw * scale[:, None, None, None]
                    err = float(np.max(np.abs(fw - w)) / (np.max(np.abs(w)) + 1e-12))
                    if b is not None:
                        fb = beta - mean * scale
                        err = max(err, float(np.max(np.abs(fb - b)) / (np.max(np.abs(b)) + 1e-12)))
                    if best is None or err < best[0]:
                        best = (err, kname, bname, eps)
            # unfused case (no BN after this conv)
            err = float(np.max(np.abs(k_oihw - w)) / (np.max(np.abs(w)) + 1e-12))
            if best is None or err < best[0]:
                best = (err, kname, None, None)
        print(f"  onnx {node.name:40s} {str(w.shape):18s} -> {best}")
        if best:
            used.add(best[1])
            worst = max(worst, best[0])
    print(f"matched TF kernels {len(used)}/{len(kernels)}; worst relative error {worst:.3g}")


if __name__ == "__main__":
    main()
