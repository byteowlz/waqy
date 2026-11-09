"""Score an exported head against its own held-out positives.

A silent export failure is worse than a training failure: the ONNX looks fine, packages fine, and
predicts a constant. This is the gate - if the head does not separate its own data, it is not a head.

Usage: verify_head.py <onnx> <positive_features_test.npy> [threshold]
Exit 0 when the median score on held-out positives is at least the threshold (default 0.5).
"""
import sys
import numpy as np
import onnxruntime as ort

onnx_path, pos_path = sys.argv[1], sys.argv[2]
threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
feature = session.get_inputs()[0].name
windows = np.load(pos_path, mmap_mode='r')[:200].astype(np.float32)
scores = session.run(None, {feature: windows})[0][:, 0]
median = float(np.median(scores))
frac = float((scores >= threshold).mean())
print(f'{onnx_path.rsplit("/", 2)[-2]}: median {median:.3f} above {threshold} {frac:.2f}')
sys.exit(0 if median >= threshold else 1)
