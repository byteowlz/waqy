"""Score a head on the shared corpus's own voice-disjoint eval windows (context-shared/*-test-*.npy)."""
import json, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
R = Path.home()/'.local/share/foxline/wake-train-1'
tag = sys.argv[1]; kind = tag.split('-')[0]; threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
s = ort.InferenceSession(str(R/'output'/tag/f'{tag}.onnx'), providers=['CPUExecutionProvider'])
out = {}
for name, path in (('wake_pos', R/'context-shared'/f'{kind}-test-pos.npy'), ('lookalike_neg', R/'context-shared'/f'{kind}-test-neg.npy')):
    a = np.load(path).astype(np.float32)
    sc = np.concatenate([s.run(None, {'embeddings': a[i:i+4096]})[0][:, 0] for i in range(0, len(a), 4096)])
    out[name] = {'n': len(a), 'rate': round(float((sc >= threshold).mean()), 4), 'median': round(float(np.median(sc)), 3)}
print(tag, json.dumps(out))
