import json, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import Featurizer
from evaluate import max_scores, Scorer
R = Path(os.path.expanduser('~/.local/share/foxline/wake-train-1'))
OWNER = Path(os.path.expanduser('~/.local/share/foxline/wake-collect-url-1'))
TEST_SESSION = sys.argv[2] if len(sys.argv) > 2 else 'd19ae2f1'
MODELS = sys.argv[1].split(',')
feat = Featurizer()
rows = [(json.loads(p.read_text()), p.parent / 'audio.wav') for p in sorted(OWNER.glob('*/meta.json'))]
rows = [(m, w) for m, w in rows if m['sessionId'].startswith(TEST_SESSION)]
def windows(onnx, path):
    if not path.exists(): return None
    a = np.load(path).astype(np.float32)
    s = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
    out = [s.run(None, {'embeddings': a[i:i+4096]})[0][:, 0] for i in range(0, len(a), 4096)]
    return np.concatenate(out) if out else np.array([])
report = {}
for tag in MODELS:
    kind = tag.split('-')[0]
    onnx = R/'output'/tag/f'{tag}.onnx'
    sc = Scorer(onnx)
    scores = max_scores(sc, feat, [w for _, w in rows])
    by = defaultdict(list)
    for (m, _), v in zip(rows, scores): by[f"{m['language']}/{m['label']}/{m['phrase']}"].append(round(float(v), 3))
    pos = [v for k, vs in by.items() if '/positive/' in k for v in vs]
    adv = [v for k, vs in by.items() if '/negative/' in k for v in vs]
    entry = {'owner_positives': f'{sum(v >= 0.5 for v in pos)}/{len(pos)}',
             'owner_lookalikes_fired': f'{sum(v >= 0.5 for v in adv)}/{len(adv)}'}
    for name, path in (('piper_ctx_pos', R/'context'/f'{kind}-test-pos.npy'), ('piper_ctx_neg', R/'context'/f'{kind}-test-neg.npy'),
                       ('spqx_ctx_pos', R/'context-spqx-eval'/f'{kind}-test-pos.npy'), ('spqx_ctx_neg', R/'context-spqx-eval'/f'{kind}-test-neg.npy')):
        w = windows(onnx, path)
        entry[name] = '-' if w is None or not len(w) else (f'{(w >= 0.5).mean():.3f}' if 'pos' in name else f'{(w >= 0.5).mean():.4f}')
    entry['by_phrase'] = {k: f'{sum(v >= 0.5 for v in vs)}/{len(vs)}' for k, vs in sorted(by.items())}
    report[tag] = entry
out = R/'output'/f'compare-{TEST_SESSION}.json'
out.write_text(json.dumps(report, indent=1))
for k, v in report.items(): print(k, {x: v[x] for x in v if x != 'by_phrase'})
