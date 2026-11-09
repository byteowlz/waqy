"""Per-sentence false wakes on held-out-voice look-alike sentences, streamed with a realistic lead-in.

Each sentence gets 1.5 s of low-level background noise before it (not digital zeros), so the
detector is warm when the sentence starts, as in a live stream. A sentence counts as a false wake
if any frame from the sentence start to 1 s after it reaches the threshold.
Usage: .venv/bin/python sentence_fa.py <bank_dir> <model,...> [--threshold 0.5] [--all-voices]
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parent))
from build_data import Featurizer, held_out_speaker

R = Path.home() / '.local/share/foxline/wake-train-1'
bank = Path(sys.argv[1]).expanduser(); models = sys.argv[2].split(',')
threshold = float(sys.argv[sys.argv.index('--threshold') + 1]) if '--threshold' in sys.argv else 0.5
all_voices = '--all-voices' in sys.argv
rows = [json.loads(l) for l in open(bank / 'aligned.jsonl')]
feat = Featurizer(); rng = np.random.default_rng(0)
noise = sorted((R / 'data/backgrounds').glob('*.wav'))
def lead():
    n, _ = sf.read(str(noise[rng.integers(len(noise))]), dtype='float32')
    n = n if n.ndim == 1 else n[:, 0]
    n = np.tile(n, 4)[: int(1.5 * 16000)]
    if "--zeros" in sys.argv: return np.zeros(int(1.5 * 16000), np.float32)
    return n / (np.abs(n).max() + 1e-9) * 0.01   # about -40 dBFS room noise
sets = {'lookalike': [r for r in rows if r['control'] == 'lookalike'], 'wake': [r for r in rows if r['control'] == 'wake' and r['wake']]}
for k in sets:
    sets[k] = [r for r in sets[k] if all_voices or held_out_speaker(r.get('voice', ''))]
cache = {}
for k, rs in sets.items():
    for r in rs:
        pcm, _ = sf.read(str(bank / 'wav' / f"{r['id']}.wav"), dtype='float32')
        x = np.concatenate([lead(), pcm, np.zeros(16000, np.float32)])
        if '--dither' in sys.argv:  # what the runtimes would add: about -70 dBFS white noise
            x = x + np.random.default_rng(len(x)).normal(0, 3e-4, len(x)).astype(np.float32)
        fr = feat.frames(x)
        w = np.lib.stride_tricks.sliding_window_view(fr.astype(np.float32), (16, 96))[:, 0]
        cache[r['id']] = w[max(0, int(1.5 / 0.08) - 15):]   # decisions from the sentence start on
for m in models:
    s = ort.InferenceSession(str(R / 'output' / m / f'{m}.onnx'), providers=['CPUExecutionProvider'])
    out = {}
    for k, rs in sets.items():
        hits = {'de': [0, 0], 'en': [0, 0]}
        for r in rs:
            peak = float(s.run(None, {'embeddings': cache[r['id']]})[0].max())
            if k == 'wake' and m.startswith('hey_') and not r['text'].lower().startswith(('hey', 'he ', 'okay, hey')):
                continue
            hits[r['lang']][0] += peak >= threshold; hits[r['lang']][1] += 1
        out[k] = {l: f'{a}/{b}' for l, (a, b) in hits.items() if b}
    print(f'{m:<15} false wakes on look-alike sentences {out["lookalike"]}   wake sentences caught {out["wake"]}')
