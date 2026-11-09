"""Measure how precisely a wake-word timing marks the word, without new models.

Signals per clip:
  energy  frame energy profile (recomputed here): speech span, onset/offset
  wh_end  the stored Whisper word end (context clips) or the energy-span end (isolated clips)
  v1      energy rule: last frame above max(6% of peak energy, noise floor + 6 dB)
  v2      like v1, but counted backwards from the wake-word region so carrier speech after the
          word cannot extend it: walk back from wh_end while frames stay speech-like
Checks:
  - lead (wh_end - offset) and lag (offset - wh_end): how far each rule moves the end
  - "tail is silence": fraction of clips where the last 120 ms after the offset is below the
    speech floor, i.e. the cut would not clip the word
  - "word inside": v1 span covers less than 0.35-1.6 s for the phrase (plausibility)
Usage: .venv/bin/python check_timings.py <bank_dir> [--limit N]
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from numpy import bool_

B = Path(sys.argv[1]).expanduser()
limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else 0
rows = [json.loads(l) for l in open(B / 'aligned.jsonl')]
rows = [r for r in rows if r['control'] == 'wake' and r.get('wake')]
if limit: rows = rows[:limit]
FRAME, HOP = 400, 160   # 25 ms window, 10 ms hop at 16 kHz

def energy(pcm):
    n = (len(pcm) - FRAME) // HOP + 1
    e = np.empty(n)
    for i in range(n):
        seg = pcm[i * HOP:i * HOP + FRAME]
        e[i] = np.sqrt((seg.astype(np.float64) ** 2).mean())
    return e

def db(x): return 20 * np.log10(np.maximum(x, 1e-9))

def rule_v1(e):
    floor = np.percentile(e, 10)
    thr = max(e.max() * 0.06, floor * 2.0)
    on = np.flatnonzero(e > thr)
    return (on[0] * HOP / 16000, (on[-1] + 1) * HOP / 16000) if len(on) else (0.0, 0.0)

def rule_v2(e, wh_end):
    floor = np.percentile(e, 10)
    thr = max(np.percentile(e, 95) * 0.10, floor * 2.0)
    i = min(len(e) - 1, int(wh_end * 16000 / HOP))
    while i > 0 and e[i] > thr: i -= 1
    return (i + 1) * HOP / 16000

out = []
for r in rows:
    pcm, _ = sf.read(str(B / 'wav' / f"{r['id']}.wav"), dtype='float32')
    e = energy(pcm)
    span = rule_v1(e)
    wh = r['wake']['t1']
    v2 = rule_v2(e, wh)
    tail = pcm[int(span[1] * 16000):int(span[1] * 16000) + 1920]
    floor_db = db(np.percentile(e, 10))
    tail_quiet = bool(len(tail)) and db(np.sqrt((tail.astype(np.float64) ** 2).mean())) < floor_db + 6
    out.append({'id': r['id'], 'kind': r['kind'], 'lang': r['lang'], 'text': r['text'], 'wh_end': round(wh, 3),
                'v1_start': round(span[0], 3), 'v1_end': round(span[1], 3), 'v2_end': round(v2, 3),
                'tail_quiet': bool(tail_quiet), 'seconds': round(len(pcm) / 16000, 2)})
Path('/tmp/timings.json').write_text(json.dumps(out, ensure_ascii=False, indent=0))
import collections
def stat(key, val):
    xs = [o[key] - o[val] for o in out]
    return f"median {np.median(xs):+.2f}s  p5 {np.percentile(xs, 5):+.2f}  p95 {np.percentile(xs, 95):+.2f}"
print(len(out), 'wake clips')
print('stored end minus energy end (v1):', stat('wh_end', 'v1_end'))
print('v2 end minus stored end      :', stat('v2_end', 'wh_end'))
print('stored end minus v2 end      :', stat('wh_end', 'v2_end'))
print('tail quiet after v1 end      :', sum(o['tail_quiet'] for o in out), '/', len(out))
print('stored end inside speech     :', sum(1 for o in out if o['tail_quiet']), 'of', len(out), 'clips have quiet audio right after the stored end')
print('span length v1 (s)           : median %.2f  p95 %.2f' % (np.median([o['v1_end'] - o['v1_start'] for o in out]), np.percentile([o['v1_end'] - o['v1_start'] for o in out], 95)))
worst = sorted(out, key=lambda o: o['wh_end'] - o['v1_end'])
print('\nlargest gaps (stored end after energy end):')
for o in worst[-6:]: print('  ', o['id'], f"{o['wh_end'] - o['v1_end']:+.2f}s", o['text'][:40], 'len', o['seconds'])
