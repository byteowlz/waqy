"""Decide which bank clips are usable and what the wake word span is (feeds context_data.bank).

From aligned2.jsonl (word timestamps from the current files), per wake clip:
  keep  if the wake word was heard exactly once, its span is 0.15-1.6 s, no digital silence falls
        inside the word, and (isolated clips) the clip holds exactly one speech group whose span
        matches the word within 0.9 s
  drop  otherwise, with the reason
Look-alike clips keep their look-alike word spans (screened the same way).
Writes <bank>/clean.jsonl: {id, kind, lang, voice, control, text, t0, t1, seconds, reasons}
Usage: .venv/bin/python decide_timings.py <bank_dir>
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf

FRAME, HOP, MERGE = 400, 160, 0.40
bank = Path(sys.argv[1]).expanduser()

def runs(pcm):
    n = (len(pcm) - FRAME) // HOP + 1
    if n <= 0: return []
    e = np.sqrt((pcm[np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]].astype(np.float32) ** 2).mean(1))
    thr = max(np.percentile(e, 95) * 0.10, max(np.percentile(e, 10), 1e-6) * 2.0)
    above, out, i = e > thr, [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]: j += 1
            if out and (i - out[-1][1]) * HOP / 16000 < MERGE: out[-1][1] = j
            else: out.append([i, j])
            i = j + 1
        else: i += 1
    return [(a * HOP / 16000, (b + 1) * HOP / 16000) for a, b in out]

def digital_gap_inside(pcm, t0, t1):
    a, b = int(t0 * 16000), int(t1 * 16000)
    seg = pcm[max(0, a):b]
    if len(seg) < FRAME: return False
    n = (len(seg) - FRAME) // HOP + 1
    e = np.sqrt((seg[np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]].astype(np.float32) ** 2).mean(1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    run = 0
    for v in db:
        run = run + 1 if v < -70 else 0
        if run * HOP / 16000 >= 0.12: return True
    return False

rows = [json.loads(l) for l in open(bank / 'aligned2.jsonl')]
out, stats = [], {}
for r in rows:
    if 'error' in r: continue
    reasons = []
    pcm, _ = sf.read(str(bank / 'wav' / f"{r['id']}.wav"), dtype='float32')
    if r['control'] == 'wake':
        if not r.get('wake'): reasons.append('no_single_wake_word')
        elif 'implausible_word_span' in r['screens']: reasons.append('implausible_span')
    if not reasons and r['control'] == 'wake':
        t0, t1 = r['wake']['t0'], r['wake']['t1']
        if digital_gap_inside(pcm, t0, t1): reasons.append('silence_inside_word')
        if r['kind'] == 'isolated':
            rs = runs(pcm)
            if len(rs) != 1: reasons.append(f'groups_{len(rs)}')
            elif abs((rs[0][1] - rs[0][0]) - (t1 - t0)) > 0.9: reasons.append('group_vs_word_mismatch')
    elif r['control'] == 'lookalike':
        look = [w for w in r['words'] if w['w'] and w['w'].strip().lower().strip('.,!?') in ('oktober', 'october', 'oktopus', 'octopus', 'oktave', 'octave')]
        if look:
            t0, t1 = look[0]['t0'], look[0]['t1']
        elif r['words']:
            # a negative whose look-alike word is not on my list (Oktaeder, lotto, octagon, Otto,
            # doctorate, ...): span the whole sentence, which is what a negative window needs
            t0, t1 = r['words'][0]['t0'], r['words'][-1]['t1']
        else:
            reasons.append('no_words_at_all')
    if r['control'] not in ('wake', 'lookalike'):
        continue
    if not reasons:
        out.append({'id': r['id'], 'kind': r['kind'], 'lang': r['lang'], 'voice': r.get('voice'), 'control': r['control'],
                    'text': r['text'], 't0': round(t0, 3), 't1': round(t1, 3), 'seconds': r['seconds']})
    for x in reasons: stats[x] = stats.get(x, 0) + 1
(bank / 'clean.jsonl').write_text('\n'.join(json.dumps(o, ensure_ascii=False) for o in out) + '\n')
import collections
print(json.dumps({'kept': len(out), 'by_control': dict(collections.Counter(o['control'] for o in out)),
                  'by_kind_kept': {f'{k[0]}_{k[1]}': v for k, v in collections.Counter((o['control'], o['kind']) for o in out).items()},
                  'by_lang': {f'{k[0]}_{k[1]}': v for k, v in collections.Counter((o['control'], o['lang']) for o in out).items()},
                  'dropped_reasons': stats}, indent=1))
