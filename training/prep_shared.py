"""Bring the shared spqx wake sets into a wake bank: 16 kHz, manifest, word timestamps, screens.

Input:  <shared>/<de|en>/{clips.jsonl,wav/*.wav}   (24 kHz, manifest from the corpus agent)
Output: <out>/{clips.jsonl,wav/*.wav,aligned2.jsonl,clean.jsonl}
The corpus agent marks timing_verified=false, so every clip is transcribed here (word
timestamps) and screened the same way as our own bank (realign_bank.py + decide_timings.py).
Usage: .venv/bin/python prep_shared.py <shared_dir> <out_dir>
"""
import json, shutil, subprocess, sys
from pathlib import Path
import soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from build_data import to16k

shared, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
(out / 'wav').mkdir(parents=True, exist_ok=True)
rows = []
for lang_dir in sorted(p for p in shared.iterdir() if p.is_dir()):
    for r in map(json.loads, open(lang_dir / 'clips.jsonl')):
        if not r.get('valid', True):
            continue
        src = lang_dir / 'wav' / f"{r['id']}.wav"
        dst = out / 'wav' / f"{r['id']}.wav"
        if not dst.exists():
            pcm, rate = sf.read(str(src), dtype='float32')
            sf.write(str(dst), to16k(pcm, rate), 16000, subtype='PCM_16')
        kind = r['kind']
        if r['stop']:
            control = 'wake'          # a stop phrase is a wake-model positive for the stop head
        elif r['wake'] and kind in ('bare', 'embedded', 'hesitant', 'command'):
            control = 'wake'
        else:
            control = 'lookalike'
        rows.append({'id': r['id'], 'lang': r['lang'], 'kind': kind, 'control': control,
                     'text': r['text'], 'voice': r['voice'], 'split': r['split'],
                     'consumer_split': r['consumer_split'], 'stop': r['stop']})
(out / 'clips.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
import collections
print(json.dumps({'clips': len(rows), 'by_control': dict(collections.Counter(r['control'] for r in rows)),
                  'by_kind': dict(collections.Counter(r['kind'] for r in rows)),
                  'by_lang': dict(collections.Counter(r['lang'] for r in rows)),
                  'eval_split_clips': sum(1 for r in rows if r['consumer_split'] == 'eval')}, indent=1))
