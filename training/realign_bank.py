"""Re-transcribe a spqx voice bank from its current WAV files (word timestamps, sha256, screens).

Replaces stale alignment: files may have been regenerated after the first pass, and isolated
clips previously carried an energy span instead of a word time. Every clip is transcribed; wake
clips must contain exactly one wake-word hit, look-alike clips must contain a look-alike word, and
digital gaps (>= 150 ms below -70 dBFS between speech) are recorded as synthesis artifacts.

Usage: .venv/bin/python realign_bank.py <bank_dir> [--word stopp]
Writes <bank>/aligned2.jsonl with {id, kind, lang, voice, text, control, wake/t0/t1, words, sha256, screens}
"""
from __future__ import annotations
import hashlib, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from build_data import WHISPER_SERVERS, heard_wake, to16k
from context_data import whisper_words, LOOKALIKE, norm

STOPP = re.compile(r'^(stopp?|stop it|stoppt)[!.?,]*$', re.I)
bank = Path(sys.argv[1]).expanduser()
word = sys.argv[sys.argv.index('--word') + 1] if '--word' in sys.argv else 'oqto'
accept = (lambda t: bool(STOPP.match(t.strip()))) if word == 'stopp' else heard_wake

def digital_gaps(pcm: np.ndarray) -> int:
    hop, frame = 160, 400
    n = (len(pcm) - frame) // hop + 1
    if n <= 0: return 0
    e = np.sqrt((pcm[np.arange(frame)[None, :] + hop * np.arange(n)[:, None]].astype(np.float32) ** 2).mean(1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    silent = db < -70
    count, run = 0, 0
    for s in silent:
        run = run + 1 if s else 0
        if run * hop / 16000 >= 0.15 and (count == 0 or True):
            pass
    # count maximal silent runs >= 150 ms that are bounded by speech on both sides
    i, gaps = 0, 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j + 1 < len(silent) and silent[j + 1]: j += 1
            if (j - i + 1) * hop / 16000 >= 0.15 and i > 0 and j + 1 < len(silent): gaps += 1
            i = j + 1
        else:
            i += 1
    return gaps

def one(item):
    i, r = item
    path = bank / 'wav' / f"{r['id']}.wav"
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    pcm, rate = sf.read(str(path), dtype='float32')
    if rate != 16000:
        pcm = to16k(pcm, rate); sf.write(str(path), pcm, 16000, subtype='PCM_16')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    words = whisper_words(WHISPER_SERVERS[i % len(WHISPER_SERVERS)], path, r['lang'])
    screens = []
    wake = None
    if r['control'] == 'wake':
        hits = [w for w in words if accept(w['w'])]
        if len(hits) == 1:
            wake = {'t0': round(hits[0]['t0'], 3), 't1': round(hits[0]['t1'], 3), 'word': hits[0]['w']}
        elif len(hits) > 1:
            screens.append(f'repeated_wake_word_x{len(hits)}')
        else:
            screens.append('wake_word_missing')
    else:
        look = [w for w in words if LOOKALIKE.search(norm(w['w']))]
        if not look: screens.append('lookalike_missing')
        elif any(accept(w['w']) for w in words): screens.append('lookalike_heard_as_wake')
    gaps = digital_gaps(pcm)
    if gaps: screens.append(f'digital_gaps_{gaps}')
    if len(pcm) / 16000 > 8: screens.append('too_long')
    if wake and not (0.15 <= wake['t1'] - wake['t0'] <= 1.6): screens.append('implausible_word_span')
    return r | {'wake': wake, 'words': words, 'sha256': sha, 'seconds': round(len(pcm) / 16000, 2), 'screens': screens}

import threading
out_path = bank / 'aligned2.jsonl'
done_ids = {json.loads(l)['id'] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
rows = [json.loads(l) for l in (bank / 'clips.jsonl').read_text().splitlines()]
todo = [(i, r) for i, r in enumerate(rows) if r['id'] not in done_ids]
print(f"{len(done_ids)} already aligned, {len(todo)} to go", flush=True)
lock = threading.Lock()

def run(item):
    try:
        d = one(item)
    except Exception as e:  # one bad clip must not end the pass
        d = {'id': item[1]['id'], 'error': f'{type(e).__name__}: {e}'}
    with lock, out_path.open('a') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
    return d

with ThreadPoolExecutor(len(WHISPER_SERVERS)) as pool:
    list(pool.map(run, todo))
done = [json.loads(l) for l in out_path.read_text().splitlines()]
wake = [d for d in done if d['control'] == 'wake']
clean = [d for d in wake if not d['screens']]
look = [d for d in done if d['control'] == 'lookalike']
import collections
print(json.dumps({
  'clips': len(done), 'wake_clean': len(clean), 'wake_dropped': len(wake) - len(clean),
  'screens': dict(collections.Counter(s for d in done for s in d['screens'])),
  'by_lang': {l: [sum(1 for d in clean if d['lang'] == l), sum(1 for d in wake if d['lang'] == l)] for l in ('de', 'en')},
  'lookalike_clips': len(look), 'lookalike_clean': sum(1 for d in look if not d['screens']),
}, indent=1))
