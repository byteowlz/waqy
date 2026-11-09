"""Re-derive wake-word boundaries from energy and screen synthetic clips for synthesis faults.

Why: stored wake-word ends came from Whisper words (context clips) or one energy span over the
whole clip (isolated clips). Two defects followed (board ears-wakeword c868b198):
  1. Whisper's segment end includes the padded tail, so t1 sat ~0.5 s past the file end; windows
     built around it were misaligned by that much.
  2. Isolated clips were never transcript-screened, so TTS repetitions ("Okto … Okto", a second
     utterance after a digital gap) silently inflated the span (one "Okto." ran 5.8 s).

Rule: 10 ms RMS envelope; speech threshold = max(95th percentile * 0.10, 10th percentile * 2);
runs of speech merged across gaps < 250 ms; digital silence = a run gap where the level drops
below -70 dBFS (an artifact or padding, not a breath).
  isolated clips: exactly one run, else drop (repetition or false start)
  context clips: the run that ends at or just before the stored end; if the stored end falls in a
                 long gap, the run before it. Word end = that run's end.
Output: <bank>/timing.jsonl  {id, kind, lang, text, runs, t0, t1, flags, seconds}

Usage: .venv/bin/python fix_timings.py <bank_dir> [--apply]
  without --apply it only reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME, HOP = 400, 160  # 25 ms window, 10 ms hop at 16 kHz
MERGE_GAP = 0.25
DIGITAL_DB = -70.0


def envelope(pcm: np.ndarray) -> np.ndarray:
    n = (len(pcm) - FRAME) // HOP + 1
    if n <= 0:
        return np.zeros(1)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = pcm[idx].astype(np.float32)
    return np.sqrt((frames ** 2).mean(axis=1))


def runs_of(e: np.ndarray) -> list[tuple[int, int, float]]:
    """(start_frame, end_frame, peak_db) speech runs after merging short gaps."""
    floor = max(np.percentile(e, 10), 1e-6)
    thr = max(np.percentile(e, 95) * 0.10, floor * 2.0, 1e-4)
    above = e > thr
    runs: list[list[int]] = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged: list[list[int]] = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * HOP / 16000 < MERGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b, 20 * np.log10(max(e[a:b + 1].max(), 1e-9))) for a, b in merged]


def main() -> None:
    bank = Path(sys.argv[1]).expanduser()
    apply = '--apply' in sys.argv
    rows = [json.loads(l) for l in (bank / 'aligned.jsonl').read_text().splitlines()]
    out, dropped, flagged = [], 0, 0
    for r in rows:
        if r['control'] != 'wake' or not r.get('wake'):
            continue
        pcm, _ = sf.read(str(bank / 'wav' / f"{r['id']}.wav"), dtype='float32')
        e = envelope(pcm)
        runs = runs_of(e)
        dur = len(pcm) / 16000
        flags = []
        if abs(r['wake']['t1'] - dur) < 0.8 and r['kind'] == 'context':
            flags.append('stored_end_clamped')
        if r['kind'] == 'isolated':
            if len(runs) != 1:
                flags.append(f'runs_{len(runs)}')
                dropped += 1
                out.append({'id': r['id'], 'kind': r['kind'], 'lang': r['lang'], 'text': r['text'], 'runs': len(runs),
                            'drop': True, 'flags': flags, 'seconds': round(dur, 2)})
                continue
            a, b, _ = runs[0]
            t0, t1 = a * HOP / 16000, (b + 1) * HOP / 16000
        else:
            stored = r['wake']['t1']
            inside = [i for i, (a, b, _) in enumerate(runs) if a * HOP / 16000 <= stored <= (b + 1) * HOP / 16000 + 0.15]
            if inside:
                i = inside[0]
            else:  # stored end fell in a gap or past the last run: take the run ending closest before it
                before = [i for i, (a, b, _) in enumerate(runs) if (b + 1) * HOP / 16000 <= stored + 0.15]
                if not before:
                    flags.append('no_run_before_end')
                    dropped += 1
                    out.append({'id': r['id'], 'kind': r['kind'], 'lang': r['lang'], 'text': r['text'], 'runs': len(runs), 'drop': True, 'flags': flags, 'seconds': round(dur, 2)})
                    continue
                i = before[-1]
                flags.append('stored_end_past_speech')
            a, b, _ = runs[i]
            t0, t1 = a * HOP / 16000, (b + 1) * HOP / 16000
            if len(runs) > 1:
                flags.append(f'runs_{len(runs)}')
        length = t1 - t0
        if length > 1.8:
            flags.append('too_long')
        if flags:
            flagged += 1
        out.append({'id': r['id'], 'kind': r['kind'], 'lang': r['lang'], 'text': r['text'], 'runs': len(runs),
                    't0': round(t0, 3), 't1': round(t1, 3), 'stored_t1': round(r['wake']['t1'], 3), 'shift': round(r['wake']['t1'] - t1, 3),
                    'drop': False, 'flags': flags, 'seconds': round(dur, 2)})
    kept = [o for o in out if not o['drop']]
    shifts = [o['shift'] for o in kept]
    print(json.dumps({
        'wake_clips': len(out), 'kept': len(kept), 'dropped': len(out) - len(kept), 'flagged': flagged,
        'stored_end_shift_s': {'median': round(float(np.median(shifts)), 3), 'p5': round(float(np.percentile(shifts, 5)), 3), 'p95': round(float(np.percentile(shifts, 95)), 3)},
        'flags': {f: sum(1 for o in out if f in o['flags']) for f in sorted({f for o in out for f in o['flags']})},
    }, indent=1))
    if apply:
        (bank / 'timing.jsonl').write_text('\n'.join(json.dumps(o, ensure_ascii=False) for o in out) + '\n')
        print('wrote', bank / 'timing.jsonl')


if __name__ == '__main__':
    main()
