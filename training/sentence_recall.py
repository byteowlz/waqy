import json, sys, os
from pathlib import Path
import numpy as np, onnxruntime as ort, soundfile as sf
sys.path.insert(0, '.')
from build_data import Featurizer, held_out_speaker
from context_data import windows_around
R = Path.home()/'.local/share/foxline/wake-train-1'
B = Path.home()/'.local/share/foxline/wake-tts-1/voice-bank-2'
rows = [json.loads(l) for l in open(B/'aligned.jsonl')]
rows = [r for r in rows if r['control'] == 'wake' and r['wake'] is not None and held_out_speaker(r['voice']) and (os.environ.get('FORM') != 'hey' or r['text'].lower().lstrip().startswith(('hey', 'he ', 'okay, hey')))]
feat = Featurizer()
out = {}
for tag in sys.argv[1].split(','):
    kind = tag.split('-')[0]
    s = ort.InferenceSession(str(R/'output'/tag/f'{tag}.onnx'), providers=['CPUExecutionProvider'])
    hit = {}
    for r in rows:
        pcm, _ = sf.read(str(B/'wav'/f"{r['id']}.wav"), dtype='float32')
        pad = np.zeros(int(1.0*16000), np.float32)
        w = windows_around(feat.frames(np.concatenate([pad, pcm])), 1.0, [(r['wake']['t0'], r['wake']['t1'], 'wake')])['pos']
        if not w: continue
        a = np.stack(w).astype(np.float32)
        sc = s.run(None, {'embeddings': a})[0][:, 0]
        reply = hit.setdefault(r['lang'], [0, 0])
        reply[0] += int((sc >= 0.5).max()); reply[1] += 1
    out[tag] = {l: f'{v[0]}/{v[1]} ({v[0]/max(v[1],1):.2f})' for l, v in hit.items()}
print(json.dumps(out, indent=1))
