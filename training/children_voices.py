"""Design fictional child voices with spqx VoiceDesign (no real child recording involved).

Why: the owner's children triggered the deployed head repeatedly, which is expected - every training
voice is a designed adult, so a child is out of distribution (higher F0, shorter vocal tract). Real
children's speech is not used: these are designed voices, so the negatives carry no minors' data and
a release needs no extra provenance exclusion for them.
"""
import json, os, subprocess
from pathlib import Path
D = Path(os.path.expanduser('~/.local/share/spqx/native-voice-design-217edd91'))
BIN = D/'source/target/release/spqx'
ENV = dict(os.environ, LD_LIBRARY_PATH=str(D/'runtime'), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2')
OUT = Path(os.path.expanduser('~/.local/share/foxline/waqy-tts-1/voices-children-1')); OUT.mkdir(parents=True, exist_ok=True)
SPECS = [
 ('de', 'k01', 'A fictional child about seven years old, high clear voice, speaking German cheerfully. No singing or whispering.'),
 ('de', 'k02', 'A fictional child about ten years old, bright high voice, speaking German quickly. No singing or whispering.'),
 ('de', 'k03', 'A fictional girl about nine years old, soft high voice, speaking German. No singing or whispering.'),
 ('en', 'k04', 'A fictional child about seven years old, high clear voice, speaking American English excitedly. No singing or whispering.'),
 ('en', 'k05', 'A fictional child about ten years old, bright high voice, speaking American English. No singing or whispering.'),
 ('es', 'k06', 'A fictional child about eight years old, high clear voice, speaking Spanish cheerfully. No singing or whispering.'),
]
TEXT = {'de': 'Wir gehen heute in den Park und spielen mit dem Ball.', 'en': 'We are going to the park today to play with the ball.',
        'es': 'Hoy vamos al parque a jugar con la pelota.'}
bank = []
for lang, name, instruct in SPECS:
    ref = OUT/f'ref-{name}.wav'
    if not ref.exists():
        r = subprocess.run([str(BIN), '--config', str(D/'config.toml'), 'design', TEXT[lang], '--instruct', instruct,
                            '--language', {'de':'german','en':'english','es':'spanish'}[lang], '--seed', str(7000+int(name[1:])), '--out', str(ref)],
                           env=ENV, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print('design failed', name, r.stderr[-200:]); continue
    bank.append({'lang': lang, 'voice': name, 'ref': str(ref), 'ref_text': TEXT[lang], 'instruct': instruct})
(OUT/'voices.json').write_text(json.dumps(bank, indent=1, ensure_ascii=False))
print(json.dumps({'voices': len(bank), 'dir': str(OUT)}))
