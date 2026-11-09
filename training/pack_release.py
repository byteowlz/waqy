"""Package a word pack: heads, front end, manifest with sha256 and per-head provenance.

Same manifest shape as a normal release (release, training_data, threshold, classifiers, files) plus
a `heads` block carrying what each head is for and the number it was verified at. Releases are
immutable: an existing directory is refused.

Usage: pack_release.py <staging dir> <out dir> <release name> [--threshold 0.5]
The staging dir holds <head>/<head>.onnx for every head and the front-end ONNX files.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

FRONTEND = Path(__file__).parent / 'frontend'
WORDS = Path(__file__).parent / 'words'


def describe(head: str) -> dict:
    """Phrase, languages and kind from the word spec, for catalogs and host UIs."""
    addressed = head.startswith(('hey_', 'he_'))
    word = head.split('_', 1)[1] if addressed else head
    spec_path = WORDS / f'{word}.json'
    if not spec_path.exists():
        return {'addressed': addressed}
    spec = json.loads(spec_path.read_text())
    display = spec.get('display', word)
    langs = sorted(l for l in (spec.get('forms') or {}) if l in ('de', 'en'))
    return {'word': word, 'addressed': addressed, 'kind': spec.get('kind', 'wake'), 'languages': langs,
            'phrase': (head.split('_', 1)[0].title() + ' ' + display) if addressed else display}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('staging')
    ap.add_argument('out')
    ap.add_argument('release')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--provenance', default='')
    args = ap.parse_args()
    staging, out = Path(args.staging).expanduser(), Path(args.out).expanduser() / args.release
    if out.exists():
        raise SystemExit(f'{out} exists; releases are immutable, pick a new name')
    heads = sorted(p.stem for p in staging.glob('*/*.onnx') if p.parent.name == p.stem)
    if not heads:
        raise SystemExit(f'no heads in {staging}')
    out.mkdir(parents=True)
    files, metadata = {}, {}
    for key, src in (('mel', FRONTEND / 'melspectrogram_raw.onnx'), ('embedding', FRONTEND / 'embedding_model.onnx')):
        shutil.copy2(src, out / src.name)
        files[key] = {'file': src.name, 'sha256': sha256(out / src.name)}
    for head in heads:
        src = staging / head / f'{head}.onnx'
        shutil.copy2(src, out / f'{head}.onnx')
        files[head] = {'file': f'{head}.onnx', 'sha256': sha256(out / f'{head}.onnx')}
        report = staging / head / 'verify.json'
        # verify.json carries the gate's evidence (owner_recordings, measured numbers).
        metadata[head] = {**describe(head), **(json.loads(report.read_text()) if report.exists() else {})}
    manifest = {
        'release': args.release,
        'training_data': args.provenance or
            'Isolated wake words from spqx Qwen3-TTS designed voices (Apache-2.0 GGUF, no cloned '
            'speaker): 3 languages, natural address plus bare form, in-sentence and negative context. '
            'Addressed heads were trained with the other words in the pack as negatives (ADR 0007).',
        'threshold': args.threshold,
        'classifiers': heads,
        'heads': metadata,
        'files': files,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'release': args.release, 'out': str(out), 'heads': len(heads),
                      'classifiers': heads}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
