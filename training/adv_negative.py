"""Cross-word adversarial negatives: a head must not fire on other wake words.

Without this, every head fires on "Hey <any word>". Measured on the first 12-word batch: 66 of 132
head/pair cells above 10%, worst hey_snake on hex positives 0.97 and hey_naomi on meryl 0.98. Adding
the other words' windows as negatives brought those to 0.02 and 0.06 at 0.97/0.95 own-word recall.

The augmented set replaces the head's negatives in place, because livekit's exporter reads
<root>/output regardless of the configured output_dir (a head trained elsewhere exports random
weights). The unaugmented arrays are kept once as *.plain.npy.

Usage: adv_negative.py <word> [<wordpack root>]
"""
import json
import os
import sys

import numpy as np

R = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/.local/share/foxline/wake-train-1/wordpack-1')
target = sys.argv[1]
# The run's word set (WAQY_ADV_WORDS=a,b,c); the default is the first 12-word pack.
words = [w for w in os.environ.get('WAQY_ADV_WORDS', '').split(',') if w] or [
    'elissa', 'pi', 'opencode', 'hex', 'governor', 'colonel', 'snake', 'mei-ling', 'meryl',
    'otacon', 'miller', 'naomi']
others = [w for w in words if w != target]
base = f'{R}/output/hey_{target}'


def load_plain(name):
    """Read the unaugmented array, so re-running does not stack duplicates."""
    plain = f'{base}/{name}'.replace('.npy', '.plain.npy')
    current = f'{base}/{name}'
    if os.path.exists(plain):
        return np.load(plain, mmap_mode='r')
    if os.path.exists(current):
        os.replace(current, plain)
        return np.load(plain, mmap_mode='r')
    return None


def windows(word, split, cap):
    path = f'{R}/output/hey_{word}/positive_features_{split}.npy'
    if not os.path.exists(path):
        return None
    return np.asarray(np.load(path, mmap_mode='r')[:cap])


def build(split, cap, name):
    plain = load_plain(name)
    if plain is None:
        return None
    extra = [w for w in (windows(o, split, cap) for o in others) if w is not None]
    if not extra:
        return np.asarray(plain)
    merged = np.concatenate([np.asarray(plain)] + extra).astype(np.float32)
    np.random.default_rng(0).shuffle(merged)
    return merged


train = build('train', 400, 'negative_features_train.npy')
if train is not None:
    np.save(f'{base}/negative_features_train.npy', train)
test = build('test', 120, 'negative_features_test.npy')
if test is not None:
    np.save(f'{base}/negative_features_test.npy', test)
print(json.dumps({'head': target, 'neg_train': 0 if train is None else int(len(train)),
                  'neg_test': 0 if test is None else int(len(test))}))
