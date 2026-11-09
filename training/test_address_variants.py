"""Inventory/collection regression tests; no inference, synthesis or private owner data."""
import json
from pathlib import Path
import tempfile
import unittest

from addresses import is_addressed
from build_collection import lookalike_problem, word_entry
from wordpack import build

WORDS = Path(__file__).parent / 'words'


class AddressVariantTests(unittest.TestCase):
    def test_greeting_and_filler_classification(self):
        for text in ('Hey Hex', 'He Hex', 'Yo Hex', 'yo, Hex', 'So, uhm, Yo Hex', 'Also, äh, Hey Hex', 'Oye Hex'):
            self.assertTrue(is_addressed(text), text)
        for text in ('Hex', 'Hex?', 'Yodel', 'I said Yo Hex', 'Heywood'):
            self.assertFalse(is_addressed(text), text)

    def test_home_assistant_never_synthesises_bare(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'homeassistant.json').write_bytes((WORDS / 'homeassistant.json').read_bytes())
            rows = build(directory, 120)
        wakes = [r for r in rows if r['control'] == 'wake']
        self.assertTrue(wakes)
        self.assertTrue(all(is_addressed(r['text']) for r in wakes))
        self.assertTrue(all('Home Assistant' in r['text'] for r in wakes))
        entry = word_entry(WORDS / 'homeassistant.json')
        self.assertTrue(all(p['form'] == 'addressed' for group in entry['languages'].values() for p in group['positives']))

    def test_yo_variants_cover_the_core_assistants_not_only_hex(self):
        names = ('hex', 'oqto', 'pi', 'elissa', 'opencode', 'jarvis', 'computer', 'homeassistant')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in names:
                (directory / (name + '.json')).write_bytes((WORDS / (name + '.json')).read_bytes())
            rows = [r for r in build(directory, 240) if r['control'] == 'wake']
        for name in names:
            spec = json.loads((WORDS / (name + '.json')).read_text())
            entry = word_entry(WORDS / (name + '.json'))
            for language in ('de', 'en'):
                with self.subTest(word=name, language=language):
                    yo = next(f for f in spec['forms'][language] if f.startswith('Yo '))
                    self.assertEqual(spec['address_variants'][language], ['Yo'])
                    self.assertEqual(spec.get('address', {}).get('en', ['Hey']), ['Hey'])
                    forms = entry['languages'][language]['positives']
                    self.assertEqual(len([p for p in forms if p['line'] == yo and p['form'] == 'addressed']), 1)
                    texts = [r['text'] for r in rows if r['word'] == name and r['lang'] == language]
                    self.assertTrue(any('Yo ' in t for t in texts))
                    self.assertTrue(any('Hey ' in t for t in texts))
                    self.assertFalse(any('Hey Yo ' in t or 'Yo Yo ' in t or 'Yo Hey ' in t for t in texts))
                    if spec.get('train_bare', True):
                        self.assertTrue(any(not is_addressed(t) for t in texts))

    def test_address_variants_cannot_become_negative_spellings(self):
        spec = json.loads((WORDS / 'hex.json').read_text())
        self.assertIsNotNone(lookalike_problem('Yo Hex', spec, 'en'))
        self.assertIsNotNone(lookalike_problem('Hex', spec, 'en'))
        spec = json.loads((WORDS / 'homeassistant.json').read_text())
        for text in ('Home Assistant', 'homeassistant', 'home-assistant'):
            self.assertIsNotNone(lookalike_problem(text, spec, 'en'))


if __name__ == '__main__':
    unittest.main()
