"""Regression checks for acoustic stress reporting; no private recordings needed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from gate_rooms import CONDITIONS, RATE, add_noise, convolve, evaluate, read_audio, score_head, select_cases, variants


class AcousticGateTests(unittest.TestCase):
    def setUp(self):
        self.takes = [
            {"id": "bare", "word": "hex", "phrase": "Hex", "label": "positive"},
            {"id": "hey", "word": "hex", "phrase": "Hey Hex", "label": "positive"},
            {"id": "look", "word": "hex", "phrase": "extra", "label": "negative"},
            {"id": "other", "word": "snake", "phrase": "Snake", "label": "positive", "other": True},
        ]
        self.scores = {t["id"]: {"hex": 0.9, "hey_hex": 0.1} for t in self.takes}
        self.scores["hey"]["hey_hex"] = 0.8

    def test_addressed_head_excludes_bare_positive(self):
        row = score_head("hey_hex", self.takes, self.scores, 0.5)
        self.assertEqual(row["wake"], {"hits": 1, "n": 1, "fired_ids": ["hey"]})
        self.assertEqual(row["lookalike"]["n"], 1)
        self.assertEqual(row["other"]["n"], 1)
        self.assertEqual(row["other"]["hits"], 0)

    def test_bare_head_counts_both_forms_and_false_fires(self):
        row = score_head("hex", self.takes, self.scores, 0.5)
        self.assertEqual((row["wake"]["hits"], row["wake"]["n"]), (2, 2))
        self.assertEqual(row["lookalike"]["hits"], 1)
        self.assertEqual(row["other"]["hits"], 1)

    def test_zero_wake_denominator_is_not_replaced_by_hits(self):
        row = score_head("hey_snake", self.takes[:-1], {}, 0.5)
        self.assertEqual(row["wake"]["n"], 0)

    def test_missing_score_is_an_error(self):
        with self.assertRaises(KeyError):
            score_head("hex", self.takes, {}, 0.5)

    def test_quiet_conditions_preserve_attenuation(self):
        pcm = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        signals = variants(pcm, np.array([1.0]), np.array([0.1, -0.1]))
        self.assertEqual(tuple(signals), CONDITIONS)
        np.testing.assert_array_equal(signals["clean"], pcm)
        for db in (12, 24):
            np.testing.assert_allclose(signals[f"quiet-{db}dB"], pcm * 10 ** (-db / 20))
        np.testing.assert_allclose(signals["quiet+room+noise"], signals["room+noise"] * 10 ** (-12 / 20))

    def test_room_preserves_tail_and_peak(self):
        out = convolve(np.array([0.5]), np.array([1.0, 0.5]))
        np.testing.assert_allclose(out, [0.5, 0.25])

    def test_noise_snr(self):
        pcm = np.ones(1000) * 0.1
        mixed = add_noise(pcm, np.array([0.1, -0.1]), 10)
        self.assertAlmostEqual(10 * np.log10(np.mean(pcm ** 2) / np.mean((mixed - pcm) ** 2)), 10, places=5)

    def test_audio_format_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            for rate, pcm in [(8000, np.ones(10)), (RATE, np.ones((10, 2))), (RATE, np.array([]))]:
                sf.write(path, pcm, rate, subtype="FLOAT")
                with self.assertRaises(ValueError):
                    read_audio(path)
            sf.write(path, np.ones(10) * 0.1, RATE, subtype="FLOAT")
            np.testing.assert_allclose(read_audio(path), 0.1)

    def test_other_negatives_only_use_frozen_eligible_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            for take_id, word in [("allowed", "snake"), ("trained", "snake"), ("same-word", "hex")]:
                directory = store / take_id
                directory.mkdir()
                (directory / "meta.json").write_text(json.dumps({"word": word, "phrase": word, "label": "positive"}))
            rows = select_cases(self.takes[:1], store, {"allowed", "same-word"}, {"hex"}, 80)
            self.assertEqual([r["id"] for r in rows], ["bare", "allowed"])

    def test_pack_evaluation_keeps_heads_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            (pack / "manifest.json").write_text(json.dumps({"classifiers": ["hex", "hey_hex"]}))
            with patch("gate_rooms.clip_max", return_value=self.scores):
                report = evaluate(pack, self.takes, {"clean": []}, "waqy", 0.5)
            self.assertEqual(report["clean"]["hex"]["wake"]["n"], 2)
            self.assertEqual(report["clean"]["hey_hex"]["wake"]["n"], 1)
            with patch("gate_rooms.clip_max", return_value={}):
                with self.assertRaises(ValueError):
                    evaluate(pack, self.takes, {"clean": []}, "waqy", 0.5)


if __name__ == "__main__":
    unittest.main()
