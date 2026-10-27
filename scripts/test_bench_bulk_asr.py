"""Synthetic parser/scoring/fixture tests; no models or recordings required."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
from bench_bulk_asr import fixtures, moonshine_text, normalize, score, summarize


class BulkBenchTests(unittest.TestCase):
    def test_scoring_declares_insertions_and_empty_reference(self):
        self.assertEqual(score("a x c extra", "a b c"), {
            "word_errors": 2, "reference_words": 3, "char_errors": 7,
            "reference_chars": 5, "output_words": 4})
        self.assertEqual(score("", "")["word_errors"], 0)
        self.assertEqual(score("Ja wirklich", "")["word_errors"], 2)

    def test_german_spelling_and_numbers_are_not_normalized_to_answers(self):
        self.assertEqual(normalize("  Grüße, TEST! "), "grüße test")
        self.assertEqual(score("Maße", "Masse")["word_errors"], 1)
        self.assertEqual(score("zwölf", "12")["word_errors"], 1)

    def test_native_partial_is_not_counted_and_same_final_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.jsonl"
            rows = [{"kind": "model"}, {"kind": "model_ready", "load_s": 0.1}]
            for complete, text in [(False, "falscher Entwurf"), (True, "richtiger Text"), (True, "richtiger Text")]:
                rows.append({"kind": "snapshot", "clip": "one",
                             "lines": [{"id": 999, "complete": complete, "text": text}]})
            trace.write_text("\n".join(map(json.dumps, rows)))
            self.assertEqual(moonshine_text(trace, [{"id": "one"}]), ["richtiger Text"])

    def test_fixtures_are_unique_and_bulk_contains_identical_quantized_pcm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "out"
            out.mkdir()
            sf.write(root / "a.wav", np.full(48000, 0.1), 16000, subtype="FLOAT")
            sf.write(root / "b.wav", np.full(48000, -0.1), 16000, subtype="FLOAT")
            tsv = root / "test.tsv"
            tsv.write_text("1\ta.wav\tErster Text\terster text\n1\ta.wav\tErster Text\terster text\n2\tb.wav\tZweiter Text\tzweiter text\n")
            cases = fixtures(tsv, root, out, 2)
            clips = [c for c in cases if c["kind"] == "clip"]
            self.assertEqual(len({c["reference"] for c in clips}), 2)
            expected = np.concatenate([sf.read(c["path"], dtype="int16")[0] for c in clips])
            actual = sf.read(out / "bulk.wav", dtype="int16")[0]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual([c["id"] for c in cases[-2:]], ["silence", "noise"])

    def test_summary_uses_corpus_weighting_and_correct_rtf_direction(self):
        initial = {"cases": [
            {"id": "clip-0", "reference_words": 2, "reference_chars": 4, "word_errors": 1, "char_errors": 1},
            {"id": "clip-1", "reference_words": 8, "reference_chars": 16, "word_errors": 0, "char_errors": 0}]}
        runs = [{"wall_s": t, "peak_rss_bytes": 1048576, "cases": [{"word_errors": 1, "reference_words": 10}]} for t in (1, 2, 3)]
        result = summarize(("synthetic",), initial, runs, {"seconds": 10})
        self.assertEqual(result["clip_wer_pct"], 10)
        self.assertEqual(result["rtfx"], 5)
        self.assertEqual(result["rtf"], 0.2)


if __name__ == "__main__":
    unittest.main()
