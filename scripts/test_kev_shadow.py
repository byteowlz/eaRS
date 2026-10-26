"""Tests for the offline report logic; no model download or production execution."""
import json
import tempfile
import unittest
from pathlib import Path

from kev_shadow import fixtures, percentile, summary


class KevShadowTests(unittest.TestCase):
    def test_percentiles_and_baseline_false_positive(self):
        rows = [
            {"label": "literal_dictation", "prediction": "registered_command_candidate",
             "deterministic_outcome": "typed", "latency_ms": 10},
            {"label": "registered_command_candidate", "prediction": "uncertain",
             "deterministic_outcome": "control", "latency_ms": 20},
        ]
        result = summary(rows)
        self.assertEqual(result["false_command_candidates"], 1)
        self.assertEqual(result["deterministic_baseline"]["false_command_candidates"], 0)
        self.assertEqual(result["missed_command_candidates"], 1)
        self.assertEqual(percentile([10, 20], .95), 19.5)

    def test_fixtures_are_completed_and_typed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.jsonl"
            path.write_text(json.dumps({"normalized": "", "label": "uncertain"}) + "\n")
            with self.assertRaisesRegex(ValueError, "completed"):
                fixtures(path)
            path.write_text(json.dumps({"normalized": "open notes", "label": "unknown"}) + "\n")
            with self.assertRaisesRegex(ValueError, "expected one of"):
                fixtures(path)
            path.write_text(json.dumps({"normalized": "open notes", "label": "uncertain"}) + "\n")
            self.assertEqual(len(fixtures(path)), 1)


if __name__ == "__main__":
    unittest.main()
