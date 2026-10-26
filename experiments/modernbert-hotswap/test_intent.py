"""Check split hygiene and the hand-reviewed ambiguous fixtures."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from compare_intent import CHALLENGE_SHA, percentile
from generate_challenge import TARGETS, key, valid
from generate_intent import NEAR_CASES
from train_intent import LABELS, load_rows


class IntentDataTests(unittest.TestCase):
    def test_fixed_near_matches_do_not_contain_an_exact_command(self):
        self.assertGreaterEqual(len(NEAR_CASES), 50)
        self.assertEqual(len(NEAR_CASES), len(set(t.casefold() for t in NEAR_CASES)))
        exact = {"open notes", "insert greeting", "show calendar", "ears cancel",
                 "ears start typing"}
        self.assertFalse(exact.intersection(t.casefold().strip(" .!?") for t in NEAR_CASES))

    def test_frozen_challenge_is_disjoint_and_structurally_labeled(self):
        here = Path(__file__).parent / "artifacts"
        content = (here / "intent-challenge.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), CHALLENGE_SHA)
        challenge = [json.loads(line) for line in content.splitlines()]
        prior = [json.loads(line) for line in (here / "intent-data.jsonl").read_bytes().splitlines()]
        self.assertEqual(len(challenge), sum(TARGETS.values()))
        self.assertEqual(len({key(r["text"]) for r in challenge + prior}),
                         len(challenge + prior))
        for family, expected in TARGETS.items():
            rows = [r for r in challenge if r["family"] == family]
            self.assertEqual(len(rows), expected)
            self.assertTrue(all(valid(r["text"], family) for r in rows))
        self.assertEqual(percentile([1, 3, 2, 4, 5], 95), 5)

    def test_split_rejects_duplicates_and_bad_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            row = {"text": "Open note", "label": "uncertain", "split": "train"}
            path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "split": "test"}) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_rows(path)
            path.write_text(json.dumps({**row, "label": "not-a-label"}) + "\n")
            with self.assertRaisesRegex(ValueError, "bad label"):
                load_rows(path)
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(load_rows(path)["train"]), 1)
            self.assertEqual(len(LABELS), 3)


if __name__ == "__main__":
    unittest.main()
