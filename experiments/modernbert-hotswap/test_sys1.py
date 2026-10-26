"""Offline sys1 decision response checks; no inference service required."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from compare_dgemma import load_challenge, selected_indices
from compare_sys1 import (HERE, QUESTION, QUESTION_SHA, load_previous,
                          parse_answer, summarize)


class Sys1TrialTests(unittest.TestCase):
    def test_question_frozen_before_full_challenge(self):
        self.assertEqual(QUESTION_SHA, hashlib.sha256(json.dumps(QUESTION, sort_keys=True).encode()).hexdigest())
        self.assertEqual(QUESTION_SHA, "c4093cd2350ad65687f2fb82836beb3cea1b4a0801dc2dc39f8f1730da49d5ff")

    def test_strict_choice_and_abstention(self):
        answer = {"answers": {"intent": {"type": "choice", "choice": "command_intent",
                                         "probabilities": {"command_intent": 0.8,
                                                           "literal_dictation": 0.1,
                                                           "uncertain": 0.1}}}}
        self.assertEqual(parse_answer(answer)[0], "command_intent")
        self.assertEqual(parse_answer({"answers": {"intent": {"type": "choice",
                          "choice": "execute", "probabilities": {}}}}), ("invalid", None))
        self.assertEqual(parse_answer({"error": "service unavailable"}), ("invalid", None))
        rows = [{"label": "literal_dictation", "predicted": "invalid", "family": "quoted",
                 "latency_ms": 10},
                {"label": "command_intent", "predicted": "command_intent", "family": "direct",
                 "latency_ms": 12}]
        self.assertEqual(summarize(rows)["invalid"], 1)
        self.assertEqual(summarize(rows)["command_recall"], 1)

    def test_resume_rejects_mismatched_question(self):
        challenge = load_challenge(HERE / "artifacts/intent-challenge.jsonl")
        indexes = selected_indices(challenge, True, None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            output.write_text(json.dumps({"index": indexes[0], "id": "wrong",
                                          "challenge_sha256": "wrong", "question_sha256": "wrong",
                                          "model": "wrong"}) + "\n")
            with self.assertRaises(ValueError):
                load_previous(output, indexes, challenge)


if __name__ == "__main__":
    unittest.main()
