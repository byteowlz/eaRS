"""Offline parser and frozen-prompt checks; no ZGX connection required."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from compare_dgemma import (HERE, PROMPT_SHA, SYSTEM_PROMPT, load_challenge,
                            load_previous, parse_label, selected_indices, summarize)


class DiffusionGemmaTests(unittest.TestCase):
    def test_prompt_is_frozen(self):
        self.assertEqual(hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), PROMPT_SHA)
        self.assertEqual(PROMPT_SHA, "c9610cb64da13e3f2b1dbee676ee879407bffd83bea4eef97d400c7c4fbe868a")

    def test_strict_label_and_channel_marker(self):
        def reply(text):
            return {"choices": [{"message": {"content": text}}]}
        self.assertEqual(parse_label(reply('{"label":"literal_dictation"}')),
                         ("literal_dictation", False))
        self.assertEqual(parse_label(reply('thought\n{"label":"command_intent"}')),
                         ("command_intent", True))
        for text in ('{"label":"command_intent","execute":true}',
                     'thought\n```json\n{"label":"command_intent"}\n```',
                     '{"label":"unknown"}', 'command_intent', '{}'):
            self.assertEqual(parse_label(reply(text)), ("invalid", False))
        self.assertEqual(parse_label({"choices": []}), ("invalid", False))

    def test_frozen_challenge_sample_and_resume_refusal(self):
        challenge = load_challenge(HERE / "artifacts/intent-challenge.jsonl")
        indexes = selected_indices(challenge, False, 25)
        self.assertEqual(len(indexes), 100)
        self.assertEqual(len({challenge[i]["family"] for i in indexes}), 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            output.write_text(json.dumps({"index": indexes[0], "id": "wrong",
                                          "challenge_sha256": "wrong", "prompt_sha256": "wrong",
                                          "parser_revision": "wrong", "model": "wrong"}) + "\n")
            with self.assertRaises(ValueError):
                load_previous(output, indexes, challenge)

    def test_invalid_output_abstains_and_stays_in_denominator(self):
        rows = [{"label": "literal_dictation", "predicted": "invalid", "family": "quoted",
                 "latency_ms": 12, "channel_marker": False},
                {"label": "command_intent", "predicted": "command_intent", "family": "direct",
                 "latency_ms": 14, "channel_marker": True}]
        summary = summarize(rows)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["invalid"], 1)
        self.assertEqual(summary["false_command_candidates"], 0)
        self.assertEqual(summary["command_recall"], 1)


if __name__ == "__main__":
    unittest.main()
