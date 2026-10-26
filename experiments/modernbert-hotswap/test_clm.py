"""Offline CLM trial wiring checks; no embedding server or GPU required."""

import unittest
from unittest.mock import patch

import compare_clm
import compare_sys1


class ClmTrialTests(unittest.TestCase):
    def test_clm_wrapper_reuses_frozen_question_and_selects_separate_result(self):
        prior = compare_sys1.MODEL
        try:
            with patch.object(compare_sys1, "main") as invoke:
                compare_clm.main()
                self.assertEqual(compare_sys1.MODEL, "clm-latest")
                self.assertEqual(invoke.call_args.kwargs["default_url"],
                                 "http://127.0.0.1:18700/v1/systemone")
                self.assertEqual(invoke.call_args.kwargs["default_output"].name,
                                 "compare-clm.jsonl")
        finally:
            compare_sys1.MODEL = prior

    def test_clm_request_uses_same_question_and_explicit_model(self):
        prior = compare_sys1.MODEL
        compare_sys1.MODEL = "clm-latest"
        payload = {"answers": {"intent": {"type": "choice", "choice": "uncertain",
                                          "probabilities": {"literal_dictation": 0.1,
                                                            "command_intent": 0.2,
                                                            "uncertain": 0.7}}}}
        try:
            with patch.object(compare_sys1, "post_json", return_value=(payload, 9.0)) as post:
                result = compare_sys1.query(1, {"text": "Maybe not yet", "mode": "dictation",
                                                 "label": "uncertain", "family": "correction"},
                                            "http://127.0.0.1:18700/v1/systemone")
                self.assertEqual(result["predicted"], "uncertain")
                self.assertEqual(result["model"], "clm-latest")
                self.assertEqual(post.call_args.args[1]["model"], "clm-latest")
                self.assertEqual(post.call_args.args[1]["questions"]["intent"],
                                 compare_sys1.QUESTION)
        finally:
            compare_sys1.MODEL = prior


if __name__ == "__main__":
    unittest.main()
