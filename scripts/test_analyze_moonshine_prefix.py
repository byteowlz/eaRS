"""Synthetic regression examples for the offline analyzer, no models/audio."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from analyze_moonshine_prefix import POLICIES, common_prefix, lexical, main, read_trace, replay_line


def event(time, text, complete=False, updated=True):
    return {"time": time, "tokens": text.split(), "complete": complete, "updated": updated}


class PrefixReplayTests(unittest.TestCase):
    def test_repeated_agreement_does_not_guarantee_a_word_is_stable(self):
        # A synthetic analogue of a word changing after >2s of agreement.
        events = [event(t, "Risswort ist in der Mitte") for t in (0.5, 1.0, 1.5, 2.0, 2.6)]
        events += [event(3.0, "Riffwort ist in der Mitte"), event(3.5, "Riffwort ist in der Mitte", True)]
        result = replay_line(events, POLICIES[-1])
        self.assertGreater(result["early_words"], 0)
        self.assertTrue(result["contradicted"])
        self.assertTrue(result["wrong_final"])
        self.assertTrue(result["lexical_wrong"])

    def test_punctuation_and_case_are_not_silently_normalized_away(self):
        events = [event(t, "Ein Test bleibt stehen") for t in (0.5, 1.0, 1.5)]
        events += [event(2.0, "ein Test bleibt stehen", True)]
        result = replay_line(events, POLICIES[2])
        self.assertTrue(result["wrong_final"])
        self.assertFalse(result["lexical_wrong"])

    def test_a_final_does_not_repair_what_was_already_committed(self):
        events = [event(0.5, "die alte Antwort"), event(1.0, "die alte Antwort")]
        events += [event(1.5, "die neue Antwort", True)]
        result = replay_line(events, POLICIES[1])
        self.assertEqual(result["early_words"], 2)
        self.assertEqual(result["example"]["typed"], ["die", "alte"])
        self.assertTrue(result["wrong_final"])

    def test_cached_polls_are_not_independent_agreement(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            rows = [{"kind": "model", "runtime": 30000}]
            for t, updated, complete in [(0.5, True, False), (0.7, False, False), (0.9, False, False), (2.0, True, True)]:
                rows.append({"kind": "snapshot", "clip": "synthetic", "audio_s": t,
                             "lines": [{"id": 1, "text": "one two three", "complete": complete,
                                        "updated": updated, "text_changed": False}]})
            path.write_text("\n".join(map(json.dumps, rows)))
            _, groups = read_trace(path)
            events = next(iter(groups.values()))
            self.assertEqual(len(events), 2)
            self.assertEqual(replay_line(events, POLICIES[1])["early_words"], 0)

    def test_empty_final_can_retract_a_nonempty_partial(self):
        events = [event(0.5, "some words"), event(1.0, "some words"), event(2.0, "", True)]
        self.assertTrue(replay_line(events, POLICIES[1])["wrong_final"])

    def test_agreement_without_new_audio_does_not_count_twice(self):
        events = [event(0.5, "one two three"), event(0.5, "one two three"), event(2.0, "one two three", True)]
        self.assertEqual(replay_line(events, POLICIES[1])["early_words"], 0)

    def test_output_is_private_and_existing_files_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            trace, out = Path(temp) / "trace.jsonl", Path(temp) / "results.json"
            rows = [{"kind": "model", "runtime": 30000},
                    {"kind": "snapshot", "clip": "synthetic", "audio_s": 1.0,
                     "lines": [{"id": 1, "text": "Ja.", "complete": True, "updated": True}]}]
            trace.write_text("\n".join(map(json.dumps, rows)))
            with patch("sys.argv", ["analyze", "--trace", str(trace), "--out", str(out)]), redirect_stdout(io.StringIO()):
                main()
                original = out.read_bytes()
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(out.read_bytes(), original)
            if os.name == "posix":
                self.assertEqual(out.stat().st_mode & 0o777, 0o600)

    def test_lexical_diagnostic_ignores_punctuation_not_german_spelling(self):
        self.assertEqual(lexical(["Wort", ","]), lexical(["wort,"]))
        self.assertNotEqual(lexical(["Maße"]), lexical(["Masse"]))

    def test_no_early_commit_for_short_reply_and_prefix_can_shrink(self):
        self.assertEqual(common_prefix([["one", "two"], ["one"]]), 1)
        events = [event(0.5, "ja"), event(1.0, "ja"), event(1.5, "Ja.", True)]
        self.assertEqual(replay_line(events, POLICIES[0])["early_words"], 0)


if __name__ == "__main__":
    unittest.main()
