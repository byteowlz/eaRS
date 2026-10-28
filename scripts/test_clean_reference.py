"""Filler detection for German reference text: removes hesitations, never real words."""
import unittest

from clean_reference import DE_FILLERS, EN_ONLY, clean, is_filler


class FillerTests(unittest.TestCase):
    def test_removes_german_hesitations_anywhere(self):
        self.assertEqual(clean("Ähm, da zählt jede Nanosekunde.")[0], "Da zählt jede Nanosekunde.")
        self.assertEqual(clean("Also mich ähm reißen diese Ankündigungen ähm nicht mit.")[0],
                         "Also mich reißen diese Ankündigungen nicht mit.")
        self.assertEqual(clean("wenn man in die Richtung geht, uh, ob das gut ist")[0],
                         "wenn man in die Richtung geht, ob das gut ist")

    def test_reported_tokens_are_returned_verbatim(self):
        text = "Ähm, das war ähm so."
        self.assertEqual(clean(text)[1], ["Ähm,", "ähm"])

    def test_german_words_are_not_removed(self):
        for sentence in ("Wir treffen uns um acht.", "Er hat das gesagt.", "Das ist um so besser."):
            self.assertEqual(clean(sentence), (sentence, []))

    def test_english_homographs_are_explicitly_outside_the_german_set(self):
        self.assertFalse(DE_FILLERS & EN_ONLY)
        self.assertFalse(is_filler("um"))
        self.assertFalse(is_filler("er"))

    def test_punctuation_and_numbers_are_preserved(self):
        self.assertEqual(clean("Ähm 3,5 Prozent, hm?")[0], "3,5 Prozent,")
        self.assertEqual(clean("Bleibt so; ähm, wirklich.")[0], "Bleibt so; wirklich.")

    def test_empty_and_filler_only(self):
        self.assertEqual(clean(""), ("", []))
        self.assertEqual(clean("Ähm hm"), ("", ["Ähm", "hm"]))

    def test_ordinary_text_is_byte_identical(self):
        text = "Die Spracherkennung läuft direkt auf deinem Gerät — ohne Cloud."
        self.assertEqual(clean(text), (text, []))


if __name__ == "__main__":
    unittest.main()
