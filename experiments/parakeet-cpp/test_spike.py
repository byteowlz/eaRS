import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf

from bench import execute, normalize_text, score_group, sha256
from fixtures import fixture_pair, noisy_pcm


class SpikeTests(unittest.TestCase):
    def test_existing_unicode_normalization(self):
        self.assertEqual(normalize_text("  Straße, ÜBER!  zwei-drei "), "straße über zweidrei")

    def test_aggregate_wer_is_word_weighted(self):
        clips = [dict(path="a", language="de", condition="clean", text="eins"),
                 dict(path="b", language="de", condition="clean", text="eins zwei drei vier")]
        outputs = [dict(path="a", text="falsch", audio_sec=1, proc_ms=100),
                   dict(path="b", text="eins zwei drei vier", audio_sec=1, proc_ms=100)]
        group = score_group(clips, outputs)["de/clean"]
        self.assertEqual(group["wer_pct"], 20)
        self.assertEqual(group["processing_rtf"], 0.1)
        self.assertEqual(group["rtfx"], 10)

    def test_output_order_and_count_must_match(self):
        clip = dict(path="a", language="de", condition="clean", text="eins")
        with self.assertRaises(ValueError):
            score_group([clip], [])
        with self.assertRaises(ValueError):
            score_group([clip], [dict(path="b")])

    def test_invalid_timing_and_empty_reference_rejected(self):
        clip = dict(path="a", language="de", condition="clean", text="eins")
        result = dict(path="a", text="eins", audio_sec=1, proc_ms=float("nan"))
        with self.assertRaises(ValueError):
            score_group([clip], [result])
        result["proc_ms"] = 100
        clip["text"] = ""
        with self.assertRaises(ValueError):
            score_group([clip], [result])

    def test_seeded_noise_has_requested_snr(self):
        audio = np.random.default_rng(1).normal(0, 0.05, 16000)
        noisy, clipped = noisy_pcm(audio, 42)
        self.assertEqual(clipped, 0)
        self.assertTrue(np.array_equal(noisy, noisy_pcm(audio, 42)[0]))
        ratio = np.mean(audio ** 2) / np.mean((noisy - audio) ** 2)
        self.assertAlmostEqual(10 * np.log10(ratio), 5, places=6)

    def test_fixture_pair_deterministic_and_bounded(self):
        raw = io.BytesIO()
        sf.write(raw, np.zeros(16000 * 3), 16000, format="WAV", subtype="PCM_16")
        with tempfile.TemporaryDirectory() as directory:
            args = (Path(directory), "de_de", ["1", "../sample.wav", "", "eins"], raw.getvalue())
            pair = fixture_pair(*args)
            self.assertEqual(pair, fixture_pair(*args))
            self.assertEqual(len(pair), 2)
            self.assertTrue(all(Path(clip["path"]).parent == Path(directory).resolve() for clip in pair))

    def test_invalid_fixture_rate_is_rejected(self):
        raw = io.BytesIO()
        sf.write(raw, np.zeros(8000 * 3), 8000, format="WAV", subtype="PCM_16")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                fixture_pair(Path(directory), "de_de", ["1", "sample.wav", "", "eins"], raw.getvalue())

    def test_process_failure_not_silently_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subprocess.CalledProcessError):
                execute(["/bin/sh", "-c", "exit 7"], Path(directory) / "bad", 1)

    def test_timeout_kills_owned_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = ["/bin/sh", "-c", 'echo $$ > "$1"; exec sleep 30', "spike", str(root / "pid")]
            with self.assertRaises(subprocess.TimeoutExpired):
                execute(command, root / "timeout", 0.3)
            pid = int((root / "pid").read_text())
            # The child is dead (may briefly remain as a zombie awaiting its parent's reaper).
            state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
            self.assertTrue(not state.strip() or state.strip().startswith("Z"), state)

    def test_native_negative_silence_gate(self):
        """Opt-in real-model gate; heads currently FAIL it, Silero passes."""
        cli, model, audio = (os.environ.get(name) for name in ("SPIKE_CLI", "SPIKE_VAD_MODEL", "SPIKE_SILENCE"))
        if not all((cli, model, audio)):
            self.skipTest("Explicit native CLI/model/silence fixture required")
        document = json.loads(subprocess.check_output(
            [cli, "vad", "--model", model, "--input", audio, "--threads", "4",
             "--threshold", "0.5", "--min-speech", "0.1", "--min-pause", "0.2", "--speech-pad", "0"], timeout=30))
        self.assertEqual(document["segments"], [], "No-speech clip falsely opened speech gate")

    def test_streaming_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hash"
            path.write_bytes(b"abc")
            self.assertEqual(sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
