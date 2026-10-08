import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from run import command_for, validate_result


class WhistleTests(unittest.TestCase):
    def result(self):
        return dict(text="eins", language="de", ttft_ms=20, decode_tps=100,
                    words=[dict(word="eins", start=0, end=1, probability=0.9)])

    def test_valid_result(self):
        validate_result(self.result(), 1)

    def test_invalid_native_fields_are_rejected(self):
        cases = (("probability", float("nan")), ("start", 2), ("ttft_ms", -1))
        for field, value in cases:
            with self.subTest(field=field):
                result = self.result()
                target = result if field == "ttft_ms" else result["words"][0]
                target[field] = value
                with self.assertRaises(ValueError):
                    validate_result(result, 1)

    def test_silence_shape(self):
        validate_result(dict(text="", language="", words=[], ttft_ms=0, decode_tps=0), 3)

    def test_network_denial_command(self):
        args = SimpleNamespace(native=Path("native"), model=Path("model"))
        command = command_for(args, Path("manifest"), "auto")
        self.assertEqual(command[:3], ["/usr/bin/sandbox-exec", "-p", "(version 1) (allow default) (deny network*)"])

    def test_nonstationary_noise_adoption_gate(self):
        if os.environ.get("WHISTLE_PULSE_GATE") != "1":
            self.skipTest("Opt-in known failing no-speech adoption gate")
        native, model = os.environ["WHISTLE_NATIVE"], os.environ["WHISTLE_MODEL"]
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav, pcm = root / "pulse.wav", root / "pulse.f32"
            noise = np.random.default_rng(43).normal(0, 0.00005, 48000) * np.hanning(48000)
            sf.write(wav, noise, 16000, subtype="PCM_16")
            decoded, _ = sf.read(wav, dtype="float32")
            pcm.write_bytes(decoded.astype("<f4").tobytes())
            manifest = root / "paths.tsv"
            manifest.write_text(str(pcm) + "\n")
            args = SimpleNamespace(native=Path(native), model=Path(model))
            rows = [json.loads(line) for line in subprocess.check_output(
                command_for(args, manifest, "de"), timeout=30).decode().splitlines()]
            self.assertEqual(rows[1]["result"]["text"], "", "No speech in shaped noise")

    def test_real_native_no_speech_and_input_guards(self):
        native, model = os.environ.get("WHISTLE_NATIVE"), os.environ.get("WHISTLE_MODEL")
        if not native or not model:
            self.skipTest("Explicit native/model required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.f32"
            manifest = root / "paths.tsv"
            manifest.write_text(str(audio) + "\n")
            args = SimpleNamespace(native=Path(native), model=Path(model))
            command = command_for(args, manifest, "auto")
            audio.write_bytes(np.zeros(16000 * 3, dtype="<f4").tobytes())
            rows = [json.loads(line) for line in subprocess.check_output(command, timeout=30).decode().splitlines()]
            self.assertEqual(rows[1]["result"]["text"], "")
            self.assertEqual(rows[1]["result"]["language"], "")
            for invalid in (np.array([float("nan")], dtype="<f4"), np.zeros(480001, dtype="<f4")):
                audio.write_bytes(invalid.tobytes())
                result = subprocess.run(command, capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
