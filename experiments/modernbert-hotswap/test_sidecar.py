"""Integration proof using only the public SlopBERT example-trained adapters."""

import json
import os
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
TEXT = "Nvidia announces new GPUs at GTC 2025"


class SidecarProof(unittest.TestCase):
    def test_select_reload_restore_and_fail_closed(self):
        for name in ("original", "inverted"):
            self.assertTrue((HERE / "artifacts" / name / "adapter_model.safetensors").is_file(),
                            "run train_demo.py before the live integration test")
        requests = [
            {"op": "classify", "adapter": "original", "text": TEXT},
            {"op": "classify", "adapter": "inverted", "text": TEXT},
            {"op": "replace", "adapter": "original", "source": "inverted"},
            {"op": "classify", "adapter": "original", "text": TEXT},
            {"op": "replace", "adapter": "original", "source": "original"},
            {"op": "classify", "adapter": "original", "text": TEXT},
            {"op": "classify", "adapter": "unknown", "text": TEXT},
            *({"op": "classify", "adapter": "inverted", "text": TEXT} for _ in range(16)),
        ]
        result = subprocess.run(
            [sys.executable, str(HERE / "sidecar.py")],
            input="\n".join(json.dumps(req) for req in requests) + "\n",
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            check=True,
        )
        outputs = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(outputs), len(requests))
        a, b, reload_result, after, restore_result, restored, invalid = outputs[:7]
        self.assertTrue(all(out["ok"] for out in outputs[:6]))
        self.assertNotEqual(a["scores"], b["scores"])
        self.assertEqual(after["scores"], b["scores"])
        self.assertEqual(restored["scores"], a["scores"])
        self.assertTrue(reload_result["replaced"] and restore_result["replaced"])
        self.assertEqual(invalid, {"ok": False, "error": "ValueError"})
        self.assertTrue(all(out["ok"] for out in outputs[7:]))
        timings = [out["elapsed_ms"] for out in outputs[7:]]
        print(f"16 warm in-process inferences (ms): median={statistics.median(timings):.2f} "
              f"max={max(timings):.2f}; cold first={a['elapsed_ms']:.2f}")

    def test_replacement_failure_stops_process(self):
        requests = [
            {"op": "replace", "adapter": "original", "source": "not-allowlisted"},
            {"op": "classify", "adapter": "original", "text": TEXT},
        ]
        result = subprocess.run(
            [sys.executable, str(HERE / "sidecar.py")],
            input="\n".join(json.dumps(req) for req in requests) + "\n",
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([json.loads(line) for line in result.stdout.splitlines()],
                         [{"ok": False, "error": "ValueError"}])
        self.assertNotIn(TEXT, result.stderr)


if __name__ == "__main__":
    unittest.main()
