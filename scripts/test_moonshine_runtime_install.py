"""Shared-runtime packaging contract; only synthetic files and mock uname."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().with_name("install-moonshine-runtime.sh")


class RuntimeInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        uname = self.bin / "uname"
        uname.write_text('#!/usr/bin/env bash\nif [[ "$1" == -s ]]; then echo "$TEST_OS"; else echo "$TEST_ARCH"; fi\n')
        uname.chmod(0o755)
        self.target = self.root / "target"
        self.destination = self.root / "installed"
        self.env = dict(os.environ, PATH=str(self.bin) + ":" + os.environ["PATH"],
                        TEST_OS="Linux", TEST_ARCH="x86_64")

    def run_install(self):
        return subprocess.run(["bash", str(SCRIPT), str(self.destination), str(self.target)],
                              env=self.env, text=True, capture_output=True)

    def seed(self, names):
        runtime = self.target / "release/build/moonshine-sys-test/out/runtime"
        runtime.mkdir(parents=True)
        for name in names:
            (runtime / name).write_bytes(name.encode())

    def test_linux_installs_the_built_vendored_pair(self):
        names = ["libmoonshine.so", "libonnxruntime.so.1"]
        self.seed(names)
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in names:
            self.assertEqual((self.destination / name).read_bytes(), name.encode())

    def test_stock_prebuilt_is_not_an_install_fallback(self):
        old = self.target / "release/build/moonshine-sys-old/out/moonshine-prebuilt/moonshine-voice-linux-x86_64/lib"
        old.mkdir(parents=True)
        for name in ["libmoonshine.so", "libonnxruntime.so.1"]:
            (old / name).write_bytes(b"unpatched")
        self.assertNotEqual(self.run_install().returncode, 0)
        self.assertFalse(self.destination.exists())

    def test_incomplete_shared_pair_cannot_install(self):
        self.seed(["libmoonshine.so"])
        self.assertNotEqual(self.run_install().returncode, 0)
        self.assertFalse(self.destination.exists())

    def test_intel_mac_installs_its_shared_pair(self):
        self.env.update(TEST_OS="Darwin", TEST_ARCH="x86_64")
        names = ["libmoonshine.dylib", "libonnxruntime.1.23.2.dylib"]
        self.seed(names)
        self.assertEqual(self.run_install().returncode, 0)
        self.assertEqual(sorted(p.name for p in self.destination.iterdir()), sorted(names))

    def test_apple_silicon_static_framework_needs_no_shared_install(self):
        self.env.update(TEST_OS="Darwin", TEST_ARCH="arm64")
        self.assertEqual(self.run_install().returncode, 0)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
