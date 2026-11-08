"""Reproducibly pack the canonical patched upstream source and built WASM.

Maintenance only: reads the selected Git commit; never edits/extracts source,
clones a repository, downloads data, or publishes an artifact.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]
REVISION = "5a1358d3253b81ef24ae2a23cb25fe47bb16b783"
UPSTREAM = "234f60faa0eb388b01cdf7e60aca232af37aefda"
VERSION = "0.1.5-ears.1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output, gzip.GzipFile(fileobj=output, filename="", mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, data in sorted(entries.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))


def git_bytes(source, name):
    data = subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=source)
    if data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        expected = data.decode().split("oid sha256:")[1].splitlines()[0]
        data = (source / name).read_bytes()
        if sha(data) != expected:
            raise ValueError(f"Missing/incorrect LFS materialization: {name}")
    return data


def source_bundle(source, target):
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REVISION], cwd=source, text=True).splitlines()
    included = [name for name in names if name.startswith(("core/", "language-bindings/wasm/"))
                and not name.startswith("core/third-party/onnxruntime/lib/")]
    entries = {"moonshine/" + name: git_bytes(source, name) for name in included}
    # Build integration only: retain the pinned external ORT path across nested
    # CMake includes, and keep Linux shared-runtime packaging relocatable.
    cmake = "moonshine/core/third-party/onnxruntime/find-ort-library-path.cmake"
    detection, helpers = entries[cmake].decode().split("function(copy_onnxruntime_dll", 1)
    entries[cmake] = ("if(NOT DEFINED ONNXRUNTIME_LIB_PATH)\n" + detection
                      + "\nendif()\nfunction(copy_onnxruntime_dll" + helpers).encode()
    tts = "moonshine/core/moonshine-tts/cmake/moonshine-tts-onnxruntime.cmake"
    entries[tts] = entries[tts].replace(b'INSTALL_RPATH "${_moonshine_tts_ort_rpath}"',
                                      b'INSTALL_RPATH "$ORIGIN"')
    entries["moonshine/LICENSE"] = git_bytes(source, "LICENSE")
    entries["moonshine/test-assets/README.md"] = b"Models and recordings are not vendored. Tests require explicit local fixtures.\n"
    archive(target / "core-source.tar.gz", entries)
    critical = ("core/transcriber.cpp", "core/moonshine-streaming-model.cpp", "core/moonshine-streaming-model.h",
                "core/third-party/onnxruntime/find-ort-library-path.cmake",
                "core/moonshine-tts/cmake/moonshine-tts-onnxruntime.cmake")
    provenance = {"version": VERSION, "upstreamRevision": UPSTREAM, "coreRevision": REVISION,
                  "sourceSha256": sha((target / "core-source.tar.gz").read_bytes()),
                  "criticalSources": {name: sha(entries["moonshine/" + name]) for name in critical}}
    return provenance


def ort_pins(source):
    groups = {
        "aarch64-apple-darwin": ("static", ["macos/arm64/libonnxruntime.a"]),
        "x86_64-apple-darwin": ("shared", ["macos/x86_64/libonnxruntime.1.23.2.dylib"]),
        "x86_64-unknown-linux-gnu": ("shared", ["linux/x86_64/libonnxruntime.so.1"]),
        "aarch64-unknown-linux-gnu": ("shared", ["linux/aarch64/libonnxruntime.so.1"]),
        "x86_64-pc-windows-msvc": ("shared", ["windows/x86_64/onnxruntime.lib", "windows/x86_64/onnxruntime.dll"]),
    }
    pins = {}
    for target, (mode, paths) in groups.items():
        files = []
        for relative in paths:
            name = "core/third-party/onnxruntime/lib/" + relative
            raw = subprocess.check_output(["git", "show", f"{UPSTREAM}:{name}"], cwd=source)
            data = git_bytes(source, name)
            host = "media.githubusercontent.com/media" if raw.startswith(b"version https://git-lfs.github.com/spec/v1") else "raw.githubusercontent.com"
            files.append({"name": Path(relative).name, "sha256": sha(data), "bytes": len(data),
                          "url": f"https://{host}/moonshine-ai/moonshine/{UPSTREAM}/{name}"})
        pins[target] = {"mode": mode, "files": files}
    return pins


def wasm_bundle(source, build, target, provenance):
    dist = source / "language-bindings/wasm/dist"
    entries = {"dist/" + p.name: p.read_bytes() for p in dist.iterdir() if p.suffix in (".js", ".mjs", ".wasm")}
    for name in ("moonshine.mjs", "moonshine.wasm"):
        entries["dist/" + name] = (build / name).read_bytes()
    if str(Path.home()).encode() in entries["dist/moonshine.wasm"]:
        raise ValueError("WASM embeds the local build home; rebuild with -ffile-prefix-map=$HOME=/build")
    entries["README.md"] = f"Moonshine {VERSION}: pinned eaRS final-audio repair. No ASR models included.\n".encode()
    archive(target / "runtime.tar.gz", entries)
    manifest = dict(provenance, sha256=sha((target / "runtime.tar.gz").read_bytes()),
                    wasmSha256=sha(entries["dist/moonshine.wasm"]), toolchain="Emscripten 4.0.8",
                    pathRemapping="$HOME=/build")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wasm-build", type=Path, required=True)
    args = parser.parse_args()
    target = REPO / "vendor/moonshine-sys"
    provenance = source_bundle(args.source, target)
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (target / "ort-artifacts.json").write_text(json.dumps(ort_pins(args.source), indent=2) + "\n")
    wasm = REPO / "packages/browser/scripts/vendor/moonshine"
    wasm.mkdir(parents=True, exist_ok=True)
    wasm_bundle(args.source, args.wasm_build, wasm, provenance)
    for p in (target / "core-source.tar.gz", wasm / "runtime.tar.gz"):
        print(p.relative_to(REPO), p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
