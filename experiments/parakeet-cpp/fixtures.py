"""Small pinned public FLEURS subset; never extracts untrusted archive paths."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
import soundfile as sf

FLEURS_REVISION = "d7c758a6dceecd54a98cac43404d3d576e721f07"
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data"


def noisy_pcm(samples, seed, snr_db=5):
    """Seeded white Gaussian noise, whole-clip RMS, no gain normalization."""
    noise = np.random.default_rng(seed).normal(size=len(samples))
    noise *= np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / (
        np.sqrt(np.mean(noise ** 2)) * 10 ** (snr_db / 20)
    )
    mixed = samples + noise
    return np.clip(mixed, -1, 1), float(np.mean(np.abs(mixed) > 1))


def fixture_pair(out, language, row, raw):
    samples, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if sr != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Expected finite mono FLEURS audio at 16kHz")
    duration = len(samples) / sr
    if not 3 <= duration <= 30:
        return []
    name = Path(row[1]).name
    identity = f"{language}-{name.removesuffix('.wav')}"
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
    mixed, clipped = noisy_pcm(samples, seed)
    collected = []
    for condition, audio in (("clean", samples), ("white5db", mixed)):
        file = out / f"{identity}-{condition}.wav"
        sf.write(file, audio, sr, subtype="PCM_16")
        collected.append({
            "id": f"{identity}-{condition}", "path": str(file.resolve()),
            "language": language, "condition": condition,
            "text": row[3], "duration_s": duration,
            "source_file": name, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "noise_seed": seed if condition != "clean" else None,
            "clipped_fraction": clipped if condition != "clean" else 0,
        })
    return collected


def collect_language(out, language, count):
    with urllib.request.urlopen(f"{BASE}/{language}/test.tsv", timeout=60) as response:
        metadata = response.read()
    refs = {row[1]: row for row in csv.reader(io.StringIO(metadata.decode()), delimiter="\t")}
    collected = []
    with urllib.request.urlopen(f"{BASE}/{language}/audio/test.tar.gz", timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name not in refs:
                    continue
                if member.size > 10_000_000:
                    raise ValueError("Oversized fixture")
                raw = archive.extractfile(member).read()
                collected.extend(fixture_pair(out, language, refs[name], raw))
                if len(collected) == count * 2:
                    break
    if len(collected) != count * 2:
        raise ValueError(f"Insufficient {language} fixtures")
    return collected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("--count must be 1..20 (bounded spike)")
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for language in ("de_de", "en_us"):
        entries.extend(collect_language(args.out, language, args.count))
    manifest = {"dataset": "google/fleurs", "revision": FLEURS_REVISION,
                "split": "test", "license": "CC-BY-4.0", "clips": entries}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    sf.write(args.out / "silence.wav", np.zeros(16000 * 3), 16000, subtype="PCM_16")
    sf.write(args.out / "noise.wav", np.random.default_rng(42).normal(0, 0.015, 16000 * 3),
             16000, subtype="PCM_16")
    print(f"Prepared {len(entries)} clips plus silence/noise controls in {args.out}")


if __name__ == "__main__":
    main()
