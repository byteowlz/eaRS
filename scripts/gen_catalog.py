# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "pyyaml"]
# ///
"""Generate eaRS's transcribe.cpp model catalog from the handy-computer HF org.

The catalog is *data, not judgment*: everything here is mechanically derived
from each model's HF card, except a tiny CURATION overlay (recommended flag +
hand-written descriptions) that a stronger model cannot infer.

For every `handy-computer/*-gguf` repo whose card declares
`library_name: transcribe.cpp` and `transcribe_cpp.streaming: true`, it reads:
  - `language:`                    -> languages label
  - `transcribe_cpp.streaming`     -> streaming flag (filter)
  - the GGUF file list             -> quants + default-quant size

and writes `src/server/catalog.json`, which is `include_str!`'d into the binary
and also served to `ears models refresh`.

Run:  HF_TOKEN=$(hf auth token) uv run scripts/gen_catalog.py [out_path]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_url, list_models
from huggingface_hub.utils import get_session

ORG = "handy-computer"
CATALOG_VERSION = 1
DEFAULT_QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "F16", "F32"]

# Editorial overlay: slug -> {recommended?, description?, default_quant?, hidden?}.
# The only non-derivable data. Unlisted models still ship, with a generated
# description and Q8_0 default.
CURATION: dict[str, dict] = {
    "multitalker-parakeet-streaming-0.6b-v1": {
        "recommended": True,
        "name": "Multitalker Parakeet 0.6B",
        "description": "Fast, accurate English streaming (recommended)",
    },
    "nemotron-3.5-asr-streaming-0.6b": {
        "recommended": True,
        "name": "Nemotron Streaming 3.5",
        "description": "Multilingual streaming across 28 languages",
    },
    "parakeet-unified-en-0.6b": {
        "name": "Parakeet Unified EN 0.6B",
        "description": "NVIDIA unified English streaming model",
    },
    "nemotron-speech-streaming-en-0.6b": {
        "name": "Nemotron Speech Streaming EN",
        "description": "English-only Nemotron streaming variant",
    },
    "Voxtral-Mini-4B-Realtime-2602": {
        "name": "Voxtral Mini 4B Realtime",
        "default_quant": "Q5_K_M",
        "description": "Highest accuracy, ~4.7x realtime; heavier 4B model",
    },
    # Benchmarked unusable for streaming dictation (17-74% WER); hide from the
    # curated catalog while leaving them reachable by explicit slug.
    "moonshine-streaming-tiny": {"hidden": True},
    "moonshine-streaming-small": {"hidden": True},
    "moonshine-streaming-medium": {"hidden": True},
}


def card_front_matter(api: HfApi, repo_id: str) -> dict | None:
    """Parse the YAML front matter of a repo's README.md, or None."""
    url = hf_hub_url(repo_id, "README.md")
    resp = get_session().get(url, headers=api._build_hf_headers())
    if resp.status_code != 200:
        return None
    text = resp.text
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None


def languages_label(langs: list[str]) -> str:
    if not langs:
        return "Unknown"
    if len(langs) == 1:
        return "English" if langs[0] == "en" else langs[0]
    return f"Multilingual ({len(langs)})"


def pick_default_quant(files: dict[str, int], curated: str | None) -> str:
    if curated and curated in files:
        return curated
    for q in DEFAULT_QUANT_ORDER:
        if q in files:
            return q
    return next(iter(files), "Q8_0")


def quant_files(api: HfApi, repo_id: str, slug: str) -> dict[str, int]:
    """Map quant -> size_bytes for the repo's GGUF files."""
    info = api.model_info(repo_id, files_metadata=True)
    out: dict[str, int] = {}
    for sib in info.siblings or []:
        name = sib.rfilename
        m = re.fullmatch(rf"{re.escape(slug)}-(.+)\.gguf", name)
        if m and sib.size:
            out[m.group(1)] = sib.size
    return out


def slug_to_name(slug: str) -> str:
    return slug.replace("-", " ").title()


def build() -> list[dict]:
    api = HfApi()
    models: list[dict] = []
    for m in list_models(author=ORG):
        repo_id = m.id
        if not repo_id.endswith("-gguf"):
            continue
        slug = repo_id[len(f"{ORG}/") :].removesuffix("-gguf")
        fm = card_front_matter(api, repo_id)
        if not fm or fm.get("library_name") != "transcribe.cpp":
            continue
        tc = fm.get("transcribe_cpp") or {}
        if not tc.get("streaming"):
            continue  # eaRS only serves streaming models
        files = quant_files(api, repo_id, slug)
        if not files:
            continue
        cur = CURATION.get(slug, {})
        if cur.get("hidden"):
            continue
        quant = pick_default_quant(files, cur.get("default_quant"))
        langs = fm.get("language") or []
        if isinstance(langs, str):
            langs = [langs]
        models.append(
            {
                "slug": slug,
                "name": cur.get("name") or slug_to_name(slug),
                "languages": languages_label(langs),
                "streaming": True,
                "default_quant": quant,
                "default_size_mb": round(files[quant] / 1_000_000),
                "description": cur.get("description")
                or f"transcribe.cpp streaming model ({languages_label(langs)})",
                "recommended": bool(cur.get("recommended")),
            }
        )
    # Recommended first, then by name for a stable diff.
    models.sort(key=lambda m: (not m["recommended"], m["name"]))
    return models


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/server/catalog.json")
    catalog = {"catalog_version": CATALOG_VERSION, "models": build()}
    out.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"wrote {len(catalog['models'])} models -> {out}")


if __name__ == "__main__":
    main()
