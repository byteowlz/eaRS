#!/usr/bin/env python3
"""Remove hesitation fillers from German reference text; detection only, no audio edit.

The filler list follows the video-editing detector (recording-edit/scripts/redit.py)
but drops English-only homographs that are ordinary German words ("um" = in order to,
"er" = he, "ah"/"mm" as lexical tokens), so it does not corrupt German transcripts.

Both the verbatim and the cleaned text are kept; nothing is rewritten beyond removing a
standalone filler token. Punctuation of neighbouring words is preserved, and a sentence
that starts with a filler keeps its capitalisation.
"""
import re

# Tokens removed when they stand alone. "hm"/"hmm" are hesitations in German too.
DE_FILLERS = frozenset({"äh", "ähm", "öh", "ehm", "ähem", "hm", "hmm", "mhm", "uh", "uhm"})
# English-oriented set from the video editor, kept only for diagnostics.
EN_ONLY = frozenset({"um", "er", "ah", "mm", "erm"})
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def token(text: str) -> str:
    """Normalised comparison form: lowercase letters only, punctuation and digits dropped."""
    return "".join(WORD.findall(text)).lower()


def is_filler(text: str) -> bool:
    return token(text) in DE_FILLERS


def clean(text: str) -> tuple[str, list[str]]:
    """Return (cleaned text, removed tokens) with only standalone fillers removed."""
    removed: list[str] = []
    words = text.split()
    kept: list[str] = []
    for index, word in enumerate(words):
        if is_filler(word):
            removed.append(word)
            continue
        if not kept and removed:
            # Sentence-initial filler removed: capitalise the new first letter of a word.
            for i, char in enumerate(word):
                if char.isalpha():
                    word = word[:i] + char.upper() + word[i + 1:]
                    break
        kept.append(word)
    return " ".join(kept), removed


if __name__ == "__main__":
    import json, sys
    path = sys.argv[1]
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    total = 0
    for row in rows:
        cleaned, removed = clean(row["human_reference"])
        if removed:
            total += len(removed)
            print(json.dumps({"clip_id": row["clip_id"], "removed": removed}, ensure_ascii=False))
    print(json.dumps({"clips": len(rows), "removed_tokens": total}), file=sys.stderr)
