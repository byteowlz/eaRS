"""Generate synthetic shadow-only completed-utterance intent examples.

The remote generator sees only fixed generic prompts: NEVER send real dictation
history, private registries or audio to this service. Labels come from prompts,
not from a human annotation or an independent held-out voice evaluation.
"""

import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LABELS = {
    "direct": "command_intent",
    "literal": "literal_dictation",
    "quoted": "literal_dictation",
    "near": "uncertain",
    "correction": "uncertain",
}
CONTEXT = {
    "train": "home chores, errands, appointments, and drafting personal notes",
    "validation": "recipes, trip planning, weather, and reminders",
    "test": "workspaces, reading, messages, and daily schedules",
}
INSTRUCTIONS = {
    "direct": "A clearly addressed instruction to the assistant, such as open notes, insert greeting, or show calendar. Mix exact phrases with polite paraphrases. Never quote a command as prose.",
    "literal": "Ordinary dictated prose. It may discuss notes, a calendar, or greetings but must NOT ask the assistant to do anything. Include some sentences with command-like words used as ordinary content.",
    "quoted": "Dictated prose that explicitly QUOTES a command phrase (e.g. 'open notes') or reports someone saying it; the speaker is talking ABOUT the words, not issuing the command.",
    "correction": "Unresolved self-correction or change of mind about a possible command (e.g. 'open notes no wait never mind'). Avoid any clean unambiguous final command.",
}
# The remote model repeatedly generated clean imperative requests when asked for
# ambiguous near commands. Keep these auditable, hand-authored perturbations
# rather than laundering wrong labels into the training set.
NEAR_CASES = [
    "open note", "open nodes", "oh pen notes", "open nots", "open noats",
    "open notes uh", "open note s", "open n otes", "open notes no", "open gnots",
    "open uh notes", "open note no", "open not", "open node", "oh pen note",
    "insert greetings", "in sert greeting", "insert greet", "insert greating",
    "insurt greeting", "insert greeting uh", "insert greet ing", "insert greeting no",
    "in search greeting", "inset greeting", "insert great ing", "insert great",
    "show calender", "show call ender", "show calen", "sure calendar",
    "show calendars", "show calandar", "show calendar uh", "show calendar no",
    "show can lender", "show callendar", "show calender no", "sho calendar",
    "ee ars cancel", "ear cancel", "eaRS canceled", "ease cancel",
    "eaRS cancelling", "eaRS can sell", "eaRS cancel no", "eaRS can cel",
    "ee ars start typing", "ear start typing", "eaRS start type", "eaRS start typin",
    "eaRS start tying", "eaRS stars typing", "eaRS start typing uh",
    "eaRS start typing no", "eaRS start typing maybe", "eaRS restart typing",
]
MODES = ("dictation", "command_aware", "command_only")
COUNT = 10  # Matches the audited cache and fixed near-command split.


def request_items(provider, prompt, minimum=10, max_tokens=800):
    """One local provider call; the caller owns scenario labels and prompt text."""
    body = json.dumps({
        "model": provider["models"][0]["id"], "temperature": 0.9, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": "Return only valid JSON with synthetic text."},
                     {"role": "user", "content": prompt}],
    }).encode()
    request = urllib.request.Request(
        provider["baseUrl"].rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {provider['apiKey']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = json.load(response)["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    items = json.loads(content)["items"]
    if not isinstance(items, list) or len(items) < minimum:
        raise ValueError("incomplete generator response")
    if any(not isinstance(t, str) or not 4 <= len(t.strip()) <= 240 for t in items):
        raise ValueError("invalid utterance from generator")
    return [t.strip() for t in items]


def query(provider, split, kind, batch):
    # No private data appears in this prompt; the generator is not the judge.
    prompt = (f"Create {COUNT} diverse, short, natural English *completed-utterance STT texts* "
              f"for an offline voice-intent experiment. Scenario: {INSTRUCTIONS[kind]} "
              f"Use topics: {CONTEXT[split]}. Batch style seed {batch * 19 + len(split)}. "
              "Return ONLY a JSON object with key 'items', an array of strings. "
              "No labels, no explanations, no personal data, no markup. "
              "Avoid repeating grammatical openings or using identical examples.")
    return request_items(provider, prompt)[:COUNT]


def batch_texts(provider, cache, split, kind, batch):
    if kind == "near":
        offset = 0 if split == "train" else 3 * COUNT if split == "validation" else 4 * COUNT
        texts = NEAR_CASES[offset + batch * COUNT:offset + (batch + 1) * COUNT]
        if len(texts) != COUNT:
            raise ValueError("not enough fixed near-command examples")
        return texts
    path = cache / f"{split}-{kind}-{batch}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            texts = query(provider, split, kind, batch)
            path.write_text(json.dumps(texts, ensure_ascii=False))
            return texts
        except (ValueError, KeyError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"generation failed for {split}/{kind}/{batch}") from exc
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


class DatasetBuilder:
    def __init__(self, provider, cache):
        self.provider = provider
        self.cache = cache
        self.rows = []
        self.seen = set()

    def append_family(self, split, kind):
        batches = (3 if kind == "near" else 4) if split == "train" else 1
        for batch in range(batches):
            for text in batch_texts(self.provider, self.cache, split, kind, batch):
                key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
                if key in self.seen:
                    continue
                self.seen.add(key)
                self.rows.append({"text": text, "mode": MODES[(len(self.rows) + batch) % len(MODES)],
                                  "label": LABELS[kind], "family": kind, "split": split,
                                  "source": "synthetic-handwritten" if kind == "near" else "synthetic-llm",
                                  "model": self.provider["models"][0]["id"]})

    def collect(self):
        for split in CONTEXT:
            for kind in LABELS:
                before = len(self.rows)
                self.append_family(split, kind)
                print(f"{split}/{kind}: {len(self.rows) - before}", flush=True)
        random.Random(42).shuffle(self.rows)
        return self.rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / ".pi/agent/models.json")
    parser.add_argument("--provider", default="rtx6000")
    parser.add_argument("--out", type=Path, default=HERE / "artifacts/intent-data.jsonl")
    args = parser.parse_args()
    provider = json.loads(args.config.read_text())["providers"][args.provider]
    cache = args.out.parent / "intent-generation-cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = DatasetBuilder(provider, cache).collect()
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} unique synthetic examples to {args.out}; NOT real held-out speech", flush=True)


if __name__ == "__main__":
    main()
