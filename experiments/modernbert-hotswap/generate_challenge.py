"""Freeze a larger, synthetic-only voice-intent challenge after training.

No private data or previous examples are sent to the generator. Its scenario
labels are weak supervision, not real annotations or held-out speech.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generate_intent import request_items

HERE = Path(__file__).parent
TARGETS = {"direct": 220, "literal": 450, "quoted": 270, "correction": 160}
LABELS = {"direct": "command_intent", "literal": "literal_dictation",
          "quoted": "literal_dictation", "correction": "uncertain"}
TOPICS = ("museum guides and books", "sports and outdoor plans", "recipes and music",
          "letters and small projects", "household logistics and train travel",
          "podcasts and local events", "personal schedules and hobbies")
COMMANDS = "open notes, insert greeting, show calendar, eaRS cancel, eaRS start typing"


def key(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def valid(text, family):
    if not isinstance(text, str) or not 5 <= len(text.strip()) <= 240:
        return False
    lower = text.strip().lower()
    if family == "literal":
        return not re.match(r"^(please\s+)?(open|show|insert|write|make|create|start|stop|cancel|remind|draft)\b", lower)
    if family == "quoted":
        return bool(re.search(r"['\"][^'\"]*(open notes|insert greeting|show calendar|ears cancel|ears start typing)[^'\"]*['\"]", lower)) and not lower.startswith("please ")
    if family == "correction":
        return bool(re.search(r"\b(no wait|never mind|scratch that|cancel that|hold on|wait stop|actually don't|forget that|not yet|i changed my mind)\b", lower))
    return bool(re.search(r"\b(open|show|insert|start typing|cancel|set|add|bring|pull|turn|draft|make|create)\b", lower)) and not re.search(r"\b(never mind|scratch that|hold on)\b", lower)


def prompt(family, batch):
    topic = TOPICS[batch % len(TOPICS)]
    tag = f"style {batch}: {('brief spoken phrasing' if batch % 2 else 'natural disfluent speech')}"
    common = ("Write ten DISTINCT generic English completed-utterance speech transcripts for a local "
              "classification stress test. No real people, institutions, URLs, or private facts. "
              f"Topic: {topic}; {tag}. Return ONLY JSON with key 'items' containing an array of ten strings. ")
    scenarios = {
        "direct": ("Each is an UNAMBIGUOUS instruction addressed to a voice assistant, often requesting "
                   f"an app action. The generic available phrases are {COMMANDS}. Vary the wording, "
                   "including paraphrases. Do not quote an instruction as prose."),
        "literal": ("Each is DECLARATIVE dictated content for a note or message, NOT a request "
                    "for the assistant to perform an action. Include vocabulary about notes, calendar, "
                    "typing, opening, reminders, or greetings naturally in some sentences. No commands."),
        "quoted": ("Each is longer DECLARATIVE prose reporting or quoting EXACTLY one of these "
                   f"phrases: {COMMANDS}. Put the phrase in single or double quotation marks. "
                   "It is reported speech, not an instruction to the assistant. Include context after the quote."),
        "correction": ("Each begins to issue a command, then visibly cancels or hesitates before completion: "
                       "include 'no wait', 'never mind', 'scratch that', 'hold on', or 'cancel that'. "
                       "Do not end in an unambiguous new command."),
    }
    return common + scenarios[family]


def request_batch(provider, family, batch, cache):
    path = cache / f"{family}-{batch:03d}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            items = request_items(provider, prompt(family, batch), minimum=8, max_tokens=1000)
            path.write_text(json.dumps(items, ensure_ascii=False))
            return items
        except (ValueError, KeyError, TimeoutError, urllib.error.URLError) as exc:
            if attempt == 2:
                raise RuntimeError(f"generator failed for {family} batch {batch}") from exc
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def append_valid(rows, seen, texts, family, accepted, target, model_id):
    for text in texts:
        if accepted >= target or not valid(text, family) or key(text) in seen:
            continue
        seen.add(key(text))
        rows.append({"text": text.strip(), "mode": ("dictation", "command_aware", "command_only")[accepted % 3],
                     "label": LABELS[family], "family": family, "split": "challenge",
                     "source": "synthetic-llm", "model": model_id})
        accepted += 1
    return accepted


def collect(provider, cache, prior):
    seen, rows = set(prior), []
    for family, target in TARGETS.items():
        accepted, batch = 0, 0
        while accepted < target and batch < target // 3:
            numbers = range(batch, batch + 6)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(request_batch, provider, family, n, cache): n for n in numbers}
                results = {futures[f]: f.result() for f in as_completed(futures)}
            for n in numbers:
                accepted = append_valid(rows, seen, results[n], family, accepted, target,
                                        provider["models"][0]["id"])
            batch += 6
            print(f"{family}: {accepted}/{target} accepted ({batch} batches)", flush=True)
        if accepted < target:
            raise RuntimeError(f"could only produce {accepted} distinct {family} examples")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / ".pi/agent/models.json")
    parser.add_argument("--provider", default="rtx6000")
    parser.add_argument("--prior", type=Path, default=HERE / "artifacts/intent-data.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/intent-challenge.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("challenge is already frozen; refusing to overwrite it")
    provider = json.loads(args.config.read_text())["providers"][args.provider]
    prior = {key(json.loads(line)["text"]) for line in args.prior.read_text().splitlines() if line.strip()}
    cache = args.output.parent / "intent-challenge-cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = collect(provider, cache, prior)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    print(f"frozen {len(rows)} synthetic examples; sha256={digest}; not real held-out speech", flush=True)


if __name__ == "__main__":
    main()
