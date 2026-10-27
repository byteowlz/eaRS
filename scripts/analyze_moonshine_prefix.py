#!/usr/bin/env python3
"""Offline Moonshine prefix-policy replay; never types or changes live routing.

uv run scripts/analyze_moonshine_prefix.py --trace trace.jsonl --out results.json

The oracle is each native line's *own final text*, not a ground-truth transcript.
A cached poll is not an independent agreement. Exact case/punctuation matter:
anything already typed is irreversible. A contradiction freezes the simulated
line; the script does not quietly repair it with the later final transcript.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


POLICIES = [
    {"name": "eager-tail1", "updates": 1, "tail": 1, "age": 0.0},
    {"name": "agree2-tail1", "updates": 2, "tail": 1, "age": 0.0},
    {"name": "agree3-tail1", "updates": 3, "tail": 1, "age": 0.0},
    {"name": "agree3-tail2-age1s", "updates": 3, "tail": 2, "age": 1.0},
    {"name": "agree3-tail2-age2s", "updates": 3, "tail": 2, "age": 2.0},
]


def common_prefix(sequences):
    if not sequences:
        return 0
    for i, words in enumerate(zip(*sequences)):
        if len(set(words)) != 1:
            return i
    return min(map(len, sequences))


def lexical(tokens):
    # Preserve German ß (casefold would conflate it with ss). Ignore standalone
    # punctuation tokens so formatting changes do not look like shifted words.
    words = ["".join(c for c in token.lower() if c.isalnum()) for token in tokens]
    return [word for word in words if word]


def read_trace(path):
    groups = defaultdict(list)
    metadata = None
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        if row["kind"] == "model":
            metadata = row
        if row["kind"] != "snapshot":
            continue
        for line in row["lines"]:
            key = (row["clip"], line["id"])
            events = groups[key]
            event = {"time": row["audio_s"], "tokens": line["text"].split(),
                     "complete": line["complete"], "updated": line["updated"]}
            if not event["tokens"] and not events:
                continue
            # Retain a genuine inference update, a revision, or a finalization;
            # not hundreds of cached polls of an unchanged hypothesis.
            if (not events or event["updated"] or event["tokens"] != events[-1]["tokens"]
                    or event["complete"] != events[-1]["complete"]):
                events.append(event)
    if metadata is None:
        raise ValueError(f"{path}: missing model/runtime metadata")
    groups = {key: events for key, events in groups.items() if events}
    for key, events in groups.items():
        if not any(e["complete"] for e in events):
            raise ValueError(f"{path}: unfinished line {key}")
    return metadata, groups


def replay_line(events, policy):
    finals = [e for e in events if e["complete"]]
    final = finals[-1]["tokens"]
    final_time = finals[0]["time"]
    history = []
    previous = []
    since = []
    committed = []
    commit_times = []
    contradiction = None
    last_update_time = -1.0
    for event in events:
        tokens, now = event["tokens"], event["time"]
        if committed != tokens[:len(committed)] and contradiction is None:
            contradiction = {"at": now, "typed": committed.copy(), "revised": tokens}
        if event["complete"] or contradiction is not None:
            continue
        # Only count independent native updates with additional consumed audio.
        if not event["updated"] or now <= last_update_time:
            continue
        last_update_time = now
        common = common_prefix([previous, tokens])
        since = since[:common] + [now] * (len(tokens) - common)
        previous = tokens
        history.append(tokens)
        history = history[-policy["updates"]:]
        if len(history) < policy["updates"]:
            continue
        agreed = common_prefix(history)
        available = min(agreed, max(0, len(tokens) - policy["tail"]))
        # Age applies to the entire prefix; never skip over an unstable token.
        count = 0
        while count < available and now - since[count] + 1e-9 >= policy["age"]:
            count += 1
        if count > len(committed):
            commit_times.extend([now] * (count - len(committed)))
            committed = tokens[:count]
    wrong_final = committed != final[:len(committed)]
    typed_words = lexical(committed)
    lexical_wrong = typed_words != lexical(final)[:len(typed_words)]
    return {
        "early_words": len(committed),
        "early_leads": [max(0.0, final_time - t) for t in commit_times],
        "first_lead": max(0.0, final_time - commit_times[0]) if commit_times else 0.0,
        "contradicted": contradiction is not None,
        "wrong_final": wrong_final,
        "lexical_wrong": lexical_wrong,
        "example": contradiction,
        "finalized_revision": any(e["tokens"] != final for e in finals),
    }


def analyze(groups):
    reports = []
    for policy in POLICIES:
        results = [(key, replay_line(events, policy)) for key, events in groups.items()]
        leads = [lead for _, r in results for lead in r["early_leads"]]
        reports.append({
            **policy,
            "lines": len(results),
            "lines_with_early_words": sum(r["early_words"] > 0 for _, r in results),
            "early_words": sum(r["early_words"] for _, r in results),
            "contradicted_lines": sum(r["contradicted"] for _, r in results),
            "wrong_final_lines": sum(r["wrong_final"] for _, r in results),
            "lexically_wrong_final_lines": sum(r["lexical_wrong"] for _, r in results),
            "finalized_revision_lines": sum(r["finalized_revision"] for _, r in results),
            "median_early_word_lead_s": statistics.median(leads) if leads else 0.0,
            "mean_first_word_lead_all_lines_s": statistics.mean(r["first_lead"] for _, r in results) if results else 0.0,
            "examples": [{"clip": key[0], "line": key[1], **r["example"]}
                         for key, r in results if r["example"] is not None][:4],
        })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    datasets = []
    for path in args.trace:
        meta, groups = read_trace(path)
        policies = analyze(groups)
        datasets.append({"trace": str(path), "metadata": meta, "policies": policies})
        print(path.name)
        for row in policies:
            print(f"  {row['name']:<24} early words={row['early_words']:4d} "
                  f"contradicted={row['contradicted_lines']:3d}/{row['lines']} "
                  f"wrong-final={row['wrong_final_lines']:3d} "
                  f"median lead={row['median_early_word_lead_s']:.2f}s")
    # Results contain transcript examples: exclusive creation, private before
    # writing any content. Never overwrite a user file or follow its symlink.
    with args.out.open("x", encoding="utf-8") as out:
        if os.name == "posix":
            os.fchmod(out.fileno(), 0o600)
        out.write(json.dumps({"scope": "offline native-final-relative replay, not ASR accuracy or OS insertion latency", "datasets": datasets}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
