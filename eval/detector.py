"""Can the model read moods on its own? The keyword rules in detector/detect.py are NOT used here.
Same call as a chat turn (one message, no history). A mood counts if its score ≥ config.THRESHOLD.
Right = exactly the expected moods, nothing missing or extra. An unreadable answer is always wrong, never neutral.
Data, all hand-written:
  data/moods_test.jsonl, data/moods_mixed_test.jsonl   seen while tuning the prompt: may flatter it, mistakes are listed
  data/moods_holdout.jsonl                             the real score: never tune on these, mistakes aren't printed
  data/moods_intensity.jsonl                           mild vs strong pairs: is a stronger mood scored higher?
Run: python -m eval.detector   → summary, and every answer in runs/eval_detector.jsonl
The probe needs python -m detector.build first."""
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import config
import detector.detect
from detector.detect import MOOD_DESC, detect_llm, detect_probe


def load(*paths):
    rows = [json.loads(line) for p in paths for line in open(p) if line.strip()]
    for r in rows:
        r["want"] = set(r.get("moods", [r.get("mood")])) - {"neutral"}
    return rows


def ask(name, detect, msg):
    """→ scores (None if unreadable) and the raw answer. Mood names we don't know make an answer unreadable too."""
    scores = detect(msg)
    if name != "llm":
        return scores, ""
    reply = detector.detect.last_reply
    obj = re.search(r"\{.*?\}", reply, re.S)
    keys = re.findall(r'"(\w+)"\s*:', obj.group()) if obj else []
    return (None if any(k not in MOOD_DESC and k != "neutral" for k in keys) else scores), reply


def present(scores):
    return {m for m, v in scores.items() if m != "neutral" and v >= config.THRESHOLD}


def share(k, n):
    """k/n with its 95% range (Wilson): how far the true rate could be, given so few messages."""
    if not n:
        return "0/0"
    p, z = k / n, 1.96
    mid, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.0%} (95% range {max(0, mid - half):.0%}-{min(1, mid + half):.0%})"


def report(results, mistakes):
    got = [g for g, _ in results]
    rows = [r for _, r in results]
    right = sum(g == r["want"] for g, r in zip(got, rows))
    neutral = [g for g, r in zip(got, rows) if not r["want"]]
    single = [(g, r) for g, r in zip(got, rows) if len(r["want"]) == 1]
    mixed = [(g, r) for g, r in zip(got, rows) if len(r["want"]) > 1]
    print(f"  exactly right          {share(right, len(rows))}")
    print(f"  always saying neutral  {share(len(neutral), len(rows))}   ← the score to beat")
    print(f"  neutral kept neutral   {share(sum(g == set() for g in neutral), len(neutral))}   ← a false mood sticks in memory")
    print(f"  one mood, right        {share(sum(g == r['want'] for g, r in single), len(single))}")
    print(f"  two moods, both right  {share(sum(g == r['want'] for g, r in mixed), len(mixed))}")
    print(f"  unreadable answers     {sum(g is None for g in got)}/{len(rows)}")
    print(f"  {'mood':<13} {'found when present':<20} right when it says so")
    for m in MOOD_DESC:
        hits = sum(m in (g or ()) and m in r["want"] for g, r in zip(got, rows))
        expected = sum(m in r["want"] for r in rows)
        said = sum(m in (g or ()) for g in got)
        print(f"  {m:<13} {f'{hits}/{expected}':<20} {hits}/{said}")
    if mistakes:
        print("  mistakes:")
        for (g, r), reply in zip(results, mistakes):
            if g != r["want"]:
                print(f"    wanted {sorted(r['want'])}, got {'unreadable' if g is None else sorted(g)}: "
                      f"{r['msg'][:60]!r} → {reply[:60]!r}")


def main():
    tuning = load("data/moods_test.jsonl", "data/moods_mixed_test.jsonl")
    holdout = load("data/moods_holdout.jsonl")
    pairs = load("data/moods_intensity.jsonl")
    log = Path("runs/eval_detector.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"model {config.MODEL_ID} · threshold {config.THRESHOLD} · keyword rules off")
    with log.open("w") as f:
        for name, detect in (("llm", detect_llm), ("probe", detect_probe)):
            if name == "probe" and not Path(config.MOODS_FILE).exists():
                print("\nprobe: not available")
                continue
            print(f"\n===== {name} =====")
            for label, rows in (("HOLDOUT (the real score)", holdout), ("TUNING SET (may flatter)", tuning)):
                t = time.time()
                results, replies = [], []
                for r in rows:
                    scores, reply = ask(name, detect, r["msg"])
                    g = None if scores is None else present(scores)
                    results.append((g, r))
                    replies.append(reply)
                    f.write(json.dumps({"detector": name, "set": label.split()[0].lower(), "msg": r["msg"],
                                        "want": sorted(r["want"]), "got": None if g is None else sorted(g),
                                        "reply": reply}) + "\n")
                print(f"\n{label}, {(time.time() - t) / len(rows):.2f}s per message")
                report(results, replies if label.startswith("TUNING") else None)

            higher = same = 0
            levels = Counter()
            for p in pairs:
                low, high = (ask(name, detect, p[k])[0] or {} for k in ("mild", "strong"))
                a, b = low.get(p["mood"], 0.0), high.get(p["mood"], 0.0)
                higher += b > a
                same += b == a
                levels.update(round(v * 10) for v in (a, b))
            print(f"\nINTENSITY, {len(pairs)} mild vs strong pairs of the same mood")
            print(f"  strong scored higher   {share(higher, len(pairs))}")
            print(f"  same / lower           {same} / {len(pairs) - higher - same}")
            print(f"  scores given, 0-10     {dict(sorted(levels.items()))}")


if __name__ == "__main__":
    main()
