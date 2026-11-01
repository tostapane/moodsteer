"""How well does each detector read the mood, and how long does it take per message?
Test set: data/moods_test.jsonl, written by hand. data/moods/ was written by the model, so it can't test the probe.
Predicted mood = the strongest one if its score ≥ config.THRESHOLD, else neutral (what the controller sees).
Run: python -m eval.detector   (the probe needs python -m detector.build first)"""
import json
import time
from collections import Counter

import config
from detector.detect import detect_llm, detect_probe


def predict(scores):
    if scores is None:  # the detector couldn't answer: no steering, same as neutral
        return "neutral"
    mood = max((m for m in scores if m != "neutral"), key=scores.get)
    return mood if scores[mood] >= config.THRESHOLD else "neutral"


def main():
    rows = [json.loads(line) for line in open("data/moods_test.jsonl") if line.strip()]
    print(f"model {config.MODEL_ID} · {len(rows)} hand-written messages · chance ≈ {1 / len(config.MOODS):.0%}")
    for name, detect in (("llm", detect_llm), ("probe", detect_probe)):
        t = time.time()
        scores = [detect(r["msg"], []) for r in rows]
        if all(s is None for s in scores):
            print(f"\n{name}: not available")
            continue
        seconds = (time.time() - t) / len(rows)
        preds = [predict(s) for s in scores]
        right = sum(p == r["mood"] for p, r in zip(preds, rows))
        print(f"\n{name}: {right}/{len(rows)} = {right / len(rows):.0%} · {seconds:.2f}s per message"
              + (f" · {sum(s is None for s in scores)} unreadable answers" if None in scores else ""))
        for mood in config.MOODS:
            got = [p for p, r in zip(preds, rows) if r["mood"] == mood]
            wrong = Counter(p for p in got if p != mood)
            print(f"  {mood:<11} {got.count(mood)}/{len(got)}" + (f"   mistaken for {dict(wrong)}" if wrong else ""))


if __name__ == "__main__":
    main()
