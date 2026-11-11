"""Mood detection: message → score per mood in 0..1.

llm    two questions to the model: the moods in the message as JSON, each with an intensity 0-10, and how long
       the next answer should be, as one letter (wants_shorter / wants_detail).
probe  experimental: projects the message's activations onto mood directions from detector/build.py.

config.DETECTOR picks the first one to try, the next is the fallback. If none answers, the message is neutral."""
import json
import re
from pathlib import Path

import config

MOOD_DESC = {
    "curious":    "wonders why or how something works, out of interest",
    "confused":   "doesn't understand the previous answer",
    "frustrated": "annoyed that things aren't working, corrects the assistant",
    "rage":       "angry, shouting, insulting",
    "sad":        "discouraged, wants to give up, feels bad about themselves",
    "rushed":     "in a hurry, has no time",
    "excited":    "happy, enthusiastic",
    "skeptical":  "doubts the assistant, asks for proof",
    "wants_shorter": "asks for a shorter or more concise answer, or complains that answers are too long",
    "wants_detail": "asks for a longer or more detailed answer, or complains that answers are too short",
}
LENGTH_MOODS = ("wants_shorter", "wants_detail")  # asked separately, see LENGTH


def with_neutral(scores):
    scores["neutral"] = 1.0 - max(scores.values())
    return scores


_probe = None


def detect_probe(msg: str) -> dict[str, float] | None:
    global _probe
    if _probe is None:
        if not Path(config.MOODS_FILE).exists():
            return None
        import torch
        _probe = torch.load(config.MOODS_FILE)
    from core.model import get_model  # here to avoid a circular import

    p = _probe
    h = get_model().acts([msg], p["layer"])[0]
    proj = p["dirs"] @ (h - p["mu"])
    s = ((proj - p["lo"]) / (p["hi"] - p["lo"])).clamp(0, 1)     # 0 = neutral, 1 = a typical example
    return with_neutral(dict(zip(p["moods"], s.tolist())))


PROMPT = ("You judge the mood of one message a user sent to an AI assistant.\n"
          "List only the moods clearly present, at most 3. Most messages have no mood.\n"
          "A plain question or request for facts, definitions or practical steps has no mood.\n"
          "Moods:\n"
          + "".join(f"- {m}: {d}\n" for m, d in MOOD_DESC.items() if m not in LENGTH_MOODS)
          + "Intensity from 0 to 10: 0 = not present, 3 = mild, 6 = clear, 10 = extreme.\n"
          "Reply with JSON only, no other text, mood → intensity, {} if there is no mood.")

# examples, sent as earlier chat turns. Kept apart from data/moods_*.jsonl so the eval stays honest.
SHOTS = [
    ("What is a tomato?", {}),
    ("How do I change a tire?", {}),
    ("Explain how a bicycle gear works.", {}),
    ("Okay, now tell me how volcanoes form.", {}),
    ("Explain what a mortgage is.", {}),
    ("Quick, my bus is leaving, is it going to rain today?", {"rushed": 8}),
    ("This is wrong again.", {"frustrated": 6}),
    ("I give up, none of this makes sense to me.", {"sad": 6, "confused": 5}),
    ("ARE YOU EVEN LISTENING?? USELESS!!", {"rage": 9}),
    ("Are you sure about that? Show me a source.", {"skeptical": 6}),
    ("Whoa, that's so cool! Why does it work like that?", {"excited": 6, "curious": 6}),
]


# length is its own question: as one mood among many, the 3B model kept reading "briefly" as wants_detail.
# One forward pass; the most likely letter wins.
LENGTH = ("Here is a message a user sent to an AI assistant:\n\n{msg}\n\n"
          "How long does the user want the assistant's next answer to be?\n"
          "A) much shorter\nB) a bit shorter\nC) normal, the message doesn't ask about length\n"
          "D) a bit longer\nE) much longer\nAnswer with one letter.")
LENGTH_SCORES = {"A": ("wants_shorter", 1.0), "B": ("wants_shorter", 0.4), "C": (None, 0.0),
                 "D": ("wants_detail", 0.4), "E": ("wants_detail", 1.0)}

last_reply = ""  # raw model output, logged in the trace


def detect_llm(msg: str) -> dict[str, float] | None:
    global last_reply
    from core.model import get_model

    model = get_model()
    model.set_alphas({})  # detect without steering
    shots = [{"role": r, "content": c} for m, a in SHOTS for r, c in (("user", m), ("assistant", json.dumps(a)))]
    reply = model.generate([{"role": "system", "content": PROMPT}, *shots, {"role": "user", "content": msg[:1000]}],
                           max_new_tokens=40)
    probs = model.choose([{"role": "user", "content": LENGTH.format(msg=msg[:1000])}], list(LENGTH_SCORES))
    letter = max(probs, key=probs.get)
    last_reply = f"{reply} | length {letter}"
    scores = {m: 0.0 for m in MOOD_DESC}
    try:
        raw = json.loads(re.search(r"\{.*?\}", reply, re.S).group())  # first object only: it sometimes keeps writing examples
        for mood, level in raw.items():
            mood = str(mood).strip().lower()
            if mood in scores and mood not in LENGTH_MOODS:
                scores[mood] = min(10.0, max(0.0, float(level))) / 10
    except (AttributeError, ValueError, TypeError):
        return None
    mood, level = LENGTH_SCORES[letter]
    if mood:
        scores[mood] = level
    return with_neutral(scores)


DETECTORS = {"probe": detect_probe, "llm": detect_llm}

def detect(msg: str) -> dict[str, float]:
    names = list(DETECTORS)
    for name in names[names.index(config.DETECTOR):]:
        scores = DETECTORS[name](msg)
        if scores is not None:
            return scores
    return with_neutral({m: 0.0 for m in MOOD_DESC})
