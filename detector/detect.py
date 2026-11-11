"""Mood detection: message → score per mood in 0..1.

llm    the model lists the moods in the message as JSON, each with an intensity 0-10.
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
    "rushed":     "in a hurry, or asks for a shorter answer, or complains that answers are too long",
    "excited":    "happy, enthusiastic",
    "skeptical":  "doubts the assistant, asks for proof",
    "wants_detail": "asks for a longer, clearer or more detailed answer, or complains that answers are too short",
}


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
          + "".join(f"- {m}: {d}\n" for m, d in MOOD_DESC.items())
          + "Intensity from 0 to 10: 0 = not present, 3 = mild, 6 = clear, 10 = extreme.\n"
          "For a shorter or longer answer, the intensity is how much shorter or longer they want it.\n"
          "Reply with JSON only, no other text, mood → intensity, {} if there is no mood.")

# examples, sent as earlier chat turns. Kept apart from data/moods_*.jsonl so the eval stays honest.
SHOTS = [
    ("What is a tomato?", {}),
    ("How do I change a tire?", {}),
    ("Explain how a bicycle gear works.", {}),
    ("That was a bit long, could you trim it?", {"rushed": 3}),
    ("Explain briefly how a dam works.", {"rushed": 6}),
    ("One sentence only, I have no time.", {"rushed": 10}),
    ("Good, maybe one more example?", {"wants_detail": 3}),
    ("Can you go into more detail on the second step?", {"wants_detail": 6}),
    ("Explain every single step in depth, as long as it takes.", {"wants_detail": 10}),
    ("This is wrong again.", {"frustrated": 6}),
    ("I give up, none of this makes sense to me.", {"sad": 6, "confused": 5}),
    ("ARE YOU EVEN LISTENING?? USELESS!!", {"rage": 9}),
    ("Are you sure about that? Show me a source.", {"skeptical": 6}),
    ("Whoa, that's so cool! Why does it work like that?", {"excited": 6, "curious": 6}),
]


last_reply = ""  # raw model output, logged in the trace


def detect_llm(msg: str) -> dict[str, float] | None:
    global last_reply
    from core.model import get_model

    model = get_model()
    model.set_alphas({})  # detect without steering
    shots = [{"role": r, "content": c} for m, a in SHOTS for r, c in (("user", m), ("assistant", json.dumps(a)))]
    reply = model.generate([{"role": "system", "content": PROMPT}, *shots, {"role": "user", "content": msg[:1000]}],
                           max_new_tokens=40)
    last_reply = reply
    scores = {m: 0.0 for m in MOOD_DESC}
    try:
        raw = json.loads(re.search(r"\{.*?\}", reply, re.S).group())  # first object only: it sometimes keeps writing examples
        for mood, level in raw.items():
            mood = str(mood).strip().lower()
            if mood in scores:
                scores[mood] = min(10.0, max(0.0, float(level))) / 10
    except (AttributeError, ValueError, TypeError):
        return None
    return with_neutral(scores)


DETECTORS = {"probe": detect_probe, "llm": detect_llm}

# explicit requests about length or format: regexes catch the direction more reliably than the 3B model
ASKS_DETAIL = re.compile(r"more (detail|depth|clear|example)|lots? of detail|in detail|detailed|clearer|explain more"
                         r"|something more|tell me more|too (short|brief)|so rushed|expand on|go deeper"
                         r"|(not|n'?t|less)( be)?( so)? (brief|concise|short)", re.I)
ASKS_SHORT = re.compile(r"\b(brief|briefly|concise|concisely|shorter|summary|tl;?dr)\b|too long|wall of text"
                        r"|keep it short|in (one|a) (sentence|line)|one[- ]line|few words|key points|cut it down", re.I)

ASKS_PROSE = re.compile(r"\b(no|without|avoid|stop using)( any)? (bullets?( points?)?|numbered lists?|lists\b)"
                        r"|n'?t (use|want|give me)( any)? (bullets?( points?)?|numbered lists?|lists\b)"
                        r"|in prose|(one|single) paragraph", re.I)


FALLBACK = 0.6  # intensity of a "shorter" request the model scored 0


def with_request(msg: str, scores: dict | None) -> dict | None:
    """An explicit request sets the direction. "Shorter" keeps the model's intensity, so it's graded.
    "Longer" is always full: its target is no steering, and anything less would leave the answers short."""
    got = scores or {}
    asked = {}
    if ASKS_DETAIL.search(msg):
        asked = {"rushed": 0.0, "wants_detail": 1.0}
    elif ASKS_SHORT.search(msg):
        asked = {"rushed": got.get("rushed") or FALLBACK, "wants_detail": 0.0}
    if ASKS_PROSE.search(msg):
        asked["wants_prose"] = 1.0
    if not asked:
        return scores
    return with_neutral({**(scores or {m: 0.0 for m in MOOD_DESC}), **asked})


def detect(msg: str) -> dict[str, float]:
    names = list(DETECTORS)
    for name in names[names.index(config.DETECTOR):]:
        scores = DETECTORS[name](msg)
        if scores is not None:
            return with_request(msg, scores)
    return with_request(msg, with_neutral({m: 0.0 for m in MOOD_DESC}))
