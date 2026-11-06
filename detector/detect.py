"""Mood detection: message → mood scores 0..1.
probe = project the model's activations, averaged over the whole user message, onto mood directions
        (built by detector/build.py). No generation, ~one forward pass.
llm   = the model reads the conversation and lists the moods present, each with intensity 0-10, as JSON.
config.DETECTOR picks where to start (probe → llm). If nothing can answer: neutral, i.e. no steering."""
import json
import re
from pathlib import Path

import config

MOOD_DESC = {
    "curious":    "wants to learn more, asks why/how",
    "confused":   "doesn't understand the previous answer",
    "frustrated": "annoyed that things aren't working, corrects the assistant",
    "rage":       "angry, shouting, insulting",
    "sad":        "discouraged, wants to give up, feels bad about themselves",
    "rushed":     "in a hurry, wants a short direct answer",
    "excited":    "happy, enthusiastic",
    "skeptical":  "doubts the assistant, asks for proof",
}


def with_neutral(scores):
    scores["neutral"] = 1.0 - max(scores.values())
    return scores


# ---- probe -------------------------------------------------------------------------------------

_probe = None


def detect_probe(msg: str, history: list) -> dict[str, float] | None:
    global _probe
    if _probe is None:
        if not Path(config.MOODS_FILE).exists():
            return None
        import torch
        _probe = torch.load(config.MOODS_FILE)
    from core.model import get_model  # lazy: avoids a circular import

    p = _probe
    h = get_model().acts([msg], p["layer"])[0]                   # mean over every token of the message
    proj = p["dirs"] @ (h - p["mu"])                             # one number per mood
    s = ((proj - p["lo"]) / (p["hi"] - p["lo"])).clamp(0, 1)     # 0 ≈ neutral, 1 ≈ typical example
    return with_neutral(dict(zip(p["moods"], s.tolist())))


# ---- llm ---------------------------------------------------------------------------------------

PROMPT = ("You read a conversation between a user and an assistant and judge the mood of the user's LAST message.\n"
          "List only the moods clearly present, at most 3. Most messages are neutral: no mood at all.\n"
          "A plain request or instruction (like 'be concise') is neutral unless its tone shows a mood.\n"
          "Moods:\n"
          + "".join(f"- {m}: {d}\n" for m, d in MOOD_DESC.items())
          + "Intensity from 0 to 10: 0 = not present, 3 = mild, 6 = clear, 10 = extreme.\n"
          'Reply with JSON only, no other text, mood → intensity.\n'
          'One mood: {"frustrated": 4}. Two moods: {"rushed": 8, "confused": 5}. Neutral: {}')


def detect_llm(msg: str, history: list) -> dict[str, float] | None:
    from core.model import get_model

    convo = "\n".join(f"{h['role'].upper()}: {h['content'][:500]}" for h in history[-4:])
    convo += f"\nUSER (last message): {msg[:1000]}"
    model = get_model()
    model.set_alphas({})  # classify with the unsteered model
    reply = model.generate([{"role": "system", "content": PROMPT}, {"role": "user", "content": convo}],
                           max_new_tokens=40)
    scores = {m: 0.0 for m in MOOD_DESC}
    try:
        raw = json.loads(re.search(r"\{.*\}", reply, re.S).group())
        for mood, level in raw.items():
            mood = str(mood).strip().lower()
            if mood in scores:  # anything unknown is ignored
                scores[mood] = min(10.0, max(0.0, float(level))) / 10
    except (AttributeError, ValueError, TypeError):
        return None
    return with_neutral(scores)


# ---- entry point -------------------------------------------------------------------------------

DETECTORS = {"probe": detect_probe, "llm": detect_llm}


def detect(msg: str, history: list) -> dict[str, float]:
    names = list(DETECTORS)
    for name in names[names.index(config.DETECTOR):]:
        scores = DETECTORS[name](msg, history)
        if scores is not None:
            return scores
    return with_neutral({m: 0.0 for m in MOOD_DESC})
