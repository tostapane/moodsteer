"""Steering memory. One state per axis (-1..1, sign = pole) that persists across turns.
A mood moves only the axes it lists in config.MAPPING. Other axes, and any neutral message, change nothing."""
import math

import config


def update(state: dict, scores: dict) -> dict:
    new = dict(state)
    moods = [(v, m) for m, v in scores.items() if m != "neutral" and v >= config.THRESHOLD]
    for v, m in sorted(moods):  # weakest first: the strongest mood has the last word on a shared axis
        for axis, (target, speed) in config.MAPPING.get(m, {}).items():
            now = new.get(axis, 0.0)
            new[axis] = now + v * speed * (target - now)
    return {a: v for a, v in sorted(new.items()) if abs(v) >= 0.01}


def push(alphas: dict) -> float:
    """Total push of these alphas, assuming the axis vectors point in unrelated directions."""
    return math.sqrt(sum((a * config.STRENGTH.get(axis, 0.1)) ** 2 for axis, a in alphas.items()))


def capped(alphas: dict) -> dict:
    """What actually steers a reply: the same alphas, scaled down if their total push is over config.MAX_PUSH."""
    p = push(alphas)
    return alphas if p <= config.MAX_PUSH else {axis: a * config.MAX_PUSH / p for axis, a in alphas.items()}
