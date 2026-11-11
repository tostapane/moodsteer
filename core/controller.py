"""Steering state: one value per axis in -1..1 (the sign picks the pole), kept across turns."""
import math

import config


def update(state: dict, scores: dict) -> dict:
    new = dict(state)
    moods = [(v, m) for m, v in scores.items() if m != "neutral" and v >= config.THRESHOLD]
    for v, m in sorted(moods):  # weakest first, so the strongest mood wins on a shared axis
        for axis, (target, speed) in config.MAPPING.get(m, {}).items():
            now = new.get(axis, 0.0)
            new[axis] = now + v * speed * (target - now)
    return {a: v for a, v in sorted(new.items()) if abs(v) >= 0.01}


def push(alphas: dict) -> float:
    """Combined push of these alphas, treating the axis vectors as orthogonal."""
    return math.sqrt(sum((a * config.STRENGTH.get(axis, 0.1)) ** 2 for axis, a in alphas.items()))


def capped(alphas: dict) -> dict:
    """The alphas scaled down so their combined push is at most config.MAX_PUSH."""
    p = push(alphas)
    return alphas if p <= config.MAX_PUSH else {axis: a * config.MAX_PUSH / p for axis, a in alphas.items()}
