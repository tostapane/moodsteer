"""Steering memory. One state per axis (-1..1, sign = pole) that persists across turns."""
import config


def wanted(scores: dict) -> dict | None:
    """Axes this message's moods ask for, or None if no mood is clearly present (neutral message)."""
    moods = {m: v for m, v in scores.items() if m != "neutral" and v >= config.THRESHOLD}
    if not moods:
        return None
    want = {}
    for m, v in moods.items():
        for axis, w in config.MAPPING.get(m, {}).items():
            want[axis] = want.get(axis, 0.0) + v * w
    return want


def update(state: dict, scores: dict, n: int) -> tuple[dict, int]:
    """n = mood messages seen so far. Returns the new state and n."""
    want = wanted(scores)
    if want is None:
        new = {a: config.HOLD * v for a, v in state.items()}
    else:
        n += 1
        w = max(1 - config.KEEP, 1 / n)  # 100%, 50%, then 40%
        new = {a: (1 - w) * state.get(a, 0.0) + w * want.get(a, 0.0) for a in set(state) | set(want)}
    return {a: max(-1.0, min(1.0, v)) for a, v in sorted(new.items()) if abs(v) >= 0.01}, n
