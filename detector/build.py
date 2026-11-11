"""Build the experimental mood probe (artifacts/moods.pt): one direction per mood in activation space.

Each message in data/moods/<mood>.txt is read at READ_LAYER and averaged over its tokens. A mood's direction is
mean(mood) − mean(neutral). Scores are calibrated so 0 is the 90th percentile of neutral messages and 1 is the
mood's median. Missing example files are written by the model. Every 5th example is held out for the accuracy check.

Run: python -m detector.build [--regen]"""
import re
import sys
from pathlib import Path

import torch

import config
from core.model import get_model
from detector.detect import MOOD_DESC

DATA = Path("data/moods")
DESC = {"neutral": "calm and neutral, just asking a normal question or giving a normal instruction", **MOOD_DESC}
N_EXAMPLES = 40


def load_examples(m, regen):
    missing = [k for k in DESC if regen or not (DATA / f"{k}.txt").exists()]
    if missing:
        prompts = [[{"role": "user", "content":
                     f"Write {N_EXAMPLES} different messages that a user who is {k} ({DESC[k]}) might send to an "
                     "AI assistant while getting help with coding, writing, studying or everyday tasks. Vary the "
                     "length, topic and style. One message per line. No numbering, no quotes, nothing else."}]
                   for k in missing]
        DATA.mkdir(parents=True, exist_ok=True)
        for k, out in zip(missing, m.generate_batch(prompts, max_new_tokens=2000)):
            lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"') for line in out.splitlines()]
            (DATA / f"{k}.txt").write_text("\n".join(x for x in lines if len(x) > 3) + "\n")
            print(f"generated data/moods/{k}.txt")
    return {k: [x for x in (DATA / f"{k}.txt").read_text().splitlines() if x.strip()] for k in DESC}


def main(regen=False):
    m = get_model()
    layer = config.READ_LAYER if config.READ_LAYER is not None else len(m.layers) // 2
    ex = load_examples(m, regen)
    train = {k: m.acts([x for i, x in enumerate(v) if i % 5], layer) for k, v in ex.items()}
    test = {k: m.acts([x for i, x in enumerate(v) if i % 5 == 0], layer) for k, v in ex.items()}

    moods = list(MOOD_DESC)
    mu = train["neutral"].mean(0)
    dirs = torch.stack([train[k].mean(0) - mu for k in moods])
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    lo = ((train["neutral"] - mu) @ dirs.T).quantile(0.9, dim=0)
    hi = torch.stack([((train[k] - mu) @ dirs[i]).median() for i, k in enumerate(moods)])
    hi = torch.maximum(hi, lo + 1e-3)
    probe = {"layer": layer, "moods": moods, "mu": mu, "dirs": dirs, "lo": lo, "hi": hi}
    Path(config.MOODS_FILE).parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe, config.MOODS_FILE)
    print(f"saved {config.MOODS_FILE} (layer {layer})")

    # predicted mood: the top score if above 0.5, otherwise neutral
    print("\nheld-out accuracy:")
    right = total = 0
    for k, X in test.items():
        s = (((X - mu) @ dirs.T - lo) / (hi - lo)).clamp(0, 1)
        best = s.max(1)
        pred = ["neutral" if v < 0.5 else moods[i] for v, i in zip(best.values.tolist(), best.indices.tolist())]
        ok = sum(p == k for p in pred)
        right, total = right + ok, total + len(pred)
        wrong = {p: pred.count(p) for p in set(pred) if p != k}
        print(f"  {k:<11} {ok}/{len(pred)}" + (f"   mistaken for {wrong}" if wrong else ""))
    print(f"  overall     {right}/{total} = {right / max(total, 1):.0%}  (chance ≈ {1 / len(DESC):.0%})")


if __name__ == "__main__":
    main(regen="--regen" in sys.argv)
