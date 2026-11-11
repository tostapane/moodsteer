"""Which layer steers best? Builds one axis's vector at every 4th layer and steers the same questions with each.
Run: python -m eval.layers [axis] [--strength 0.2]   → table, replies in runs/eval_layers.jsonl"""
import sys
import time
from pathlib import Path

import config
from core.model import get_model
from eval.steer import TESTS, report, run
from vectors.build import build


def main():
    args = sys.argv[1:]
    strength = 0.2
    if "--strength" in args:
        i = args.index("--strength")
        strength = float(args[i + 1])
        del args[i:i + 2]
    axis = args[0] if args else "concise"
    t = time.time()
    m = get_model()
    layers = list(range(2, len(m.layers), 4))
    print(f"model {config.MODEL_ID} · {axis} · strength {strength} · layers {layers}\n")
    print(f"{'vector':<11} pairs  strength")
    conditions = {"baseline": ({}, config.SYSTEM_PROMPT)}
    for layer in layers:
        name = f"{axis}@{layer}"
        m.vectors[name] = build(axis, m, layer)
        alpha = TESTS[axis][0] * strength / config.STRENGTH.get(name, 0.1)
        conditions[f"layer {layer}"] = ({name: alpha}, config.SYSTEM_PROMPT)
    Path("runs").mkdir(exist_ok=True)
    with open("runs/eval_layers.jsonl", "a") as log:
        report(m, axis, run(m, conditions), log)
    print(f"\ndone in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
