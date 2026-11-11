"""One reply per strength for each axis, to pick config.STRENGTH: the largest strength where the reply has
clearly changed and is still good.
Run: python -m vectors.tune [axis ...] [--prompt "..."] [--neg]   (--neg pushes toward the second pole)"""
import shutil
import sys

import config
from core.model import get_model

PROMPT = "My code throws a KeyError and I don't know why. What should I do?"
STRENGTHS = [0.1, 0.2, 0.4, 0.8]  # fraction of a typical hidden state


def main():
    args = sys.argv[1:]
    prompt = PROMPT
    if "--prompt" in args:
        i = args.index("--prompt")
        prompt = args[i + 1]
        del args[i:i + 2]
    sign = -1 if "--neg" in args else 1
    names = [a for a in args if not a.startswith("--")]

    print(f"loading {config.MODEL_ID}…", flush=True)
    m = get_model()
    names = names or sorted(m.vectors)
    width = shutil.get_terminal_size((100, 24)).columns - 14
    msgs = [{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    def show(label, alphas):
        m.set_alphas(alphas)
        out = m.generate(msgs, max_new_tokens=80)
        print(f"  {label:<5} {len(out.split()):>3}w  {' '.join(out.split())[:width]}", flush=True)

    print(f"prompt: {prompt}\n")
    show("0", {})
    for name in names:
        print(f"\n{name} ({'−' if sign < 0 else '+'} pole) · config.STRENGTH = {config.STRENGTH.get(name)}")
        for s in STRENGTHS:
            show(f"{s}", {name: sign * s / config.STRENGTH.get(name, 0.1)})


if __name__ == "__main__":
    main()
