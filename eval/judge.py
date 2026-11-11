"""LLM judge for what eval.steer can't count (warm, direct, accountable) and for the cost in answer quality.

Same conditions as eval.steer. The judge (the same model, unsteered) compares each condition's reply with the
baseline reply twice: is it more <pole>, and is it more correct and helpful. Each pair is asked in both A/B orders
to cancel position bias. Win rate: 100% = wins in both orders every time, 50% = no difference.
A 3B model judging its own replies is noisy: trust the direction more than the number.

Run: python -m eval.judge [axis ...] [--strengths 0.2,0.4,0.8]   → table, verdicts in runs/eval_judge.jsonl"""
import json
import re
import sys
import time
from pathlib import Path

import config
from core.model import get_model
from eval.steer import ANSWER_TOKENS, QUESTIONS, UNCERTAIN, answer
from vectors.build import AXES

# axis → (pole pushed: +1 first, −1 second; what the judge asks about)
POLES = {
    "concise":     (+1, "more concise, saying it in fewer words"),
    "warm":        (+1, "warmer, showing more empathy, care and encouragement"),
    "calm":        (-1, "more enthusiastic and excited"),  # the baseline is already calm
    "simple":      (+1, "simpler, in plainer everyday words that a beginner would understand"),
    "direct":      (+1, "more direct, getting to the answer sooner with less preamble"),
    "cautious":    (+1, "more cautious, admitting uncertainty instead of stating guesses as facts"),
    "structured":  (+1, "more structured, organised as steps or points"),
    "accountable": (+1, "more accountable, owning the mistake instead of defending the earlier answer"),
}
QUALITY = ("more correct and more helpful. Judge only the content: ignore length, tone, formatting "
           "and how confident it sounds")
# accountable needs a complaint about an earlier answer; none of these were used to build the vector
COMPLAINTS = [
    "The command you gave me to delete a folder didn't work.", "Your explanation of photosynthesis was wrong.",
    "I followed your budget plan and I'm even more in debt.", "The SQL query you wrote returns the wrong rows.",
    "Your translation of that sentence was way off.", "The workout you suggested gave me a back ache.",
    "You said Python lists are immutable, that's not true.", "Your fix made my website look broken on phones.",
    "The recipe you gave me came out completely raw.", "Your summary left out the most important part.",
    "The regex you gave me matches nothing.", "You told me the meeting tool was free, it isn't.",
]
AXIS_QUESTIONS = {"cautious": UNCERTAIN, "accountable": COMPLAINTS}
JUDGE_SYSTEM = "You are a careful, impartial judge."
JUDGE = """Here are two replies to the same user message.

User message: {q}

Reply A:
{a}

Reply B:
{b}

Which reply is {criterion}? Answer with one letter, A or B, and nothing else."""


def pick(verdict):
    """'A', 'B', or None if the judge didn't answer with one of them."""
    m = re.match(r"\W*([AB])\b", verdict.strip())
    return m.group(1) if m else None


def judge(m, questions, base, other, criterion):
    """Per question: the two raw verdicts and the score of `other` against `base`:
    1 = wins in both orders, 0.5 = split or unreadable, 0 = loses in both."""
    prompts = [JUDGE.format(q=q, a=a, b=b, criterion=criterion)
               for q, x, y in zip(questions, base, other) for a, b in ((x, y), (y, x))]
    verdicts = answer(m, prompts, {}, [JUDGE_SYSTEM] * len(prompts), 4)
    out = []
    for i in range(0, len(verdicts), 2):
        v1, v2 = pick(verdicts[i]), pick(verdicts[i + 1])  # other is B, then A
        wins = (v1 == "B") + (v2 == "A")
        losses = (v1 == "A") + (v2 == "B")
        out.append(((verdicts[i], verdicts[i + 1]), 0.5 + (wins - losses) / 4))
    return out


def main():
    args = sys.argv[1:]
    strengths = [0.2, 0.4, 0.8]
    if "--strengths" in args:
        i = args.index("--strengths")
        strengths = [float(x) for x in args[i + 1].split(",")]
        del args[i:i + 2]
    t = time.time()
    m = get_model()
    axes = [a for a in args if a in POLES] or [a for a in POLES if a in getattr(m, "vectors", POLES)]
    print(f"model {config.MODEL_ID} · judge: same model, unsteered · strengths {strengths} · axes {', '.join(axes)}")
    Path("runs").mkdir(exist_ok=True)
    with open("runs/eval_judge.jsonl", "a") as log:
        for axis in axes:
            sign, pole = POLES[axis]
            questions = AXIS_QUESTIONS.get(axis, QUESTIONS)
            # the model multiplies alpha by STRENGTH, so this gives strength s
            conditions = {"baseline": ({}, config.SYSTEM_PROMPT)}
            conditions.update({f"steer {s}": ({axis: sign * s / config.STRENGTH.get(axis, 0.1)}, config.SYSTEM_PROMPT)
                               for s in strengths})
            conditions["prompt"] = ({}, AXES[axis][0 if sign > 0 else 1])
            names = list(conditions)
            rows = [(n, q) for n in names for q in questions]
            replies = answer(m, [q for _, q in rows], [conditions[n][0] for n, _ in rows],
                             [conditions[n][1] for n, _ in rows], ANSWER_TOKENS)
            replies = {n: replies[i * len(questions):(i + 1) * len(questions)] for i, n in enumerate(names)}

            print(f"\n{axis} · is it {pole}?")
            print(f"  {'condition':<12} {'axis win':>9} {'quality win':>12}   (vs baseline, 50% = no difference)")
            picked_a = total = unreadable = 0
            for name in names[1:]:
                res = {c: judge(m, questions, replies["baseline"], replies[name], crit)
                       for c, crit in (("axis", pole), ("quality", QUALITY))}
                rate = {c: 100 * sum(s for _, s in r) / len(r) for c, r in res.items()}
                print(f"  {name:<12} {rate['axis']:>8.0f}% {rate['quality']:>11.0f}%")
                for r in res.values():
                    for pair, _ in r:
                        for v in map(pick, pair):
                            picked_a += v == "A"
                            unreadable += v is None
                            total += 1
                log.write(json.dumps({"axis": axis, "condition": name, "axis_win": rate["axis"],
                                      "quality_win": rate["quality"], "questions": questions,
                                      "baseline": replies["baseline"], "replies": replies[name],
                                      "verdicts": {c: [p for p, _ in r] for c, r in res.items()}}) + "\n")
                log.flush()
            print(f"  judge bias: picked A {100 * picked_a / max(1, total):.0f}% of the time, "
                  f"{unreadable}/{total} verdicts unreadable")
    print(f"\ndone in {time.time() - t:.0f}s · every verdict in runs/eval_judge.jsonl")


if __name__ == "__main__":
    main()
