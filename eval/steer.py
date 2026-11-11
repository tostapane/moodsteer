"""Does each axis change what it claims to, and what does it cost?

For each axis with a countable property, held-out questions are answered with no steering (baseline), with the
vector at several strengths, and with the pole's persona in the system prompt instead. The cost is how many simple
factual questions are still answered correctly under each condition.

Run: python -m eval.steer [axis ...] [--strengths 0.2,0.4,0.8]   → table, replies in runs/eval_steer.jsonl"""
import json
import re
import sys
import time
from pathlib import Path

import config
from core.model import get_model
from vectors.build import AXES

# none of these were used to build the vectors
QUESTIONS = [
    "How do I reverse a string in Python?", "What is a hash table?", "Why do cats purr?",
    "How do I make a good cup of coffee?", "What is the difference between a list and a tuple in Python?",
    "How does a refrigerator work?", "What is machine learning?", "How do I improve my sleep?",
    "What is an operating system?", "How do I start learning the guitar?", "Why do we have time zones?",
    "What is version control?",
]
# cautious only shows on questions with no sure answer
UNCERTAIN = [
    "Will the stock market go up next month?", "Who will be the next US president?",
    "Is a glass of milk a day good for adults?", "What did Cleopatra look like?",
    "Will humans live on Mars by 2050?", "Does listening to music help you study?",
    "How many trees are there in the Amazon?", "Why were the Maya cities abandoned?",
    "Is it better to rent or buy a house?", "Will electric cars be cheaper than petrol cars in five years?",
    "What do dreams mean?", "How old is the oldest tree alive today?",
]
AXIS_QUESTIONS = {"cautious": UNCERTAIN}
# (question, accepted answers as a regex): the capability cost
FACTS = [
    ("What is the capital of France?", "paris"), ("What is the capital of Japan?", "tokyo"),
    ("What is the capital of Italy?", "rome"), ("What is the capital of Canada?", "ottawa"),
    ("What is the capital of Australia?", "canberra"), ("How many legs does a spider have?", "8|eight"),
    ("How many days are in a week?", "7|seven"), ("What is 12 times 12?", "144"),
    ("What is 15 plus 27?", "42"), ("What is 100 divided by 4?", "25"),
    ("Which planet is known as the Red Planet?", "mars"), ("What is the largest planet in the Solar System?", "jupiter"),
    ("Which gas do plants absorb from the air?", "carbon dioxide|co2|co₂"), ("What is the chemical symbol for gold?", "au"),
    ("What is the chemical formula of water?", "h2o|h₂o"), ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("Who painted the Mona Lisa?", "leonardo|da vinci"), ("At what temperature in Celsius does water freeze?", "0|zero"),
    ("At what temperature in Celsius does water boil?", "100"), ("How many continents are there?", "7|seven"),
    ("What is the largest ocean on Earth?", "pacific"), ("What language is mainly spoken in Brazil?", "portuguese"),
    ("In which year did World War II end?", "1945"), ("What is the square root of 81?", "9|nine"),
    ("What is the smallest prime number?", "2|two"), ("How many sides does a hexagon have?", "6|six"),
    ("What is the currency of Japan?", "yen"), ("Who developed the theory of relativity?", "einstein"),
    ("What is the tallest mountain in the world?", "everest"), ("What does CPU stand for?", "central processing unit"),
]
HEDGES = ["might", "may", "could", "possibly", "perhaps", "probably", "likely", "not sure", "i think", "i believe",
          "it depends", "generally", "typically", "usually", "often", "in some cases", "uncertain", "not certain"]


def words(text):
    return len(text.split())


def list_lines(text):
    lines = [x for x in text.splitlines() if x.strip()]
    return 100 * sum(bool(re.match(r"\s*(\d+[.)]|[-*•])\s", x)) for x in lines) / max(1, len(lines))


def reading_ease(text):
    """Flesch reading ease: higher = simpler. Syllables ≈ groups of vowels."""
    ws = re.findall(r"[A-Za-z]+", text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in ws)
    return 206.835 - 1.015 * len(ws) / sentences - 84.6 * syllables / max(1, len(ws))


def exclamations(text):
    return 100 * text.count("!") / max(1, words(text))


def hedges(text):
    t = text.lower()
    return 100 * sum(len(re.findall(rf"\b{h}\b", t)) for h in HEDGES) / max(1, words(text))


# axis → (pole we push: +1 first, −1 second, what we count, how it should move)
TESTS = {
    "concise":    (+1, words, "words", "down"),
    "structured": (+1, list_lines, "% lines that are list items", "up"),
    "simple":     (+1, reading_ease, "reading ease (Flesch)", "up"),
    "calm":       (-1, exclamations, "! per 100 words (enthusiastic pole)", "up"),
    "cautious":   (+1, hedges, "hedge words per 100 words", "up"),
}
ANSWER_TOKENS = config.MAX_NEW_TOKENS  # same as the chat, so answers aren't cut short
FACT_TOKENS = 100  # room for a hedge before the answer


def answer(m, questions, alphas, systems, max_new_tokens):
    """Answer questions[i] with steering alphas[i] and system prompt systems[i], in one batch."""
    m.set_alphas(alphas)
    convos = [[{"role": "system", "content": s}, {"role": "user", "content": q}] for q, s in zip(questions, systems)]
    out = m.generate_batch(convos, max_new_tokens)
    m.set_alphas({})
    return out


def n_cut(m, replies, limit):
    """Replies that hit the token limit, whose metric is unreliable."""
    if not hasattr(m, "n_tokens"):  # FakeModel
        return 0
    return sum(m.n_tokens(r) >= limit - 2 for r in replies)  # re-tokenizing can be off by a token


def correct(reply, expected):
    return bool(re.search(rf"\b({expected})\b", reply.lower()))


def run(m, conditions, questions=QUESTIONS):
    """{name: (alphas, system prompt)} → {name: (replies, fact replies)}. All conditions share one batch."""
    names = list(conditions)
    out = {}
    for qs, tokens in ((questions, ANSWER_TOKENS), ([q for q, _ in FACTS], FACT_TOKENS)):
        rows = [(n, q) for n in names for q in qs]
        replies = answer(m, [q for _, q in rows], [conditions[n][0] for n, _ in rows],
                         [conditions[n][1] for n, _ in rows], tokens)
        for i, n in enumerate(names):
            out.setdefault(n, []).append(replies[i * len(qs):(i + 1) * len(qs)])
    return out


def report(m, axis, results, log):
    sign, metric, label, better = TESTS[axis]
    print(f"\n{axis} ({'+' if sign > 0 else '−'} pole) · {label}, should go {better}")
    print(f"  {'condition':<12} {'metric':>7}   facts right   cut off (answers, facts)")
    for name, (replies, facts) in results.items():
        value = sum(map(metric, replies)) / len(replies)
        right = sum(correct(r, e) for r, (_, e) in zip(facts, FACTS))
        cut = n_cut(m, replies, ANSWER_TOKENS), n_cut(m, facts, FACT_TOKENS)
        print(f"  {name:<12} {value:>7.1f}   {right:>5}/{len(FACTS)}   {cut[0]}/{len(replies)}, {cut[1]}/{len(facts)}")
        log.write(json.dumps({"axis": axis, "condition": name, "metric": value, "facts_right": right,
                              "facts_total": len(FACTS), "cut": cut, "replies": replies, "facts": facts}) + "\n")


def main():
    args = sys.argv[1:]
    strengths = [0.2, 0.4, 0.8]
    if "--strengths" in args:
        i = args.index("--strengths")
        strengths = [float(x) for x in args[i + 1].split(",")]
        del args[i:i + 2]
    t = time.time()
    m = get_model()
    axes = [a for a in args if a in TESTS] or [a for a in TESTS if a in m.vectors]
    print(f"model {config.MODEL_ID} · strengths {strengths} · axes {', '.join(axes)}")
    Path("runs").mkdir(exist_ok=True)
    with open("runs/eval_steer.jsonl", "a") as log:
        for axis in axes:
            sign = TESTS[axis][0]
            # a baseline per axis, since cautious uses its own questions
            conditions = {"baseline": ({}, config.SYSTEM_PROMPT)}
            # the model multiplies alpha by STRENGTH, so this gives strength s
            conditions.update({f"steer {s}": ({axis: sign * s / config.STRENGTH.get(axis, 0.1)}, config.SYSTEM_PROMPT)
                               for s in strengths})
            conditions["prompt"] = ({}, AXES[axis][0 if sign > 0 else 1])
            report(m, axis, run(m, conditions, AXIS_QUESTIONS.get(axis, QUESTIONS)), log)
            log.flush()
    print(f"\ndone in {time.time() - t:.0f}s · every reply in runs/eval_steer.jsonl")


if __name__ == "__main__":
    main()
