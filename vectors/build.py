"""Build the steering vectors (contrastive activation addition).

Each axis has two poles, e.g. concise ↔ long: +α pushes toward the first, −α toward the second.
1. Pairs: for each question the model writes one answer per pole, prompted with the personas in AXES, into
   data/pairs/<axis>.jsonl. They're plain text, so they're committed and reused; --regen rewrites them.
   Pairs whose two answers are nearly identical are dropped.
2. Vector: each answer goes through the model after a neutral system prompt (no persona), and the hidden state at
   STEER_LAYER is averaged over the answer tokens. vec = mean(positive) − mean(negative), rescaled to the size of
   a typical hidden state so a given strength is the same push on every axis.

Run: python -m vectors.build [axis ...] [--regen], then pick strengths with python -m vectors.tune."""
import difflib
import json
import sys
import time
from pathlib import Path

import torch

import config
from core.model import get_model

DATA = Path("data/pairs")

# axis → (persona for the + pole, persona for the − pole). Names must match config.MAPPING / config.STRENGTH.
AXES = {
    "concise":     ("Answer in as few words as possible.",
                    "Answer at great length, with lots of detail and digressions."),
    "warm":        ("You are a warm, caring friend. Show empathy and encouragement in every sentence.",
                    "You are cold and curt. No empathy, no pleasantries, no encouragement. Just the facts."),
    "calm":        ("Be calm, quiet and soothing. Speak softly and slowly, never use exclamation marks.",
                    "Be EXTREMELY enthusiastic and excited!!! Use lots of exclamation marks and hype!!!"),
    "simple":      ("Explain to a complete beginner, in plain everyday words.",
                    "Explain like an expert, with dense technical jargon."),
    "direct":      ("Give the answer in the very first sentence. No preamble.",
                    "Start with a long preamble and filler before slowly getting to the point."),
}
QUESTIONS = [
    "How do I sort a list in Python?", "Why is the sky blue?", "My code throws a KeyError, what do I do?",
    "Explain what an API is.", "How do I center a div?", "What's the difference between TCP and UDP?",
    "How can I learn to cook?", "Why does my program run slowly?", "What is a good way to start running?",
    "How do vaccines work?", "Can you explain recursion?", "How do I write a cover letter?",
    "What does git rebase do?", "Is coffee bad for you?", "How do I fix a merge conflict?",
    "What is inflation?", "How do I make my React app faster?", "Why did my sourdough not rise?",
    "What is a neural network?", "How should I prepare for a job interview?", "How do I read a CSV in pandas?",
    "What causes the seasons?", "How do I stop procrastinating?", "What is the difference between RAM and storage?",
]
ANSWER_TOKENS = 60  # the style shows in the first sentences
MAX_SIMILARITY = 0.8  # drop pairs whose answers are more alike than this (difflib ratio)


def generate_pairs(names, m):
    """Write data/pairs/<name>.jsonl for each name, in one batched generation."""
    convos = [[{"role": "system", "content": persona}, {"role": "user", "content": q}]
              for name in names for persona in AXES[name] for q in QUESTIONS]
    t = time.time()
    print(f"generating {len(convos)} example answers ({', '.join(names)})…", end=" ", flush=True)
    answers = iter(m.generate_batch(convos, max_new_tokens=ANSWER_TOKENS))
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        qs = QUESTIONS
        pos, neg = [next(answers) for _ in qs], [next(answers) for _ in qs]
        rows = [{"question": q, "positive": p, "negative": n} for q, p, n in zip(qs, pos, neg)
                if p and n and difflib.SequenceMatcher(None, p, n).ratio() <= MAX_SIMILARITY]
        (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"  {name:<11} kept {len(rows)}/{len(qs)} pairs")
    print(f"done in {time.time() - t:.0f}s → {DATA}/")


def load_pairs(name):
    path = DATA / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(name, m, layer=None):
    rows = load_pairs(name)
    if layer is None:
        layer = config.STEER_LAYER if config.STEER_LAYER is not None else len(m.layers) // 2
    prefixes = [m.chat_text([{"role": "system", "content": config.SYSTEM_PROMPT},
                             {"role": "user", "content": r["question"]}]) for r in rows]
    starts = [m.n_tokens(p) for p in prefixes]
    pos = m.acts([p + r["positive"] for p, r in zip(prefixes, rows)], layer, starts)
    neg = m.acts([p + r["negative"] for p, r in zip(prefixes, rows)], layer, starts)
    vec = pos.mean(0) - neg.mean(0)
    size = torch.cat([pos, neg]).norm(dim=1).mean()  # a typical hidden state
    strength = (vec.norm() / size).item()            # distance between the poles, before rescaling
    print(f"{name:<11} {len(rows):>5}  {'█' * max(1, round(strength * 40))} {strength:.0%}"
          + ("  weak" if strength < 0.08 else ""))
    return {"vec": vec / vec.norm() * size, "layer": layer}


def similarity(vectors):
    """Cosine similarity between axes. Near ±1, two axes push the same direction and add up when combined."""
    names = list(vectors)
    print("\nsimilarity   " + " ".join(f"{n[:6]:>6}" for n in names))
    for a in names:
        row = [torch.cosine_similarity(vectors[a]["vec"], vectors[b]["vec"], dim=0).item() for b in names]
        print(f"{a:<11}  " + " ".join(f"{v:+6.2f}" for v in row))


if __name__ == "__main__":
    t = time.time()
    regen = "--regen" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(AXES)
    print(f"loading {config.MODEL_ID}…", end=" ", flush=True)
    m = get_model()
    layer = config.STEER_LAYER if config.STEER_LAYER is not None else len(m.layers) // 2
    missing = [n for n in names if regen or not (DATA / f"{n}.jsonl").exists()]
    print(f"{len(m.layers)} layers · steering layer {layer} · examples: "
          + (f"generating {len(missing)}" if missing else f"cached in {DATA}/"))
    if missing:
        generate_pairs(missing, m)
    Path(config.VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n{'vector':<11} pairs  strength")
    vectors = {name: build(name, m) for name in names}
    for name, v in vectors.items():
        torch.save(v, Path(config.VECTORS_DIR) / f"{name}.pt")
    similarity(vectors)
    print(f"\nsaved {len(names)} vectors → {config.VECTORS_DIR}/  ({time.time() - t:.0f}s)")
