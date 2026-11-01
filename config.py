"""Shared knobs. Everything reads from here."""
import os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")  # the committed vectors are built for this model
FAKE = os.getenv("FAKE") == "1"  # FAKE=1 → no GPU/torch, canned replies
DTYPE = os.getenv("DTYPE", "float16")  # float16 works on a Colab T4; float32 on CPU
VECTORS_DIR = "artifacts/vectors"
MOODS_FILE = "artifacts/moods.pt"
READ_LAYER = None   # layer the detector reads (None = middle)
STEER_LAYER = None  # layer the vectors are added to (None = middle)
TRACE_FILE = "runs/trace.jsonl"

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 400
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # rows generated at once: bounds memory on long lists

# mood detection: the model reads the conversation and names ONE dominant mood + intensity 0-10 (→ 0..1).
#   "probe" = experimental activation reader (needs detector/build.py), falls back to "llm".
DETECTOR = os.getenv("DETECTOR", "llm")
MOODS = ["neutral", "curious", "confused", "frustrated", "rage", "sad", "rushed", "excited", "skeptical"]

# steering memory, per axis (core/controller.py)
#   message shows a mood (intensity ≥ THRESHOLD): state = (1 − w) × state + w × axes that mood wants
#     w = max(1 − KEEP, 1/n), n = mood messages so far → 100% on the first, 50%, then capped at 40%
#   neutral message (no new info):                 state = HOLD × state   (slow fade)
KEEP = 0.6
HOLD = 0.9
THRESHOLD = 0.3

# mood → axis weights. Each axis is one vector (vectors/build.py AXES) with two poles:
#   + pushes toward the first pole, − toward the second. Missing vectors are skipped.
MAPPING = {
    "neutral":    {},
    "curious":    {"cautious": 0.5},
    "confused":   {"simple": 1.0, "structured": 0.7},
    "frustrated": {"concise": 0.8, "direct": 0.6, "accountable": 0.6},
    "rage":       {"calm": 1.0, "concise": 0.7, "accountable": 0.8},
    "sad":        {"warm": 1.0, "calm": 0.5},
    "rushed":     {"concise": 1.0, "direct": 1.0},
    "excited":    {"calm": -0.5},
    "skeptical":  {"cautious": 1.0},
}

# strength: how hard each axis pushes at α = ±1, as a fraction of a typical hidden state (0.1 = 10%).
#   Vectors are rescaled when built, so the same number means the same push on every axis.
#   Pick values with: python -m vectors.tune, check them with: python -m eval.steer
#   From eval.steer on 3B: concise 0.4 halves words with no fact loss (0.8 → 25/30 facts, 2.0 breaks the model);
#   simple and calm keep gaining up to 0.8, but kept at 0.6 since a mood pushes 2-3 axes at once and they add up.
#   cautious 0.4 hedges more with no fact loss; at 0.8 it doubts even plain facts (21/30).
#   The others aren't measured yet (warm, direct, accountable need a judge).
STRENGTH = {
    "concise": 0.4, "warm": 0.3, "calm": 0.6, "simple": 0.6,
    "direct": 0.3, "cautious": 0.4, "structured": 0.3, "accountable": 0.3,
}
