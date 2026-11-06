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

# mood detection: the model reads the latest message and lists the moods present, each with intensity 0-10 (→ 0..1).
#   "probe" = experimental activation reader (needs detector/build.py), falls back to "llm".
DETECTOR = os.getenv("DETECTOR", "llm")
MOODS = ["neutral", "curious", "confused", "frustrated", "rage", "sad", "rushed", "excited", "skeptical",
         "wants_detail", "wants_prose"]

# steering memory, per axis (core/controller.py)
#   a mood with intensity ≥ THRESHOLD moves only the axes it lists: state += intensity × speed × (target − state)
#   everything else stays put, and a neutral message changes nothing. Moods in one message apply weakest first.
THRESHOLD = 0.3

# mood → {axis: (target, speed)}. Each axis is one vector (vectors/build.py AXES) with two poles:
#   + pushes toward the first pole, − toward the second. Missing vectors are skipped.
#   speed 1 at intensity 10/10 jumps straight to the target. Target 0 with a low speed eases off an axis.
MAPPING = {
    "neutral":    {},
    "curious":    {"cautious": (0.5, 0.5), "concise": (0.0, 0.25), "direct": (0.0, 0.25)},
    "confused":   {"simple": (1.0, 0.8), "structured": (0.7, 0.8), "concise": (0.0, 0.5)},
    "frustrated": {"concise": (0.55, 0.8), "direct": (0.6, 0.8), "accountable": (0.6, 0.8)},
    "rage":       {"calm": (1.0, 1.0), "concise": (0.45, 1.0), "accountable": (0.8, 1.0)},
    "sad":        {"warm": (1.0, 0.8), "calm": (0.5, 0.8), "concise": (0.0, 0.5), "direct": (0.0, 0.5)},
    "rushed":     {"concise": (1.0, 1.0), "direct": (1.0, 1.0)},
    "excited":    {"calm": (-0.5, 0.8)},
    "skeptical":  {"cautious": (1.0, 0.8)},
    "wants_detail": {"concise": (0.0, 1.0), "direct": (0.0, 1.0)},
    "wants_prose":  {"structured": (-1.0, 1.0)},  # keyword rule only (detector/detect.py), the model isn't asked
}

# the other pole of each axis, for display
POLES = {"concise": "long", "warm": "cold", "calm": "enthusiastic", "simple": "technical", "direct": "roundabout",
         "cautious": "overconfident", "structured": "freeform", "accountable": "defensive"}

# strength: how hard each axis pushes at α = ±1, as a fraction of a typical hidden state (0.1 = 10%).
#   Vectors are rescaled when built, so the same number means the same push on every axis.
#   Pick values with: python -m vectors.tune, check them with: python -m eval.steer
#   From eval.steer on 3B: concise 0.4 cuts words 35% with no fact loss, 0.8 cuts 62% (25/30 facts), 2.0 breaks the
#   model. It's 0.6 so a very rushed user gets clearly shorter answers; 0.6 itself hasn't been measured yet.
#   simple and calm keep gaining up to 0.8, but kept at 0.6 since a mood pushes 2-3 axes at once and they add up.
#   cautious 0.4 hedges more with no fact loss; at 0.8 it doubts even plain facts (21/30).
#   The others aren't measured yet (warm, direct, accountable need a judge).
STRENGTH = {
    "concise": 0.6, "warm": 0.3, "calm": 0.6, "simple": 0.6,
    "direct": 0.3, "cautious": 0.4, "structured": 0.3, "accountable": 0.3,
}

# limit on the total push of one reply, same unit as STRENGTH: moods add up, and too much breaks the model.
#   Above it, all axes shrink together for that reply. The steering memory itself isn't touched.
MAX_PUSH = 0.5
