"""Settings shared by every module."""
import os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")  # the committed vectors only fit this model
FAKE = os.getenv("FAKE") == "1"  # canned replies, no GPU or torch
DTYPE = os.getenv("DTYPE", "float16")  # float32 on CPU
VECTORS_DIR = "artifacts/vectors"
MOODS_FILE = "artifacts/moods.pt"
READ_LAYER = None   # layer the mood probe reads, None = middle
STEER_LAYER = None  # layer the vectors are added to, None = middle
TRACE_FILE = "runs/trace.jsonl"

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 400
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # rows per generate call: lower it if memory runs out

# "llm": the model lists the message's moods as JSON. "probe": experimental, needs detector/build.py.
DETECTOR = os.getenv("DETECTOR", "llm")
MOODS = ["neutral", "curious", "confused", "frustrated", "rage", "sad", "rushed", "excited", "skeptical",
         "wants_shorter", "wants_detail"]

# a mood counts only at this intensity (0..1) or above
THRESHOLD = 0.3

# mood → {axis: (target, speed)}. Each turn: state += intensity × speed × (target − state).
# A positive target pushes toward the axis's first pole, a negative one toward the second (POLES).
MAPPING = {
    "neutral":    {},
    "curious":    {"cautious": (0.5, 0.5), "concise": (0.0, 0.25), "direct": (0.0, 0.25)},
    "confused":   {"simple": (1.0, 0.8), "concise": (0.0, 0.5)},
    "frustrated": {"concise": (0.55, 0.8), "direct": (0.6, 0.8), "accountable": (0.6, 0.8)},
    "rage":       {"calm": (1.0, 1.0), "concise": (0.45, 1.0), "accountable": (0.8, 1.0)},
    "sad":        {"warm": (1.0, 0.8), "calm": (0.5, 0.8), "concise": (0.0, 0.5), "direct": (0.0, 0.5)},
    "rushed":     {"concise": (1.0, 1.0), "direct": (1.0, 1.0)},
    "excited":    {"calm": (-0.5, 0.8)},
    "skeptical":  {"cautious": (1.0, 0.8)},
    "wants_shorter": {"concise": (1.0, 1.0), "direct": (1.0, 1.0)},
    "wants_detail": {"concise": (0.0, 1.0), "direct": (0.0, 1.0)},
}

# name of each axis's negative pole, for display
POLES = {"concise": "long", "warm": "cold", "calm": "enthusiastic", "simple": "technical", "direct": "roundabout",
         "cautious": "overconfident", "accountable": "defensive"}

# how hard each axis pushes at α = ±1, as a fraction of a typical hidden state.
# Pick with vectors.tune, check with eval.steer. On 3B, 0.8 starts costing facts (cautious the most) and 2.0 breaks
# the model; 0.6 leaves room because a mood pushes 2-3 axes at once. warm, direct and accountable are unmeasured.
STRENGTH = {
    "concise": 0.6, "warm": 0.3, "calm": 0.6, "simple": 0.6,
    "direct": 0.3, "cautious": 0.4, "accountable": 0.3,
}

# cap on the combined push of one reply, same unit as STRENGTH. Above it all axes shrink together for that
# reply; the stored state is unchanged.
MAX_PUSH = 0.35
