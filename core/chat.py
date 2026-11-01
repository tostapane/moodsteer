"""One turn: detect the mood, update the steering memory, reply steered (and unsteered with compare=True), log.
on_event(dict) gets live progress (used by the CLI and the web demo):
  {"type": "stage", "stage": "detect" | "steered" | "baseline"}
  {"type": "detected", "scores", "alphas", "seconds"}   alphas = the steering state applied this turn
  {"type": "token", "stream": "steered" | "baseline", "text"}
  {"type": "replied", "stream", "seconds", "words"}"""
import json
import threading
import time
from pathlib import Path

import config
from core.controller import update
from core.model import get_model
from detector.detect import detect

SESSIONS = {}            # session_id → {"history": [...], "baseline_history": [...], "state": {axis: α}, "n": mood msgs}
LOCK = threading.Lock()  # one turn at a time: steering is global state on the model


def chat(session_id: str, msg: str, steer: bool = True, compare: bool = False, on_event=None) -> dict:
    """compare=True also generates the unsteered reply (own history), with a single mood detection."""
    emit = on_event or (lambda e: None)
    with LOCK:
        t = time.time()
        s = SESSIONS.setdefault(session_id, {"history": [], "baseline_history": [], "state": {}, "n": 0})
        emit({"type": "stage", "stage": "detect"})
        scores = detect(msg, s["history"])
        s["state"], s["n"] = update(s["state"], scores, s["n"])  # memory is kept even with steer off
        alphas = dict(s["state"]) if steer else {}
        emit({"type": "detected", "scores": scores, "alphas": alphas, "seconds": time.time() - t})
        out = {"reply": reply(s["history"], msg, alphas, "steered", emit),
               "scores": scores, "alphas": alphas}
        if compare:
            out["baseline"] = reply(s["baseline_history"], msg, {}, "baseline", emit)
        log({"session_id": session_id, "steer": steer, "msg": msg, "latency": time.time() - t, **out})
        return out


def reply(history: list, msg: str, alphas: dict, stream: str, emit) -> str:
    emit({"type": "stage", "stage": stream})
    t = time.time()
    model = get_model()
    model.set_alphas(alphas)
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}, *history, {"role": "user", "content": msg}]
    text = model.generate(messages, on_token=lambda chunk: emit({"type": "token", "stream": stream, "text": chunk}))
    history += [{"role": "user", "content": msg}, {"role": "assistant", "content": text}]
    emit({"type": "replied", "stream": stream, "seconds": time.time() - t, "words": len(text.split())})
    return text


def reset(session_id: str):
    SESSIONS.pop(session_id, None)


def log(row):
    path = Path(config.TRACE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
