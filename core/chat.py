"""One chat turn: detect the mood, update the steering state, write the steered and/or baseline reply, log it.

The two replies keep separate histories. With steering off the mood isn't detected and the state doesn't change.
Setting core.model.STOP ends the turn early and keeps the partial reply.

on_event receives progress events:
    {"type": "stage", "stage": "detect" | "steered" | "baseline"}
    {"type": "detected", "scores", "alphas", "seconds"}   alphas = the state after this message
    {"type": "token", "stream": "steered" | "baseline", "text"}
    {"type": "replied", "stream", "seconds", "words"}"""
import json
import threading
import time
from pathlib import Path

import config
from core.controller import capped, update
from core.model import STOP, get_model
import detector.detect
from detector.detect import detect

SESSIONS = {}            # session_id → {"history": [...], "baseline_history": [...], "state": {axis: α}}
LOCK = threading.Lock()  # one turn at a time: steering is global state on the model


def chat(session_id: str, msg: str, steered: bool = True, baseline: bool = False, on_event=None) -> dict:
    emit = on_event or (lambda e: None)
    with LOCK:
        STOP.clear()
        t = time.time()
        s = SESSIONS.setdefault(session_id, {"history": [], "baseline_history": [], "state": {}})
        scores = {}
        if steered:
            emit({"type": "stage", "stage": "detect"})
            scores = detect(msg)
            s["state"] = update(s["state"], scores)
            emit({"type": "detected", "scores": scores, "alphas": s["state"], "seconds": time.time() - t})
        alphas = dict(s["state"])
        out = {"scores": scores, "alphas": alphas}
        if steered and not STOP.is_set():
            out["reply"] = reply(s["history"], msg, alphas, "steered", emit)
        if baseline and not STOP.is_set():
            out["baseline"] = reply(s["baseline_history"], msg, {}, "baseline", emit)
        log({"session_id": session_id, "msg": msg, "latency": time.time() - t,
             "detector_reply": detector.detect.last_reply, **out})
        return out


def reply(history: list, msg: str, alphas: dict, stream: str, emit) -> str:
    emit({"type": "stage", "stage": stream})
    t = time.time()
    model = get_model()
    model.set_alphas(capped(alphas))
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
