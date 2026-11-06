"""Web demo: the same view as the CLI. Steering state, mood of each message, steered and/or baseline reply.
  python app.py            # local: http://localhost:7860
  python app.py --share    # public link (e.g. from a notebook)
Notebook: import app; app.demo.launch(share=True)   (MODEL_ID / DTYPE env vars, if any, before importing)"""
import queue
import sys
import threading
import uuid

import gradio as gr

import config
from core.chat import chat, reset
from core.controller import push
from core.model import STOP, get_model

STAGES = {"detect": "detecting mood…", "steered": "generating · steered…", "baseline": "generating · baseline…"}


def meter(v, width=12):
    full = round(max(0.0, min(1.0, abs(v))) * width)
    return "█" * full + "░" * (width - full)


def panel(scores, alphas):
    """Two monospace tables: the steering state per axis, kept across messages, and this message's moods."""
    state = ["STEERING STATE (kept across messages)"]
    for axis in config.STRENGTH:
        a = alphas.get(axis, 0.0)
        left, right = (meter(a, 6)[::-1], "░" * 6) if a < 0 else ("░" * 6, meter(a, 6))
        line = f"{config.POLES[axis]:>13} {left}|{right} {axis:<11} {a:+.2f}"
        state.append(f"<b style='color:#0891b2'>{line}</b>" if abs(a) >= 0.05 else line)
    p = push(alphas)
    state += ["", f"total push {min(p, config.MAX_PUSH):.2f}" + (f" (capped from {p:.2f})" if p > config.MAX_PUSH else "")]
    mood = ["MOOD (this message only)"]
    for m in config.MOODS:
        v = scores.get(m, 0.0)
        line = f"{m:<13} {meter(v)} {v:.2f}"
        mood.append(f"<b style='color:#d97706'>{line}</b>" if v >= config.THRESHOLD else line)
    pre = "<pre style='margin:0;font-size:13px;line-height:1.35'>{}</pre>"
    return (f"<div style='display:flex;gap:40px;flex-wrap:wrap'>"
            f"{pre.format(chr(10).join(state))}{pre.format(chr(10).join(mood))}</div>")


def respond(msg, sid, use_steered, use_baseline, steered, baseline, mood_html):
    msg = msg.strip()
    if not msg:
        yield "", sid, steered, baseline, mood_html, ""
        return
    if not (use_steered or use_baseline):
        yield msg, sid, steered, baseline, mood_html, "steered and baseline are both off: turn one on"
        return
    STOP.set()  # a new message interrupts the reply in progress
    sid = sid or uuid.uuid4().hex
    events = queue.Queue()

    def run():
        try:
            chat(sid, msg, steered=use_steered, baseline=use_baseline, on_event=events.put)
        except Exception as e:  # show it instead of hanging the UI
            events.put({"type": "error", "error": repr(e)})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()
    user = {"role": "user", "content": msg}
    if use_steered:
        steered = steered + [user, {"role": "assistant", "content": ""}]
    if use_baseline:
        baseline = baseline + [user, {"role": "assistant", "content": ""}]
    done, status = [], ""
    while (e := events.get()) is not None:
        if e["type"] == "stage":
            status = STAGES[e["stage"]]
        elif e["type"] == "detected":
            mood_html = panel(e["scores"], e["alphas"])
            done.append(f"mood {e['seconds']:.1f}s")
        elif e["type"] == "token":
            (steered if e["stream"] == "steered" else baseline)[-1]["content"] += e["text"]
        elif e["type"] == "replied":
            done.append(f"{e['stream']} {e['words']} words in {e['seconds']:.1f}s")
        elif e["type"] == "error":
            done.append(f"error: {e['error']}")
        yield "", sid, steered, baseline, mood_html, " · ".join(done + ([status] if status else []))
        status = ""
    for pane, on in ((steered, use_steered), (baseline, use_baseline)):
        if on and not pane[-1]["content"]:
            pane[-1]["content"] = "(stopped)"
    yield "", sid, steered, baseline, mood_html, " · ".join(done + (["stopped"] if STOP.is_set() else []))


def new_chat(sid):
    STOP.set()
    if sid:
        reset(sid)
    return "", [], [], panel({}, {}), ""


get_model()  # load once at startup, not on the first message

with gr.Blocks(title="Moodsteer") as demo:
    gr.Markdown("## Moodsteer\nChat assistant that reads your mood and changes its behavior using vector steering. "
                f"Model: `{'FAKE' if config.FAKE else config.MODEL_ID}`")
    sid = gr.State("")
    mood = gr.HTML(panel({}, {}))
    with gr.Row():
        use_steered = gr.Checkbox(True, label="steered")
        use_baseline = gr.Checkbox(True, label="baseline")
    with gr.Row():
        steered = gr.Chatbot(label="steered", height=480)
        baseline = gr.Chatbot(label="baseline (no steering)", height=480)
    status = gr.Markdown()
    with gr.Row():
        box = gr.Textbox(placeholder="Type a message and press Enter…", show_label=False, scale=8)
        stop = gr.Button("Stop", scale=1)
        clear = gr.Button("New chat", scale=1)
    turn = box.submit(respond, [box, sid, use_steered, use_baseline, steered, baseline, mood],
                      [box, sid, steered, baseline, mood, status], trigger_mode="multiple", concurrency_limit=None)
    stop.click(STOP.set, queue=False)
    clear.click(new_chat, [sid], [box, steered, baseline, mood, status], cancels=[turn])

if __name__ == "__main__":
    demo.queue().launch(share="--share" in sys.argv)
