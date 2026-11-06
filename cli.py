"""Terminal chat that shows every step live: mood detection, steering, streamed replies, status bar.
Runs the whole pipeline in-process.
  python cli.py                          # Qwen2.5-3B on GPU (override with MODEL_ID / DTYPE env vars)
  MODEL_ID=... python cli.py             # another model: rebuild the vectors for it first (python -m vectors.build)
  FAKE=1 python cli.py                   # no GPU, canned replies
Commands: /steer  toggle the steered reply · /compare  toggle the baseline reply · /reset  new conversation · /quit
Ctrl+C stops the reply in progress."""
import itertools
import re
import shutil
import signal
import sys
import threading
import time

import config
from core.chat import chat, reset
from core.controller import push
from core.model import STOP, get_model

TTY = sys.stdout.isatty()
MAX_WIDTH = 100  # replies wrap at the terminal width, capped here so lines stay readable full screen
INDENT = len("baseline › ")
COLORS = {"dim": 2, "bold": 1, "cyan": 36, "green": 32, "yellow": 33, "red": 31, "magenta": 35}


def c(name, text):
    return f"\033[{COLORS[name]}m{text}\033[0m" if TTY else text


class Screen:
    """Normal scrolling output, plus a status bar pinned to the last row (ANSI scroll region)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.busy = None        # what we're doing right now, None = idle
        self.status = ""        # right part of the bar
        self.frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        if TTY:
            sys.stdout.write("\n")
            self.region()
            signal.signal(signal.SIGWINCH, lambda *_: self.region())
            threading.Thread(target=self.spin, daemon=True).start()

    def region(self):
        rows = shutil.get_terminal_size().lines
        with self.lock:
            sys.stdout.write(f"\0337\033[1;{rows - 1}r\0338")  # scroll everything except the last row
        self.bar()

    def write(self, text):
        with self.lock:
            sys.stdout.write(text)
            sys.stdout.flush()

    def bar(self, busy=..., status=None):
        if busy is not ...:
            self.busy = busy
        if status is not None:
            self.status = status
        if not TTY:
            return
        cols, rows = shutil.get_terminal_size()
        state = f"{next(self.frames)} {self.busy}" if self.busy else "● idle"
        line = f" {state} │ {self.status}"[:cols - 1].ljust(cols - 1)
        with self.lock:
            sys.stdout.write(f"\0337\033[{rows};1H\033[2K\033[7m{line}\033[0m\0338")
            sys.stdout.flush()

    def spin(self):
        while True:
            time.sleep(0.1)
            if self.busy:
                self.bar()

    def close(self):
        if TTY:
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[r\033[{rows};1H\033[2K")
            sys.stdout.flush()


def meter(v, width=12):
    full = round(max(0.0, min(1.0, v)) * width)
    return "█" * full + "░" * (width - full)


class Turn:
    """Renders the events of one chat() call."""

    def __init__(self, screen, model, n):
        self.screen, self.model, self.n = screen, model, n
        self.words = 0
        self.times = {}
        self.col, self.space = 0, False  # current column, a space is pending before the next word

    def __call__(self, e):
        s = self.screen
        if e["type"] == "stage":
            label = {"detect": "detecting mood", "steered": "generating · steered",
                     "baseline": "generating · baseline"}[e["stage"]]
            s.bar(busy=label)
            if e["stage"] != "detect":
                self.words, self.col, self.space = 0, INDENT, False
                label = f"{e['stage']} ›".ljust(INDENT)
                s.write("\n" + (c("green", label) if e["stage"] == "steered" else c("dim", label)))
        elif e["type"] == "detected":
            self.times["detect"] = e["seconds"]
            s.write(self.mood_table(e["scores"], e["alphas"]))
        elif e["type"] == "token":
            text = self.wrap(e["text"])
            s.write(text if e["stream"] == "steered" else c("dim", text))
            self.words += e["text"].count(" ") + e["text"].count("\n")
            s.bar(busy=f"generating · {e['stream']} · {self.words} words")
        elif e["type"] == "replied":
            self.times[e["stream"]] = e["seconds"]
            s.write("\n" + c("dim", f"  └ {e['words']} words in {e['seconds']:.1f}s") + "\n")

    def wrap(self, text):
        """Word-wrap streamed text at the terminal width (max MAX_WIDTH); wrapped lines are indented."""
        width = min(shutil.get_terminal_size((MAX_WIDTH, 24)).columns, MAX_WIDTH)
        out = ""
        for part in re.split(r"(\s+)", text):
            if not part:
                continue
            if part.isspace():
                if "\n" in part:
                    out += "\n" * part.count("\n") + " " * INDENT
                    self.col, self.space = INDENT, False
                else:
                    self.space = True
                continue
            if self.space and self.col + 1 + len(part) > width:
                out += "\n" + " " * INDENT
                self.col = INDENT
            elif self.space:
                out += " "
                self.col += 1
            out += part
            self.col += len(part)
            self.space = False
        return out

    def mood_table(self, scores, alphas):
        vectors = getattr(self.model, "vectors", None)
        if alphas:
            parts = []
            for b, a in sorted(alphas.items(), key=lambda kv: -abs(kv[1])):
                missing = vectors is not None and b not in vectors
                parts.append(f"{b if a >= 0 else config.POLES.get(b, b)} {abs(a):.2f}"
                             + (c("red", " (no vector)") if missing else ""))
            p = push(alphas)
            parts.append(c("dim", f"total push {min(p, config.MAX_PUSH):.2f}"
                                  + (f" (capped from {p:.2f})" if p > config.MAX_PUSH else "")))
            rows = [c("cyan", "\n  steering state → ") + " · ".join(parts)]
        else:
            rows = [c("cyan", "\n  steering state → ") + c("dim", "none")]
        rows.append(c("dim", "\n  mood          this message only"))
        for m in config.MOODS:
            v = scores.get(m, 0.0)
            line = f"  {m:<13} {meter(v)} {v:.2f}"
            if v >= config.THRESHOLD:
                line = c("yellow", line)
            elif v < 0.05:
                line = c("dim", line)
            rows.append(line)
        return "\n".join(rows) + "\n"


def main():
    screen = Screen()
    screen.bar(busy=f"loading {config.MODEL_ID}", status="")
    model = get_model()
    vectors = sorted(getattr(model, "vectors", {}) or [])
    screen.write(
        c("bold", "moodsteer") + c("dim", " · /steer /compare /reset /quit · Ctrl+C stops the reply") + "\n"
        + c("dim", f"model {'FAKE' if config.FAKE else config.MODEL_ID} · detector {config.DETECTOR} · "
                   f"vectors {', '.join(vectors) or 'none (run: python -m vectors.build)'}") + "\n")
    steer, compare, n, session = True, True, 0, f"cli-{int(time.time())}"

    def status(times=None):
        t = times or {}
        parts = [f"turn {n}", f"steer {'on' if steer else 'off'}", f"compare {'on' if compare else 'off'}"]
        parts += [f"{k} {v:.1f}s" for k, v in t.items()]
        return " · ".join(parts)

    screen.bar(busy=None, status=status())
    try:
        while True:
            try:
                msg = input(c("bold", "\nyou › ")).strip()
            except EOFError:
                break
            if not msg:
                continue
            if msg == "/quit":
                break
            if msg in ("/steer", "/compare", "/reset"):
                if msg == "/steer":
                    steer = not steer
                elif msg == "/compare":
                    compare = not compare
                else:
                    reset(session)
                    n = 0
                screen.write(c("dim", f"  {msg[1:]}: done") + "\n")
                screen.bar(status=status())
                continue
            if not (steer or compare):
                screen.write(c("dim", "  steered and baseline are both off: /steer or /compare turns one on") + "\n")
                continue
            n += 1
            turn = Turn(screen, model, n)
            worker = threading.Thread(target=chat, args=(session, msg), daemon=True,
                                      kwargs=dict(steered=steer, baseline=compare, on_event=turn))
            worker.start()
            while worker.is_alive():
                try:
                    worker.join(0.1)
                except KeyboardInterrupt:
                    STOP.set()
                    screen.write("\n" + c("dim", "  stopping") + "\n")
            screen.bar(busy=None, status=status(turn.times))
    except KeyboardInterrupt:
        pass
    finally:
        screen.close()
        print()


if __name__ == "__main__":
    main()
