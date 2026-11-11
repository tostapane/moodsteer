"""End-to-end test in FAKE mode, stdlib only: python tests/smoke.py"""
import os
import sys

os.environ["FAKE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.TRACE_FILE = "runs/smoke.jsonl"

import core.model  # noqa: E402
from core.chat import SESSIONS, chat  # noqa: E402
from core.controller import capped, push  # noqa: E402
from detector.detect import PROMPT, detect, with_request  # noqa: E402

# fake model doesn't answer JSON → detector can't answer → neutral, no steering
r = chat("smoke-a", "WTF THIS IS USELESS!!!")
assert r["scores"]["neutral"] == 1.0 and r["alphas"] == {}, r


class JsonModel(core.model.FakeModel):
    """Answers the mood prompt with JSON, everything else like the fake model."""
    answer = None
    seen = None

    def generate(self, messages, max_new_tokens=None, on_token=None):
        if messages[0]["content"] == PROMPT:
            last = JsonModel.seen = messages[-1]["content"]
            return self.answer or ('{"rage": 10}' if "WTF" in last else '{"frustrated": 9}' if "wrong" in last else '{}')
        return super().generate(messages, max_new_tokens, on_token)


core.model._model = JsonModel()

r = chat("smoke-b", "my code crashes")
assert set(r) == {"reply", "scores", "alphas"} and r["alphas"] == {}, r
r = chat("smoke-b", "no that's wrong")  # concise: .9 × .8 × (.55 − 0)
assert r["scores"]["frustrated"] == 0.9 and r["scores"]["rage"] == 0.0, r
assert abs(r["alphas"]["concise"] - 0.396) < 1e-9, r
before = r["alphas"]
r = chat("smoke-b", "ok, next topic")  # neutral → nothing changes
assert r["alphas"] == before, r
r = chat("smoke-b", "still wrong")  # moves further toward the target
assert abs(r["alphas"]["concise"] - (0.396 + 0.72 * (0.55 - 0.396))) < 1e-9, r

r = chat("smoke-c", "WTF THIS IS USELESS")
assert abs(r["alphas"]["calm"] - 1.0) < 1e-9, r

r = chat("smoke-d", "WTF", steered=False, baseline=True)  # steered off: no detection, no steered reply, memory untouched
assert "reply" not in r and r["scores"] == {} and r["alphas"] == {} and "steering: none" in r["baseline"], r
assert JsonModel.seen == "WTF THIS IS USELESS", JsonModel.seen  # the detector wasn't called, and it only sees the message

events = []
r = chat("smoke-e", "WTF!!!", baseline=True, on_event=events.append)
kinds = [e["type"] for e in events]
assert "baseline" in r and kinds.count("detected") == 1, kinds  # one detection for both replies
assert any(e["type"] == "token" and e["stream"] == "baseline" for e in events), events
assert "[fake | steering: none]" in r["baseline"] and "calm" in r["reply"], r

JsonModel.answer = '{"excited": 10}'  # excited → calm −0.5: the other pole
r = chat("smoke-f", "yay")
assert r["alphas"]["calm"] < 0, r

JsonModel.answer = '{"rushed": 10}'  # very rushed jumps to the target
r = chat("smoke-g", "quick")
assert r["alphas"]["concise"] == 1.0 and "concise=0.75" in r["reply"], r  # memory 1.0, capped for the reply
assert abs(push(capped(r["alphas"])) - config.MAX_PUSH) < 1e-9 and capped({"concise": 0.5}) == {"concise": 0.5}
JsonModel.answer = '{}'  # neutral changes nothing
assert chat("smoke-g", "and then?")["alphas"]["concise"] == 1.0
JsonModel.answer = '{"curious": 6}'  # curious eases concise a little, and adds its own axis
r = chat("smoke-g", "why?")
assert abs(r["alphas"]["concise"] - 0.85) < 1e-9 and abs(r["alphas"]["cautious"] - 0.15) < 1e-9, r

JsonModel.answer = '{"wants_detail": 10}'  # "tell me more" eases concise by the request's amount, keeps the rest
r = chat("smoke-g", "tell me more")
assert abs(r["alphas"]["concise"] - 0.85 * 0.4) < 1e-9 and "cautious" in r["alphas"], r

JsonModel.answer = '{"rushed": 8, "frustrated": 6}'  # both count, the stronger applies last
r = chat("smoke-h", "still broken, hurry")
assert r["scores"]["rushed"] == 0.8 and r["scores"]["frustrated"] == 0.6, r
assert abs(r["alphas"]["concise"] - 0.8528) < 1e-9 and abs(r["alphas"]["accountable"] - 0.288) < 1e-9, r

JsonModel.answer = 'Sure! {"sad": "x"}'  # not a number → detector gives up → neutral
assert detect("I give up")["neutral"] == 1.0
JsonModel.answer = '{"confused": 30}'  # clamped to 10
s = detect("I don't get it")
assert s["confused"] == 1.0 and s["neutral"] == 0.0 and s["sad"] == 0.0, s
JsonModel.answer = '{"bored": 8}'  # unknown mood → neutral
assert detect("meh")["neutral"] == 1.0

JsonModel.answer = '{"sad": 6}\n"What is rain?" → {}'  # it kept writing examples: the first object counts
assert detect("meh")["sad"] == 0.6

# explicit requests override the model; how much shorter depends on the wording
JsonModel.answer = '{}'
r = chat("smoke-k", "explain briefly how a helicopter flies")
assert r["scores"]["rushed"] == 0.6 and r["alphas"]["concise"] == 0.6, r
r = chat("smoke-k", "from now on explain every single detail")
assert r["scores"]["wants_detail"] == 1.0 and r["scores"]["rushed"] == 0.0 and "concise" not in r["alphas"], r
JsonModel.answer = '{"excited": 6, "wants_detail": 8}'  # the model got the direction wrong: the request wins
r = chat("smoke-m", "this is wayy too much, keep it a little shorter")
assert r["scores"]["rushed"] == 0.3 and r["scores"]["wants_detail"] == 0.0 and abs(r["alphas"]["concise"] - 0.3) < 1e-9, r
JsonModel.answer = '{}'
for msg, mood, level in (("be very very concise", "rushed", 1.0), ("one sentence only", "rushed", 1.0),
                         ("please dont procude wall of text", "rushed", 0.6), ("a bit shorter please", "rushed", 0.3),
                         ("explain it in depth", "wants_detail", 1.0), ("dont be so brief", "wants_detail", 0.6),
                         ("okay with little more detail?", "wants_detail", 0.3)):
    assert with_request(msg, None)[mood] == level, msg
assert with_request("what is rain?", None) is None
assert with_request("what is a list comprehension? no list comprehension please", None) is None

# relative requests move concise step by step, both ways
steps = [("explain to a 8 years old briefly how butter is made", 0.6), ("okay with little more detail?", 0.42),
         ("still too long", 0.42 + 0.6 * 0.58), ("now too short", 0.768 * 0.4)]
for msg, want in steps:
    got = chat("smoke-n", msg)["alphas"].get("concise", 0.0)
    assert abs(got - want) < 1e-9, (msg, got, want)

# stop: the reply ends early, the baseline is skipped, the partial reply stays, the next turn runs normally
JsonModel.answer = '{}'
r = chat("smoke-i", "hello", baseline=True, on_event=lambda e: e["type"] == "token" and core.model.STOP.set())
assert "baseline" not in r and "you said" not in r["reply"], r
assert SESSIONS["smoke-i"]["history"][-1]["content"] == r["reply"] and SESSIONS["smoke-i"]["baseline_history"] == []
assert "you said" in chat("smoke-i", "again")["reply"]

# CLI word wrap: no line longer than the width, words kept intact
import cli  # noqa: E402

turn = cli.Turn(None, None, 0)
turn.col = cli.INDENT
text = "".join(turn.wrap(chunk) for chunk in ["Waterfalls are natural ", "formations where a river ",
                                               "flows over a steep drop. " * 12, "\nNext line."])
lines = text.split("\n")
assert all(len(line) <= cli.MAX_WIDTH for line in lines) and len(lines) > 3, lines
assert "steep drop." in text and lines[-1].strip() == "Next line.", lines

print("smoke ok")
