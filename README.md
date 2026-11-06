# Moodsteer

A chat assistant that adapts to the user's mood by steering the model's activations, not its prompt.
Frustrated? It gets shorter and more direct. Confused? Simpler. Sad? Warmer. The weights and the prompt never change:
a vector is added to the model's hidden state while it writes the reply.

## Quick start (Colab T4 GPU or any CUDA GPU, Qwen2.5-3B-Instruct)

The steering vectors for Qwen2.5-3B are in the repo, so there's nothing to build.

```bash
git clone https://github.com/tostapane/moodsteer && cd moodsteer
pip install -r requirements.txt     # torch, transformers, accelerate, gradio
python app.py --share               # web demo: public link printed in the output
python cli.py                       # or the terminal chat. /steer /compare /reset /quit, Ctrl+C stops the reply
FAKE=1 python cli.py                # no GPU: canned replies, shows the mood → steering pipeline
```

From a notebook: `import app; app.demo.launch(share=True)`.

The demo shows the steering state per axis, the moods it detected in the latest message, and the steered reply next to
the same model's reply without steering. Either reply can be turned off to save time, and Stop ends a reply early.

## Results (Qwen2.5-3B, `python -m eval.steer`)

Each axis is measured on 12 held-out questions, at three steering strengths, against simply writing the same instruction
in the system prompt. "Facts" = how many of 30 simple factual questions (capital of France, 12 × 12, …) it still gets
right under the same condition: the cost of the change.

| axis | metric | baseline | steer 0.2 | steer 0.4 | steer 0.8 | prompt |
|---|---|---|---|---|---|---|
| simple | reading ease (higher = simpler) | 37 | 45 | 63 | **74** | 65 |
| | facts | 30 | 30 | 30 | **30** | 23 |
| concise | words per answer | 91 | 86 | 59 | 34 | 12 |
| | facts | 30 | 29 | 30 | 25 | 30 |
| calm (enthusiastic pole) | `!` per 100 words | 0.1 | 0.3 | 1.3 | 3.7 | 4.6 |
| | facts | 30 | 29 | 30 | 28 | 29 |
| cautious | hedge words per 100 words | 1.1 | 1.2 | 1.8 | 1.6 | 1.2 |
| | facts | 30 | 29 | 30 | 21 | 30 |
| structured | % lines that are list items | 57 | 40 | 49 | 27 | 100 |
| | facts | 30 | 28 | 29 | 30 | 29 |

What it shows:
- **Simple, concise, calm and cautious move the intended way at almost no cost up to 0.4.** Past that it starts to
  hurt. At 0.8
  cautious doubts even plain facts ("the largest planet is a bit tricky to pinpoint"), and at 2.0 the model breaks.
- **Simple: steering beats the prompt.** It's simpler than the prompted version and keeps every fact. Asking for
  "plain everyday words" in the prompt costs 7 of 30.
- **Prompting is stronger for format.** It's far shorter, and gives perfect lists. Steered "structured" answers do
  become short one-idea lines, but without the `-`/`1.` markers the metric counts.
- **Middle layer is right.** On the concise vector (`python -m eval.layers`), layers 2–14 and 26–34 do almost nothing.
  Layer 18 (the middle, used everywhere) and 22 do the work.

Warm, direct and accountable can't be measured by counting, so they need an LLM judge and aren't measured yet.
Every reply behind these numbers is saved to `runs/` when you run the scripts.

## How it works

Every turn:
1. **Detect:** the model reads the latest message and lists the user's moods, each with an intensity
   (frustrated 6/10 and rushed 8/10, …).
2. **Remember:** a mood moves only the axes it lists, toward a target at its own speed:
   `state += intensity × speed × (target − state)`. Very rushed jumps straight to short answers, curious eases them
   a little, and a neutral message changes nothing.
3. **Steer:** each axis vector × its α × its strength is added to the hidden state at the middle layer
   while the reply is generated. If the axes add up to more than `MAX_PUSH`, they shrink together for that reply.
4. **Compare:** the same question is also answered without steering, side by side.

The 8 steering axes (+ pole ↔ − pole): concise ↔ long · warm ↔ cold · calm ↔ enthusiastic ·
simple ↔ technical · direct ↔ roundabout · cautious ↔ overconfident · structured ↔ freeform ·
accountable ↔ defensive. Which mood pushes which axes is `MAPPING` in `config.py`.

Each vector is built with contrastive completions: the model answers the same questions in both poles
(e.g. "answer in as few words as possible" vs "answer at great length"). The vector is the mean hidden state over the
answer tokens of the + answers minus the − answers, read **without** the instruction in the text, so it captures the
model producing the behaviour, not reading about it.

## Rebuild and evaluate

```bash
python -m vectors.build             # vectors from the committed example answers in data/pairs/ (no generation, fast)
python -m vectors.build --regen     # also rewrite the example answers with the model (needed for another model)
python -m vectors.tune              # each axis at strength 0.1/0.2/0.4/0.8 on one prompt, to pick config.STRENGTH
python -m detector.build            # optional experimental mood probe (the default LLM detector needs nothing)

python -m eval.steer                # the table above
python -m eval.layers               # which layer steers best
python -m eval.detector             # mood detector accuracy: 74 hand-written single-mood messages, 12 mixed
```

Another model: `MODEL_ID=... python -m vectors.build --regen`. The committed vectors only fit Qwen2.5-3B.
Environment variables are read once, when `config` is first imported.

How each axis is measured (`eval/steer.py`): concise → words · structured → share of list lines ·
simple → Flesch reading ease · calm → exclamation marks · cautious → hedge words, on questions with no sure answer
(which are also what its vector is built from).

## Knobs (`config.py`)

- `MAPPING`: which axes each mood moves, toward what target, how fast.
- `STRENGTH`: how hard each axis pushes at α = ±1, as a fraction of a typical hidden state (tuned from the table).
- `THRESHOLD`: the intensity a mood needs to count.
- `MAX_PUSH`: limit on the total push of one reply.
- `MODEL_ID`, `DTYPE`, `SYSTEM_PROMPT`, `MAX_NEW_TOKENS`, `BATCH_SIZE`.

## Code

- `detector/detect.py`: mood detection (the model answers with JSON).
- `core/controller.py`: steering memory.
- `core/model.py`: model loading, the steering hook (each row of a batch can be steered differently), generation.
- `core/chat.py`: one turn.
- `vectors/build.py`, `vectors/tune.py`: build and inspect the vectors. `artifacts/vectors/`: the built vectors.
- `eval/`: the measurements above.
- `cli.py`, `app.py`: terminal chat and web demo.
- `tests/smoke.py`: end-to-end test in FAKE mode, no GPU or torch needed.
