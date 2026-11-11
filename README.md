# Moodsteer

A chat assistant that adapts to the user's mood by steering the model's activations, not its prompt.
Frustrated? It gets shorter and more direct. Confused? Simpler. Sad? Warmer. The weights and the prompt never change:
a vector is added to the model's hidden state while it writes the reply.

## Quick start (Kaggle, Colab or any CUDA GPU, Qwen2.5-3B-Instruct)

Kaggle notebook: [kaggle.com/code/tostapane/moodsteer](https://www.kaggle.com/code/tostapane/moodsteer).
Copy & Edit, set the accelerator to a GPU with internet on, and run all.

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

## Results (Qwen2.5-3B)

`python -m eval.steer`: each axis on 12 held-out questions, at three steering strengths, against writing the same
instruction in the system prompt. Answers up to 400 tokens. "Facts" = how many of 30 simple factual questions (capital
of France, 12 × 12, …) it still gets right under the same condition: the cost of the change.

| axis | metric | baseline | steer 0.2 | steer 0.4 | steer 0.8 | prompt |
|---|---|---|---|---|---|---|
| concise | words per answer | 252* | 150 | 68 | 53 | 12 |
| | facts | 30 | 30 | 30 | 25 | 30 |
| simple | reading ease (higher = simpler) | 38 | 54 | 70 | **79** | 65 |
| | facts | 30 | 30 | 30 | **30** | 28 |
| calm (enthusiastic pole) | `!` per 100 words | 0.0 | 0.3 | 1.9 | **4.7** | 4.7 |
| | facts | 30 | 30 | 30 | **30** | 29 |

\* half the baseline answers hit the 400-token limit, so they're really longer.

`python -m eval.judge`: warm and direct have nothing to count, so the same model, unsteered, compares each reply with
the baseline reply to the same question, in both A/B orders. Share of wins on the axis, 50% = no difference:

| axis | steer 0.2 | steer 0.4 | steer 0.8 | prompt |
|---|---|---|---|---|
| warm | 62% | 67% | 62% | 88% |
| direct | 75% | 71% | 58% | 79% |

What it shows:
- **Concise, simple and calm move the intended way with no fact lost up to 0.4.** At 0.8 concise loses 5 of 30, and
  at 2.0 the model breaks. Several axes at once break it sooner (a reply drifted into Chinese at a combined 0.48), so
  `MAX_PUSH` caps a reply at 0.35.
- **Simple: steering beats the prompt.** Simpler than the prompted version, and every fact kept.
- **Calm: steering matches the prompt** at 0.8, at no cost.
- **Prompting is stronger for length:** 12 words against 53.
- **Warm and direct work, less than the prompt.** The judge's other question, which answer is better, isn't usable:
  it prefers the longer baseline even over the prompted answer (8-25%).
- **Middle layer is right.** On the concise vector (`python -m eval.layers`), layer 18 (the middle, used everywhere)
  cuts 252 words to 150 at 0.2 and layer 22 to 167. Layers 2-14 and 26-34 do almost nothing.
- **Three axes were dropped.** Structured (lists ↔ prose): list lines went only from 57% to 27% at 0.8 (prompt 100%),
  and in a chat the model copies the format of its earlier replies. Cautious: hedging barely moved (1.8 → 2.2 per 100
  words) and cost 5 facts at 0.8. Accountable: no change up to 0.4 by the judge, and the opposite at 0.8.

`python -m eval.detector`, 160 hand-written messages never used for tuning (40 of them neutral):
- exactly the right moods: 59% (always answering "neutral" gets 25%)
- neutral messages kept neutral: 75%. A false mood stays in the steering state, so this one matters most.
- "more detail": found 10/14, and right every time it was reported. "Shorter": found 6/6, but also reported on 24 other
  messages, mostly hurried or angry ones.
- the stronger of two versions of a message scored higher in 16/18 pairs

Every reply behind these numbers is saved to `runs/` when you run the scripts.

## How it works

Every turn:
1. **Detect:** the model answers two questions about the latest message: the user's moods, each with an intensity
   (frustrated 6/10 and rushed 8/10, …), and how long the next answer should be (much shorter … much longer).
2. **Remember:** a mood moves only the axes it lists, toward a target at its own speed:
   `state += intensity × speed × (target − state)`. Very rushed jumps straight to short answers, curious eases them
   a little, and a neutral message changes nothing.
3. **Steer:** each axis vector × its α × its strength is added to the hidden state at the middle layer
   while the reply is generated. If the axes add up to more than `MAX_PUSH`, they shrink together for that reply.
4. **Compare:** the same question is also answered without steering, side by side.

The 5 steering axes (+ pole ↔ − pole): concise ↔ long · warm ↔ cold · calm ↔ enthusiastic ·
simple ↔ technical · direct ↔ roundabout. Which mood pushes which axes is `MAPPING` in `config.py`.

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

python -m eval.steer                # the first table above
python -m eval.layers               # which layer steers best
python -m eval.judge                # LLM judge: warm and direct
python -m eval.detector             # can the model read moods? scored on messages never used for tuning
```

Another model: `MODEL_ID=... python -m vectors.build --regen`. The committed vectors only fit Qwen2.5-3B.
Environment variables are read once, when `config` is first imported.

How each axis is measured (`eval/steer.py`): concise → words · simple → Flesch reading ease · calm → exclamation
marks.

## Knobs (`config.py`)

- `MAPPING`: which axes each mood moves, toward what target, how fast.
- `STRENGTH`: how hard each axis pushes at α = ±1, as a fraction of a typical hidden state (tuned from the table).
- `THRESHOLD`: the intensity a mood needs to count.
- `MAX_PUSH`: limit on the total push of one reply.
- `MODEL_ID`, `DTYPE`, `SYSTEM_PROMPT`, `MAX_NEW_TOKENS`, `BATCH_SIZE`.

## Code

- `detector/detect.py`: mood detection (moods as JSON, length as one letter).
- `core/controller.py`: steering memory.
- `core/model.py`: model loading, the steering hook (each row of a batch can be steered differently), generation.
- `core/chat.py`: one turn.
- `vectors/build.py`, `vectors/tune.py`: build and inspect the vectors. `artifacts/vectors/`: the built vectors.
- `eval/`: the measurements above.
- `cli.py`, `app.py`: terminal chat and web demo.
- `tests/smoke.py`: end-to-end test in FAKE mode, no GPU or torch needed.
