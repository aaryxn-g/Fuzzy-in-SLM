# Fuzzy Confidence Gating for Small Language Models

A neuro-fuzzy system that reads a small language model's own internal signals and
decides, with human-readable rules, whether to trust each answer. The language
model is never retrained; we only add a trainable fuzzy decision layer (an ANFIS)
on top.

The gate reads four cheap signals from a frozen model while it answers a
multiple-choice question, combines them with an Adaptive Neuro-Fuzzy Inference
System (ANFIS), and outputs a calibrated confidence together with the fuzzy rule
behind it. The confidence is used to decide whether to answer, abstain, or
escalate the question to a larger model.

## What is in here

```
fcg_slm/
  src/
    fuzzy_anfis.py          the fuzzy gate (pure NumPy ANFIS, trainable)
    baselines_metrics.py    baselines + calibration metrics
    extract_signals.py      run the model, record the four signals + correctness
    make_questions.py       download a multiple-choice question set (SciQ)
    synthetic_signals.py    realistic synthetic signals for testing without a model
    plots.py                all figures
    run_demo.py             runs the whole experiment end to end
    demo.py                 interactive question -> confidence demo
  data/        questions.jsonl, signals.csv (produced)
  figures/     the six figures (produced)
  results/     metrics.csv, rule_table.csv, summary.json, gate.npz (produced)
  requirements.txt
```

## How to run the full experiment

These commands run in a terminal, from inside the `src` folder. The model runs on
CPU (no GPU required) for a 0.5B model.

```bash
pip install numpy scikit-learn matplotlib torch transformers datasets

# 1. build a multiple-choice question set (writes data/questions.jsonl)
python make_questions.py --n 800

# 2. run the model and extract the four signals (writes data/signals.csv)
python extract_signals.py --data ../data/questions.jsonl --out ../data/signals.csv

# 3. train the fuzzy gate + baselines, write the metrics table and all figures
python run_demo.py
```

Step 2 is the slow part (about 15 to 30 minutes on CPU for 800 questions). Steps 1
and 3 take seconds. `run_demo.py` averages every metric over ten random splits, so
the numbers are stable, and it saves the trained gate to `results/gate.npz`.

## Quick check with no model

To verify the fuzzy gate, baselines and figures without any model download:

```bash
python run_demo.py --force_synth
```

## Interactive demo

After `run_demo.py` has produced `results/gate.npz`:

```bash
python demo.py
```

Type a multiple-choice question, or press Enter for a random one from the set. It
shows the model's chosen answer, the fuzzy confidence, the decision (answer /
abstain / escalate), and the fuzzy rule behind the rating.

## The four signals (what the fuzzy gate reads)

| Signal | Meaning | High value suggests |
|---|---|---|
| entropy | spread of the answer distribution | the model is unsure |
| margin | gap between the top two options | small gap = nearly tied |
| attention_spread | how diffuse the final attention is | no clear evidence used |
| layer_instability | how often the internal guess changed across layers | unstable reasoning |

## Check the fuzzy gate maths

The hand-derived gradients are verified against numerical gradients:

```bash
cd src
python fuzzy_anfis.py      # prints "gradient check passed"
```

## Data and model credits

- Questions come from the **SciQ** dataset (Welbl, Liu and Gardner, 2017),
  released under CC BY-NC 3.0. The included `data/questions.jsonl` is built from
  SciQ by `make_questions.py` and is used here for non-commercial research only.
- The language model is **Qwen2.5-0.5B-Instruct** (Qwen Team, 2024).
