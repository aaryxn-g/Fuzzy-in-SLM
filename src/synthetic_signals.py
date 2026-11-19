"""
synthetic_signals.py
===================
Generate a realistic synthetic signals.csv so the full pipeline (training the
fuzzy gate, the baselines, every plot) can be run and checked even when the real
model cannot be run in time.

IMPORTANT:
These are SIMULATED signals, not the output of a real language model. They are
built so that the four signals carry genuine but noisy information about
correctness, which is exactly the situation on real data. When you run
extract_signals.py with the real model you get a real signals.csv, and you re-run
run_demo.py to regenerate every figure with real numbers.

Design of the simulation
-------------------------
Real models fail for different reasons, and different signals catch different
failures. We model three independent hidden factors:
  d1 = knowledge gap        -> shows up in entropy, margin, max_softmax
  d2 = attention / retrieval failure -> shows up in attention_spread
  d3 = internal instability -> shows up in layer_instability
An answer is correct with probability sigmoid(a - 0.8 d1 - 0.7 d2 - 0.7 d3).

The point: no single signal reveals correctness on its own, so a method that
combines all four (the fuzzy gate, or the MLP) should beat the raw top
probability, which only reflects d1. This is the realistic case and a fair test.
"""

import csv
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n=1500, base_acc=0.60, noise=0.40, seed=42, out_path="data/signals.csv"):
    rng = np.random.default_rng(seed)

    # Three independent hidden failure factors.
    d1 = rng.standard_normal(n)   # knowledge gap
    d2 = rng.standard_normal(n)   # attention / retrieval failure
    d3 = rng.standard_normal(n)   # internal instability

    # Correctness depends on all three, so no single signal is sufficient.
    a = np.log(base_acc / (1 - base_acc))
    p_correct = sigmoid(a - 0.8 * d1 - 0.7 * d2 - 0.7 * d3)
    correct = (rng.random(n) < p_correct).astype(int)

    # Signals: each tied mainly to one factor, with noise.
    entropy = np.clip(1.3 + 0.9 * d1 + noise * rng.standard_normal(n), 0.01, None)
    margin = sigmoid(1.2 - 0.9 * d1 + noise * rng.standard_normal(n))
    max_softmax = sigmoid(1.0 - 0.95 * d1 + 0.30 * rng.standard_normal(n))  # d1 only
    attention_spread = sigmoid(0.1 + 1.0 * d2 + noise * rng.standard_normal(n))
    layer_instability = np.clip(
        sigmoid(-0.3 + 1.0 * d3 + noise * rng.standard_normal(n)), 0, 1)

    cols = ["entropy", "margin", "attention_spread", "layer_instability",
            "max_softmax", "correct"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([round(float(entropy[i]), 6), round(float(margin[i]), 6),
                        round(float(attention_spread[i]), 6),
                        round(float(layer_instability[i]), 6),
                        round(float(max_softmax[i]), 6), int(correct[i])])
    print(f"[synthetic] wrote {out_path}  n={n}  base accuracy={correct.mean():.3f}")
    return out_path


if __name__ == "__main__":
    generate()
