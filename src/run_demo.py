"""
run_demo.py
==========
End-to-end driver for the Fuzzy Confidence Gating experiment.

Pipeline
--------
1. Load data/signals.csv. If it is missing, generate a realistic synthetic one
   (see synthetic_signals.py) so the whole thing still runs.
2. Standardise the four signals; split into train and test.
3. Train the ANFIS fuzzy gate.
4. Fit the three baselines (raw confidence, temperature scaling, MLP).
5. Compute the metrics table and write results/metrics.csv.
6. Save the learned rule base to results/rule_table.csv.
7. Draw every figure into figures/.

Run:
    python run_demo.py                 # uses real signals.csv if present, else synthetic
    python run_demo.py --force_synth   # always use synthetic demonstration data
"""

import argparse
import csv
import json
import os

import numpy as np

import plots
from baselines_metrics import (apply_temperature, fit_mlp, fit_temperature,
                               mlp_proba, raw_confidence, summarise)
from fuzzy_anfis import ANFIS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "signals.csv")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
FEATURES = ["entropy", "margin", "attention_spread", "layer_instability"]


def load_signals(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    X = np.array([[r[f] for f in FEATURES] for r in rows])
    y = np.array([r["correct"] for r in rows])
    max_softmax = np.array([r["max_softmax"] for r in rows])
    return X, y, max_softmax


def train_and_score(X, y, max_softmax, seed, epochs):
    """One random 70/30 split: train the fuzzy gate and all baselines, return
    each method's test-set confidences. Used both for the figures (one split)
    and, looped over seeds, for the averaged metrics table."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    vcut = int(0.8 * len(tr))          # hold out part of train for early stopping
    ftr, val = tr[:vcut], tr[vcut:]

    # fuzzy gate: zero-order + early stopping (prevents overfitting)
    net = ANFIS(n_inputs=4, n_mf=2, first_order=False, l2=1e-3, seed=seed)
    net.fit(Xs[ftr], y[ftr], epochs=epochs, lr=0.05,
            X_val=Xs[val], y_val=y[val], patience=80)
    conf_fuzzy = net.predict_proba(Xs[te])

    # baselines
    conf_raw = raw_confidence(max_softmax[te])
    logit_tr = np.log(max_softmax[tr] / (1 - max_softmax[tr] + 1e-9) + 1e-9)
    logit_te = np.log(max_softmax[te] / (1 - max_softmax[te] + 1e-9) + 1e-9)
    T = fit_temperature(logit_tr, y[tr])
    conf_temp = apply_temperature(logit_te, T)
    clf = fit_mlp(Xs[ftr], y[ftr], seed=seed)
    conf_mlp = mlp_proba(clf, Xs[te])

    methods = {
        "Raw confidence": conf_raw,
        "Temperature scaling": conf_temp,
        "MLP calibrator": conf_mlp,
        "Fuzzy gate (ANFIS)": conf_fuzzy,
    }
    return methods, y[te], net, mu, sd, T, Xs[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force_synth", action="store_true",
                    help="ignore any real signals.csv and use synthetic data")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=10,
                    help="number of random splits to average the metrics over")
    args = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    if args.force_synth or not os.path.exists(DATA):
        from synthetic_signals import generate
        generate(out_path=DATA)
        data_kind = "SYNTHETIC demonstration data"
    else:
        data_kind = "real signals.csv"
    X, y, max_softmax = load_signals(DATA)
    print(f"[data] {data_kind}: {len(y)} rows, base accuracy {y.mean():.3f}")

    # 2. train + score one split (used for the figures, the saved gate and the
    #    rule table) -------------------------------------------------------
    methods, y_te, net, mu, sd, T, Xs_te = train_and_score(X, y, max_softmax,
                                                           args.seed, args.epochs)
    conf_fuzzy = methods["Fuzzy gate (ANFIS)"]
    print(f"[temp] fitted temperature T = {T:.3f}")

    # save the trained gate so demo.py can reuse it on new questions
    np.savez(os.path.join(RES, "gate.npz"),
             c=net.c, s=net.s, P=net.P, b=net.b, sel=net.sel,
             first_order=net.first_order, M=net.M, J=net.J,
             mu=mu, sd=sd, features=np.array(FEATURES))
    print(f"[gate] saved trained gate to {os.path.join(RES, 'gate.npz')}")

    # 5. metrics table: averaged over several random splits ----------------
    # With only a few hundred test points a single split is noisy, so the
    # reported numbers are the mean over args.seeds random splits (std shown).
    per_seed = {name: [] for name in methods}
    pooled = {name: [] for name in methods}
    pooled_y = []
    for s in range(args.seeds):
        m_s, yte_s, *_ = train_and_score(X, y, max_softmax, s, args.epochs)
        pooled_y.append(yte_s)
        for name, conf in m_s.items():
            per_seed[name].append(summarise(name, conf, yte_s))
            pooled[name].append(conf)
    pooled_y = np.concatenate(pooled_y)
    methods_pooled = {name: np.concatenate(v) for name, v in pooled.items()}
    metric_keys = ["ECE", "Brier", "AUROC", "AURC", "acc_at_50pct_coverage"]
    rows = []
    for name in methods:
        agg = {"method": name}
        for mk in metric_keys:
            vals = np.array([d[mk] for d in per_seed[name]])
            agg[mk] = round(float(vals.mean()), 4)
            agg[mk + "_std"] = round(float(vals.std()), 4)
        rows.append(agg)
    with open(os.path.join(RES, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n=== Metrics: mean over {args.seeds} splits (test set) ===")
    print(f"{'method':>22} " + " ".join(f"{mk:>16}" for mk in metric_keys))
    for r in rows:
        print(f"{r['method']:>22} " +
              " ".join(f"{r[mk]:.3f}+-{r[mk+'_std']:.3f}".rjust(16)
                       for mk in metric_keys))

    # 6. readable rule base -------------------------------------------------
    rule_rows = net.rule_table(FEATURES)
    # add average normalised firing on the test set, to show which rules matter
    net.forward(Xs_te)
    wn_mean = net._cache["wn"].mean(0)
    for rr in rule_rows:
        rr["avg_firing"] = round(float(wn_mean[rr["rule"]]), 4)
    rule_rows.sort(key=lambda d: -d["avg_firing"])
    with open(os.path.join(RES, "rule_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rule", "antecedent",
                                          "bias_confidence", "avg_firing"])
        w.writeheader()
        w.writerows(rule_rows)
    print("\n=== Top fired rules ===")
    for rr in rule_rows[:5]:
        print(f"  [{rr['avg_firing']:.3f}] IF {rr['antecedent']} "
              f"-> conf~{rr['bias_confidence']}")

    # 7. figures ------------------------------------------------------------
    plots.plot_architecture(os.path.join(FIG, "fig_architecture.png"), FEATURES)
    plots.plot_membership(os.path.join(FIG, "fig_membership.png"), net, FEATURES,
                          mu, sd)
    plots.plot_reliability(os.path.join(FIG, "fig_reliability.png"),
                           methods_pooled, pooled_y)
    plots.plot_risk_coverage(os.path.join(FIG, "fig_risk_coverage.png"),
                             methods_pooled, pooled_y)
    plots.plot_ece_bar(os.path.join(FIG, "fig_ece_bar.png"), rows)
    plots.plot_training_loss(os.path.join(FIG, "fig_training_loss.png"),
                             net.history_)

    # machine-readable summary for the writeup
    summary = dict(data_kind=data_kind, n=len(y), base_accuracy=round(float(y.mean()), 4),
                   temperature=round(T, 4), metrics=rows,
                   improvement_ECE_vs_raw=round(rows[0]["ECE"] - rows[3]["ECE"], 4))
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] figures in {FIG}, results in {RES}")


if __name__ == "__main__":
    main()
