"""
baselines_metrics.py
====================
Baselines to compare the fuzzy gate against, plus the calibration and
selective-prediction metrics used to compare the methods.

Baselines
---------
1. Raw confidence     : use the model's own max softmax probability. No learning.
2. Temperature scaling: one learned scalar T that divides the logit before the
                        sigmoid. The standard calibration baseline (Guo et al., 2017).
3. MLP calibrator     : a small black-box neural network (scikit-learn) on the
                        same four signals the fuzzy gate uses. Strong baseline,
                        but gives no readable reason for its output.

Metrics
-------
- ECE  : Expected Calibration Error (lower is better).
- Brier: mean squared error between confidence and correctness (lower is better).
- AUROC: how well the confidence score separates correct from wrong answers.
- Risk-coverage: accuracy on the answers kept, as we allow the system to abstain
                 on the least confident ones. Summarised by AURC (area under the
                 risk-coverage curve, lower is better).
"""

import numpy as np
from sklearn.neural_network import MLPClassifier


# ----------------------------------------------------------------- baselines
def raw_confidence(max_softmax):
    """Baseline 1: the model's own top probability, used directly as confidence."""
    return np.clip(np.asarray(max_softmax, dtype=float), 1e-6, 1 - 1e-6)


def fit_temperature(logit_conf, y, iters=500, lr=0.05):
    """Baseline 2: learn a single temperature T on a confidence-logit.

    logit_conf : log(p / (1-p)) of the raw confidence (the network's certainty).
    Returns the fitted T. We optimise T by gradient descent on BCE.
    """
    z = np.asarray(logit_conf, dtype=float)
    y = np.asarray(y, dtype=float)
    logT = 0.0  # optimise log T so T stays positive
    for _ in range(iters):
        T = np.exp(logT)
        p = 1.0 / (1.0 + np.exp(-z / T))
        # d BCE/dz_scaled * dz_scaled/dlogT ; z_scaled = z / T = z * exp(-logT)
        dBCE_dzs = (p - y) / len(y)
        dzs_dlogT = -z * np.exp(-logT)
        grad = np.sum(dBCE_dzs * dzs_dlogT)
        logT -= lr * grad
    return float(np.exp(logT))


def apply_temperature(logit_conf, T):
    z = np.asarray(logit_conf, dtype=float)
    return 1.0 / (1.0 + np.exp(-z / T))


def fit_mlp(X, y, seed=0):
    """Baseline 3: small black-box MLP calibrator on the four signals."""
    clf = MLPClassifier(hidden_layer_sizes=(8, 8), activation="relu",
                        max_iter=1500, random_state=seed)
    clf.fit(X, y)
    return clf


def mlp_proba(clf, X):
    return clf.predict_proba(X)[:, 1]


# ------------------------------------------------------------------- metrics
def expected_calibration_error(conf, y, n_bins=10):
    """ECE with equal-width bins over [0, 1]."""
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc - avg_conf)
    return float(ece)


def reliability_curve(conf, y, n_bins=10):
    """Return (bin_centres, bin_accuracy, bin_count) for a reliability diagram."""
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        centres.append((lo + hi) / 2)
        if mask.sum() == 0:
            accs.append(np.nan)
        else:
            accs.append(y[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(centres), np.array(accs), np.array(counts)


def brier_score(conf, y):
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((conf - y) ** 2))


def auroc(conf, y):
    """Area under ROC for confidence predicting correctness. Rank-based, no sklearn
    dependency so the team can read the implementation."""
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=float).astype(int)
    pos = conf[y == 1]
    neg = conf[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U statistic via rank sum.
    order = np.argsort(conf)
    ranks = np.empty(len(conf), dtype=float)
    ranks[order] = np.arange(1, len(conf) + 1)
    # average ties
    _, inv, counts = np.unique(conf, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = sum_ranks / counts
    ranks = avg_ranks[inv]
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = r_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def risk_coverage(conf, y):
    """Sort by confidence descending; report coverage vs accuracy-on-kept.

    Returns (coverage, accuracy, aurc). AURC = mean risk (1-acc) over coverages,
    a single number where lower is better.
    """
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(-conf)
    y_sorted = y[order]
    N = len(y)
    coverage = np.arange(1, N + 1) / N
    acc = np.cumsum(y_sorted) / np.arange(1, N + 1)
    risk = 1.0 - acc
    aurc = float(np.mean(risk))
    return coverage, acc, aurc


def summarise(name, conf, y):
    """One row of the results table."""
    cov, acc, aurc = risk_coverage(conf, y)
    return dict(
        method=name,
        ECE=round(expected_calibration_error(conf, y), 4),
        Brier=round(brier_score(conf, y), 4),
        AUROC=round(auroc(conf, y), 4),
        AURC=round(aurc, 4),
        acc_at_50pct_coverage=round(float(acc[len(acc) // 2]), 4),
    )
