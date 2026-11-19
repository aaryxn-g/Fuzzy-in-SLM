"""
plots.py
========
All figures. Every figure is saved at 300 dpi so it is sharp when printed.

Figures produced:
  fig_architecture.png      schematic of the 5-layer ANFIS gate
  fig_membership.png        learned Gaussian membership functions per signal
  fig_reliability.png       reliability diagram comparing all methods
  fig_risk_coverage.png     accuracy-on-kept vs coverage (selective prediction)
  fig_ece_bar.png           ECE comparison bar chart
  fig_training_loss.png     ANFIS training loss curve
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 300
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.autolayout": True,
})


def plot_architecture(path, feature_names):
    """Schematic of the five-layer ANFIS gate."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    layer_x = [0.05, 0.28, 0.5, 0.72, 0.9]
    titles = ["Inputs\n(signals)", "L1 Fuzzify", "L2-L3\nRules + norm",
              "L4 Consequent", "L5 Output"]
    for x, t in zip(layer_x, titles):
        ax.text(x, 1.02, t, ha="center", va="bottom", fontsize=9, weight="bold")

    # Input nodes
    ys = np.linspace(0.15, 0.85, len(feature_names))
    for y, name in zip(ys, feature_names):
        ax.add_patch(plt.Circle((layer_x[0], y), 0.025, color="#1f4e79"))
        ax.text(layer_x[0] - 0.04, y, name, ha="right", va="center", fontsize=8)

    # Membership nodes (2 per input)
    mf_ys = np.linspace(0.08, 0.92, len(feature_names) * 2)
    for i, y in enumerate(mf_ys):
        ax.add_patch(plt.Rectangle((layer_x[1] - 0.02, y - 0.015), 0.04, 0.03,
                                   fill=False, ec="#444"))
        # connect to its input
        src_y = ys[i // 2]
        ax.plot([layer_x[0], layer_x[1] - 0.02], [src_y, y], color="#999", lw=0.5)

    # Rule nodes (show a representative subset)
    rule_ys = np.linspace(0.1, 0.9, 6)
    for y in rule_ys:
        ax.add_patch(plt.Circle((layer_x[2], y), 0.018, color="#c55a11"))
        for my in mf_ys:
            ax.plot([layer_x[1] + 0.02, layer_x[2]], [my, y], color="#ddd", lw=0.3)
    ax.text(layer_x[2], 0.0, "16 rules (2^4)", ha="center", fontsize=8)

    # Consequent nodes
    for y in rule_ys:
        ax.add_patch(plt.Rectangle((layer_x[3] - 0.02, y - 0.015), 0.04, 0.03,
                                   fill=False, ec="#444"))
        ax.plot([layer_x[2], layer_x[3] - 0.02], [y, y], color="#999", lw=0.5)

    # Output node
    ax.add_patch(plt.Circle((layer_x[4], 0.5), 0.03, color="#1f4e79"))
    ax.text(layer_x[4] + 0.05, 0.5, "confidence\n(0-1)", ha="left", va="center",
            fontsize=8)
    for y in rule_ys:
        ax.plot([layer_x[3] + 0.02, layer_x[4]], [y, 0.5], color="#999", lw=0.5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.12)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_membership(path, anfis, feature_names, mu=None, sd=None):
    """Learned Gaussian membership functions, one panel per input signal."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    xs = np.linspace(-3, 3, 400)
    mf_names = ["low", "high"] if anfis.M == 2 else [f"MF{m}" for m in range(anfis.M)]
    for j, ax in enumerate(axes.ravel()):
        if j >= anfis.J:
            ax.axis("off")
            continue
        for m in range(anfis.M):
            c, s = anfis.c[j, m], anfis.s[j, m]
            ax.plot(xs, np.exp(-0.5 * ((xs - c) / s) ** 2), label=mf_names[m])
        ax.set_title(feature_names[j], fontsize=10)
        ax.set_xlabel("standardised signal value")
        ax.set_ylabel("degree of membership")
        ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_reliability(path, methods, y, n_bins=10):
    """Reliability diagram: predicted confidence vs observed accuracy."""
    from baselines_metrics import reliability_curve
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    markers = {"Raw confidence": "o", "Temperature scaling": "s",
               "MLP calibrator": "^", "Fuzzy gate (ANFIS)": "D"}
    for name, conf in methods.items():
        centres, accs, counts = reliability_curve(conf, y, n_bins)
        keep = counts > 0
        ax.plot(centres[keep], accs[keep], marker=markers.get(name, "o"),
                label=name, lw=1.5, markersize=6)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_risk_coverage(path, methods, y):
    from baselines_metrics import risk_coverage
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    styles = {"Raw confidence": dict(lw=1.6),
              "Temperature scaling": dict(lw=1.6, ls="--"),
              "MLP calibrator": dict(lw=1.8),
              "Fuzzy gate (ANFIS)": dict(lw=1.8)}
    for name, conf in methods.items():
        cov, acc, aurc = risk_coverage(conf, y)
        ax.plot(cov, acc, label=f"{name} (AURC={aurc:.3f})",
                **styles.get(name, dict(lw=1.6)))
    ax.set_xlabel("coverage (fraction of questions answered)")
    ax.set_ylabel("accuracy on answered questions")
    ax.set_title("Accuracy vs coverage (abstaining on low confidence)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0.05, 1.0)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_ece_bar(path, rows):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    names = [r["method"] for r in rows]
    eces = [r["ECE"] for r in rows]
    colors = ["#9aa0a6", "#6fa8dc", "#f6b26b", "#1f4e79"][:len(names)]
    bars = ax.bar(names, eces, color=colors)
    for b, e in zip(bars, eces):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{e:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_title("Calibration error by method")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_training_loss(path, history):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(history, color="#1f4e79", lw=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy loss")
    ax.set_title("ANFIS training convergence")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
