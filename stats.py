"""
Statistical comparison of the methods (Section 5.2.1, Fig. 11).

Runs the Friedman test over the per-task accuracies of all methods, then the
post-hoc Nemenyi test, and draws a critical-difference (CD) diagram. Methods
whose average ranks differ by less than the CD are not significantly different
and are linked in the diagram.

    CD = q_alpha * sqrt( k (k + 1) / (6 N) )                     (Eq. 35)

where k = number of methods and N = number of cross-domain tasks.

Input: a benchmark CSV produced by `train.py --benchmark` (rows = methods,
columns = tasks + AVG). Combine several CSVs (e.g. one per label percentage) by
passing them all -- their task columns are concatenated to raise N.

Usage:
    python stats.py results/benchmark_case2_label1.csv results/benchmark_case2_label10.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata

# Nemenyi q_alpha critical values at p=0.05 (two-tailed), indexed by #methods.
Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
              8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def load_accuracy_matrix(csv_paths):
    """Return (methods, acc_matrix [k x N]) from one or more benchmark CSVs."""
    blocks, methods = [], None
    for path in csv_paths:
        df = pd.read_csv(path)
        task_cols = [c for c in df.columns if c not in ("method", "AVG")]
        df = df.set_index("method")
        if methods is None:
            methods = list(df.index)
        blocks.append(df.loc[methods, task_cols].to_numpy(dtype=float))
    return methods, np.concatenate(blocks, axis=1)


def cd_diagram(methods, acc, save="results/cd_diagram.png"):
    k, N = acc.shape
    # ranks per task (1 = best); average over tasks
    ranks = np.array([rankdata(-acc[:, j], method="average") for j in range(N)]).T
    avg_rank = ranks.mean(1)

    stat, p = friedmanchisquare(*[acc[i] for i in range(k)])
    cd = Q_ALPHA_05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6.0 * N))
    print(f"Friedman chi2={stat:.3f}  p={p:.3e}  (N={N} tasks, k={k} methods)")
    print(f"Critical difference CD={cd:.3f} at p=0.05")
    order = np.argsort(avg_rank)
    for i in order:
        print(f"  {methods[i]:12s}  avg-rank={avg_rank[i]:.3f}")

    # ----- draw the CD diagram ----- #
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(9, 0.5 * k + 2))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 2)
    ax.axis("off")
    y_axis = k + 1
    ax.plot([lo, hi], [y_axis, y_axis], "k-", lw=1.5)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y_axis, y_axis + 0.15], "k-", lw=1.2)
        ax.text(r, y_axis + 0.45, str(r), ha="center", va="bottom", fontsize=9)
    # CD bar
    ax.plot([lo, lo + cd], [y_axis + 1.0, y_axis + 1.0], "k-", lw=2)
    ax.text(lo + cd / 2, y_axis + 1.15, f"CD = {cd:.2f}", ha="center", fontsize=9)

    srt = np.argsort(avg_rank)
    left = srt[: (k + 1) // 2]
    right = srt[(k + 1) // 2:][::-1]
    for row, i in enumerate(left):
        yy = y_axis - 1 - row
        ax.plot([avg_rank[i], avg_rank[i]], [y_axis, yy], "k-", lw=1)
        ax.plot([avg_rank[i], lo - 0.4], [yy, yy], "k-", lw=1)
        ax.text(lo - 0.5, yy, methods[i], ha="right", va="center", fontsize=9)
    for row, i in enumerate(right):
        yy = y_axis - 1 - row
        ax.plot([avg_rank[i], avg_rank[i]], [y_axis, yy], "k-", lw=1)
        ax.plot([avg_rank[i], hi + 0.4], [yy, yy], "k-", lw=1)
        ax.text(hi + 0.5, yy, methods[i], ha="left", va="center", fontsize=9)

    # connect groups that are within CD of each other
    level = 0.12
    for a in range(k):
        grp = [b for b in range(k) if abs(avg_rank[a] - avg_rank[b]) <= cd]
        if len(grp) > 1:
            rr = [avg_rank[b] for b in grp]
            ax.plot([min(rr) - 0.05, max(rr) + 0.05],
                    [y_axis - 0.25 - level, y_axis - 0.25 - level], "r-", lw=3, alpha=0.6)
            level += 0.12

    plt.title(f"Critical-difference diagram (Nemenyi, p=0.05, N={N})", fontsize=10)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print("Saved ->", save)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="benchmark CSV(s) from train.py --benchmark")
    ap.add_argument("--save", default="results/cd_diagram.png")
    args = ap.parse_args()
    methods, acc = load_accuracy_matrix(args.csv)
    cd_diagram(methods, acc, save=args.save)


if __name__ == "__main__":
    main()
