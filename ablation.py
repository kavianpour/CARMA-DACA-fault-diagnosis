"""
Ablation studies from the paper.

  coeff     : Section 5.2.3 / Fig. 13 -- accuracy over the alpha-beta grid
              {0, 0.02, 0.05, 0.1, 0.5, 1} for task A1->D2 with 1% label.
  adistance : Section 5.2.2 / Fig. 12, Eq. 36-37 -- A-distance and AL-distance of
              the learned features (lower = better domain / subdomain alignment).
  labels    : Section 5 -- accuracy vs. the labelled-target fraction {0,1,5,10}%.

Examples
--------
  python ablation.py coeff     --source A1 --target D2 --label 0.01
  python ablation.py adistance --source C1 --target D2 --label 0.10 --method CARMA-DACA
  python ablation.py labels    --source A1 --target D2 --method CARMA-DACA
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.svm import LinearSVC

from config import cfg
import data as D
from methods import DEEP_METHODS
from train import fit_model, run_method


@torch.no_grad()
def _extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device))["struct"].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def coeff(args):
    """Accuracy over the alpha-beta grid (Fig. 13)."""
    grid = [0, 0.02, 0.05, 0.1, 0.5, 1]
    rows = []
    for beta in grid:
        for alpha in grid:
            cfg.alpha, cfg.beta = alpha, beta
            _, _, _, acc, _ = fit_model(cfg, DEEP_METHODS["CARMA-DACA"],
                                        args.source, args.target, args.epochs, cfg.seed)
            rows.append({"beta": beta, "alpha": alpha, "accuracy": round(acc, 2)})
            print(rows[-1])
    out = os.path.join(cfg.out_dir, "ablation_coeff.csv")
    pd.DataFrame(rows).pivot(index="beta", columns="alpha", values="accuracy").to_csv(out)
    print("Saved ->", out)


def adistance(args):
    """
    A-distance  = 2 (1 - 2*eps)                          (Eq. 36)
    AL-distance = 2 * sum_c p(c) (1 - 2*eps_c)           (Eq. 37)
    eps is the source-vs-target classification error of a linear SVM on the
    learned features; eps_c restricts that to class c.
    """
    spec = DEEP_METHODS[args.method]
    model, loaders, device, _, _ = fit_model(cfg, spec, args.source, args.target,
                                             args.epochs, cfg.seed)
    fs, ys = _extract(model, loaders["source_train"], device)
    ft, yt = _extract(model, loaders["target_test"], device)

    X = np.concatenate([fs, ft])
    dom = np.concatenate([np.ones(len(fs)), np.zeros(len(ft))])
    eps = 1.0 - LinearSVC(max_iter=5000).fit(X, dom).score(X, dom)
    a_dist = 2 * (1 - 2 * eps)

    al = 0.0
    for c in np.unique(yt):
        sc, tc = fs[ys == c], ft[yt == c]
        if len(sc) < 2 or len(tc) < 2:
            continue
        Xc = np.concatenate([sc, tc])
        dc = np.concatenate([np.ones(len(sc)), np.zeros(len(tc))])
        eps_c = 1.0 - LinearSVC(max_iter=5000).fit(Xc, dc).score(Xc, dc)
        al += (len(tc) / len(ft)) * (1 - 2 * eps_c)
    al_dist = 2 * al

    print(f"{args.method}: A-distance={a_dist:.3f}  AL-distance={al_dist:.3f}")
    out = os.path.join(cfg.out_dir, "ablation_adistance.csv")
    row = {"method": args.method, "A_distance": round(a_dist, 3), "AL_distance": round(al_dist, 3)}
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out, index=False)
    print("Saved ->", out)


def labels(args):
    """Accuracy vs. labelled-target fraction (Section 5)."""
    rows = []
    for frac in [0.0, 0.01, 0.05, 0.10]:
        cfg.label_fraction = frac
        mean, std = run_method(cfg, args.method, args.source, args.target, args.epochs, args.runs)
        rows.append({"label_pct": int(frac * 100), "mean_acc": round(mean, 2),
                     "std_acc": round(std, 2)})
        print(rows[-1])
    out = os.path.join(cfg.out_dir, f"ablation_labels_{args.method}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved ->", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=["coeff", "adistance", "labels"])
    p.add_argument("--method", default="CARMA-DACA")
    p.add_argument("--source", default="A1", choices=D.COND_NAMES)
    p.add_argument("--target", default="D2", choices=D.COND_NAMES)
    p.add_argument("--label", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--data_root", default=None)
    args = p.parse_args()
    if args.data_root: cfg.data_root = args.data_root
    if args.label is not None: cfg.label_fraction = args.label
    args.epochs = args.epochs if args.epochs is not None else cfg.epochs
    os.makedirs(cfg.out_dir, exist_ok=True)

    {"coeff": coeff, "adistance": adistance, "labels": labels}[args.study](args)


if __name__ == "__main__":
    main()
