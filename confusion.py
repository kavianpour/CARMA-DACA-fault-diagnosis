"""
Confusion matrix for a transfer task (reproduces Fig. 5 / Fig. 9).

Trains a CARMA-DACA model and plots the per-class confusion matrix on the target
evaluation set, with the paper's class order
(NC, IF007, IF014, IF021, OF007, OF014, OF021, BF007, BF014, BF021).

Usage:
    python confusion.py --source D1 --target D2 --label 0.10 --epochs 500
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from config import cfg
import data as D
from methods import DEEP_METHODS
from train import fit_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    yp, yt = [], []
    for x, y in loader:
        yp.append(model(x.to(device))["logits"].argmax(1).cpu().numpy())
        yt.append(y.numpy())
    return np.concatenate(yt), np.concatenate(yp)


def plot_confusion(args):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model, loaders, _, acc, _ = fit_model(cfg, DEEP_METHODS[args.method],
                                          args.source, args.target, args.epochs, cfg.seed)
    y_true, y_pred = predict(model, loaders["target_test"], device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(cfg.num_classes)); ax.set_yticks(range(cfg.num_classes))
    ax.set_xticklabels(D.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(D.CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    thresh = cm.max() / 2.0
    for i in range(cfg.num_classes):
        row = cm[i].sum()
        for j in range(cfg.num_classes):
            pct = 100.0 * cm[i, j] / row if row else 0.0
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > thresh else "black")
    ax.set_title(f"Confusion matrix  {args.source}->{args.target}  "
                 f"({args.method}, {int(cfg.label_fraction * 100)}% label)  acc={acc:.2f}%")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = f"{cfg.out_dir}/confusion_{args.source}_{args.target}.png"
    plt.savefig(out, dpi=150)
    print("Saved ->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="CARMA-DACA")
    p.add_argument("--source", default="D1", choices=D.COND_NAMES)
    p.add_argument("--target", default="D2", choices=D.COND_NAMES)
    p.add_argument("--label", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--data_root", default=None)
    args = p.parse_args()
    if args.data_root: cfg.data_root = args.data_root
    if args.label is not None: cfg.label_fraction = args.label
    args.epochs = args.epochs if args.epochs is not None else cfg.epochs
    os.makedirs(cfg.out_dir, exist_ok=True)
    plot_confusion(args)
