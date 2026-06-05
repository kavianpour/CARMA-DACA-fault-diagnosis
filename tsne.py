"""
t-SNE feature visualisation (reproduces Fig. 6 / Fig. 10).

Trains a CARMA-DACA model for a transfer task and plots a 2-D t-SNE embedding at
four stages -- (a) input, (b) GAP layer, (c) ARMA3 structured features, (d) FC6
logits -- with source (o) and target (+) markers, coloured by class.

Usage:
    python tsne.py --source C1 --target C2 --label 0.10 --epochs 500
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import cfg
import data as D
from methods import DEEP_METHODS
from train import fit_model


@torch.no_grad()
def collect(model, loader, device, max_batches=6):
    model.eval()
    raws, gaps, structs, logits, labels = [], [], [], [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        node = model.cnn(x).squeeze(-1) if hasattr(model, "cnn") else None
        out = model(x)
        raws.append(x.cpu().numpy().reshape(x.size(0), -1))
        gaps.append(node.cpu().numpy() if node is not None else out["struct"].cpu().numpy())
        structs.append(out["struct"].cpu().numpy())
        logits.append(out["logits"].cpu().numpy())
        labels.append(y.numpy())
    cat = lambda L: np.concatenate(L)
    return cat(raws), cat(gaps), cat(structs), cat(logits), cat(labels)


def _embed(M):
    n = M.shape[0]
    return TSNE(n_components=2, init="pca",
               perplexity=min(30, max(5, n // 4))).fit_transform(M)


def plot_tsne(args):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model, loaders, _, _, _ = fit_model(cfg, DEEP_METHODS[args.method],
                                        args.source, args.target, args.epochs, cfg.seed)
    rs, gs, ss, ls, ys = collect(model, loaders["source_train"], device)
    rt, gt, st, lt, yt = collect(model, loaders["target_test"], device)

    stages = [("(a) input", rs, rt, ys, yt),
              ("(b) GAP", gs, gt, ys, yt),
              ("(c) ARMA3", ss, st, ys, yt),
              ("(d) FC6", ls, lt, ys, yt)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (title, src, tgt, ys_, yt_) in zip(axes.ravel(), stages):
        emb = _embed(np.concatenate([src, tgt]))
        es, et = emb[:len(src)], emb[len(src):]
        ax.scatter(es[:, 0], es[:, 1], c=ys_, marker="o", cmap="tab10", s=14, alpha=0.7)
        ax.scatter(et[:, 0], et[:, 1], c=yt_, marker="+", cmap="tab10", s=22)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"t-SNE  {args.source}->{args.target}  ({args.method}, "
                 f"{int(cfg.label_fraction * 100)}% label)   o=source  +=target")
    plt.tight_layout()
    out = f"{cfg.out_dir}/tsne_{args.source}_{args.target}.png"
    plt.savefig(out, dpi=150)
    print("Saved ->", out)


if __name__ == "__main__":
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="CARMA-DACA")
    p.add_argument("--source", default="C1", choices=D.COND_NAMES)
    p.add_argument("--target", default="C2", choices=D.COND_NAMES)
    p.add_argument("--label", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--data_root", default=None)
    args = p.parse_args()
    if args.data_root: cfg.data_root = args.data_root
    if args.label is not None: cfg.label_fraction = args.label
    args.epochs = args.epochs if args.epochs is not None else cfg.epochs
    os.makedirs(cfg.out_dir, exist_ok=True)
    plot_tsne(args)
