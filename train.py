"""
Training / evaluation for CARMA-DACA and all comparison methods (Section 3.2, 4).

Total loss (Eq. 29, optimised with a GRL so a single backward pass realises the
two-step min-max of Eq. 30-31):

    L_total = L_C  +  alpha * L_DA  +  beta * L_CA

  L_C  : cross-entropy on the labelled source AND the few labelled target samples (Eq. 23)
  L_DA : domain-discriminator CE through the GRL (if the method is adversarial)   (Eq. 24)
  L_CA : class-alignment loss in {mklmmd, mkmmd, lmmd, coral, hdan, mmd, none}     (Eq. 25-28)
         applied at the FC4 (z4) and/or FC5 (z5) layers.

Examples
--------
  # proposed method, hardest task of case study 2, 10% labels
  python train.py --method CARMA-DACA --source A1 --target D2 --label 0.10 --epochs 500

  # one method across a case study (case1 = missing only, case2 = + changing conditions)
  python train.py --method CARMA-DACA --case case2 --label 0.01 --runs 1

  # full comparison table: every method x every task -> results/benchmark_*.csv
  python train.py --benchmark --case case2 --label 0.10 --runs 1
"""

import os
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import cfg
import data as D
from methods import DEEP_METHODS, ALL_METHODS
from model import build_model, xavier_init, calc_coeff
from mklmmd import LMMDLoss
import da_losses


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)["logits"]
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def alignment_loss(spec, out_s, out_t, ys, lmmd_mk, lmmd_sk):
    """Class-alignment term L_CA applied at the spec's layers (Eq. 25-28)."""
    name = spec.align_loss
    if name == "none":
        return torch.tensor(0.0, device=out_s["z5"].device)
    t_soft = torch.softmax(out_t["logits"], 1)
    total = 0.0
    for layer in spec.align_layers:
        fs, ft = out_s[layer], out_t[layer]
        if name == "mklmmd":
            total = total + lmmd_mk(fs, ft, ys, t_soft)
        elif name == "lmmd":
            total = total + lmmd_sk(fs, ft, ys, t_soft)
        elif name == "mkmmd":
            total = total + da_losses.mk_mmd_loss(fs, ft)
        elif name == "mmd":
            total = total + da_losses.mmd_loss(fs, ft)
        elif name == "coral":
            total = total + da_losses.coral_loss(fs, ft)
        elif name == "hdan":
            total = total + da_losses.hdan_loss(fs, ft)
        else:
            raise ValueError(name)
    return total


def fit_model(cfg, spec, source, target, epochs, seed, verbose=False):
    """Train one model; return (model, loaders, device, final_acc, best_acc)."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    loaders = D.transfer_loaders(cfg, source, target, seed)
    model = build_model(cfg, spec, num_classes=cfg.num_classes).to(device)
    xavier_init(model)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()
    lmmd_mk = LMMDLoss(num_class=cfg.num_classes, sigmas=cfg.mk_sigmas, multi_kernel=True).to(device)
    lmmd_sk = LMMDLoss(num_class=cfg.num_classes, multi_kernel=False).to(device)

    iter_num, best_acc = 0, 0.0
    for epoch in range(epochs):
        model.train()
        tgt_iter = itertools.cycle(loaders["target_unlabelled"])
        lab_iter = (itertools.cycle(loaders["target_labelled"])
                    if loaders["target_labelled"] is not None else None)

        for xs, ys in loaders["source_train"]:
            xt, _ = next(tgt_iter)
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            out_s = model(xs)
            out_t = model(xt)
            loss = ce(out_s["logits"], ys)

            # supervised loss on the few labelled target samples (Eq. 23, 2nd term)
            if lab_iter is not None:
                xtl, ytl = next(lab_iter)
                xtl, ytl = xtl.to(device), ytl.to(device)
                loss = loss + ce(model(xtl)["logits"], ytl)

            if epoch >= cfg.warmup_epochs:
                if spec.use_adversarial:
                    iter_num += 1
                    coeff = calc_coeff(iter_num, gamma=cfg.grl_gamma, max_iter=cfg.grl_max_iter)
                    d_s = model.domain_logits(out_s["struct"], coeff)
                    d_t = model.domain_logits(out_t["struct"], coeff)
                    dom_logits = torch.cat([d_s, d_t], dim=0)
                    dom_y = torch.cat([torch.zeros(d_s.size(0), dtype=torch.long, device=device),
                                       torch.ones(d_t.size(0), dtype=torch.long, device=device)])
                    loss = loss + cfg.alpha * ce(dom_logits, dom_y)      # Eq. 24 (2-class form)
                if spec.align_loss != "none":
                    loss = loss + cfg.beta * alignment_loss(spec, out_s, out_t, ys, lmmd_mk, lmmd_sk)

            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = evaluate(model, loaders["target_test"], device)
        best_acc = max(best_acc, acc)
        if verbose:
            print(f"  epoch {epoch + 1:3d}/{epochs}  target_acc={acc:6.2f}  best={best_acc:6.2f}")

    final_acc = evaluate(model, loaders["target_test"], device)
    return model, loaders, device, final_acc, best_acc


def run_method(cfg, method, source, target, epochs, runs, seed0=0):
    accs = []
    for r in range(runs):
        _, _, _, acc, _ = fit_model(cfg, DEEP_METHODS[method], source, target,
                                    epochs, seed0 + r)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="CARMA-DACA", choices=ALL_METHODS)
    p.add_argument("--source", default="A1", choices=D.COND_NAMES)
    p.add_argument("--target", default="D2", choices=D.COND_NAMES)
    p.add_argument("--label", type=float, default=None, help="labelled target fraction (0/0.01/0.05/0.10)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--case", default=None, choices=["case1", "case2", "all"],
                   help="run every task of a case study instead of one task")
    p.add_argument("--benchmark", action="store_true", help="all methods x all tasks of --case")
    p.add_argument("--data_root", default=None)
    args = p.parse_args()

    if args.data_root: cfg.data_root = args.data_root
    if args.label is not None: cfg.label_fraction = args.label
    if args.epochs is not None: cfg.epochs = args.epochs
    epochs = cfg.epochs
    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = f"label{int(round(cfg.label_fraction * 100))}"

    def tname(s, t):
        return f"{s}->{t}"

    if args.benchmark:
        tasks = D.get_tasks(args.case or "case2")
        rows = []
        for method in ALL_METHODS:
            row = {"method": method}
            vals = []
            for s, t in tasks:
                mean, _ = run_method(cfg, method, s, t, epochs, args.runs)
                row[tname(s, t)] = round(mean, 2)
                vals.append(mean)
            row["AVG"] = round(float(np.mean(vals)), 2)
            print(method, "AVG=", row["AVG"])
            rows.append(row)
        out = os.path.join(cfg.out_dir, f"benchmark_{args.case or 'case2'}_{tag}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("Saved ->", out)

    elif args.case:
        tasks = D.get_tasks(args.case)
        rows = []
        for s, t in tasks:
            mean, std = run_method(cfg, args.method, s, t, epochs, args.runs)
            print(f"{tname(s, t)}: {mean:.2f} +/- {std:.2f}")
            rows.append({"task": tname(s, t), "mean_acc": mean, "std_acc": std})
        df = pd.DataFrame(rows)
        df.loc[len(df)] = {"task": "AVG", "mean_acc": df["mean_acc"].mean(),
                           "std_acc": df["std_acc"].mean()}
        out = os.path.join(cfg.out_dir, f"{args.method}_{args.case}_{tag}.csv")
        df.to_csv(out, index=False)
        print(f"\nAVG: {df.iloc[-1]['mean_acc']:.2f}%  ->  {out}")

    else:
        _, _, _, acc, best = fit_model(cfg, DEEP_METHODS[args.method], args.source,
                                       args.target, epochs, cfg.seed, verbose=True)
        print(f"\n{args.method} {tname(args.source, args.target)} ({tag}): "
              f"final={acc:.2f}%  best={best:.2f}%")


if __name__ == "__main__":
    main()
