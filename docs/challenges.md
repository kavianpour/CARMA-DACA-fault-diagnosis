# The three coupled challenges

Deploying deep-learning bearing fault diagnosis in the real world means facing
three problems at the same time. Most prior methods handle one and assume the
others away; CARMA-DACA is built to handle all three together.

---

## 1. Changing working conditions

Plain deep-learning models assume the training (source) and test (target) data
share one distribution. Rotating machinery breaks that assumption constantly —
load and speed change during operation, shifting the data distribution, so a
model trained under one condition degrades sharply under another.

**Why standard fixes fall short.** Domain-adaptation criteria (MMD, MK-MMD,
CORAL, Wasserstein, adversarial) reduce the *global* distribution gap, but they
**ignore subdomain matching** — aligning each class across domains. After global
alignment, a sample from one class can end up next to another class in latent
space, causing misclassification and even negative transfer.

**CARMA-DACA's response:** combine adversarial **domain adaptation** (global)
with **MK-LMMD class alignment** (per-class subdomains) — see
[method.md](method.md#3-three-optimization-objectives).

---

## 2. Scarce labels

Deep models need substantial labeled data, but in industry most collected data is
**unlabeled** — labeling everything wastes cost and expertise. Purely supervised
methods are ineffective when labels are scarce.

**CARMA-DACA's response:** a **semi-supervised** formulation. The source is fully
labeled; the target has only a tiny labeled fraction (the paper tests 0%, 1%, 5%,
10%) plus a large unlabeled pool. Knowledge transferred from the source lifts
target accuracy even at 1% labels — where each class has just **one** labeled
sample and 116 unlabeled ones.

---

## 3. Missing data (multi-rate sampling)

When sensors sample at different rates (plus sensor failure, packet dropout, etc.),
the dataset has gaps. With two sensors at a 4:1 rate ratio, only ~25% of samples
are structurally complete — a huge waste.

**Why standard fixes fall short:**

- **Listwise deletion** throws away every sample with any missing value — and a
  lot of useful information with it.
- **Imputation** (mean / regression / KNN) assumes structure that may not hold
  (normality, linearity), suffers in high dimensions, and degrades badly when a
  large fraction is missing.
- A prior transfer-learning approach for missing data only handled the *same*
  working condition and risked negative transfer.

**CARMA-DACA's response:** rather than fill the gaps, it **absorbs** them. The
paper's hardest setting drops **75%** of target-domain points (up to six
consecutive points missing per window), and the
[ARMA-GCN structural features](method.md#2-feature-extractor-cnn--arma-gcn) plus
domain/class alignment still recover accurate diagnosis.

---

## Why an *integrated* solution is needed

Earlier work treats these threads separately: DL methods ignore domain shift and
data structure; domain-adaptation methods ignore subdomain (per-class) matching;
GCN methods use polynomial filters that don't transfer well; and missing-data
methods either delete/impute information or assume a fixed working condition.
CARMA-DACA unifies CNN + ARMA-GCN feature extraction, global adversarial
alignment, and MK-LMMD class alignment into one end-to-end model, so changing
conditions, label scarcity, and missing data are addressed **together**.
