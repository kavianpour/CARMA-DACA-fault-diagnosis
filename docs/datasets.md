# Dataset

CARMA-DACA is evaluated on the **CWRU** bearing benchmark under a combined
missing-data and cross-working-condition protocol. This repository does **not**
redistribute the dataset — use the official source.

---

## CWRU (Case Western Reserve University)

Vibration from an induction-motor test rig (motor + torque transducer +
dynamometer), recorded by drive-end and fan-end accelerometers.

| Property | Value |
|---|---|
| Health states | 10 — NC + {IF, OF, BF} × 3 severities (0.007, 0.014, 0.021 in.) |
| Sampling rates | 12 kHz (source) and 48 kHz (target) |
| Sample length | 1024 points, non-overlapping sliding window |
| Per speed | 1170 samples (80% train / 20% test) |

🔗 Official source: the CWRU Bearing Data Center.

### Working conditions (domains)

| Domain | Sampling | Load | Speed |
|---|---|---|---|
| A1 | 12 kHz | 0 hp | 1797 rpm |
| B1 | 12 kHz | 1 hp | 1772 rpm |
| C1 | 12 kHz | 2 hp | 1750 rpm |
| D1 | 12 kHz | 3 hp | 1730 rpm |
| C2 | 48 kHz | 2 hp | 1750 rpm |
| D2 | 48 kHz | 3 hp | 1730 rpm |

12 kHz domains act as the (complete, labeled) **source**; 48 kHz domains act as
the **target**. Only six conditions are used to keep classes balanced for a fair
comparison.

---

## The missing-data protocol

The target sensor samples at **4×** the source rate, so the protocol simulates
multi-rate loss by **randomly dropping 3 of every 4 consecutive points** in each
target sample — i.e. **75% of target points are missing** (up to six consecutive
points may be lost in a window, making it harder). Source and target keep the
same sample/point counts; only the target is degraded.

A transfer task like **B1→D2** therefore means: train on complete, labeled 1 hp /
12 kHz data, and adapt to 3 hp / 48 kHz data that is mostly unlabeled **and** 75%
missing.

---

## Two case studies

- **Case study 1 — missing data only.** Tasks **C1→C2** and **D1→D2** keep the
  working condition fixed (so the only shift is the 75% missingness), at label
  budgets 0% / 1% / 5% / 10%.
- **Case study 2 — missing data + changing conditions.** Six cross-condition
  tasks (**A1→C2, A1→D2, B1→C2, B1→D2, C1→D2, D1→C2**) combine the 75%
  missingness with a load/speed change — the full difficulty.

See [`method.md`](method.md) for preprocessing and model details.
