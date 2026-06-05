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

---

## Getting the data (this repo)

```bash
python download_cwru.py --root ./CWRU
```

[`download_cwru.py`](../download_cwru.py) fetches the exact 12 kHz **and** 48 kHz
drive-end `.mat` files (plus the Normal baseline) and validates each by checking
for a `*_DE_time` signal. If the download fails on your network, fetch the files
manually from the CWRU Bearing Data Center and drop them — named like `97.mat`,
`109.mat`, … — into `./CWRU/` (the loader also globs for `*<number>*.mat`).

**Preprocessing order in [`data.py`](../data.py):** read DE signal → 1024-point
non-overlapping windows (117/class) → per-window normalisation → (target only)
zero-fill the multi-rate missing mask → semi-supervised split.

### CWRU file numbers

Index order = `[0 hp, 1 hp, 2 hp, 3 hp]`. The Normal (NC) class uses the 48 kHz
baseline files for both sampling rates.

**12 kHz drive-end (source A1/B1/C1/D1):**

| Class | Files | | Class | Files |
|---|---|---|---|---|
| NC | 97, 98, 99, 100 | | OF014 | 197, 198, 199, 200 |
| IF007 | 105, 106, 107, 108 | | OF021 | 234, 235, 236, 237 |
| IF014 | 169, 170, 171, 172 | | BF007 | 118, 119, 120, 121 |
| IF021 | 209, 210, 211, 212 | | BF014 | 185, 186, 187, 188 |
| OF007 | 130, 131, 132, 133 | | BF021 | 222, 223, 224, 225 |

**48 kHz drive-end (target C2/D2):**

| Class | Files | | Class | Files |
|---|---|---|---|---|
| NC | 97, 98, 99, 100 | | OF014 | 201, 202, 203, 204 |
| IF007 | 109, 110, 111, 112 | | OF021 | 238, 239, 240, 241 |
| IF014 | 174, 175, 176, 177 | | BF007 | 122, 123, 124, 125 |
| IF021 | 213, 214, 215, 217 | | BF014 | 189, 190, 191, 192 |
| OF007 | 135, 136, 137, 138 | | BF021 | 226, 227, 228, 229 |

(For IF021 at 48 kHz, file 216 is absent in CWRU; 217 is the 3 hp file.)
