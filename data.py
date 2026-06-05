"""
CWRU bearing dataset loader for CARMA-DACA (Section 4.1, Table 2).

Pre-processing implemented here:
  * Drive-end (DE) vibration signal at 12 kHz (source) AND 48 kHz (target).
  * 10 health states: NC + {IF, OF, BF} x {0.007, 0.014, 0.021} in.
    (class order matches the paper's confusion matrices, Fig. 5 / Fig. 9:
     NC, IF007, IF014, IF021, OF007, OF014, OF021, BF007, BF014, BF021)
  * 6 working conditions (Table 2):
        A1 = 12 kHz / 0 hp / 1797 rpm        (source)
        B1 = 12 kHz / 1 hp / 1772 rpm        (source)
        C1 = 12 kHz / 2 hp / 1750 rpm        (source)
        D1 = 12 kHz / 3 hp / 1730 rpm        (source)
        C2 = 48 kHz / 2 hp / 1750 rpm        (target)
        D2 = 48 kHz / 3 hp / 1730 rpm        (target)
    48 kHz at 1797/1772 rpm is deliberately NOT used (class imbalance, Section 4.1).
  * 1024-point non-overlapping sliding window, 117 windows/class (1170 / 10).
  * Multi-rate MISSING DATA on the target: keep 1 of every 4 consecutive points
    (75% missing, up to 6 consecutive gaps), zero-filled (Fig. 4).
  * SEMI-SUPERVISED target: a small labelled fraction (0/1/5/10%) + the rest
    unlabelled. Accuracy is reported on the unlabelled target (transductive).

Put the raw CWRU *.mat files in `cfg.data_root` (12 kHz DE + 48 kHz DE + Normal
baseline). `download_cwru.py` fetches the exact files automatically; the loader
also globs for "*<number>*.mat", so the precise file naming does not matter.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------------------------------- #
# CWRU file numbers.  Index = load [0 hp, 1 hp, 2 hp, 3 hp].
# --------------------------------------------------------------------------- #
# 12 kHz drive-end fault files (+ 48 kHz normal baseline used for the NC class).
CWRU_12K = {
    "NC":    [97, 98, 99, 100],      # Normal baseline (sampled at 48 kHz natively)
    "IF007": [105, 106, 107, 108],   # inner race 0.007"
    "IF014": [169, 170, 171, 172],   # inner race 0.014"
    "IF021": [209, 210, 211, 212],   # inner race 0.021"
    "OF007": [130, 131, 132, 133],   # outer race 0.007" @6 o'clock
    "OF014": [197, 198, 199, 200],   # outer race 0.014" @6
    "OF021": [234, 235, 236, 237],   # outer race 0.021" @6
    "BF007": [118, 119, 120, 121],   # ball 0.007"
    "BF014": [185, 186, 187, 188],   # ball 0.014"
    "BF021": [222, 223, 224, 225],   # ball 0.021"
}
# 48 kHz drive-end fault files (+ same 48 kHz normal baseline for NC).
CWRU_48K = {
    "NC":    [97, 98, 99, 100],
    "IF007": [109, 110, 111, 112],
    "IF014": [174, 175, 176, 177],
    "IF021": [213, 214, 215, 217],   # 216 is absent in CWRU; 217 is the 3 hp file
    "OF007": [135, 136, 137, 138],
    "OF014": [201, 202, 203, 204],
    "OF021": [238, 239, 240, 241],
    "BF007": [122, 123, 124, 125],
    "BF014": [189, 190, 191, 192],
    "BF021": [226, 227, 228, 229],
}
CLASS_NAMES = list(CWRU_12K.keys())          # label index == position in this list

# Working conditions: name -> (sampling_rate_kHz, load_index, rpm, domain_kind)
CONDITIONS = {
    "A1": (12, 0, 1797, "source"),
    "B1": (12, 1, 1772, "source"),
    "C1": (12, 2, 1750, "source"),
    "D1": (12, 3, 1730, "source"),
    "C2": (48, 2, 1750, "target"),
    "D2": (48, 3, 1730, "target"),
}
COND_NAMES = list(CONDITIONS.keys())


# --------------------------------------------------------------------------- #
# raw-file utilities
# --------------------------------------------------------------------------- #
def _find_mat(data_root: str, file_number: int) -> str:
    candidates = [
        os.path.join(data_root, f"{file_number}.mat"),
        os.path.join(data_root, f"{file_number:03d}.mat"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(data_root, f"*{file_number}*.mat"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"Could not find CWRU file {file_number}.mat in {data_root}. "
        f"Run `python download_cwru.py --root {data_root}` first, or place the "
        f"file there manually (named like {file_number}.mat)."
    )


def _read_de_signal(mat_path: str) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if "DE_time" in k]
    if not keys:                                   # some normal files only carry FE/BA
        keys = [k for k in mat.keys() if k.endswith("_time")]
    if not keys:
        raise KeyError(f"No *_DE_time signal in {mat_path}. Keys: {list(mat.keys())}")
    return mat[keys[0]].reshape(-1).astype(np.float32)


def _normalize(window: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean-std":
        return (window - window.mean()) / (window.std() + 1e-8)
    if mode == "0-1":
        rng = window.max() - window.min()
        return (window - window.min()) / (rng + 1e-8)
    return window


def _windows(signal, length, step, n_max):
    out, idx = [], 0
    while idx + length <= len(signal) and len(out) < n_max:
        out.append(signal[idx:idx + length])
        idx += step
    return np.stack(out, axis=0)                    # (n, length) -- raw, un-normalised


def apply_multirate_missing(windows: np.ndarray, ratio: int, rng: np.random.Generator):
    """
    Zero out (ratio-1) of every `ratio` consecutive points (Section 4.1, Fig. 4).
    For ratio=4 this drops 75% of points, keeping 1 random point per group of 4,
    so up to 6 consecutive points can be missing. Operates per window.
    """
    n, L = windows.shape
    out = np.zeros_like(windows)
    n_groups = L // ratio
    for i in range(n):
        for g in range(n_groups):
            keep = rng.integers(0, ratio)           # which point in the group survives
            pos = g * ratio + keep
            out[i, pos] = windows[i, pos]
        # tail (if L not divisible by ratio): keep one random surviving point
        rem = L - n_groups * ratio
        if rem > 0:
            keep = rng.integers(0, rem)
            out[i, n_groups * ratio + keep] = windows[i, n_groups * ratio + keep]
    return out


# --------------------------------------------------------------------------- #
# per-condition dataset
# --------------------------------------------------------------------------- #
def build_condition(cfg, cond_name: str, rng: np.random.Generator):
    """
    Return (X, y) for one working condition. X is (N, 1, L) float32, normalised,
    with the multi-rate missing pattern already applied if the condition is a
    target (48 kHz) one and cfg.apply_missing_to_target is True.
    """
    rate, load_idx, _, kind = CONDITIONS[cond_name]
    file_map = CWRU_48K if rate == 48 else CWRU_12K

    X, y = [], []
    for label, cls in enumerate(CLASS_NAMES):
        sig = _read_de_signal(_find_mat(cfg.data_root, file_map[cls][load_idx]))
        win = _windows(sig, cfg.sample_length, cfg.window_step, cfg.n_per_class)
        if len(win) < cfg.n_per_class:
            raise ValueError(
                f"{cond_name}/{cls}: only {len(win)} windows available, "
                f"need {cfg.n_per_class}. Reduce cfg.n_per_class or window_step."
            )
        # Normalise each window FIRST, then apply the zero-fill missing mask, so
        # that missing samples are represented by exactly 0 at the network input
        # (per-sample mean/std is estimated from the complete window; with a
        # random 1-of-4 sub-sampling this is statistically the same as using the
        # observed points only).
        win = np.stack([_normalize(w, cfg.normalize) for w in win], axis=0)
        if kind == "target" and cfg.apply_missing_to_target:
            win = apply_multirate_missing(win, cfg.missing_ratio, rng)
        X.append(win)
        y.append(np.full(len(win), label))

    X = np.concatenate(X, axis=0)[:, None, :].astype(np.float32)   # (N, 1, L)
    y = np.concatenate(y, axis=0).astype(np.int64)
    return X, y


def _split_target(X, y, label_fraction, num_classes, rng):
    """
    Split the target condition into (labelled, unlabelled) subsets with an equal
    number of labelled samples per class (Section 4.2/5).
    """
    lab_idx, unlab_idx = [], []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(label_fraction * len(idx)))
        lab_idx.extend(idx[:k])
        unlab_idx.extend(idx[k:])
    return np.array(lab_idx, dtype=int), np.array(unlab_idx, dtype=int)


def _tensor_ds(X, y):
    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())


def transfer_loaders(cfg, source_cond: str, target_cond: str, seed: int):
    """
    Build dataloaders for one transfer task `source -> target`.

    Returns a dict with:
      source_train     : all source samples, labelled
      target_labelled  : the small labelled target fraction (may be empty at 0%)
      target_unlabelled: the unlabelled target samples (used for adv/align + eval)
      target_test      : the held-out evaluation set (= unlabelled target, or the
                         full target if cfg.eval_on_full_target)
    """
    rng = np.random.default_rng(seed)
    Xs, ys = build_condition(cfg, source_cond, rng)
    Xt, yt = build_condition(cfg, target_cond, rng)

    lab_idx, unlab_idx = _split_target(Xt, yt, cfg.label_fraction, cfg.num_classes, rng)

    src_train = _tensor_ds(Xs, ys)
    tgt_unlab = _tensor_ds(Xt[unlab_idx], yt[unlab_idx]) if len(unlab_idx) else _tensor_ds(Xt, yt)
    tgt_lab = _tensor_ds(Xt[lab_idx], yt[lab_idx]) if len(lab_idx) else None
    eval_X, eval_y = (Xt, yt) if cfg.eval_on_full_target else (Xt[unlab_idx], yt[unlab_idx])
    tgt_test = _tensor_ds(eval_X, eval_y)

    bs = cfg.batch_size
    loaders = {
        "source_train": DataLoader(src_train, batch_size=bs, shuffle=True, drop_last=True),
        "target_unlabelled": DataLoader(tgt_unlab, batch_size=bs, shuffle=True, drop_last=True),
        "target_labelled": (DataLoader(tgt_lab, batch_size=bs, shuffle=True, drop_last=False)
                            if tgt_lab is not None else None),
        "target_test": DataLoader(tgt_test, batch_size=bs, shuffle=False, drop_last=False),
    }
    return loaders


# --------------------------------------------------------------------------- #
# transfer-task lists (Section 5.1 / 5.2)
# --------------------------------------------------------------------------- #
CASE1_TASKS = [("C1", "C2"), ("D1", "D2")]                    # missing data only
CASE2_TASKS = [("A1", "C2"), ("A1", "D2"), ("B1", "C2"),
               ("B1", "D2"), ("C1", "D2"), ("D1", "C2")]      # + changing conditions
ALL_TASKS = CASE1_TASKS + CASE2_TASKS


def get_tasks(case: str = "all"):
    return {"case1": CASE1_TASKS, "case2": CASE2_TASKS, "all": ALL_TASKS}[case]
