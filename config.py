"""
Central configuration for the CARMA-DACA implementation.

CARMA-DACA = Convolution neural network + Auto-Regressive Moving Average graphs
             with Domain Adaptation and Class Alignment.
Reference: M. Kavianpour, A. Ramezani, M.T.H. Beheshti,
"A class alignment method based on graph convolution neural network for bearing
fault diagnosis in presence of missing data and changing working conditions",
Measurement 199 (2022) 111536.  DOI: 10.1016/j.measurement.2022.111536

Every hyper-parameter below follows Section 4.2 ("Implementation details") and
Table 1 of the paper. Points where the paper is ambiguous are flagged with a
NOTE and exposed here so they can be changed without touching the code.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- data (Section 4.1, Table 2) ------------------------------------- #
    data_root: str = "./CWRU"        # directory holding the raw CWRU *.mat files
    sample_length: int = 1024        # sliding-window length (Section 4.1)
    window_step: int = 1024          # NON-overlapping windows (Section 4.1)
    n_per_class: int = 117           # windows per class per condition (1170 / 10 classes)
    num_classes: int = 10            # NC + {IF, OF, BF} x {0.007, 0.014, 0.021}
    normalize: str = "mean-std"      # per-sample normalisation: "mean-std" | "0-1" | "none"

    # ----- missing data (Section 4.1 / Fig. 4) ----------------------------- #
    # The 48 kHz target sensor samples 4x faster than the 12 kHz source, so for
    # every 4 consecutive points only 1 survives -> 75% of the target samples are
    # missing, with up to 6 consecutive missing points. Missing points are
    # zero-filled so the CNN keeps a fixed 1024-length input.
    # NOTE: zero-filling is our explicit modelling choice (the paper feeds the
    # incomplete signal to the network rather than imputing it).
    missing_ratio: int = 4           # keep 1 of every `missing_ratio` points (4 -> 75% missing)
    apply_missing_to_target: bool = True

    # ----- semi-supervised setting (Section 4.2 / 5) ----------------------- #
    # Fraction of LABELLED samples in the target domain. The paper reports
    # 0% (unsupervised), 1%, 5% and 10%. With n_per_class=117 this gives
    # roughly 0 / 1 / 6 / 12 labelled samples per class, matching the paper's
    # "1 labelled and 116 unlabelled samples in each class" for the 1% mode.
    label_fraction: float = 0.10
    # Accuracy is reported on the UNLABELLED target samples (transductive SSDA
    # protocol). Set eval_on_full_target=True to evaluate on every target sample
    # instead (this reproduces the per-class counts of the paper's confusion
    # matrices, e.g. 117 samples/class in Fig. 5 / Fig. 9).
    eval_on_full_target: bool = False

    # ----- model (Table 1) ------------------------------------------------- #
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])  # Conv1..4
    cnn_kernels:  List[int] = field(default_factory=lambda: [128, 64, 3, 3])    # wide -> narrow
    cnn_stride: int = 2              # Table 1: stride 2 in every conv layer
    pool_size: int = 2              # Table 1: max-pool size 2 after Conv1..3
    node_feat_dim: int = 128        # CNN + GAP output  (Table 1: Conv4 -> N*128*1)
    graph_topk: int = 2             # Top-K(.) sparse adjacency in the GGL (Eq. 20, K=2)

    # ARMA graph convolution (Section 2.1 / 3.1, Table 1)
    arma_hidden: List[int] = field(default_factory=lambda: [128, 256, 256])  # ARMA1/2/3 widths
    arma_num_stacks: int = 3        # third-order ARMA filter K=3 (Section 4.2)
    arma_num_layers: int = 1        # recursion depth T of the ARMA_1 filter (Eq. 5)
    # NOTE: the paper says both "three stacked layers of ARMA_K" (-> 3 sequential
    # ARMA layers, listed as ARMA1/ARMA2/ARMA3 in Table 1) and "third-order ARMA
    # filter" (-> K=3 parallel stacks per layer). We implement both: 3 sequential
    # ARMAConv layers, each with num_stacks=3. Change arma_num_stacks to revert to
    # a first-order (ARMA_1) filter if your reading differs.
    cheby_k: int = 2                # ChebConv order for the DAGCN comparison method

    # heads (Table 1)
    fc_domain: List[int] = field(default_factory=lambda: [128, 128, 2])   # FC1, FC2, FC3
    fc_class:  List[int] = field(default_factory=lambda: [128, 128])      # FC4, FC5
    disc_hidden: int = 128          # domain-discriminator backbone width (FC1/FC2)

    # ----- objective (Section 3.2, Eq. 29) --------------------------------- #
    alpha: float = 1.0              # trade-off for the adversarial loss  (paper: alpha = 1)
    beta: float = 0.05              # trade-off for the MK-LMMD loss      (paper: beta = 0.05)
    # MK-LMMD / MK-MMD: linear combination of 5 Gaussian kernels with the fixed
    # bandwidths named in Section 4.2.
    mk_sigmas: List[float] = field(default_factory=lambda: [0.001, 0.01, 1.0, 10.0, 100.0])

    # ----- optimisation (Section 4.2) -------------------------------------- #
    lr: float = 1e-3                # Adam, learning rate 0.001
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 500               # all methods trained for 500 epochs
    grl_gamma: float = 10.0         # GRL ramp-up steepness (DANN schedule)
    grl_max_iter: float = 10000.0   # GRL ramp-up horizon
    warmup_epochs: int = 1          # source-only epochs before the DA terms switch on

    # ----- experiment ------------------------------------------------------ #
    runs: int = 10                  # repeat each task 10 times (Section 4.2)
    seed: int = 0
    device: str = "cuda"            # "cuda" | "cpu"
    out_dir: str = "./results"


cfg = Config()
