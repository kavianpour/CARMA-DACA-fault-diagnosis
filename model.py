"""
CARMA-DACA model (Section 3.1, Table 1).

Pipeline for a mini-batch of N samples (= N graph nodes):
    raw signal (N, 1, 1024)
      -> Spatial Feature block : 4 Conv1d (wide->narrow) + BN + ReLU + MaxPool,
                                 last layer ends with Global Average Pooling     (Eq. 18)
                                 -> node feature matrix X (N, 128)
      -> Structural Feature block:
           Graph Generation Layer (GGL): A = normalise(X X^T), Top-K=2           (Eq. 19-20)
           3 x ARMA graph convolution (ARMA1/ARMA2/ARMA3) + BN + ReLU            (Eq. 21-22)
                                 -> structured features H (N, 256)
    Heads (operate per node):
      Domain adaptation block : GRL -> FC1 -> FC2 -> FC3 (2) domain logits       (Eq. 24)
      Class alignment block   : FC4 -> z4, FC5 -> z5   (MK-LMMD sits here)        (Eq. 25-26)
      Classification block    : FC6 -> 10 class logits                           (Eq. 23)

The SAME backbone serves every deep comparison method through `methods.MethodSpec`
switches (use_graph / graph_type / use_adversarial / align_loss). With the default
spec it IS the proposed CARMA-DACA. WDCNN keeps a dedicated wide-kernel CNN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    from torch_geometric.nn import ARMAConv, ChebConv
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required (ARMAConv, ChebConv). Install it following "
        "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    ) from e


# --------------------------------------------------------------------------- #
# Gradient Reversal Layer  (adversarial domain adaptation, Eq. 29-31)
# --------------------------------------------------------------------------- #
def calc_coeff(iter_num, high=1.0, low=0.0, gamma=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-gamma * iter_num / max_iter))
                 - (high - low) + low)


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def grad_reverse(x, coeff=1.0):
    return _GradReverse.apply(x, coeff)


# --------------------------------------------------------------------------- #
# Graph Generation Layer  (Eq. 19-20)
# --------------------------------------------------------------------------- #
def generate_graph(feat: torch.Tensor, topk: int):
    """A = normalise(feat feat^T); keep Top-K entries per row -> sparse graph."""
    n = feat.size(0)
    k = min(topk, n)
    A = feat @ feat.t()                                   # (N, N)
    A = A / (A.max(dim=1, keepdim=True).values + 1e-8)    # row-wise normalisation
    vals, idx = A.topk(k, dim=1, largest=True)            # Top-K(A)  (Eq. 20)
    rows = torch.arange(n, device=feat.device).repeat_interleave(k)
    cols = idx.reshape(-1)
    edge_index = torch.stack([rows, cols], dim=0)
    edge_weight = vals.reshape(-1)
    return edge_index, edge_weight


# --------------------------------------------------------------------------- #
# Spatial feature block  (4-layer wide->narrow CNN, Table 1, Eq. 18)
# --------------------------------------------------------------------------- #
class SpatialCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        chans, kers = cfg.cnn_channels, cfg.cnn_kernels
        layers, in_c = [], 1
        for i, (c, k) in enumerate(zip(chans, kers)):
            layers += [nn.Conv1d(in_c, c, kernel_size=k, stride=cfg.cnn_stride, padding=k // 2),
                       nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
            if i < len(chans) - 1:
                layers.append(nn.MaxPool1d(kernel_size=cfg.pool_size, stride=cfg.pool_size))
            else:
                layers.append(nn.AdaptiveAvgPool1d(1))    # Global Average Pooling
            in_c = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)                    # (N, 128)


# --------------------------------------------------------------------------- #
# ARMA / Cheby graph block  (Structural feature block, Table 1, Eq. 21-22)
# --------------------------------------------------------------------------- #
class GraphBlock(nn.Module):
    def __init__(self, cfg, graph_type="arma"):
        super().__init__()
        self.graph_type = graph_type
        dims = [cfg.node_feat_dim] + list(cfg.arma_hidden)   # 128 -> 128 -> 256 -> 256
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for din, dout in zip(dims[:-1], dims[1:]):
            if graph_type == "arma":
                conv = ARMAConv(din, dout, num_stacks=cfg.arma_num_stacks,
                                num_layers=cfg.arma_num_layers, act=None)
            elif graph_type == "cheby":
                conv = ChebConv(din, dout, K=cfg.cheby_k)    # DAGCN spectral filter
            else:
                raise ValueError(graph_type)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(dout))
        self.out_dim = dims[-1]

    def forward(self, x, edge_index, edge_weight):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(bn(x))                                # BN + ReLU after each layer
        return x


# --------------------------------------------------------------------------- #
# Full CARMA-DACA model (configurable backbone)
# --------------------------------------------------------------------------- #
class CarmaDaca(nn.Module):
    def __init__(self, cfg, spec=None, num_classes=None):
        super().__init__()
        self.cfg = cfg
        self.use_graph = True if spec is None else spec.use_graph
        self.graph_type = "arma" if spec is None else spec.graph_type
        self.use_adversarial = True if spec is None else spec.use_adversarial
        n_cls = num_classes if num_classes is not None else cfg.num_classes

        self.cnn = SpatialCNN(cfg)
        if self.use_graph:
            self.graph = GraphBlock(cfg, graph_type=self.graph_type)
            struct_dim = self.graph.out_dim                  # 256
        else:
            self.graph = None
            struct_dim = cfg.node_feat_dim                   # 128

        # class alignment block: FC4 -> z4, FC5 -> z5  (Eq. 25-26)
        f4, f5 = cfg.fc_class
        self.fc4 = nn.Sequential(nn.Linear(struct_dim, f4), nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(nn.Linear(f4, f5), nn.ReLU(inplace=True))
        self.fc6 = nn.Linear(f5, n_cls)                      # classifier (Eq. 23)

        # domain adaptation block: FC1 -> FC2 -> FC3 (2)    (Eq. 24)
        d1, d2, d3 = cfg.fc_domain
        self.domain_net = nn.Sequential(
            nn.Linear(struct_dim, d1), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(d1, d2), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(d2, d3),
        )

    def feature(self, x):
        node = self.cnn(x)                                   # (N, 128)
        if self.use_graph:
            ei, ew = generate_graph(node, self.cfg.graph_topk)
            struct = self.graph(node, ei, ew)                # (N, 256)
        else:
            struct = node                                    # (N, 128)
        return struct

    def forward(self, x):
        struct = self.feature(x)
        z4 = self.fc4(struct)
        z5 = self.fc5(z4)
        logits = self.fc6(z5)
        return {"struct": struct, "z4": z4, "z5": z5, "logits": logits}

    def domain_logits(self, struct, coeff):
        return self.domain_net(grad_reverse(struct, coeff))  # (N, 2)


# --------------------------------------------------------------------------- #
# WDCNN baseline (Zhang et al. 2017, comparison method in Section 4.3)
# Wide first-layer kernel, then narrow kernels; CE only, no graph / no DA.
# Same dict interface as CarmaDaca so train.py is unchanged.
# --------------------------------------------------------------------------- #
class WDCNN(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        self.use_graph = False
        self.use_adversarial = False
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24), nn.BatchNorm1d(16),
            nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32),
            nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64),
            nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64),
            nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3), nn.BatchNorm1d(64),
            nn.ReLU(True), nn.AdaptiveMaxPool1d(1),
        )
        self.fc4 = nn.Sequential(nn.Linear(64, 100), nn.ReLU(True))   # FC layer (100 units)
        self.fc5 = nn.Identity()
        self.fc6 = nn.Linear(100, num_classes)

    def feature(self, x):
        return self.cnn(x).squeeze(-1)

    def forward(self, x):
        struct = self.feature(x)
        z4 = self.fc4(struct)
        z5 = self.fc5(z4)
        logits = self.fc6(z5)
        return {"struct": struct, "z4": z4, "z5": z5, "logits": logits}


# --------------------------------------------------------------------------- #
def build_model(cfg, spec, num_classes):
    """Factory used by train.py."""
    if spec.name == "WDCNN":
        return WDCNN(cfg, num_classes=num_classes)
    return CarmaDaca(cfg, spec=spec, num_classes=num_classes)


def xavier_init(module):
    """Xavier initialisation (Section 4.2)."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
