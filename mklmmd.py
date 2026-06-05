"""
Multi-Kernel Local Maximum Mean Discrepancy (MK-LMMD) -- Section 2.2 / 3.2,
Eq. 14-17, 25-28. This is the CLASS-ALIGNMENT term of CARMA-DACA.

LMMD (Zhu et al. 2021) extends MMD by weighting every sample with its class
membership, so it aligns the distribution of each class (subdomain) across the
source and target domains instead of only the global distribution. Source
weights use the true labels; target weights use the classifier's softmax
outputs as pseudo-labels (Eq. 16). CARMA-DACA uses the MULTI-KERNEL variant
(Eq. 27): a linear combination of Gaussian kernels with the five fixed
bandwidths {0.001, 0.01, 1, 10, 100} (Section 4.2), applied at FC4 and FC5 and
summed (Eq. 28).

Convention for a kernel of bandwidth s: k(x, y) = exp(-||x-y||^2 / (2 s)), matching
the multi-kernel MMD convention used by the comparison methods in da_losses.py.
"""

import numpy as np
import torch
import torch.nn as nn


def _multi_gaussian(source, target, sigmas):
    """Sum of fixed-bandwidth Gaussian kernels over the joint [source; target]."""
    total = torch.cat([source, target], dim=0)
    t0 = total.unsqueeze(0)
    t1 = total.unsqueeze(1)
    l2 = ((t0 - t1) ** 2).sum(2)                              # (M, M) pairwise sq. distance
    return sum(torch.exp(-l2 / (2.0 * s + 1e-12)) for s in sigmas)


def _median_gaussian(source, target, kernel_mul=2.0, kernel_num=5):
    """Median-heuristic multi-bandwidth kernel (used by single-kernel LMMD)."""
    n = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    l2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    bandwidth = l2.detach().sum() / (n * n - n + 1e-8)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bws = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    return sum(torch.exp(-l2 / (bw + 1e-8)) for bw in bws)


class LMMDLoss(nn.Module):
    """
    LMMD / MK-LMMD between source and target features.

    multi_kernel=True  -> MK-LMMD with the paper's fixed bandwidths (CARMA-DACA).
    multi_kernel=False -> single-kernel LMMD with the median heuristic (DSACNN).
    """

    def __init__(self, num_class=10, sigmas=None, multi_kernel=True):
        super().__init__()
        self.num_class = num_class
        self.sigmas = sigmas if sigmas is not None else [0.001, 0.01, 1.0, 10.0, 100.0]
        self.multi_kernel = multi_kernel

    @staticmethod
    def _onehot(labels, num_class):
        return np.eye(num_class)[labels]

    def _weights(self, s_label, t_soft, batch_size):
        """Per-sample class weights (Eq. 16) -> (w_ss, w_tt, w_st) matrices."""
        s_vec = self._onehot(s_label.cpu().numpy(), self.num_class)        # hard labels
        s_sum = s_vec.sum(0, keepdims=True); s_sum[s_sum == 0] = 100
        s_vec = s_vec / s_sum

        t_vec = t_soft.detach().cpu().numpy()                              # softmax pseudo-labels
        t_sca = t_vec.argmax(1)
        t_sum = t_vec.sum(0, keepdims=True); t_sum[t_sum == 0] = 100
        t_vec = t_vec / t_sum

        index = list(set(s_label.cpu().numpy()) & set(t_sca))             # classes in both
        mask = np.zeros((batch_size, self.num_class)); mask[:, index] = 1
        s_vec, t_vec = s_vec * mask, t_vec * mask

        w_ss = s_vec @ s_vec.T
        w_tt = t_vec @ t_vec.T
        w_st = s_vec @ t_vec.T
        n = len(index)
        if n != 0:
            w_ss, w_tt, w_st = w_ss / n, w_tt / n, w_st / n
        else:
            w_ss = w_tt = w_st = np.array([0.0])
        return (w_ss.astype("float32"), w_tt.astype("float32"), w_st.astype("float32"))

    def forward(self, source, target, s_label, t_soft):
        bs = source.size(0)
        if bs != target.size(0):
            raise ValueError(f"LMMD needs equal source/target batch sizes, "
                             f"got {bs} vs {target.size(0)}.")
        w_ss, w_tt, w_st = self._weights(s_label, t_soft, bs)
        dev = source.device
        w_ss = torch.from_numpy(w_ss).to(dev)
        w_tt = torch.from_numpy(w_tt).to(dev)
        w_st = torch.from_numpy(w_st).to(dev)

        if self.multi_kernel:
            kernels = _multi_gaussian(source, target, self.sigmas)
        else:
            kernels = _median_gaussian(source, target)

        if torch.isnan(kernels).any():
            return torch.tensor(0.0, device=dev)
        SS = kernels[:bs, :bs]
        TT = kernels[bs:, bs:]
        ST = kernels[:bs, bs:]
        return torch.sum(w_ss * SS + w_tt * TT - 2 * w_st * ST)
