"""
Domain-adaptation losses used by the comparison methods (Section 4.3).

  * MMD    : single multi-kernel RBF MMD (median heuristic) -> DAGCN's global term
  * MK-MMD : multi-kernel MMD with the FIVE fixed bandwidths {0.001,0.01,1,10,100}
             the paper specifies                              -> CARMA-M, DTLCNN
  * CORAL  : deep correlation alignment (Sun & Saenko 2016)   -> CARMA-C
  * Wasserstein : sliced Wasserstein-1 distance              -> HDAN (with MK-MMD)

MK-LMMD (the proposed class-alignment loss) lives in mklmmd.py.
"""

import torch

MK_SIGMAS = [0.001, 0.01, 1.0, 10.0, 100.0]


def _median_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    n = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    l2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    bandwidth = l2.detach().sum() / (n * n - n + 1e-8)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bws = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    return sum(torch.exp(-l2 / (bw + 1e-8)) for bw in bws)


def mmd_loss(source, target):
    """Global (unweighted) multi-kernel RBF MMD -> DAGCN."""
    bs = source.size(0)
    k = _median_kernel(source, target)
    XX, YY = k[:bs, :bs], k[bs:, bs:]
    XY, YX = k[:bs, bs:], k[bs:, :bs]
    return torch.mean(XX + YY - XY - YX)


def mk_mmd_loss(source, target, sigmas=MK_SIGMAS):
    """Multi-kernel MMD with the paper's fixed bandwidths -> CARMA-M / DTLCNN."""
    def kernel(x, y, s):
        d = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        return torch.exp(-d / (2.0 * s + 1e-12))
    cost = 0.0
    for s in sigmas:
        cost = cost + kernel(source, source, s).mean()
        cost = cost + kernel(target, target, s).mean()
        cost = cost - 2.0 * kernel(source, target, s).mean()
    return cost / len(sigmas)


def coral_loss(source, target):
    """Deep CORAL (Sun & Saenko 2016) -> CARMA-C."""
    d = source.size(1)

    def cov(x):
        xm = x - x.mean(0, keepdim=True)
        return (xm.t() @ xm) / (x.size(0) - 1 + 1e-8)

    return (cov(source) - cov(target)).pow(2).sum() / (4 * d * d)


def sliced_wasserstein(source, target, n_proj=128):
    """
    Sliced Wasserstein-1 distance: project onto random directions, sort, take
    the mean absolute difference. A light, differentiable proxy for the
    Wasserstein term of HDAN (which originally uses a critic network).
    """
    d = source.size(1)
    proj = torch.randn(d, n_proj, device=source.device)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
    ps = (source @ proj).sort(dim=0).values
    pt = (target @ proj).sort(dim=0).values
    m = min(ps.size(0), pt.size(0))
    return (ps[:m] - pt[:m]).abs().mean()


def hdan_loss(source, target):
    """HDAN hybrid distance: sliced-Wasserstein + MK-MMD (Section 4.3)."""
    return sliced_wasserstein(source, target) + mk_mmd_loss(source, target)
