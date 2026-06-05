"""
Method registry (Section 4.3 of the paper).

Every deep method is a combination of switches over the SAME backbone, so they
share code and use the same hyper-parameters:
  * use_graph        : include the GGL + graph-conv branch?
  * graph_type       : 'arma' (CARMA family) or 'cheby' (DAGCN spectral filter)
  * use_adversarial  : include the GRL domain discriminator (Eq. 24)?
  * align_loss       : third loss in {none, mmd, mkmmd, lmmd, mklmmd, coral, hdan}
  * align_layers     : feature layers the alignment loss is applied at (z4 / z5)

This reproduces the paper's definitions:
  WDCNN      : wide-kernel CNN, source+labelled-target CE only, no graph, no DA.
  CARMA      : CNN + ARMA graph + classifier, CE only (no DA modules).
  DTLCNN     : CNN + multi-layer MK-MMD (no graph, no adversarial).
  DSACNN     : CNN + single-kernel LMMD (no graph, no adversarial).
  HDAN       : CNN + adversarial + (Wasserstein + MK-MMD) in the last two layers.
  DAGCN      : Cheby graph + adversarial + MMD.
  CARMA-M    : ARMA graph + adversarial + MK-MMD       (ablation of the class loss).
  CARMA-C    : ARMA graph + adversarial + CORAL        (ablation of the class loss).
  CARMA-DACA : ARMA graph + adversarial + MK-LMMD      (the PROPOSED method).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MethodSpec:
    name: str
    use_graph: bool
    graph_type: str = "arma"               # 'arma' | 'cheby'
    use_adversarial: bool = False
    align_loss: str = "none"               # none|mmd|mkmmd|lmmd|mklmmd|coral|hdan
    align_layers: List[str] = field(default_factory=lambda: ["z4", "z5"])


DEEP_METHODS = {
    "WDCNN":      MethodSpec("WDCNN",      use_graph=False),
    "CARMA":      MethodSpec("CARMA",      use_graph=True,  graph_type="arma"),
    "DTLCNN":     MethodSpec("DTLCNN",     use_graph=False, use_adversarial=False, align_loss="mkmmd"),
    "DSACNN":     MethodSpec("DSACNN",     use_graph=False, use_adversarial=False, align_loss="lmmd"),
    "HDAN":       MethodSpec("HDAN",       use_graph=False, use_adversarial=True,  align_loss="hdan"),
    "DAGCN":      MethodSpec("DAGCN",      use_graph=True,  graph_type="cheby",
                             use_adversarial=True, align_loss="mmd", align_layers=["z5"]),
    "CARMA-M":    MethodSpec("CARMA-M",    use_graph=True,  graph_type="arma",
                             use_adversarial=True, align_loss="mkmmd"),
    "CARMA-C":    MethodSpec("CARMA-C",    use_graph=True,  graph_type="arma",
                             use_adversarial=True, align_loss="coral"),
    "CARMA-DACA": MethodSpec("CARMA-DACA", use_graph=True,  graph_type="arma",
                             use_adversarial=True, align_loss="mklmmd"),
}

# order preserved as in Table 3 / Fig. 7
ALL_METHODS = ["WDCNN", "CARMA", "DTLCNN", "DSACNN", "HDAN",
               "DAGCN", "CARMA-M", "CARMA-C", "CARMA-DACA"]
