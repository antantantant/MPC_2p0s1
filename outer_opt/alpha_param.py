# outer_opt/alpha_param.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..core.types import Tensor
from ..tree.indexing import FullIaryTreeIndexer


@dataclass
class AlphaParamConfig:
    """
    Configuration for the α/logit parameterization.

    Attributes
    ----------
    init_scale:
        Standard deviation of the Gaussian initialization for logits.
        A small value (e.g. 0.01) keeps α close to uniform initially.
    """

    init_scale: float = 0.01


class AlphaParam(nn.Module):
    """
    Logit parameterization of the belief-splitting coefficients α_{k,ω,i}^a.

    Shape convention
    ----------------
    logits : (K, max_nodes_per_depth, I, I)

        - logits[k, node, i, a] is the (unnormalized) log-probability of
          playing prototype a when the game is at node (k, ω) and P1’s
          type is i.
        - K is the number of time steps.
        - max_nodes_per_depth is the maximum node count over depths
          (I^K for a full I-ary tree), as given by the tree indexer.
        - I is both the number of types and the number of prototypes.

    The corresponding α is obtained via a softmax over the last dimension:

        α_{k,ω,i,*} = softmax_a logits[k, node, i, a].

    Only the first num_nodes_per_depth[k] nodes are used at each depth k;
    logits for unused nodes are ignored but remain trainable parameters.
    """

    def __init__(
        self,
        indexer: FullIaryTreeIndexer,
        alpha_cfg: AlphaParamConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if alpha_cfg is None:
            alpha_cfg = AlphaParamConfig()

        self.indexer = indexer
        self.alpha_cfg = alpha_cfg

        K = indexer.K
        I = indexer.I
        max_nodes = indexer.max_nodes_per_depth

        device = torch.device(device) if isinstance(device, str) else device

        logits = torch.empty(
            K,
            max_nodes,
            I,
            I,
            dtype=dtype,
            device=device,
        )

        # Small Gaussian initialization around zero → almost uniform α.
        torch.nn.init.normal_(logits, mean=0.0, std=alpha_cfg.init_scale)

        self.logits = nn.Parameter(logits)

    def forward(self) -> Tensor:
        """
        Compute α from logits via softmax over the prototype dimension.

        Returns
        -------
        Tensor
            α tensor of shape (K, max_nodes_per_depth, I, I), where the last
            dimension sums to 1 for each (k, node, i).
        """
        return torch.softmax(self.logits, dim=-1)

    @property
    def K(self) -> int:
        return self.indexer.K

    @property
    def I(self) -> int:
        return self.indexer.I

    @property
    def max_nodes_per_depth(self) -> int:
        return self.indexer.max_nodes_per_depth