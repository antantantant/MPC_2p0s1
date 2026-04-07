"""α (belief-splitting) parameterisation.

Identical to ``nl_sqp/alpha_param.py``.

Shape:  logits[k, node, type, action] → softmax over actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor

from .indexing import FullIaryTreeIndexer


@dataclass(frozen=True)
class AlphaParamConfig:
    init_scale: float = 0.01


class AlphaParam(nn.Module):
    def __init__(
        self,
        indexer: FullIaryTreeIndexer,
        cfg: AlphaParamConfig | None = None,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = AlphaParamConfig()

        self.indexer = indexer
        self.cfg = cfg

        K = indexer.K
        I = indexer.I
        max_nodes = indexer.max_nodes_per_depth

        device_t = torch.device(device) if isinstance(device, str) else device

        logits = torch.empty(K, max_nodes, I, I, dtype=dtype, device=device_t)
        torch.nn.init.normal_(logits, mean=0.0, std=cfg.init_scale)
        self.logits = nn.Parameter(logits)

    def forward(self) -> Tensor:
        """Return α = softmax(logits, dim=-1).  Shape (K, max_nodes, I, I)."""
        return torch.softmax(self.logits, dim=-1)
