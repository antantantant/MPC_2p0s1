"""Action-space utilities (box constraints for u and v).

Identical to ``nl_sqp/action_spaces.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BoxActionSpace:
    """Simple per-dimension box constraint for u and v."""

    u_min: Tensor   # (du,)
    u_max: Tensor   # (du,)
    v_min: Tensor   # (dv,)
    v_max: Tensor   # (dv,)

    def clip_u(self, u: Tensor) -> Tensor:
        return torch.clamp(u, self.u_min, self.u_max)

    def clip_v(self, v: Tensor) -> Tensor:
        return torch.clamp(v, self.v_min, self.v_max)

    def clip_uv(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.clip_u(u), self.clip_v(v)
