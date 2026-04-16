from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from MPC_2p0s1.core.types import Tensor


@dataclass
class DualNoSplitP1PolicyConfig:
    """
    Config for no-split P1 policy parameterization.
    """

    init_scale_u: float = 0.05


class DualNoSplitP1PolicyParam(nn.Module):
    """
    No-split P1 policy: one control per time-step.

    u_seq[k] in R^{du}.
    """

    def __init__(
        self,
        K: int,
        du: int,
        cfg: DualNoSplitP1PolicyConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = DualNoSplitP1PolicyConfig()
        if int(K) < 0:
            raise ValueError(f"K must be nonnegative, got {K}.")
        if int(du) <= 0:
            raise ValueError(f"du must be positive, got {du}.")

        self.K = int(K)
        self.du = int(du)
        self.cfg = cfg
        device_t = torch.device(device) if isinstance(device, str) else device

        u_seq = torch.empty(self.K, self.du, dtype=dtype, device=device_t)
        torch.nn.init.normal_(u_seq, mean=0.0, std=float(cfg.init_scale_u))
        self.u_seq = nn.Parameter(u_seq)

    def action_at(self, depth: int) -> Tensor:
        if not (0 <= int(depth) < self.K):
            raise ValueError(f"depth must be in [0, {self.K - 1}], got {depth}.")
        return self.u_seq[int(depth)]


@dataclass
class DualNoSplitP2PolicyConfig:
    """
    Config for no-split P2 policy parameterization.
    """

    init_scale_v: float = 0.05


class DualNoSplitP2PolicyParam(nn.Module):
    """
    No-split P2 policy: one control per time-step.

    v_seq[k] in R^{dv}.
    """

    def __init__(
        self,
        K: int,
        dv: int,
        cfg: DualNoSplitP2PolicyConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = DualNoSplitP2PolicyConfig()
        if int(K) < 0:
            raise ValueError(f"K must be nonnegative, got {K}.")
        if int(dv) <= 0:
            raise ValueError(f"dv must be positive, got {dv}.")

        self.K = int(K)
        self.dv = int(dv)
        self.cfg = cfg
        device_t = torch.device(device) if isinstance(device, str) else device

        v_seq = torch.empty(self.K, self.dv, dtype=dtype, device=device_t)
        torch.nn.init.normal_(v_seq, mean=0.0, std=float(cfg.init_scale_v))
        self.v_seq = nn.Parameter(v_seq)

    def action_at(self, depth: int) -> Tensor:
        if not (0 <= int(depth) < self.K):
            raise ValueError(f"depth must be in [0, {self.K - 1}], got {depth}.")
        return self.v_seq[int(depth)]


@dataclass
class DualSplitP2PolicyConfig:
    """
    Config for split dual P2 policy parameterization.

    At each depth k, P2 chooses one atom a in {0, ..., A-1} where A is the
    split branching factor (typically I+1 in the dual game).
    """

    init_scale_v: float = 0.05
    init_scale_logits: float = 0.05
    init_scale_phat: float = 0.05


class DualSplitP2PolicyParam(nn.Module):
    """
    Split dual P2 policy with per-depth atomic branching.

    Parameters per depth k:
      - logits_seq[k, a]: unnormalized log-probability of atom a.
      - v_seq[k, a]: control associated with atom a.
      - phat_next_raw_seq[k, a, i]: proposed next dual variable for type i.

    The rollout enforces a martingale correction so that
      sum_a lam_k[a] * phat_next[k, a] = current phat_k.
    """

    def __init__(
        self,
        K: int,
        dv: int,
        I: int,
        branching: int,
        cfg: DualSplitP2PolicyConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = DualSplitP2PolicyConfig()
        if int(K) < 0:
            raise ValueError(f"K must be nonnegative, got {K}.")
        if int(dv) <= 0:
            raise ValueError(f"dv must be positive, got {dv}.")
        if int(I) <= 0:
            raise ValueError(f"I must be positive, got {I}.")
        if int(branching) <= 1:
            raise ValueError(f"branching must be >= 2, got {branching}.")

        self.K = int(K)
        self.dv = int(dv)
        self.I = int(I)
        self.branching = int(branching)
        self.cfg = cfg
        device_t = torch.device(device) if isinstance(device, str) else device

        logits_seq = torch.empty(self.K, self.branching, dtype=dtype, device=device_t)
        v_seq = torch.empty(self.K, self.branching, self.dv, dtype=dtype, device=device_t)
        phat_next_raw_seq = torch.empty(
            self.K, self.branching, self.I, dtype=dtype, device=device_t
        )
        torch.nn.init.normal_(logits_seq, mean=0.0, std=float(cfg.init_scale_logits))
        torch.nn.init.normal_(v_seq, mean=0.0, std=float(cfg.init_scale_v))
        torch.nn.init.normal_(phat_next_raw_seq, mean=0.0, std=float(cfg.init_scale_phat))

        self.logits_seq = nn.Parameter(logits_seq)
        self.v_seq = nn.Parameter(v_seq)
        self.phat_next_raw_seq = nn.Parameter(phat_next_raw_seq)

    def logits_at(self, depth: int) -> Tensor:
        if not (0 <= int(depth) < self.K):
            raise ValueError(f"depth must be in [0, {self.K - 1}], got {depth}.")
        return self.logits_seq[int(depth)]

    def action_table_at(self, depth: int) -> Tensor:
        if not (0 <= int(depth) < self.K):
            raise ValueError(f"depth must be in [0, {self.K - 1}], got {depth}.")
        return self.v_seq[int(depth)]

    def phat_raw_table_at(self, depth: int) -> Tensor:
        if not (0 <= int(depth) < self.K):
            raise ValueError(f"depth must be in [0, {self.K - 1}], got {depth}.")
        return self.phat_next_raw_seq[int(depth)]
