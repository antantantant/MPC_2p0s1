from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from MPC_2p0s1.core.types import Tensor


def _make_activation(name: str) -> nn.Module:
    k = str(name).strip().lower()
    if k == "relu":
        return nn.ReLU()
    if k == "gelu":
        return nn.GELU()
    if k == "silu":
        return nn.SiLU()
    if k == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"Unsupported activation '{name}'. Expected one of: relu, gelu, silu, tanh."
    )


def _normalize_t(
    t: Tensor | float | int,
    *,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Normalize time feature to shape (B, 1).
    """
    if isinstance(t, (float, int)):
        return torch.full((B, 1), float(t), device=device, dtype=dtype)
    if t.ndim == 0:
        return torch.full((B, 1), float(t.item()), device=device, dtype=dtype)
    if t.ndim == 1:
        if t.shape[0] == B:
            return t.view(B, 1).to(device=device, dtype=dtype)
        if t.shape[0] == 1:
            return t.view(1, 1).expand(B, 1).to(device=device, dtype=dtype)
    if t.ndim == 2 and t.shape == (B, 1):
        return t.to(device=device, dtype=dtype)
    raise ValueError(f"Unsupported t shape for B={B}: {tuple(t.shape)}")


@dataclass
class DualSplitP1BRNetConfig:
    """
    Config for dual split P1 best-response network.
    """

    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "silu"
    action_scale_u: float | None = None


class DualSplitP1BRNet(nn.Module):
    """
    P1 BR network conditioned on (x, phat, t_norm).

    Input
    -----
    x:    (B, dx)
    phat: (B, I)
    t:    scalar or (B,) or (B,1), normalized in [0,1]

    Output
    ------
    u:    (B, du)

    If `cfg.action_scale_u` is provided, output is bounded as
      u = tanh(u_raw) * action_scale_u.
    """

    def __init__(
        self,
        dx: int,
        I: int,
        du: int,
        cfg: DualSplitP1BRNetConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = DualSplitP1BRNetConfig()
        if int(dx) <= 0:
            raise ValueError(f"dx must be positive, got {dx}.")
        if int(I) <= 0:
            raise ValueError(f"I must be positive, got {I}.")
        if int(du) <= 0:
            raise ValueError(f"du must be positive, got {du}.")
        if any(int(h) <= 0 for h in cfg.hidden_dims):
            raise ValueError(f"All hidden dims must be positive, got {cfg.hidden_dims}.")

        self.dx = int(dx)
        self.I = int(I)
        self.du = int(du)
        self.cfg = cfg
        self.action_scale_u = (
            float(cfg.action_scale_u) if cfg.action_scale_u is not None else None
        )
        if self.action_scale_u is not None and self.action_scale_u <= 0.0:
            raise ValueError(f"action_scale_u must be positive, got {self.action_scale_u}.")

        in_dim = self.dx + self.I + 1
        layers: list[nn.Module] = []
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(_make_activation(cfg.activation))
            in_dim = int(h)
        self.trunk = nn.Sequential(*layers)
        self.head_u = nn.Linear(in_dim, self.du)

        self.to(device=device, dtype=dtype)

    def forward(self, x: Tensor, phat: Tensor, t: Tensor | float | int) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.dx:
            raise ValueError(f"x must have shape (B,{self.dx}), got {tuple(x.shape)}")
        if phat.ndim != 2 or phat.shape[1] != self.I:
            raise ValueError(f"phat must have shape (B,{self.I}), got {tuple(phat.shape)}")
        if x.shape[0] != phat.shape[0]:
            raise ValueError("x and phat batch dimensions must match.")

        B = x.shape[0]
        t_feat = _normalize_t(t, B=B, device=x.device, dtype=x.dtype)
        inp = torch.cat([x, phat, t_feat], dim=1)
        h = self.trunk(inp)
        u = self.head_u(h)
        if self.action_scale_u is not None:
            u = torch.tanh(u) * self.action_scale_u
        return u


@dataclass
class DualSplitP2NetConfig:
    """
    Config for split dual P2 network.
    """

    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "silu"
    action_scale_v: float | None = None


class DualSplitP2Net(nn.Module):
    """
    P2 dual network conditioned on (x, phat, t_norm).

    Outputs per sample:
      - branch logits:         (A,)
      - branch controls:       (A, dv)
      - branch raw phat_next:  (A, I)

    If `cfg.action_scale_v` is provided, branch controls are bounded as
      v = tanh(v_raw) * action_scale_v.
    """

    def __init__(
        self,
        dx: int,
        I: int,
        dv: int,
        branching: int,
        cfg: DualSplitP2NetConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = DualSplitP2NetConfig()
        if int(dx) <= 0:
            raise ValueError(f"dx must be positive, got {dx}.")
        if int(I) <= 0:
            raise ValueError(f"I must be positive, got {I}.")
        if int(dv) <= 0:
            raise ValueError(f"dv must be positive, got {dv}.")
        if int(branching) <= 1:
            raise ValueError(f"branching must be >= 2, got {branching}.")
        if any(int(h) <= 0 for h in cfg.hidden_dims):
            raise ValueError(f"All hidden dims must be positive, got {cfg.hidden_dims}.")

        self.dx = int(dx)
        self.I = int(I)
        self.dv = int(dv)
        self.branching = int(branching)
        self.cfg = cfg
        self.action_scale_v = (
            float(cfg.action_scale_v) if cfg.action_scale_v is not None else None
        )
        if self.action_scale_v is not None and self.action_scale_v <= 0.0:
            raise ValueError(f"action_scale_v must be positive, got {self.action_scale_v}.")

        in_dim = self.dx + self.I + 1
        layers: list[nn.Module] = []
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(_make_activation(cfg.activation))
            in_dim = int(h)
        self.trunk = nn.Sequential(*layers)

        self.head_logits = nn.Linear(in_dim, self.branching)
        self.head_v = nn.Linear(in_dim, self.branching * self.dv)
        self.head_phat = nn.Linear(in_dim, self.branching * self.I)

        self.to(device=device, dtype=dtype)

    def forward(
        self, x: Tensor, phat: Tensor, t: Tensor | float | int
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 2 or x.shape[1] != self.dx:
            raise ValueError(f"x must have shape (B,{self.dx}), got {tuple(x.shape)}")
        if phat.ndim != 2 or phat.shape[1] != self.I:
            raise ValueError(f"phat must have shape (B,{self.I}), got {tuple(phat.shape)}")
        if x.shape[0] != phat.shape[0]:
            raise ValueError("x and phat batch dimensions must match.")

        B = x.shape[0]
        t_feat = _normalize_t(t, B=B, device=x.device, dtype=x.dtype)
        inp = torch.cat([x, phat, t_feat], dim=1)
        h = self.trunk(inp)

        logits = self.head_logits(h)                              # (B,A)
        v = self.head_v(h).view(B, self.branching, self.dv)      # (B,A,dv)
        if self.action_scale_v is not None:
            v = torch.tanh(v) * self.action_scale_v
        phat_raw = self.head_phat(h).view(B, self.branching, self.I)  # (B,A,I)
        return logits, v, phat_raw


@dataclass
class DualNoSplitP1NetConfig:
    """
    Config for no-split dual P1 network.
    """

    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "silu"
    action_scale_u: float | None = None


class DualNoSplitP1Net(DualSplitP1BRNet):
    """
    No-split P1 policy network.

    This is the same architecture as `DualSplitP1BRNet`: u = pi_1(x, phat, t).
    """

    def __init__(
        self,
        dx: int,
        I: int,
        du: int,
        cfg: DualNoSplitP1NetConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        split_cfg = DualSplitP1BRNetConfig() if cfg is None else DualSplitP1BRNetConfig(
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            action_scale_u=cfg.action_scale_u,
        )
        super().__init__(dx=dx, I=I, du=du, cfg=split_cfg, dtype=dtype, device=device)


@dataclass
class DualNoSplitP2NetConfig:
    """
    Config for no-split dual P2 network.
    """

    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "silu"
    action_scale_v: float | None = None


class DualNoSplitP2Net(nn.Module):
    """
    No-split P2 policy network conditioned on (x, phat, t_norm).

    Output
    ------
    v: (B, dv)

    If `cfg.action_scale_v` is provided, output is bounded as
      v = tanh(v_raw) * action_scale_v.
    """

    def __init__(
        self,
        dx: int,
        I: int,
        dv: int,
        cfg: DualNoSplitP2NetConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = DualNoSplitP2NetConfig()
        if int(dx) <= 0:
            raise ValueError(f"dx must be positive, got {dx}.")
        if int(I) <= 0:
            raise ValueError(f"I must be positive, got {I}.")
        if int(dv) <= 0:
            raise ValueError(f"dv must be positive, got {dv}.")
        if any(int(h) <= 0 for h in cfg.hidden_dims):
            raise ValueError(f"All hidden dims must be positive, got {cfg.hidden_dims}.")

        self.dx = int(dx)
        self.I = int(I)
        self.dv = int(dv)
        self.cfg = cfg
        self.action_scale_v = (
            float(cfg.action_scale_v) if cfg.action_scale_v is not None else None
        )
        if self.action_scale_v is not None and self.action_scale_v <= 0.0:
            raise ValueError(f"action_scale_v must be positive, got {self.action_scale_v}.")

        in_dim = self.dx + self.I + 1
        layers: list[nn.Module] = []
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(_make_activation(cfg.activation))
            in_dim = int(h)
        self.trunk = nn.Sequential(*layers)
        self.head_v = nn.Linear(in_dim, self.dv)

        self.to(device=device, dtype=dtype)

    def forward(self, x: Tensor, phat: Tensor, t: Tensor | float | int) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.dx:
            raise ValueError(f"x must have shape (B,{self.dx}), got {tuple(x.shape)}")
        if phat.ndim != 2 or phat.shape[1] != self.I:
            raise ValueError(f"phat must have shape (B,{self.I}), got {tuple(phat.shape)}")
        if x.shape[0] != phat.shape[0]:
            raise ValueError("x and phat batch dimensions must match.")

        B = x.shape[0]
        t_feat = _normalize_t(t, B=B, device=x.device, dtype=x.dtype)
        inp = torch.cat([x, phat, t_feat], dim=1)
        h = self.trunk(inp)
        v = self.head_v(h)
        if self.action_scale_v is not None:
            v = torch.tanh(v) * self.action_scale_v
        return v
