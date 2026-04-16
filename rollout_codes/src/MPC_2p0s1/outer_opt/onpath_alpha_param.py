from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ..core.types import Tensor


@dataclass
class OnPathAlphaConfig:
    """
    Configuration for an on-path alpha policy network.

    The network maps current context to per-type signal logits:
        f(x_k, p_k, t_k) -> logits in R^{I x I}

    where:
      - x_k is the current physical state,
      - p_k is the current public belief,
      - t_k is normalized time in [0, 1].
    """

    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "silu"
    dropout: float = 0.0


def _make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"Unsupported activation '{name}'. Expected one of: relu, gelu, silu, tanh."
    )


class OnPathAlphaParam(nn.Module):
    """
    Compact on-path alpha parameterization with no full-tree allocation.

    This model avoids tensors of shape (K, I**K, I, I). Instead, it predicts
    only the local alpha matrix for the current step:
        alpha_k = softmax(logits_k, dim=-1),   alpha_k shape = (I, I).
    """

    def __init__(
        self,
        dx: int,
        I: int,
        cfg: Optional[OnPathAlphaConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = OnPathAlphaConfig()
        if dx <= 0:
            raise ValueError(f"dx must be positive, got {dx}.")
        if I <= 0:
            raise ValueError(f"I must be positive, got {I}.")

        self.dx = int(dx)
        self.I = int(I)
        self.cfg = cfg

        hidden_dims = tuple(int(h) for h in cfg.hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one positive integer.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"All hidden dims must be positive; got {hidden_dims}.")

        input_dim = self.dx + self.I + 1
        output_dim = self.I * self.I

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_make_activation(cfg.activation))
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(p=float(cfg.dropout)))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        device_t = torch.device(device) if isinstance(device, str) else device
        self.to(device=device_t, dtype=dtype)

    def _normalize_inputs(self, x: Tensor, belief: Tensor, t: Tensor | float) -> Tuple[Tensor, Tensor, Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if belief.ndim == 1:
            belief = belief.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"x must have shape (dx,) or (B, dx); got {tuple(x.shape)}")
        if belief.ndim != 2:
            raise ValueError(
                f"belief must have shape (I,) or (B, I); got {tuple(belief.shape)}"
            )
        if x.shape[-1] != self.dx:
            raise ValueError(f"x last dim must be dx={self.dx}, got {x.shape[-1]}")
        if belief.shape[-1] != self.I:
            raise ValueError(f"belief last dim must be I={self.I}, got {belief.shape[-1]}")
        if x.shape[0] != belief.shape[0]:
            raise ValueError(
                f"x and belief batch sizes must match; got {x.shape[0]} and {belief.shape[0]}"
            )

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype)

        if t.ndim == 0:
            t = t.expand(x.shape[0]).unsqueeze(-1)
        elif t.ndim == 1:
            if t.shape[0] == 1 and x.shape[0] > 1:
                t = t.expand(x.shape[0]).unsqueeze(-1)
            elif t.shape[0] == x.shape[0]:
                t = t.unsqueeze(-1)
            else:
                raise ValueError(
                    f"t batch size must be 1 or {x.shape[0]}, got {t.shape[0]}"
                )
        elif t.ndim == 2:
            if t.shape != (x.shape[0], 1):
                raise ValueError(
                    f"t must have shape ({x.shape[0]}, 1), got {tuple(t.shape)}"
                )
        else:
            raise ValueError(f"t must be scalar, (B,), or (B,1); got shape {tuple(t.shape)}")

        return x, belief, t

    def forward_logits(self, x: Tensor, belief: Tensor, t: Tensor | float) -> Tensor:
        """
        Predict local logits with shape (B, I, I).
        """
        x, belief, t = self._normalize_inputs(x=x, belief=belief, t=t)
        h = torch.cat([x, belief, t], dim=-1)
        logits = self.net(h)
        return logits.view(x.shape[0], self.I, self.I)

    def forward(self, x: Tensor, belief: Tensor, t: Tensor | float) -> Tensor:
        """
        Predict local alpha probabilities with shape (B, I, I).
        """
        logits = self.forward_logits(x=x, belief=belief, t=t)
        return torch.softmax(logits, dim=-1)

    def single_alpha(self, x: Tensor, belief: Tensor, t: Tensor | float) -> Tensor:
        """
        Convenience helper returning alpha with shape (I, I) for one context.
        """
        alpha_b = self.forward(x=x, belief=belief, t=t)
        return alpha_b[0]


def posterior_from_signal(
    belief: Tensor,
    alpha: Tensor,
    action: int,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """
    Bayes update for one public signal.

    Parameters
    ----------
    belief:
        Prior belief p with shape (I,).
    alpha:
        Local signal probabilities with shape (I, I), where alpha[i, a] is
        P(signal=a | type=i).
    action:
        Observed signal index a.
    eps:
        Stability floor for normalization.

    Returns
    -------
    (posterior, signal_mass):
        posterior has shape (I,), signal_mass is scalar tensor.
    """
    if belief.ndim != 1:
        raise ValueError(f"belief must have shape (I,), got {tuple(belief.shape)}")
    if alpha.ndim != 2:
        raise ValueError(f"alpha must have shape (I, I), got {tuple(alpha.shape)}")
    I = belief.shape[0]
    if alpha.shape != (I, I):
        raise ValueError(f"alpha shape must be ({I}, {I}), got {tuple(alpha.shape)}")
    if not (0 <= int(action) < I):
        raise ValueError(f"action must be in [0, {I - 1}], got {action}")

    numer = belief * alpha[:, int(action)]
    mass = numer.sum()
    denom = mass.clamp_min(eps)
    posterior = numer / denom

    if bool((mass <= eps).item()):
        posterior = belief / belief.sum().clamp_min(eps)

    return posterior, mass
