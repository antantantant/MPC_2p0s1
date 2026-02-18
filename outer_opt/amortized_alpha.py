from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import warnings

import torch
from torch import nn

from ..core.action_spaces import BoxActionSpace
from ..core.types import Tensor
from ..games.base_lq_game import BaseLQGame
from ..tree.averaged_costs import compute_averaged_costs
from ..tree.belief_tree import BeliefTree, build_belief_tree
from ..tree.indexing import FullIaryTreeIndexer
from ..tree.riccati_tree import RiccatiSolution, riccati_backward

_VMAP_DISABLED = False


@dataclass
class AmortizedAlphaConfig:
    """
    Configuration for the amortized alpha network f(x0, p0) -> logits.

    Attributes
    ----------
    hidden_dims:
        Width of hidden layers in the MLP encoder/decoder.
    activation:
        Nonlinearity used after each hidden layer ("relu", "gelu", "silu", "tanh").
    dropout:
        Dropout probability applied after hidden activations.
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


class AmortizedAlphaParam(nn.Module):
    """
    Context-conditioned alpha/logit parameterization.

    Given an initial state x0 and prior p0, predicts full tree logits of shape:
        (K, max_nodes_per_depth, I, I)
    and alpha via softmax over the last axis.
    """

    def __init__(
        self,
        indexer: FullIaryTreeIndexer,
        dx: int,
        I: int,
        cfg: Optional[AmortizedAlphaConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = AmortizedAlphaConfig()
        if dx <= 0:
            raise ValueError(f"dx must be positive, got {dx}.")
        if I <= 0:
            raise ValueError(f"I must be positive, got {I}.")

        self.indexer = indexer
        self.dx = int(dx)
        self.I = int(I)
        self.cfg = cfg

        self._input_dim = self.dx + self.I
        self._output_shape = (
            self.indexer.K,
            self.indexer.max_nodes_per_depth,
            self.I,
            self.I,
        )
        self._output_dim = int(torch.tensor(self._output_shape).prod().item())

        hidden_dims = tuple(int(h) for h in cfg.hidden_dims)
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(
                f"All hidden layer widths must be positive, got {cfg.hidden_dims}."
            )

        layers: list[nn.Module] = []
        in_dim = self._input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_make_activation(cfg.activation))
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(p=float(cfg.dropout)))
            in_dim = h
        layers.append(nn.Linear(in_dim, self._output_dim))
        self.net = nn.Sequential(*layers)

        device_t = torch.device(device) if isinstance(device, str) else device
        self.to(device=device_t, dtype=dtype)

    @property
    def output_shape(self) -> Tuple[int, int, int, int]:
        return self._output_shape

    def _normalize_inputs(self, x0: Tensor, p0: Tensor) -> Tuple[Tensor, Tensor]:
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)
        if p0.ndim == 1:
            p0 = p0.unsqueeze(0)
        if x0.ndim != 2:
            raise ValueError(f"x0 must have shape (dx,) or (B, dx); got {tuple(x0.shape)}")
        if p0.ndim != 2:
            raise ValueError(f"p0 must have shape (I,) or (B, I); got {tuple(p0.shape)}")
        if x0.shape[-1] != self.dx:
            raise ValueError(f"x0 last dim must be dx={self.dx}, got {x0.shape[-1]}")
        if p0.shape[-1] != self.I:
            raise ValueError(f"p0 last dim must be I={self.I}, got {p0.shape[-1]}")
        if x0.shape[0] != p0.shape[0]:
            raise ValueError(
                f"x0 and p0 batch sizes must match; got {x0.shape[0]} and {p0.shape[0]}"
            )
        return x0, p0

    def forward_logits(self, x0: Tensor, p0: Tensor) -> Tensor:
        """
        Predict unnormalized logits with shape (B, K, max_nodes, I, I).
        """
        x0, p0 = self._normalize_inputs(x0=x0, p0=p0)
        h = torch.cat([x0, p0], dim=-1)
        logits_flat = self.net(h)
        return logits_flat.view(x0.shape[0], *self._output_shape)

    def forward(self, x0: Tensor, p0: Tensor) -> Tensor:
        """
        Predict alpha probabilities with shape (B, K, max_nodes, I, I).
        """
        logits = self.forward_logits(x0=x0, p0=p0)
        return torch.softmax(logits, dim=-1)

    def single_alpha(self, x0: Tensor, p0: Tensor) -> Tensor:
        """
        Convenience helper returning alpha with shape (K, max_nodes, I, I).
        """
        alpha_batch = self.forward(x0=x0, p0=p0)
        return alpha_batch[0]


def primal_objective_from_alpha(
    game: BaseLQGame,
    alpha: Tensor,
    indexer: FullIaryTreeIndexer,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Evaluate primal objective for a provided alpha tensor.
    """
    device = game.device_resolved
    dtype = game.dtype

    if x0 is None:
        x0 = game.default_initial_state()
    if p0 is None:
        p0 = game.default_prior()

    x0 = x0.to(device=device, dtype=dtype)
    p0 = p0.to(device=device, dtype=dtype)
    alpha = alpha.to(device=device, dtype=dtype)

    belief_tree: BeliefTree = build_belief_tree(alpha=alpha, p0=p0, indexer=indexer)
    avg_costs = compute_averaged_costs(game=game, belief_tree=belief_tree)
    riccati_sol: RiccatiSolution = riccati_backward(
        game=game,
        belief_tree=belief_tree,
        avg_costs=avg_costs,
        action_space=action_space,
    )
    loss = riccati_sol.value_at_root(x0)

    if not return_details:
        return loss

    details: Dict[str, object] = {
        "alpha": alpha,
        "belief_tree": belief_tree,
        "avg_costs": avg_costs,
        "riccati_solution": riccati_sol,
    }
    return loss, details


def amortized_batch_objective(
    game: BaseLQGame,
    alpha_model: AmortizedAlphaParam,
    indexer: FullIaryTreeIndexer,
    x0_batch: Tensor,
    p0_batch: Tensor,
    action_space: Optional[BoxActionSpace] = None,
    reduction: str = "mean",
    vectorize: bool = True,
    vmap_chunk_size: Optional[int] = None,
    strict_vectorize: bool = False,
) -> Tensor:
    """
    Evaluate the amortized objective over a mini-batch of (x0, p0) contexts.
    """
    alpha_batch = alpha_model(x0=x0_batch, p0=p0_batch)
    batch_size = alpha_batch.shape[0]

    def _single_loss(alpha: Tensor, x0: Tensor, p0: Tensor) -> Tensor:
        return primal_objective_from_alpha(
            game=game,
            alpha=alpha,
            indexer=indexer,
            x0=x0,
            p0=p0,
            action_space=action_space,
            return_details=False,
        )

    def _loop_eval() -> Tensor:
        losses: list[Tensor] = []
        for b in range(batch_size):
            losses.append(_single_loss(alpha_batch[b], x0_batch[b], p0_batch[b]))
        return torch.stack(losses, dim=0)

    global _VMAP_DISABLED

    if vectorize and not _VMAP_DISABLED:
        try:
            from torch.func import vmap

            if vmap_chunk_size is None or int(vmap_chunk_size) <= 0:
                vmap_chunk_size = batch_size

            chunk = int(vmap_chunk_size)
            loss_chunks: list[Tensor] = []
            vmapped_single = vmap(_single_loss, in_dims=(0, 0, 0))
            for start in range(0, batch_size, chunk):
                end = min(start + chunk, batch_size)
                loss_chunks.append(
                    vmapped_single(
                        alpha_batch[start:end],
                        x0_batch[start:end],
                        p0_batch[start:end],
                    )
                )
            loss_vec = torch.cat(loss_chunks, dim=0) if len(loss_chunks) > 1 else loss_chunks[0]
        except Exception as exc:
            if strict_vectorize:
                raise
            _VMAP_DISABLED = True
            warnings.warn(
                "amortized_batch_objective: vectorized path failed; falling back to "
                "looped evaluation for subsequent calls. "
                f"Error: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            loss_vec = _loop_eval()
    else:
        loss_vec = _loop_eval()

    key = reduction.lower()
    if key == "mean":
        return loss_vec.mean()
    if key == "sum":
        return loss_vec.sum()
    if key == "none":
        return loss_vec
    raise ValueError(f"Unsupported reduction '{reduction}'. Expected mean, sum, or none.")
