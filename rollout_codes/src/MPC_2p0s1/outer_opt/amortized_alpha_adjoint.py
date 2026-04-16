from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..core.action_spaces import BoxActionSpace
from ..core.types import Tensor
from ..games.base_lq_game import BaseLQGame
from ..tree.averaged_costs import compute_averaged_costs
from ..tree.belief_tree import BeliefTree, build_belief_tree
from ..tree.indexing import FullIaryTreeIndexer
from ..tree.riccati_tree import RiccatiSolution, riccati_backward
from ..tree.riccati_tree_adjoint import riccati_root_value_adjoint
from .amortized_alpha import AmortizedAlphaParam


def primal_objective_from_alpha_adjoint(
    game: BaseLQGame,
    alpha: Tensor,
    indexer: FullIaryTreeIndexer,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    return_details: bool = False,
    include_rollout_riccati: bool = True,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Evaluate primal objective for a provided alpha tensor using manual adjoint.
    """
    del action_space  # kept for API compatibility

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
    need_rollout_riccati = bool(return_details and include_rollout_riccati)
    use_rollout_solver_only = bool(need_rollout_riccati and (not torch.is_grad_enabled()))

    riccati_sol: RiccatiSolution | None = None
    if use_rollout_solver_only:
        # In no-grad eval/rollout codepaths, avoid doing both adjoint root-value
        # and classic Riccati solves. Reuse classic solve for loss+details.
        riccati_sol = riccati_backward(
            game=game,
            belief_tree=belief_tree,
            avg_costs=avg_costs,
        )
        loss = riccati_sol.value_at_root(x0)
    else:
        loss = riccati_root_value_adjoint(
            game=game,
            belief_tree=belief_tree,
            avg_costs=avg_costs,
            x0=x0,
        )

    if not return_details:
        return loss

    if need_rollout_riccati and riccati_sol is None:
        # Keep rollout-policy extraction API-compatible with the original objective
        # while preserving adjoint differentiation for the loss path.
        with torch.no_grad():
            riccati_sol = riccati_backward(
                game=game,
                belief_tree=belief_tree,
                avg_costs=avg_costs,
            )

    details: Dict[str, object] = {
        "alpha": alpha,
        "belief_tree": belief_tree,
        "avg_costs": avg_costs,
        "solver": "riccati_root_value_adjoint",
    }
    if riccati_sol is not None:
        details["riccati_solution"] = riccati_sol
    return loss, details


def amortized_batch_objective_adjoint(
    game: BaseLQGame,
    alpha_model: AmortizedAlphaParam,
    indexer: FullIaryTreeIndexer,
    x0_batch: Tensor,
    p0_batch: Tensor,
    action_space: Optional[BoxActionSpace] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Amortized objective over a mini-batch using manual-adjoint root solver.

    Uses a looped evaluation path intentionally for stability while validating
    the adjoint implementation.
    """
    alpha_batch = alpha_model(x0=x0_batch, p0=p0_batch)
    batch_size = alpha_batch.shape[0]

    losses: list[Tensor] = []
    for b in range(batch_size):
        losses.append(
            primal_objective_from_alpha_adjoint(
                game=game,
                alpha=alpha_batch[b],
                indexer=indexer,
                x0=x0_batch[b],
                p0=p0_batch[b],
                action_space=action_space,
                return_details=False,
            )
        )
    loss_vec = torch.stack(losses, dim=0)

    key = reduction.lower()
    if key == "mean":
        return loss_vec.mean()
    if key == "sum":
        return loss_vec.sum()
    if key == "none":
        return loss_vec
    raise ValueError(f"Unsupported reduction '{reduction}'. Expected mean, sum, or none.")
