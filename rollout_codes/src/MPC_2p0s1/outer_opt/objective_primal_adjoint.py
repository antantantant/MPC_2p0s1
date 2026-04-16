from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..core.action_spaces import BoxActionSpace
from ..core.types import Tensor
from ..games.base_lq_game import BaseLQGame
from ..tree.averaged_costs import AveragedCostData, compute_averaged_costs
from ..tree.belief_tree import BeliefTree, build_belief_tree
from ..tree.indexing import FullIaryTreeIndexer
from ..tree.riccati_tree import RiccatiSolution, riccati_backward
from ..tree.riccati_tree_adjoint import riccati_root_value_adjoint
from .alpha_param import AlphaParam


def primal_objective_adjoint(
    game: BaseLQGame,
    alpha_module: AlphaParam,
    indexer: FullIaryTreeIndexer,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    return_details: bool = False,
    include_rollout_riccati: bool = True,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """
    Primal objective using the experimental manual-adjoint Riccati root-value path.

    This is an additive alternative to `objective_primal.primal_objective`.
    """
    del action_space  # kept for API compatibility; unconstrained Riccati path.

    device = game.device_resolved
    dtype = game.dtype

    if x0 is None:
        x0 = game.default_initial_state()
    if p0 is None:
        p0 = game.default_prior()

    x0 = x0.to(device=device, dtype=dtype)
    p0 = p0.to(device=device, dtype=dtype)

    alpha: Tensor = alpha_module()  # (K, max_nodes, I, I)

    belief_tree: BeliefTree = build_belief_tree(alpha=alpha, p0=p0, indexer=indexer)
    avg_costs: AveragedCostData = compute_averaged_costs(game=game, belief_tree=belief_tree)
    need_rollout_riccati = bool(return_details and include_rollout_riccati)
    use_rollout_solver_only = bool(need_rollout_riccati and (not torch.is_grad_enabled()))

    riccati_sol: RiccatiSolution | None = None
    if use_rollout_solver_only:
        # In no-grad rollout/eval paths, avoid double-solving by using the
        # classic Riccati result for both loss and rollout details.
        riccati_sol = riccati_backward(
            game=game,
            belief_tree=belief_tree,
            avg_costs=avg_costs,
        )
        loss: Tensor = riccati_sol.value_at_root(x0)
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
        # `return_details=True` is used by rollout/evaluation code that expects
        # explicit gains, so attach the classic Riccati solution here.
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
