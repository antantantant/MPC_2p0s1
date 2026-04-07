"""Outer objective for training α in the nonlinear-SQP tree game.

Mirrors ``nl_sqp/objective_primal_sqp.py``.

    α → belief tree → cost data → SQP layer (unrolled) → expected cost

The returned scalar is differentiable w.r.t. α logits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..game.quadrotor_game import Hexner3DQuadrotorGame
from ..tree.indexing import FullIaryTreeIndexer
from ..tree.signaling import AlphaParam
from ..solvers.action_spaces import BoxActionSpace
from ..solvers.sqp_tree import SQPTreeResult, sqp_tree_layer


def _expected_cost_from_sqp_result(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    sqp_res: SQPTreeResult,
) -> Tensor:
    """Expected cost of the nominal tree trajectories (scalar, differentiable)."""

    K = indexer.K
    I = indexer.I
    total = torch.zeros((), device=game.device, dtype=game.dtype)

    for k in range(K):
        x_k = sqp_res.x_nodes[k]
        Nk = x_k.shape[0]

        x_edge = x_k.unsqueeze(1).expand(Nk, I, game.dx).reshape(Nk * I, game.dx)
        u_edge = sqp_res.u_edges[k].reshape(Nk * I, game.du)
        v_edge = sqp_res.v_edges[k].reshape(Nk * I, game.dv)

        Q = sqp_res.cost_data.Q[k].reshape(Nk * I, game.dx, game.dx)
        q = sqp_res.cost_data.q[k].reshape(Nk * I, game.dx)
        c = sqp_res.cost_data.c[k].reshape(Nk * I)
        R = sqp_res.cost_data.R[k].reshape(Nk * I, game.du, game.du)
        S = sqp_res.cost_data.S[k].reshape(Nk * I, game.dv, game.dv)

        w = sqp_res.belief_tree.lambda_node[k + 1]

        Qx = torch.bmm(Q, x_edge.unsqueeze(-1)).squeeze(-1)
        Ru = torch.bmm(R, u_edge.unsqueeze(-1)).squeeze(-1)
        Sv = torch.bmm(S, v_edge.unsqueeze(-1)).squeeze(-1)

        stage = (
            0.5 * (x_edge * Qx).sum(dim=-1)
            + (q * x_edge).sum(dim=-1) + c
            + 0.5 * (u_edge * Ru).sum(dim=-1)
            - 0.5 * (v_edge * Sv).sum(dim=-1)
        )
        stage = game.cfg.tau * stage
        total = total + (w * stage).sum()

    xK = sqp_res.x_nodes[K]
    wK = sqp_res.belief_tree.lambda_node[K]
    P = sqp_res.cost_data.P_leaf
    r = sqp_res.cost_data.r_leaf
    c = sqp_res.cost_data.c_leaf

    Px = torch.bmm(P, xK.unsqueeze(-1)).squeeze(-1)
    terminal = 0.5 * (xK * Px).sum(dim=-1) + (r * xK).sum(dim=-1) + c
    total = total + (wK * terminal).sum()

    return total


def primal_objective_sqp(
    *,
    game: Hexner3DQuadrotorGame,
    alpha_module: AlphaParam,
    indexer: FullIaryTreeIndexer,
    x0: Optional[Tensor] = None,
    p0: Optional[Tensor] = None,
    action_space: Optional[BoxActionSpace] = None,
    num_sqp_iters: int = 3,
    sqp_step_size: float = 1.0,
    riccati_reg: float = 1e-3,
    max_riccati_reg_tries: int = 5,
    riccati_reg_factor: float = 10.0,
    collect_sqp_diagnostics: bool = False,
    sqp_tol_u: float = 1e-3,
    sqp_tol_v: float = 1e-3,
    sqp_tol_rel_cost: float = 1e-4,
    sqp_verbose: bool = False,
    sqp_early_stop: bool = False,
    sqp_line_search: bool = True,
    ls_alpha_min: float = 0.05,
    ls_backtrack: float = 0.5,
    ls_max_steps: int = 6,
    line_search_accept_worse: Optional[bool] = None,
    u_init: Optional[List[Tensor]] = None,
    v_init: Optional[List[Tensor]] = None,
    return_details: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """Compute the nonlinear-SQP primal objective (differentiable).

    Returns a scalar loss differentiable w.r.t. ``alpha_module.logits``.
    """

    if x0 is None:
        x0 = game.default_initial_state()
    if p0 is None:
        p0 = game.default_prior()

    x0 = x0.to(device=game.device, dtype=game.dtype)
    p0 = p0.to(device=game.device, dtype=game.dtype)

    alpha = alpha_module()

    sqp_res = sqp_tree_layer(
        game=game,
        indexer=indexer,
        alpha=alpha,
        x0=x0,
        p0=p0,
        num_sqp_iters=num_sqp_iters,
        step_size=sqp_step_size,
        action_space=action_space,
        riccati_reg=riccati_reg,
        max_riccati_reg_tries=max_riccati_reg_tries,
        riccati_reg_factor=riccati_reg_factor,
        collect_diagnostics=collect_sqp_diagnostics,
        sqp_tol_u=sqp_tol_u,
        sqp_tol_v=sqp_tol_v,
        sqp_tol_rel_cost=sqp_tol_rel_cost,
        verbose=sqp_verbose,
        early_stop=sqp_early_stop,
        line_search=sqp_line_search,
        ls_alpha_min=ls_alpha_min,
        ls_backtrack=ls_backtrack,
        ls_max_steps=ls_max_steps,
        line_search_accept_worse=line_search_accept_worse,
        u_init=u_init,
        v_init=v_init,
    )

    loss = _expected_cost_from_sqp_result(
        game=game, indexer=indexer, sqp_res=sqp_res
    )

    if not return_details:
        return loss

    details: Dict[str, object] = {
        "alpha": alpha,
        "sqp_result": sqp_res,
    }
    return loss, details
