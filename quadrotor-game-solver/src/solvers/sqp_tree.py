"""Differentiable SQP layer on a public I-ary tree (vectorized).

This is the nonlinear extension of the LQ Riccati layer:

    α → belief tree (Bayes) → belief-averaged costs
      → SQP iterations:
            (1) linearise nonlinear dynamics on every edge
            (2) solve time-varying tree LQ saddle game via Riccati
            (3) forward rollout under feedback to update the nominal
      → return tree solution

The whole pipeline is differentiable w.r.t. α logits (unrolled SQP).

Mirrors ``nl_sqp/sqp_tree.py`` but uses the quadrotor game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from ..game.quadrotor_game import Hexner3DQuadrotorGame
from ..tree.indexing import FullIaryTreeIndexer
from ..tree.belief_tree import BeliefTree, build_belief_tree_vectorized
from ..objectives.cost_tree import QuadraticCostData, compute_quadratic_cost_data
from ..solvers.riccati import RiccatiSolution, riccati_backward_time_varying
from ..solvers.action_spaces import BoxActionSpace


# ── Diagnostics ─────────────────────────────────────────────────────────────

@dataclass
class SQPDiagnostics:
    cost_hist: List[float]
    rel_cost_impr_hist: List[float]
    du_max_hist: List[float]
    dv_max_hist: List[float]
    root_value_pred_hist: List[float]
    accepted_step_hist: List[float]
    used_prev_iterate_hist: List[bool]
    riccati_retry_hist: List[bool]
    converged: bool
    converged_iter: int
    nan_or_inf_encountered: bool


@dataclass
class SQPTreeResult:
    alpha: Tensor
    belief_tree: BeliefTree
    cost_data: QuadraticCostData
    riccati_sol: RiccatiSolution
    x_nodes: List[Tensor]    # len K+1
    u_edges: List[Tensor]    # len K
    v_edges: List[Tensor]    # len K
    diagnostics: Optional[SQPDiagnostics] = None


def _riccati_solution_is_numerically_safe(
    sol: RiccatiSolution,
    *,
    max_abs: float = 1e8,
) -> bool:
    """Basic numerical sanity checks for Riccati gains/offsets."""
    tensors = (
        list(sol.K_u)
        + list(sol.K_v)
        + list(sol.kappa_u)
        + list(sol.kappa_v)
    )
    for t in tensors:
        if not torch.isfinite(t).all():
            return False
        if float(t.abs().max().item()) > max_abs:
            return False
    return True


# ── Expected cost from nominal trajectories ────────────────────────────────

def _expected_cost_from_nominal_tree(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    belief_tree: BeliefTree,
    cost_data: QuadraticCostData,
    x_nodes: List[Tensor],
    u_edges: List[Tensor],
    v_edges: List[Tensor],
) -> Tensor:
    """Expected cost of the nominal tree trajectories (for backprop)."""

    K = indexer.K
    I = indexer.I
    total = torch.zeros((), device=game.device, dtype=game.dtype)

    for k in range(K):
        x_k = x_nodes[k]
        Nk = x_k.shape[0]

        x_edge = x_k.unsqueeze(1).expand(Nk, I, game.dx).reshape(Nk * I, game.dx)
        u_edge = u_edges[k].reshape(Nk * I, game.du)
        v_edge = v_edges[k].reshape(Nk * I, game.dv)

        Q = cost_data.Q[k].reshape(Nk * I, game.dx, game.dx)
        q = cost_data.q[k].reshape(Nk * I, game.dx)
        c = cost_data.c[k].reshape(Nk * I)
        R = cost_data.R[k].reshape(Nk * I, game.du, game.du)
        S = cost_data.S[k].reshape(Nk * I, game.dv, game.dv)

        w = belief_tree.lambda_node[k + 1]

        Qx = torch.bmm(Q, x_edge.unsqueeze(-1)).squeeze(-1)
        Ru = torch.bmm(R, u_edge.unsqueeze(-1)).squeeze(-1)
        Sv = torch.bmm(S, v_edge.unsqueeze(-1)).squeeze(-1)

        stage = (
            0.5 * (x_edge * Qx).sum(dim=-1)
            + (q * x_edge).sum(dim=-1)
            + c
            + 0.5 * (u_edge * Ru).sum(dim=-1)
            - 0.5 * (v_edge * Sv).sum(dim=-1)
        )
        stage = game.cfg.tau * stage

        total = total + (w * stage).sum()

    # Terminal
    xK = x_nodes[K]
    wK = belief_tree.lambda_node[K]
    P = cost_data.P_leaf
    r = cost_data.r_leaf
    c = cost_data.c_leaf

    Px = torch.bmm(P, xK.unsqueeze(-1)).squeeze(-1)
    terminal = 0.5 * (xK * Px).sum(dim=-1) + (r * xK).sum(dim=-1) + c

    total = total + (wK * terminal).sum()
    return total


# ── Forward rollout under feedback ──────────────────────────────────────────
# @torch.compile()
def _tree_forward_rollout(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    x0: Tensor,
    riccati_sol: RiccatiSolution,
    belief_tree: BeliefTree,
    u_prev: Optional[List[Tensor]] = None,
    v_prev: Optional[List[Tensor]] = None,
    step_size: float = 1.0,
    action_space: Optional[BoxActionSpace] = None,
) -> tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Forward-propagate states on the full tree under feedback prototypes."""

    K = indexer.K
    I = indexer.I
    dx = game.dx
    du = game.du
    dv = game.dv

    x_nodes: List[Tensor] = [None] * (K + 1)   # type: ignore
    u_edges: List[Tensor] = [None] * K          # type: ignore
    v_edges: List[Tensor] = [None] * K          # type: ignore

    x_nodes[0] = x0.view(1, dx)

    for k in range(K):
        x_k = x_nodes[k]
        Nk = x_k.shape[0]

        # Aggregate feedforward using prior edge probs
        lam = belief_tree.lambda_edge[k]
        kappa_u_all = riccati_sol.kappa_u[k]
        kappa_v_all = riccati_sol.kappa_v[k]
        kappa_u_agg = torch.einsum("na, nad -> nd", lam, kappa_u_all)
        kappa_v_agg = torch.einsum("na, nae -> ne", lam, kappa_v_all)

        # Action-specific feedback
        Ku = riccati_sol.K_u[k]
        Kv = riccati_sol.K_v[k]

        u_lin = torch.einsum("naud, nd -> nau", Ku, x_k)
        v_lin = torch.einsum("naed, nd -> nae", Kv, x_k)

        u_opt = u_lin + kappa_u_agg.unsqueeze(1)
        v_opt = v_lin + kappa_v_agg.unsqueeze(1)

        if action_space is not None:
            u_opt = action_space.clip_u(u_opt)
            v_opt = action_space.clip_v(v_opt)

        if u_prev is None:
            u_new = u_opt
        else:
            u_new = (1.0 - step_size) * u_prev[k] + step_size * u_opt

        if v_prev is None:
            v_new = v_opt
        else:
            v_new = (1.0 - step_size) * v_prev[k] + step_size * v_opt

        if action_space is not None:
            u_new = action_space.clip_u(u_new)
            v_new = action_space.clip_v(v_new)

        u_edges[k] = u_new
        v_edges[k] = v_new

        # Propagate to children (parent-major)
        x_parent = x_k.unsqueeze(1).expand(Nk, I, dx).reshape(Nk * I, dx)
        u_flat = u_new.reshape(Nk * I, du)
        v_flat = v_new.reshape(Nk * I, dv)
        x_next = game.step(x_parent, u_flat, v_flat)
        x_nodes[k + 1] = x_next

    return x_nodes, u_edges, v_edges


# ── Main SQP entry point ───────────────────────────────────────────────────
# @torch.compile()
def sqp_tree_layer(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    alpha: Tensor,
    x0: Tensor,
    p0: Tensor,
    num_sqp_iters: int = 3,
    step_size: float = 1.0,
    action_space: Optional[BoxActionSpace] = None,
    riccati_reg: float = 1e-3,
    max_riccati_reg_tries: int = 5,
    riccati_reg_factor: float = 10.0,
    collect_diagnostics: bool = False,
    sqp_tol_u: float = 1e-3,
    sqp_tol_v: float = 1e-3,
    sqp_tol_rel_cost: float = 1e-4,
    verbose: bool = False,
    early_stop: bool = False,
    line_search: bool = True,
    ls_alpha_min: float = 0.05,
    ls_backtrack: float = 0.5,
    ls_max_steps: int = 6,
    line_search_accept_worse: Optional[bool] = None,
    u_init: Optional[List[Tensor]] = None,
    v_init: Optional[List[Tensor]] = None,
) -> SQPTreeResult:
    """Run SQP iterations and return the final tree solution.

    Parameters
    ----------
    game : Hexner3DQuadrotorGame
    indexer : FullIaryTreeIndexer
    alpha : (K, max_nodes, I, I)   signaling matrix
    x0 : (dx,)
    p0 : (I,)
    num_sqp_iters : int
        Number of SQP iterations (linearise → Riccati → rollout).
    step_size : float
        Initial/max damping factor for control updates (0 = keep old, 1 = full Newton).
    riccati_reg : float
        Diagonal regularisation for the saddle Hessian.
    line_search : bool
        If True, use backtracking line search to ensure cost improvement.
    ls_alpha_min : float
        Minimum step size for line search before giving up.
    ls_backtrack : float
        Factor to reduce step size by on each line search backtrack.
    ls_max_steps : int
        Maximum number of backtracking steps.
    line_search_accept_worse : bool
        If False, keep the previous iterate when no tested line-search step
        improves nominal cost.
    u_init : List[Tensor] | None
        Initial guess for P1 controls (warm-start from previous solve).
        If None, initializes to hover controls.
    v_init : List[Tensor] | None
        Initial guess for P2 controls (warm-start from previous solve).
        If None, initializes to hover controls.
    """

    if num_sqp_iters < 1:
        raise ValueError("num_sqp_iters must be >= 1")
    if line_search_accept_worse is None:
        line_search_accept_worse = game.cfg.line_search_accept_worse

    # 1) Belief tree
    belief_tree = build_belief_tree_vectorized(
        alpha=alpha, p0=p0, indexer=indexer
    )

    # 2) Initial belief-averaged cost data (will be re-computed in SQP loop)
    # For faithful SQP, costs should be quadratized at each iteration around
    # the current nominal. For quadratic costs (Hexner), this gives the same result.
    cost_data = compute_quadratic_cost_data(
        game=game, belief_tree=belief_tree,
        x_nodes=None, u_edges=None, v_edges=None
    )

    # 3) Initialise nominal controls
    # If warm-start trajectories provided, use them; otherwise use hover
    if u_init is not None and v_init is not None:
        # Warm-start from previous solution (clone to avoid aliasing)
        u_nom: List[Tensor] = [u.clone() for u in u_init]
        v_nom: List[Tensor] = [v.clone() for v in v_init]
    else:
        # Cold start from hover controls
        u_hover, v_hover = game.default_hover_control()
        u_nom: List[Tensor] = []
        v_nom: List[Tensor] = []
        for k in range(indexer.K):
            Nk = indexer.node_count(k)
            u_nom.append(
                u_hover.unsqueeze(0).unsqueeze(0).expand(Nk, indexer.I, game.du).clone()
            )
            v_nom.append(
                v_hover.unsqueeze(0).unsqueeze(0).expand(Nk, indexer.I, game.dv).clone()
            )

    # Initial forward rollout under hover controls
    x_nodes: List[Tensor] = [None] * (indexer.K + 1)   # type: ignore
    x_nodes[0] = x0.view(1, game.dx)
    for k in range(indexer.K):
        Nk = indexer.node_count(k)
        x_k = x_nodes[k]
        x_parent = x_k.unsqueeze(1).expand(Nk, indexer.I, game.dx).reshape(
            Nk * indexer.I, game.dx
        )
        u_flat = u_nom[k].reshape(Nk * indexer.I, game.du)
        v_flat = v_nom[k].reshape(Nk * indexer.I, game.dv)
        x_nodes[k + 1] = game.step(x_parent, u_flat, v_flat)

    riccati_sol: Optional[RiccatiSolution] = None

    # Diagnostics storage
    cost_hist: List[float] = []
    rel_hist: List[float] = []
    du_hist: List[float] = []
    dv_hist: List[float] = []
    pred_hist: List[float] = []
    accepted_step_hist: List[float] = []
    used_prev_iterate_hist: List[bool] = []
    riccati_retry_hist: List[bool] = []
    converged_iter = -1
    nan_or_inf = False
    riccati_reg_cur = float(riccati_reg)

    if collect_diagnostics:
        with torch.no_grad():
            c0 = _expected_cost_from_nominal_tree(
                game=game, indexer=indexer, belief_tree=belief_tree,
                cost_data=cost_data, x_nodes=x_nodes,
                u_edges=u_nom, v_edges=v_nom,
            )
            cost_hist.append(float(c0.detach().cpu().item()))
            rel_hist.append(0.0)
            du_hist.append(float("inf"))
            dv_hist.append(float("inf"))
            pred_hist.append(float("nan"))

    # 4) SQP iterations
    for it in range(num_sqp_iters):
        # Linearise dynamics on all edges
        A_list: List[Tensor] = []
        B1_list: List[Tensor] = []
        B2_list: List[Tensor] = []
        d_list: List[Tensor] = []

        for k in range(indexer.K):
            x_k = x_nodes[k]
            Nk = x_k.shape[0]
            I = indexer.I

            x_edge = x_k.unsqueeze(1).expand(Nk, I, game.dx).reshape(
                Nk * I, game.dx
            )
            u_edge = u_nom[k].reshape(Nk * I, game.du)
            v_edge = v_nom[k].reshape(Nk * I, game.dv)

            A_f, B1_f, B2_f, d_f = game.linearize(x_edge, u_edge, v_edge)

            A_list.append(A_f.view(Nk, I, game.dx, game.dx))
            B1_list.append(B1_f.view(Nk, I, game.dx, game.du))
            B2_list.append(B2_f.view(Nk, I, game.dx, game.dv))
            d_list.append(d_f.view(Nk, I, game.dx))

        # Quadratize costs around current nominal trajectory
        # For faithful SQP, this should be done at each iteration.
        # For quadratic costs (Hexner), this gives the same result every time,
        # but it's the principled approach and verifies correctness.
        cost_data = compute_quadratic_cost_data(
            game=game,
            belief_tree=belief_tree,
            x_nodes=x_nodes,
            u_edges=u_nom,
            v_edges=v_nom,
        )

        # Riccati backward (with one retry at higher regularization when unstable)
        riccati_retried = False
        riccati_sol = riccati_backward_time_varying(
            A=A_list, B1=B1_list, B2=B2_list, d=d_list,
            Q=cost_data.Q, q=cost_data.q, c=cost_data.c,
            R=cost_data.R, S=cost_data.S,
            P_leaf=cost_data.P_leaf, r_leaf=cost_data.r_leaf,
            c_leaf=cost_data.c_leaf,
            lambda_edge=belief_tree.lambda_edge,
            tau=game.cfg.tau,
            reg=riccati_reg_cur,
            max_reg_tries=max_riccati_reg_tries,
            reg_factor=riccati_reg_factor,
        )
        riccati_safe = _riccati_solution_is_numerically_safe(riccati_sol)

        if not riccati_safe:
            riccati_retried = True
            retry_reg = min(riccati_reg_cur * riccati_reg_factor, 1e9)
            riccati_sol = riccati_backward_time_varying(
                A=A_list, B1=B1_list, B2=B2_list, d=d_list,
                Q=cost_data.Q, q=cost_data.q, c=cost_data.c,
                R=cost_data.R, S=cost_data.S,
                P_leaf=cost_data.P_leaf, r_leaf=cost_data.r_leaf,
                c_leaf=cost_data.c_leaf,
                lambda_edge=belief_tree.lambda_edge,
                tau=game.cfg.tau,
                reg=retry_reg,
                max_reg_tries=max_riccati_reg_tries,
                reg_factor=riccati_reg_factor,
            )
            riccati_safe = _riccati_solution_is_numerically_safe(riccati_sol)
            if riccati_safe:
                riccati_reg_cur = retry_reg

        with torch.no_grad():
            c_prev_nominal = _expected_cost_from_nominal_tree(
                game=game, indexer=indexer, belief_tree=belief_tree,
                cost_data=cost_data, x_nodes=x_nodes,
                u_edges=u_nom, v_edges=v_nom,
            )
            c_prev_nominal_f = float(c_prev_nominal.detach().cpu().item())

        # Forward rollout with new feedback + optional line search
        x_prev = x_nodes
        u_prev = u_nom
        v_prev = v_nom

        accepted_step = 0.0
        used_prev_iterate = True

        if riccati_safe:
            if line_search:
                ls_step = step_size
                best_x: List[Tensor] | None = None
                best_u: List[Tensor] | None = None
                best_v: List[Tensor] | None = None
                best_cost = float("inf")
                best_step = 0.0

                for _ls in range(ls_max_steps):
                    x_try, u_try, v_try = _tree_forward_rollout(
                        game=game, indexer=indexer, x0=x0,
                        riccati_sol=riccati_sol, belief_tree=belief_tree,
                        u_prev=u_prev, v_prev=v_prev,
                        step_size=ls_step, action_space=action_space,
                    )
                    with torch.no_grad():
                        c_try = _expected_cost_from_nominal_tree(
                            game=game, indexer=indexer, belief_tree=belief_tree,
                            cost_data=cost_data, x_nodes=x_try,
                            u_edges=u_try, v_edges=v_try,
                        )
                    c_try_f = float(c_try.detach().cpu().item())

                    if torch.isfinite(c_try) and c_try_f < best_cost:
                        best_x, best_u, best_v = x_try, u_try, v_try
                        best_cost = c_try_f
                        best_step = ls_step

                    ls_step *= ls_backtrack
                    if ls_step < ls_alpha_min:
                        break

                if best_x is None:
                    nan_or_inf = True
                else:
                    improved = best_cost < c_prev_nominal_f
                    if improved or line_search_accept_worse:
                        x_nodes, u_nom, v_nom = best_x, best_u, best_v  # type: ignore
                        accepted_step = best_step
                        used_prev_iterate = False
            else:
                x_nodes, u_nom, v_nom = _tree_forward_rollout(
                    game=game, indexer=indexer, x0=x0,
                    riccati_sol=riccati_sol, belief_tree=belief_tree,
                    u_prev=u_prev, v_prev=v_prev,
                    step_size=step_size, action_space=action_space,
                )
                accepted_step = float(step_size)
                used_prev_iterate = False
        else:
            nan_or_inf = True

        # Keep previous iterate if a candidate produced invalid tensors.
        if not used_prev_iterate:
            invalid = False
            for kk in range(indexer.K + 1):
                if not torch.isfinite(x_nodes[kk]).all():
                    invalid = True
                    break
            if not invalid:
                for kk in range(indexer.K):
                    if (not torch.isfinite(u_nom[kk]).all()) or (not torch.isfinite(v_nom[kk]).all()):
                        invalid = True
                        break
            if invalid:
                nan_or_inf = True
                x_nodes, u_nom, v_nom = x_prev, u_prev, v_prev
                accepted_step = 0.0
                used_prev_iterate = True

        if collect_diagnostics:
            with torch.no_grad():
                c_now = _expected_cost_from_nominal_tree(
                    game=game, indexer=indexer, belief_tree=belief_tree,
                    cost_data=cost_data, x_nodes=x_nodes,
                    u_edges=u_nom, v_edges=v_nom,
                )
                if not torch.isfinite(c_now):
                    nan_or_inf = True

                c_now_f = float(c_now.detach().cpu().item())
                c_prev_f = cost_hist[-1]
                denom = max(1.0, abs(c_prev_f))
                rel_impr = (c_prev_f - c_now_f) / denom

                du_max = dv_max = 0.0
                for kk in range(indexer.K):
                    du_max = max(du_max, float(
                        (u_nom[kk] - u_prev[kk]).norm(dim=-1).max().item()
                    ))
                    dv_max = max(dv_max, float(
                        (v_nom[kk] - v_prev[kk]).norm(dim=-1).max().item()
                    ))

                if riccati_safe:
                    pred = riccati_sol.value_at_root(x0).detach().cpu().item()
                else:
                    pred = float("nan")

                cost_hist.append(c_now_f)
                rel_hist.append(float(rel_impr))
                du_hist.append(du_max)
                dv_hist.append(dv_max)
                pred_hist.append(float(pred))
                accepted_step_hist.append(float(accepted_step))
                used_prev_iterate_hist.append(bool(used_prev_iterate))
                riccati_retry_hist.append(bool(riccati_retried))

                is_conv = (
                    du_max <= sqp_tol_u
                    and dv_max <= sqp_tol_v
                    and abs(rel_impr) <= sqp_tol_rel_cost
                )
                if converged_iter < 0 and is_conv:
                    converged_iter = it

                if verbose:
                    print(
                        f"[SQP] it={it:02d}  cost={c_now_f:.6f}  "
                        f"step={accepted_step:.3f}  "
                        f"rel_impr={rel_impr:+.3e}  "
                        f"du={du_max:.3e}  dv={dv_max:.3e}  "
                        f"pred={pred:.6f}  "
                        f"{'KEEP_PREV ' if used_prev_iterate else ''}"
                        f"{'RIC_RETRY ' if riccati_retried else ''}"
                        f"{'CONV' if is_conv else ''}"
                        f"{'  (NaN!)' if nan_or_inf else ''}"
                    )

                if early_stop and is_conv:
                    break

    assert riccati_sol is not None

    diagnostics: Optional[SQPDiagnostics] = None
    if collect_diagnostics:
        diagnostics = SQPDiagnostics(
            cost_hist=cost_hist,
            rel_cost_impr_hist=rel_hist,
            du_max_hist=du_hist,
            dv_max_hist=dv_hist,
            root_value_pred_hist=pred_hist,
            accepted_step_hist=accepted_step_hist,
            used_prev_iterate_hist=used_prev_iterate_hist,
            riccati_retry_hist=riccati_retry_hist,
            converged=(converged_iter >= 0),
            converged_iter=converged_iter,
            nan_or_inf_encountered=nan_or_inf,
        )

    return SQPTreeResult(
        alpha=alpha,
        belief_tree=belief_tree,
        cost_data=cost_data,
        riccati_sol=riccati_sol,
        x_nodes=x_nodes,
        u_edges=u_nom,
        v_edges=v_nom,
        diagnostics=diagnostics,
    )
