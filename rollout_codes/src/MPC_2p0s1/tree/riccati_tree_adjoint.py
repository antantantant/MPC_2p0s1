from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from MPC_2p0s1.core.tensor_ops import symmetrize
from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.tree.averaged_costs import AveragedCostData
from MPC_2p0s1.tree.belief_tree import BeliefTree

# @torch.compile()
# @torch._dynamo.dont_skip_tracing()
# @torch.compile(fullgraph=True)
def _local_lq_saddle_batched_with_cache(
    A: Tensor,
    B1: Tensor,
    B2: Tensor,
    tau: float,
    R_bar: Tensor,   # (..., du, du)
    S_bar: Tensor,   # (..., dv, dv)
    P_plus: Tensor,  # (..., dx, dx)
    r_plus: Tensor,  # (..., dx)
    c_plus: Tensor,  # (...,)
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """
    Batched local post-reveal saddle solve with explicit cache for adjoint backward.

    Returns
    -------
    P_loc, r_loc, c_loc:
        Optimized local value coefficients.
    cache:
        Tensors needed by the manual backward pass.
    """
    dx = A.shape[0]
    du = B1.shape[1]
    dv = B2.shape[1]

    batch_shape = P_plus.shape[:-2]
    P_plus_flat = P_plus.reshape(-1, dx, dx)
    r_plus_flat = r_plus.reshape(-1, dx)
    c_plus_flat = c_plus.reshape(-1)

    R_bar_flat = R_bar.reshape(-1, du, du)
    S_bar_flat = S_bar.reshape(-1, dv, dv)

    b_flat = P_plus_flat.shape[0]

    A_b = A.unsqueeze(0).expand(b_flat, dx, dx)
    B1_b = B1.unsqueeze(0).expand(b_flat, dx, du)
    B2_b = B2.unsqueeze(0).expand(b_flat, dx, dv)

    P_B1 = P_plus_flat @ B1_b
    P_B2 = P_plus_flat @ B2_b
    P_A = P_plus_flat @ A_b

    B1T = B1_b.transpose(-1, -2)
    B2T = B2_b.transpose(-1, -2)
    AT = A_b.transpose(-1, -2)

    H_uu = tau * R_bar_flat + B1T @ P_B1
    H_uv = B1T @ P_B2
    H_vv = -tau * S_bar_flat + B2T @ P_B2

    H = P_plus_flat.new_empty(b_flat, du + dv, du + dv)
    H[:, :du, :du] = H_uu
    H[:, :du, du:] = H_uv
    H[:, du:, :du] = H_uv.transpose(-1, -2)
    H[:, du:, du:] = H_vv

    F_u = B1T @ P_A
    F_v = B2T @ P_A
    F = torch.cat([F_u, F_v], dim=1)

    r_plus_col = r_plus_flat.unsqueeze(-1)
    f_u = (B1T @ r_plus_col).squeeze(-1)
    f_v = (B2T @ r_plus_col).squeeze(-1)
    f = torch.cat([f_u, f_v], dim=1)

    Q = AT @ P_A
    q = (AT @ r_plus_col).squeeze(-1)

    rhs = torch.cat([F, f.unsqueeze(-1)], dim=2)
    sol = torch.linalg.solve(H, rhs)
    sol_F = sol[..., :dx]
    sol_f = sol[..., dx:]

    P_loc_flat = Q - F.transpose(-1, -2) @ sol_F
    P_loc_flat = symmetrize(P_loc_flat)

    r_loc_flat = q - (F.transpose(-1, -2) @ sol_f).squeeze(-1)
    c_loc_flat = c_plus_flat - 0.5 * (f * sol_f.squeeze(-1)).sum(dim=-1)

    P_loc = P_loc_flat.view(*batch_shape, dx, dx)
    r_loc = r_loc_flat.view(*batch_shape, dx)
    c_loc = c_loc_flat.view(*batch_shape)

    cache = {
        "H": H,
        "F": F,
        "f": f,
        "sol_F": sol_F,
        "sol_f": sol_f,
    }
    return P_loc, r_loc, c_loc, cache

@torch._dynamo.dont_skip_tracing()
# @torch.compile(fullgraph=True)
def _local_lq_saddle_batched_backward_from_cache(
    A: Tensor,
    B1: Tensor,
    B2: Tensor,
    tau: float,
    H: Tensor,        # (B, m, m)
    F: Tensor,        # (B, m, dx)
    f: Tensor,        # (B, m)
    sol_F: Tensor,    # (B, m, dx)
    sol_f: Tensor,    # (B, m, 1)
    gP_loc: Tensor,   # (..., dx, dx)
    gr_loc: Tensor,   # (..., dx)
    gc_loc: Tensor,   # (...,)
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Vectorized manual adjoint for local saddle map.

    Returns gradients wrt local inputs:
        P_plus, r_plus, c_plus, R_bar, S_bar
    with the same leading batch shape as gP_loc/gr_loc/gc_loc.
    """
    dx = A.shape[0]
    du = B1.shape[1]
    dv = B2.shape[1]
    m = du + dv

    batch_shape = gP_loc.shape[:-2]
    b_flat = H.shape[0]

    gP = symmetrize(gP_loc.reshape(b_flat, dx, dx))
    gr = gr_loc.reshape(b_flat, dx)
    gc = gc_loc.reshape(b_flat)

    X = sol_F
    y = sol_f.squeeze(-1)

    gQ = gP
    gq = gr
    gc_plus = gc

    gF = -(X @ gP.transpose(-1, -2)) - (y.unsqueeze(-1) @ gr.unsqueeze(1))
    gX = -(F @ gP)

    gy = -(F @ gr.unsqueeze(-1)).squeeze(-1) - 0.5 * gc.unsqueeze(-1) * f
    gf = -0.5 * gc.unsqueeze(-1) * y

    H_t = H.transpose(-1, -2)

    # Solve H^T [U | v] = [gX | gy] in one batched call.
    rhs_adj = torch.cat([gX, gy.unsqueeze(-1)], dim=2)  # (B, m, dx+1)
    sol_adj = torch.linalg.solve(H_t, rhs_adj)          # (B, m, dx+1)
    U = sol_adj[..., :dx]                               # (B, m, dx)
    v = sol_adj[..., dx:].squeeze(-1)                   # (B, m)

    gF = gF + U
    gH = -(U @ X.transpose(-1, -2))

    gf = gf + v
    gH = gH - (v.unsqueeze(-1) @ y.unsqueeze(1))

    gF_u = gF[:, :du, :]
    gF_v = gF[:, du:, :]
    gf_u = gf[:, :du]
    gf_v = gf[:, du:]

    gH_uu = gH[:, :du, :du]
    gH_uv_eff = gH[:, :du, du:] + gH[:, du:, :du].transpose(-1, -2)
    gH_vv = gH[:, du:, du:]

    gR = tau * gH_uu
    gS = -tau * gH_vv

    AT = A.transpose(0, 1)
    B1T = B1.transpose(0, 1)
    B2T = B2.transpose(0, 1)

    gP_plus = torch.einsum("ab,nbc,cd->nad", A, gQ, AT)
    gP_plus = gP_plus + torch.einsum("ab,nbc,cd->nad", B1, gF_u, AT)
    gP_plus = gP_plus + torch.einsum("ab,nbc,cd->nad", B2, gF_v, AT)

    gP_plus = gP_plus + torch.einsum("ab,nbc,cd->nad", B1, gH_uu, B1T)
    gP_plus = gP_plus + torch.einsum("ab,nbc,cd->nad", B1, gH_uv_eff, B2T)
    gP_plus = gP_plus + torch.einsum("ab,nbc,cd->nad", B2, gH_vv, B2T)

    gr_plus = torch.einsum("ab,nb->na", A, gq)
    gr_plus = gr_plus + torch.einsum("ab,nb->na", B1, gf_u)
    gr_plus = gr_plus + torch.einsum("ab,nb->na", B2, gf_v)

    gP_plus = gP_plus.view(*batch_shape, dx, dx)
    gr_plus = gr_plus.view(*batch_shape, dx)
    gc_plus = gc_plus.view(*batch_shape)
    gR = gR.view(*batch_shape, du, du)
    gS = gS.view(*batch_shape, dv, dv)

    return gP_plus, gr_plus, gc_plus, gR, gS


class _RiccatiRootValueAdjointFn(torch.autograd.Function):
    """
    Custom autograd for root value tree Riccati recursion.

    Input layout:
      x0, A, B1, B2, P_leaf, r_leaf, c_leaf,
      *R_bar[0:K], *S_bar[0:K], *lambda_edge[0:K], K(int), tau(float)
    """

    @staticmethod
    def forward(ctx, x0: Tensor, A: Tensor, B1: Tensor, B2: Tensor, P_leaf: Tensor, r_leaf: Tensor, c_leaf: Tensor, *args):
        if len(args) < 2:
            raise ValueError("Expected trailing metadata (K, tau).")

        K = int(args[-2])
        tau = float(args[-1])
        tensor_args = args[:-2]

        if len(tensor_args) != 3 * K:
            raise ValueError(
                f"Expected 3*K tensor args (R,S,lambda), got {len(tensor_args)} for K={K}."
            )

        R_bar = list(tensor_args[:K])
        S_bar = list(tensor_args[K : 2 * K])
        lambda_edge = list(tensor_args[2 * K : 3 * K])

        if K <= 0:
            raise ValueError(f"K must be positive, got {K}.")

        dx = A.shape[0]
        I = R_bar[0].shape[1]

        caches: List[Dict[str, Tensor]] = [None for _ in range(K)]  # type: ignore

        with torch.no_grad():
            P_nodes: List[Tensor] = [None for _ in range(K + 1)]  # type: ignore
            r_nodes: List[Tensor] = [None for _ in range(K + 1)]  # type: ignore
            c_nodes: List[Tensor] = [None for _ in range(K + 1)]  # type: ignore

            P_nodes[K] = P_leaf.clone()
            r_nodes[K] = r_leaf.clone()
            c_nodes[K] = c_leaf.clone()

            for k in reversed(range(K)):
                num_nodes = R_bar[k].shape[0]

                # Full I-ary layout: children of node n are contiguous as n*I:(n+1)*I.
                # Use reshape views instead of explicit child-index gather.
                P_plus = P_nodes[k + 1].reshape(num_nodes, I, dx, dx)
                r_plus = r_nodes[k + 1].reshape(num_nodes, I, dx)
                c_plus = c_nodes[k + 1].reshape(num_nodes, I)

                P_loc, r_loc, c_loc, local_cache = _local_lq_saddle_batched_with_cache(
                    A=A,
                    B1=B1,
                    B2=B2,
                    tau=tau,
                    R_bar=R_bar[k],
                    S_bar=S_bar[k],
                    P_plus=P_plus,
                    r_plus=r_plus,
                    c_plus=c_plus,
                )

                lam_k = lambda_edge[k]
                P_k = (lam_k.unsqueeze(-1).unsqueeze(-1) * P_loc).sum(dim=1)
                r_k = (lam_k.unsqueeze(-1) * r_loc).sum(dim=1)
                c_k = (lam_k * c_loc).sum(dim=1)

                P_nodes[k] = P_k
                r_nodes[k] = r_k
                c_nodes[k] = c_k

                caches[k] = {
                    "lam": lam_k,
                    "P_loc": P_loc,
                    "r_loc": r_loc,
                    "c_loc": c_loc,
                    "H": local_cache["H"],
                    "F": local_cache["F"],
                    "f": local_cache["f"],
                    "sol_F": local_cache["sol_F"],
                    "sol_f": local_cache["sol_f"],
                }

            root_P = P_nodes[0][0]
            root_r = r_nodes[0][0]
            root_c = c_nodes[0][0]

            loss = 0.5 * torch.dot(x0, root_P @ x0) + torch.dot(root_r, x0) + root_c

            num_nodes_by_depth = [int(P_nodes[d].shape[0]) for d in range(K + 1)]

        ctx.K = K
        ctx.tau = tau
        ctx.dx = int(dx)
        ctx.num_nodes_by_depth = num_nodes_by_depth
        ctx.caches = caches
        ctx.save_for_backward(x0, A, B1, B2, root_P, root_r)

        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x0, A, B1, B2, root_P, root_r = ctx.saved_tensors
        K: int = ctx.K
        tau: float = ctx.tau
        dx: int = ctx.dx
        num_nodes_by_depth: Sequence[int] = ctx.num_nodes_by_depth
        caches: Sequence[Dict[str, Tensor]] = ctx.caches

        device = x0.device
        dtype = x0.dtype

        g = grad_output

        gP_nodes: List[Tensor] = []
        gr_nodes: List[Tensor] = []
        gc_nodes: List[Tensor] = []
        for n in num_nodes_by_depth:
            gP_nodes.append(torch.zeros((n, dx, dx), device=device, dtype=dtype))
            gr_nodes.append(torch.zeros((n, dx), device=device, dtype=dtype))
            gc_nodes.append(torch.zeros((n,), device=device, dtype=dtype))

        x0_outer = torch.outer(x0, x0)
        gP_nodes[0][0] = g * (0.5 * x0_outer)
        gr_nodes[0][0] = g * x0
        gc_nodes[0][0] = g

        root_P_sym = 0.5 * (root_P + root_P.transpose(0, 1))
        gx0 = g * (root_P_sym @ x0 + root_r)

        gR_list: List[Tensor] = [None for _ in range(K)]  # type: ignore
        gS_list: List[Tensor] = [None for _ in range(K)]  # type: ignore
        gLam_list: List[Tensor] = [None for _ in range(K)]  # type: ignore

        for k in range(K):
            cache_k = caches[k]

            lam_k = cache_k["lam"]
            P_loc = cache_k["P_loc"]
            r_loc = cache_k["r_loc"]
            c_loc = cache_k["c_loc"]

            gP_k = gP_nodes[k]
            gr_k = gr_nodes[k]
            gc_k = gc_nodes[k]

            gP_loc = lam_k.unsqueeze(-1).unsqueeze(-1) * gP_k.unsqueeze(1)
            gr_loc = lam_k.unsqueeze(-1) * gr_k.unsqueeze(1)
            gc_loc = lam_k * gc_k.unsqueeze(1)

            gLam = (gP_k.unsqueeze(1) * P_loc).sum(dim=(-1, -2))
            gLam = gLam + (gr_k.unsqueeze(1) * r_loc).sum(dim=-1)
            gLam = gLam + gc_k.unsqueeze(1) * c_loc
            gLam_list[k] = gLam

            gP_plus, gr_plus, gc_plus, gR_k, gS_k = _local_lq_saddle_batched_backward_from_cache(
                A=A,
                B1=B1,
                B2=B2,
                tau=tau,
                H=cache_k["H"],
                F=cache_k["F"],
                f=cache_k["f"],
                sol_F=cache_k["sol_F"],
                sol_f=cache_k["sol_f"],
                gP_loc=gP_loc,
                gr_loc=gr_loc,
                gc_loc=gc_loc,
            )

            gR_list[k] = gR_k
            gS_list[k] = gS_k

            # Full I-ary layout yields a one-to-one contiguous mapping from
            # (node, action) to depth-(k+1) indices. No scatter is needed.
            gP_nodes[k + 1] = gP_plus.reshape_as(gP_nodes[k + 1])
            gr_nodes[k + 1] = gr_plus.reshape_as(gr_nodes[k + 1])
            gc_nodes[k + 1] = gc_plus.reshape_as(gc_nodes[k + 1])

        gP_leaf = gP_nodes[K]
        gr_leaf = gr_nodes[K]
        gc_leaf = gc_nodes[K]

        grads: List[Tensor | None] = [
            gx0,
            None,  # A
            None,  # B1
            None,  # B2
            gP_leaf,
            gr_leaf,
            gc_leaf,
        ]

        grads.extend(gR_list)
        grads.extend(gS_list)
        grads.extend(gLam_list)

        # trailing non-tensor args: K, tau
        grads.extend([None, None])

        return tuple(grads)


def riccati_root_value_adjoint(
    game: BaseLQGame,
    belief_tree: BeliefTree,
    avg_costs: AveragedCostData,
    x0: Tensor,
) -> Tensor:
    """
    Compute root value using post-reveal Riccati recursion with manual adjoint.

    This function is additive/experimental and leaves the original
    `riccati_tree.riccati_backward` path untouched.
    """
    K = int(belief_tree.indexer.K)
    if len(avg_costs.R_bar) != K or len(avg_costs.S_bar) != K or len(belief_tree.lambda_edge) != K:
        raise ValueError(
            "Inconsistent tree depth between avg_costs / belief_tree and indexer."
        )

    return _RiccatiRootValueAdjointFn.apply(
        x0,
        game.A,
        game.B1,
        game.B2,
        avg_costs.P_leaf,
        avg_costs.r_leaf,
        avg_costs.c_leaf,
        *avg_costs.R_bar,
        *avg_costs.S_bar,
        *belief_tree.lambda_edge,
        K,
        float(game.cfg.tau),
    )
