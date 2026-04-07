"""Tree-structured Riccati backward pass for time-varying affine dynamics.

This is the "inner" layer used inside each SQP iteration.

Given, for each edge (k, node, a):
    x⁺ = A x + B1 u + B2 v + d                             (linearised dynamics)
    stage cost (belief-averaged):
        τ [ ½ x^T Q x + q^T x + c  +  ½ u^T R u − ½ v^T S v ]
    next value at child:
        V₊(x⁺) = ½ x⁺ᵀ P₊ x⁺ + r₊ᵀ x⁺ + c₊

we solve the one-step  min_u max_v  saddle game in closed form (KKT system)
and aggregate over actions a using λ_edge.

Mirrors ``nl_sqp/riccati_timevarying.py`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from ..utils.types import ValueQuad
from ..utils.linalg import symmetrize


@dataclass
class RiccatiSolution:
    """Output of the Riccati backward pass on the full tree."""

    # Node value quads
    P_nodes: List[Tensor]   # len K+1; P_nodes[k]: (N_k, dx, dx)
    r_nodes: List[Tensor]   # len K+1; r_nodes[k]: (N_k, dx)
    c_nodes: List[Tensor]   # len K+1; c_nodes[k]: (N_k,)

    # Edge feedback prototypes
    K_u: List[Tensor]       # len K; K_u[k]: (N_k, I, du, dx)
    kappa_u: List[Tensor]   # len K; kappa_u[k]: (N_k, I, du)
    K_v: List[Tensor]       # len K; K_v[k]: (N_k, I, dv, dx)
    kappa_v: List[Tensor]   # len K; kappa_v[k]: (N_k, I, dv)

    def root_value_quad(self) -> ValueQuad:
        return ValueQuad(
            P=self.P_nodes[0][0],
            r=self.r_nodes[0][0],
            c=self.c_nodes[0][0],
        )

    def value_at_root(self, x0: Tensor) -> Tensor:
        return self.root_value_quad().evaluate(x0)


# ── Batched one-step LQ saddle solve ────────────────────────────────────────
# @torch.compile()
def _local_affine_lq_saddle_batched(
    *,
    A: Tensor,       # (B, dx, dx)
    B1: Tensor,      # (B, dx, du)
    B2: Tensor,      # (B, dx, dv)
    d: Tensor,       # (B, dx)
    tau: float,
    Q: Tensor,       # (B, dx, dx) or (dx, dx)
    q: Tensor,       # (B, dx)
    c: Tensor,       # (B,)
    R: Tensor,       # (B, du, du)
    S: Tensor,       # (B, dv, dv)
    P_plus: Tensor,  # (B, dx, dx)
    r_plus: Tensor,  # (B, dx)
    c_plus: Tensor,  # (B,)
    reg: float = 1e-3,
    max_tries: int = 5,
    reg_factor: float = 10.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Solve batched one-step LQ saddle problem with affine dynamics.

    Returns (P_loc, r_loc, c_loc, K_u, kappa_u, K_v, kappa_v).
    """
    B, dx, _ = A.shape
    du = B1.shape[-1]
    dv = B2.shape[-1]

    AT  = A.transpose(-1, -2)
    B1T = B1.transpose(-1, -2)
    B2T = B2.transpose(-1, -2)

    P_A  = P_plus @ A
    P_B1 = P_plus @ B1
    P_B2 = P_plus @ B2

    # Hessian blocks
    H_uu0 = tau * R + B1T @ P_B1
    H_uv  = B1T @ P_B2
    H_vv0 = -tau * S + B2T @ P_B2

    # Cross terms with x
    F_u = B1T @ P_A
    F_v = B2T @ P_A
    F   = torch.cat([F_u, F_v], dim=1)   # (B, du+dv, dx)

    # Linear terms from affine + r_plus
    Pd_plus_r = (P_plus @ d.unsqueeze(-1)).squeeze(-1) + r_plus
    f_u = (B1T @ Pd_plus_r.unsqueeze(-1)).squeeze(-1)
    f_v = (B2T @ Pd_plus_r.unsqueeze(-1)).squeeze(-1)
    f   = torch.cat([f_u, f_v], dim=1)   # (B, du+dv)

    # Pure x-terms
    if Q.ndim == 2:
        Qb = Q.unsqueeze(0).expand(B, dx, dx)
    else:
        Qb = Q
    Q_total = tau * Qb + AT @ P_A
    q_total = tau * q + (AT @ Pd_plus_r.unsqueeze(-1)).squeeze(-1)

    quad_d = 0.5 * (d.unsqueeze(1) @ (P_plus @ d.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    lin_d  = (r_plus * d).sum(dim=-1)
    c_total = tau * c + c_plus + quad_d + lin_d

    RHS = torch.cat([F, f.unsqueeze(-1)], dim=-1)   # (B, du+dv, dx+1)

    # Regularisation
    Iu = torch.eye(du, device=A.device, dtype=A.dtype).expand(B, du, du)
    Iv = torch.eye(dv, device=A.device, dtype=A.dtype).expand(B, dv, dv)

    H_uu0 = symmetrize(H_uu0)
    H_vv0 = symmetrize(H_vv0)

    sol: Tensor | None = None
    reg_cur = float(reg)

    for _ in range(max_tries):
        H_uu = H_uu0 + reg_cur * Iu
        H_vv = H_vv0 - reg_cur * Iv

        H = torch.empty(B, du + dv, du + dv, device=A.device, dtype=A.dtype)
        H[:, :du, :du] = H_uu
        H[:, :du, du:] = H_uv
        H[:, du:, :du] = H_uv.transpose(-1, -2)
        H[:, du:, du:] = H_vv
        H = symmetrize(H)

        try:
            sol_try = torch.linalg.solve(H, RHS)
        except RuntimeError:
            sol_try = None

        if sol_try is not None and torch.isfinite(sol_try).all():
            sol = sol_try
            break

        reg_cur = min(reg_cur * reg_factor, 1e9)

    if sol is None:
        H_uu = H_uu0 + reg_cur * Iu
        H_vv = H_vv0 - reg_cur * Iv

        H = torch.empty(B, du + dv, du + dv, device=A.device, dtype=A.dtype)
        H[:, :du, :du] = H_uu
        H[:, :du, du:] = H_uv
        H[:, du:, :du] = H_uv.transpose(-1, -2)
        H[:, du:, du:] = H_vv
        H = symmetrize(H)

        sol = torch.linalg.lstsq(H, RHS).solution
        sol = torch.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)

    sol_F = sol[..., :dx]
    sol_f = sol[..., dx:]

    K_opt = -sol_F
    b_opt = -sol_f.squeeze(-1)

    K_u = K_opt[:, :du, :]
    K_v = K_opt[:, du:, :]
    kappa_u = b_opt[:, :du]
    kappa_v = b_opt[:, du:]

    P_loc = Q_total - F.transpose(-1, -2) @ sol_F
    P_loc = symmetrize(P_loc)

    r_loc = q_total - (F.transpose(-1, -2) @ sol_f).squeeze(-1)
    c_loc = c_total - 0.5 * (f * sol_f.squeeze(-1)).sum(dim=-1)

    return P_loc, r_loc, c_loc, K_u, kappa_u, K_v, kappa_v


# ── Full Riccati backward sweep ─────────────────────────────────────────────

def riccati_backward_time_varying(
    *,
    # Dynamics (edge-wise)
    A: List[Tensor],    # len K; A[k] = (N_k, I, dx, dx)
    B1: List[Tensor],
    B2: List[Tensor],
    d: List[Tensor],    # affine offset
    # Costs (edge-wise + terminal)
    Q: List[Tensor],
    q: List[Tensor],
    c: List[Tensor],
    R: List[Tensor],
    S: List[Tensor],
    P_leaf: Tensor,
    r_leaf: Tensor,
    c_leaf: Tensor,
    # Tree probabilities
    lambda_edge: List[Tensor],   # len K; (N_k, I)
    tau: float,
    # Stabilisation
    reg: float = 1e-3,
    max_reg_tries: int = 5,
    reg_factor: float = 10.0,
) -> RiccatiSolution:
    """Vectorized Riccati recursion for a full I-ary tree."""

    K = len(A)
    dx = P_leaf.shape[-1]

    P_nodes: List[Tensor] = [None] * (K + 1)   # type: ignore
    r_nodes: List[Tensor] = [None] * (K + 1)   # type: ignore
    c_nodes: List[Tensor] = [None] * (K + 1)   # type: ignore

    K_u_list: List[Tensor] = [None] * K         # type: ignore
    kappa_u_list: List[Tensor] = [None] * K     # type: ignore
    K_v_list: List[Tensor] = [None] * K         # type: ignore
    kappa_v_list: List[Tensor] = [None] * K     # type: ignore

    # Terminal
    P_nodes[K] = P_leaf
    r_nodes[K] = r_leaf
    c_nodes[K] = c_leaf

    for k in reversed(range(K)):
        lam = lambda_edge[k]              # (N_k, I)
        Nk, I = lam.shape

        P_plus = P_nodes[k + 1].view(Nk, I, dx, dx)
        r_plus = r_nodes[k + 1].view(Nk, I, dx)
        c_plus = c_nodes[k + 1].view(Nk, I)

        B_flat = Nk * I

        A_f  = A[k].reshape(B_flat, dx, dx)
        B1_f = B1[k].reshape(B_flat, dx, B1[k].shape[-1])
        B2_f = B2[k].reshape(B_flat, dx, B2[k].shape[-1])
        d_f  = d[k].reshape(B_flat, dx)

        Q_f = Q[k].reshape(B_flat, dx, dx)
        q_f = q[k].reshape(B_flat, dx)
        c_f = c[k].reshape(B_flat)

        R_f = R[k].reshape(B_flat, R[k].shape[-1], R[k].shape[-1])
        S_f = S[k].reshape(B_flat, S[k].shape[-1], S[k].shape[-1])

        Pp_f = P_plus.reshape(B_flat, dx, dx)
        rp_f = r_plus.reshape(B_flat, dx)
        cp_f = c_plus.reshape(B_flat)

        (
            P_loc_f, r_loc_f, c_loc_f,
            K_u_f, kappa_u_f, K_v_f, kappa_v_f,
        ) = _local_affine_lq_saddle_batched(
            A=A_f, B1=B1_f, B2=B2_f, d=d_f,
            tau=tau,
            Q=Q_f, q=q_f, c=c_f,
            R=R_f, S=S_f,
            P_plus=Pp_f, r_plus=rp_f, c_plus=cp_f,
            reg=reg, max_tries=max_reg_tries, reg_factor=reg_factor,
        )

        du_k = B1[k].shape[-1]
        dv_k = B2[k].shape[-1]

        P_loc = P_loc_f.view(Nk, I, dx, dx)
        r_loc = r_loc_f.view(Nk, I, dx)
        c_loc = c_loc_f.view(Nk, I)

        K_u_k     = K_u_f.view(Nk, I, du_k, dx)
        kappa_u_k = kappa_u_f.view(Nk, I, du_k)
        K_v_k     = K_v_f.view(Nk, I, dv_k, dx)
        kappa_v_k = kappa_v_f.view(Nk, I, dv_k)

        # Aggregate node values
        lam_P = lam.unsqueeze(-1).unsqueeze(-1)
        lam_r = lam.unsqueeze(-1)

        P_nodes[k] = (lam_P * P_loc).sum(dim=1)
        r_nodes[k] = (lam_r * r_loc).sum(dim=1)
        c_nodes[k] = (lam * c_loc).sum(dim=1)

        K_u_list[k]     = K_u_k
        kappa_u_list[k] = kappa_u_k
        K_v_list[k]     = K_v_k
        kappa_v_list[k] = kappa_v_k

    return RiccatiSolution(
        P_nodes=P_nodes,
        r_nodes=r_nodes,
        c_nodes=c_nodes,
        K_u=K_u_list,
        kappa_u=kappa_u_list,
        K_v=K_v_list,
        kappa_v=kappa_v_list,
    )
