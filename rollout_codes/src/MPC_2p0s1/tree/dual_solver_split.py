from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame
from MPC_2p0s1.outer_opt.dual_split_network import DualSplitP1BRNet, DualSplitP2Net
from MPC_2p0s1.outer_opt.dual_split_param import DualNoSplitP1PolicyParam, DualSplitP2PolicyParam


@dataclass
class DualSplitConfig:
    """
    Runtime options for split dual rollout.
    """

    terminal_smooth_temp: float = 0.0
    branching: Optional[int] = None


@dataclass
class DualSplitRolloutResult:
    """
    Outputs of a split dual rollout over an explicit path set.
    """

    dual_value: Tensor             # ()
    path_value: Tensor             # (S,)
    prob_seq: Tensor               # (S,)
    terminal_terms: Tensor         # (S, I)
    expected_type_cost: Tensor     # (I,)
    type_cost_by_path: Tensor      # (S, I)
    x_final: Tensor                # (S, dx)
    phat_final: Tensor             # (S, I)
    paths: Tensor                  # (S, K)


def all_dual_paths(branching: int, K: int, device: torch.device | None = None) -> Tensor:
    """
    Enumerate all full paths of length K with symbols in {0, ..., branching-1}.
    """
    if int(branching) <= 1:
        raise ValueError(f"branching must be >= 2, got {branching}.")
    if int(K) < 0:
        raise ValueError(f"K must be nonnegative, got {K}.")

    branching = int(branching)
    K = int(K)
    if K == 0:
        return torch.zeros((1, 0), dtype=torch.long, device=device)

    S = branching**K
    idx = torch.arange(S, dtype=torch.long, device=device)
    cols = []
    for power in range(K - 1, -1, -1):
        cols.append((idx // (branching**power)) % branching)
    return torch.stack(cols, dim=1)


def _parent_index_from_prefix(paths: Tensor, depth: int, branching: int) -> Tensor:
    """
    Parent node ids in base-`branching` for each path at given depth.
    """
    S = int(paths.shape[0])
    device = paths.device
    if depth <= 0:
        return torch.zeros((S,), dtype=torch.long, device=device)
    coef = (branching ** torch.arange(depth - 1, -1, -1, device=device, dtype=torch.long))
    return (paths[:, :depth] * coef.unsqueeze(0)).sum(dim=1)


def _running_cost_per_type_batched(game: BaseLQGame, u: Tensor, v: Tensor) -> Tensor:
    """
    Batched type-wise running cost l_i(u,v), returns shape (S, I).
    """
    tau = float(game.cfg.tau)
    run_u = 0.5 * tau * torch.einsum("sd,idf,sf->si", u, game.R, u)
    run_v = 0.5 * tau * torch.einsum("sd,idf,sf->si", v, game.S, v)
    return run_u - run_v


def _terminal_cost_per_type_batched(game: BaseLQGame, x: Tensor) -> Tensor:
    """
    Batched type-wise terminal costs g_i(x), returns shape (S, I).
    """
    quad = 0.5 * torch.einsum("sd,idf,sf->si", x, game.Q, x)
    lin = torch.einsum("sd,id->si", x, game.q)
    return quad + lin + game.c.view(1, -1)


def rollout_dual_split_paths(
    game: BaseLQGame,
    p1_policy: DualNoSplitP1PolicyParam,
    p2_policy: DualSplitP2PolicyParam,
    *,
    paths: Tensor,
    x0: Tensor,
    phat0: Tensor,
    cfg: DualSplitConfig,
    action_space: Optional[BoxActionSpace] = None,
    edge_allow_bitmaps: Optional[list[Tensor | None]] = None,
) -> DualSplitRolloutResult:
    """
    Roll out split dual game over a fixed explicit set of P2 atomic paths.

    Each path is a sequence a_0,...,a_{K-1} with a_k in {0,...,A-1}, A=branching.
    """
    device = game.device_resolved
    dtype = game.dtype
    K = int(game.cfg.K)
    I = int(game.I)

    if paths.ndim != 2 or paths.shape[1] != K:
        raise ValueError(f"paths must have shape (S, K={K}), got {tuple(paths.shape)}")
    if x0.shape != (game.dx,):
        raise ValueError(f"x0 must have shape ({game.dx},), got {tuple(x0.shape)}")
    if phat0.shape != (I,):
        raise ValueError(f"phat0 must have shape ({I},), got {tuple(phat0.shape)}")
    if p1_policy.K != K:
        raise ValueError(f"p1_policy.K={p1_policy.K} does not match game K={K}.")
    if p2_policy.K != K:
        raise ValueError(f"p2_policy.K={p2_policy.K} does not match game K={K}.")

    branching = int(cfg.branching) if cfg.branching is not None else int(p2_policy.branching)
    if branching != int(p2_policy.branching):
        raise ValueError(
            f"cfg branching={branching} mismatches p2_policy.branching={p2_policy.branching}."
        )
    if int(paths.max().item()) >= branching or int(paths.min().item()) < 0:
        raise ValueError("paths entries must be in [0, branching).")

    if edge_allow_bitmaps is None:
        edge_allow_bitmaps = [None] * K
    if len(edge_allow_bitmaps) < K:
        edge_allow_bitmaps = list(edge_allow_bitmaps) + [None] * (K - len(edge_allow_bitmaps))
    elif len(edge_allow_bitmaps) > K:
        edge_allow_bitmaps = list(edge_allow_bitmaps[:K])

    paths = paths.to(device=device, dtype=torch.long)
    x0 = x0.to(device=device, dtype=dtype)
    phat0 = phat0.to(device=device, dtype=dtype)

    S = int(paths.shape[0])
    x = x0.view(1, -1).repeat(S, 1)
    phat = phat0.view(1, -1).repeat(S, 1)
    prob_seq = torch.ones((S,), device=device, dtype=dtype)
    type_cost = torch.zeros((S, I), device=device, dtype=dtype)

    for k in range(K):
        logits_k = p2_policy.logits_at(k).to(device=device, dtype=dtype)         # (A,)
        v_table = p2_policy.action_table_at(k).to(device=device, dtype=dtype)     # (A,dv)
        phat_raw_table = p2_policy.phat_raw_table_at(k).to(device=device, dtype=dtype)  # (A,I)

        bm = edge_allow_bitmaps[k]
        if bm is None:
            lam = torch.softmax(logits_k, dim=0).view(1, branching).expand(S, branching)
        else:
            bm = bm.to(device=device, dtype=torch.bool)
            parent = _parent_index_from_prefix(paths, k, branching)  # (S,)
            keys = parent.view(-1, 1) * branching + torch.arange(
                branching, device=device, dtype=torch.long
            ).view(1, -1)
            allow = torch.zeros((S, branching), device=device, dtype=torch.bool)
            valid = keys < int(bm.numel())
            allow[valid] = bm[keys[valid]]
            none_allowed = ~allow.any(dim=-1, keepdim=True)
            allow = torch.where(none_allowed, torch.ones_like(allow), allow)

            logits = logits_k.view(1, branching).expand(S, branching)
            logits = torch.where(allow, logits, torch.full_like(logits, -1e9))
            lam = torch.softmax(logits, dim=-1)

        a_k = paths[:, k]  # (S,)
        lam_k = lam.gather(dim=1, index=a_k.view(-1, 1)).squeeze(1)  # (S,)

        v_k = v_table.index_select(dim=0, index=a_k)  # (S,dv)
        if action_space is not None:
            v_k = action_space.clip_v(v_k)

        # Martingale correction on p_hat split:
        # E_a[raw_next_a] is shifted so E_a[next_a] = current phat.
        expected_raw = lam @ phat_raw_table  # (S,I)
        phat_raw_sel = phat_raw_table.index_select(dim=0, index=a_k)  # (S,I)
        correction = phat - expected_raw
        phat_next_pre_cost = phat_raw_sel + correction

        u_k = p1_policy.action_at(k).to(device=device, dtype=dtype).view(1, -1).expand(S, -1)
        if action_space is not None:
            u_k = action_space.clip_u(u_k)

        l_type = _running_cost_per_type_batched(game, u_k, v_k)  # (S,I)
        type_cost = type_cost + l_type

        x = game.step_dynamics(x, u_k, v_k)  # (S,dx)
        phat = phat_next_pre_cost - l_type
        prob_seq = prob_seq * lam_k

    g_type = _terminal_cost_per_type_batched(game, x)   # (S,I)
    type_cost = type_cost + g_type
    terminal_terms = phat - g_type                      # (S,I)

    temp = float(cfg.terminal_smooth_temp)
    if temp > 0.0:
        path_value = temp * torch.logsumexp(terminal_terms / temp, dim=1)  # (S,)
    else:
        path_value = terminal_terms.max(dim=1).values  # (S,)

    dual_value = torch.sum(prob_seq * path_value)
    expected_type_cost = torch.sum(prob_seq.view(-1, 1) * type_cost, dim=0)  # (I,)

    return DualSplitRolloutResult(
        dual_value=dual_value,
        path_value=path_value,
        prob_seq=prob_seq,
        terminal_terms=terminal_terms,
        expected_type_cost=expected_type_cost,
        type_cost_by_path=type_cost,
        x_final=x,
        phat_final=phat,
        paths=paths,
    )


def build_edge_allow_bitmaps_from_paths(paths: Tensor, branching: int) -> list[Tensor]:
    """
    Build per-depth edge-allow bitmaps from kept paths.
    """
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2D, got shape {tuple(paths.shape)}")
    S, K = int(paths.shape[0]), int(paths.shape[1])
    device = paths.device
    bitmaps: list[Tensor] = []
    if S == 0:
        return [torch.zeros((0,), dtype=torch.bool, device=device) for _ in range(K)]

    for k in range(K):
        parent = _parent_index_from_prefix(paths, k, branching)  # (S,)
        child = paths[:, k]
        max_parent = int(parent.max().item()) if parent.numel() > 0 else -1
        if max_parent < 0:
            bitmaps.append(torch.zeros((0,), dtype=torch.bool, device=device))
            continue
        bm = torch.zeros(((max_parent + 1) * branching,), dtype=torch.bool, device=device)
        bm[parent * branching + child] = True
        bitmaps.append(bm)
    return bitmaps


def topk_prune_paths(
    paths: Tensor,
    prob_seq: Tensor,
    *,
    keep_ratio: float = 0.5,
    min_paths: int = 2,
    min_drop: int = 1,
    max_paths: Optional[int] = None,
) -> tuple[Tensor, dict[str, float]]:
    """
    Keep top-k paths by probability mass.
    """
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2D, got {tuple(paths.shape)}")
    if prob_seq.ndim != 1 or prob_seq.shape[0] != paths.shape[0]:
        raise ValueError(
            f"prob_seq must have shape ({paths.shape[0]},), got {tuple(prob_seq.shape)}"
        )
    if not (0.0 < float(keep_ratio) <= 1.0):
        raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}.")

    S = int(paths.shape[0])
    if S <= int(min_paths):
        return paths, {"kept_k": float(S), "target_k": float(S), "kept_mass_frac": 1.0}

    k_target = int(torch.ceil(torch.tensor(float(keep_ratio) * S)).item())
    k_target = max(int(min_paths), min(S, k_target))
    if max_paths is not None:
        k_target = min(k_target, int(max_paths))
        k_target = max(int(min_paths), k_target)
    if (S - k_target) < int(min_drop) and S > int(min_paths):
        k_target = max(int(min_paths), S - int(min_drop))

    if k_target >= S:
        return paths, {"kept_k": float(S), "target_k": float(S), "kept_mass_frac": 1.0}

    prob = prob_seq.detach()
    order = torch.argsort(prob, descending=True)
    keep_idx = order[:k_target]
    kept = paths.index_select(0, keep_idx)
    kept = torch.unique(kept, dim=0)

    denom = prob.sum().clamp_min(1e-12)
    kept_mass_frac = float((prob.index_select(0, keep_idx).sum() / denom).item())
    return kept, {
        "kept_k": float(int(kept.shape[0])),
        "target_k": float(k_target),
        "kept_mass_frac": kept_mass_frac,
    }


def rollout_dual_split_paths_network(
    game: BaseLQGame,
    p1_model: DualSplitP1BRNet,
    p2_model: DualSplitP2Net,
    *,
    paths: Tensor,
    x0: Tensor,
    phat0: Tensor,
    cfg: DualSplitConfig,
    action_space: Optional[BoxActionSpace] = None,
    edge_allow_bitmaps: Optional[list[Tensor | None]] = None,
) -> DualSplitRolloutResult:
    """
    Split dual rollout with state-conditioned neural policies.
    """
    device = game.device_resolved
    dtype = game.dtype
    K = int(game.cfg.K)
    I = int(game.I)

    if paths.ndim != 2 or paths.shape[1] != K:
        raise ValueError(f"paths must have shape (S, K={K}), got {tuple(paths.shape)}")
    if x0.shape != (game.dx,):
        raise ValueError(f"x0 must have shape ({game.dx},), got {tuple(x0.shape)}")
    if phat0.shape != (I,):
        raise ValueError(f"phat0 must have shape ({I},), got {tuple(phat0.shape)}")
    if int(p2_model.branching) <= 1:
        raise ValueError(f"p2_model.branching must be >=2, got {p2_model.branching}.")

    branching = int(cfg.branching) if cfg.branching is not None else int(p2_model.branching)
    if branching != int(p2_model.branching):
        raise ValueError(
            f"cfg branching={branching} mismatches p2_model.branching={p2_model.branching}."
        )
    if int(paths.max().item()) >= branching or int(paths.min().item()) < 0:
        raise ValueError("paths entries must be in [0, branching).")

    if edge_allow_bitmaps is None:
        edge_allow_bitmaps = [None] * K
    if len(edge_allow_bitmaps) < K:
        edge_allow_bitmaps = list(edge_allow_bitmaps) + [None] * (K - len(edge_allow_bitmaps))
    elif len(edge_allow_bitmaps) > K:
        edge_allow_bitmaps = list(edge_allow_bitmaps[:K])

    paths = paths.to(device=device, dtype=torch.long)
    x = x0.to(device=device, dtype=dtype).view(1, -1).repeat(int(paths.shape[0]), 1)
    phat = phat0.to(device=device, dtype=dtype).view(1, -1).repeat(int(paths.shape[0]), 1)

    S = int(paths.shape[0])
    prob_seq = torch.ones((S,), device=device, dtype=dtype)
    type_cost = torch.zeros((S, I), device=device, dtype=dtype)

    for k in range(K):
        t_norm = float(k) / float(max(1, K))
        logits, v_table, phat_raw_table = p2_model(x, phat, t_norm)  # (S,A), (S,A,dv), (S,A,I)

        bm = edge_allow_bitmaps[k]
        if bm is None:
            lam = torch.softmax(logits, dim=-1)
        else:
            bm = bm.to(device=device, dtype=torch.bool)
            parent = _parent_index_from_prefix(paths, k, branching)  # (S,)
            keys = parent.view(-1, 1) * branching + torch.arange(
                branching, device=device, dtype=torch.long
            ).view(1, -1)
            allow = torch.zeros((S, branching), device=device, dtype=torch.bool)
            valid = keys < int(bm.numel())
            allow[valid] = bm[keys[valid]]
            none_allowed = ~allow.any(dim=-1, keepdim=True)
            allow = torch.where(none_allowed, torch.ones_like(allow), allow)
            logits = torch.where(allow, logits, torch.full_like(logits, -1e9))
            lam = torch.softmax(logits, dim=-1)

        a_k = paths[:, k]  # (S,)
        lam_k = lam.gather(dim=1, index=a_k.view(-1, 1)).squeeze(1)      # (S,)
        gather_idx_v = a_k.view(-1, 1, 1).expand(S, 1, game.dv)
        gather_idx_p = a_k.view(-1, 1, 1).expand(S, 1, I)
        v_k = v_table.gather(dim=1, index=gather_idx_v).squeeze(1)        # (S,dv)
        phat_raw_sel = phat_raw_table.gather(dim=1, index=gather_idx_p).squeeze(1)  # (S,I)

        expected_raw = torch.sum(lam.unsqueeze(-1) * phat_raw_table, dim=1)  # (S,I)
        phat_next_pre_cost = phat_raw_sel + (phat - expected_raw)

        u_k = p1_model(x, phat, t_norm)  # (S,du)
        if action_space is not None:
            u_k = action_space.clip_u(u_k)
            v_k = action_space.clip_v(v_k)

        l_type = _running_cost_per_type_batched(game, u_k, v_k)  # (S,I)
        type_cost = type_cost + l_type
        x = game.step_dynamics(x, u_k, v_k)
        phat = phat_next_pre_cost - l_type
        prob_seq = prob_seq * lam_k

    g_type = _terminal_cost_per_type_batched(game, x)
    type_cost = type_cost + g_type
    terminal_terms = phat - g_type

    temp = float(cfg.terminal_smooth_temp)
    if temp > 0.0:
        path_value = temp * torch.logsumexp(terminal_terms / temp, dim=1)
    else:
        path_value = terminal_terms.max(dim=1).values

    dual_value = torch.sum(prob_seq * path_value)
    expected_type_cost = torch.sum(prob_seq.view(-1, 1) * type_cost, dim=0)

    return DualSplitRolloutResult(
        dual_value=dual_value,
        path_value=path_value,
        prob_seq=prob_seq,
        terminal_terms=terminal_terms,
        expected_type_cost=expected_type_cost,
        type_cost_by_path=type_cost,
        x_final=x,
        phat_final=phat,
        paths=paths,
    )
