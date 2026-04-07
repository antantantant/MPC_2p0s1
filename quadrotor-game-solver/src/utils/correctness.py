"""Shared correctness metrics and gating utilities for train/eval scripts."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..game.quadrotor_game import Hexner3DQuadrotorGame
from ..rollout.trajectory import rollout_trajectory
from ..solvers.action_spaces import BoxActionSpace
from ..solvers.riccati import RiccatiSolution
from ..tree.belief_tree import BeliefTree
from ..tree.indexing import FullIaryTreeIndexer


def compute_config_fingerprint(config: Dict[str, object]) -> str:
    """Stable hash for a JSON-serializable config dictionary."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def alpha_separation_stats(
    *,
    alpha: Tensor,
    indexer: FullIaryTreeIndexer,
) -> Dict[str, object]:
    """Compute action-separation statistics over reachable nodes."""
    I = indexer.I
    K = indexer.K
    total_nodes = 0
    separated_nodes = 0

    for k in range(K):
        n_k = indexer.node_count(k)
        actions = torch.argmax(alpha[k, :n_k], dim=-1)  # (N_k, I_types)
        is_separated = actions.max(dim=-1).values != actions.min(dim=-1).values
        separated_nodes += int(is_separated.sum().item())
        total_nodes += n_k

    frac = (separated_nodes / total_nodes) if total_nodes > 0 else 0.0
    return {
        "separated_nodes": separated_nodes,
        "total_nodes": total_nodes,
        "separation_fraction": float(frac),
    }


def type_path_divergence_step(
    *,
    alpha: Tensor,
    indexer: FullIaryTreeIndexer,
    type0: int = 0,
    type1: int = 1,
) -> int:
    """First depth where greedy argmax paths for two types diverge, else -1."""
    if indexer.I < 2:
        return -1

    node0 = 0
    node1 = 0
    for k in range(indexer.K):
        a0 = int(torch.argmax(alpha[k, node0, type0]).item())
        a1 = int(torch.argmax(alpha[k, node1, type1]).item())
        if a0 != a1:
            return k
        node0 = indexer.child_index(k, node0, a0)
        node1 = indexer.child_index(k, node1, a1)
    return -1


def rollout_correctness_metrics(
    *,
    game: Hexner3DQuadrotorGame,
    indexer: FullIaryTreeIndexer,
    belief_tree: BeliefTree,
    riccati_sol: RiccatiSolution,
    alpha: Tensor,
    x0: Tensor,
    action_space: Optional[BoxActionSpace],
    altitude_floor: float = -5.0,
) -> Dict[str, object]:
    """Compute rollout-based correctness metrics for each payoff type."""
    targets = game.type_target_positions()  # (I, 3)
    per_type: List[Dict[str, object]] = []
    all_finite = True
    min_alt_p1 = float("inf")
    min_alt_p2 = float("inf")
    consistency_flags: List[bool] = []

    for type_idx in range(game.cfg.I):
        ro = rollout_trajectory(
            game=game,
            indexer=indexer,
            belief_tree=belief_tree,
            riccati_sol=riccati_sol,
            alpha=alpha,
            x0=x0,
            type_index=type_idx,
            action_space=action_space,
            sample_actions=False,
        )

        finite = bool(
            torch.isfinite(ro.x_traj).all()
            and torch.isfinite(ro.u_traj).all()
            and torch.isfinite(ro.v_traj).all()
        )
        all_finite = all_finite and finite

        p1_final = ro.x_traj[-1, :3]
        p2_final = ro.x_traj[-1, 12:15]

        dists_p1 = torch.norm(targets - p1_final.unsqueeze(0), dim=-1)
        true_d = float(dists_p1[type_idx].item())
        other_d = float(
            torch.min(
                torch.cat([dists_p1[:type_idx], dists_p1[type_idx + 1:]])
            ).item()
        ) if game.cfg.I > 1 else float("inf")
        terminal_ok = bool(true_d < other_d)
        consistency_flags.append(terminal_ok)

        min_alt_p1 = min(min_alt_p1, float(ro.x_traj[:, 2].min().item()))
        min_alt_p2 = min(min_alt_p2, float(ro.x_traj[:, 14].min().item()))

        per_type.append(
            {
                "type_index": type_idx,
                "theta": float(game.cfg.theta_values[type_idx]),
                "p1_final_pos": [float(v) for v in p1_final.tolist()],
                "p2_final_pos": [float(v) for v in p2_final.tolist()],
                "dist_true_target_p1": true_d,
                "dist_best_other_target_p1": other_d,
                "terminal_target_consistent": terminal_ok,
                "finite": finite,
                "proto_indices": [int(v) for v in ro.proto_indices.tolist()],
            }
        )

    consistency_rate = (
        float(sum(consistency_flags) / len(consistency_flags))
        if consistency_flags else 0.0
    )
    physical_plausible = bool(
        all_finite and (min_alt_p1 >= altitude_floor) and (min_alt_p2 >= altitude_floor)
    )

    sep = alpha_separation_stats(alpha=alpha, indexer=indexer)
    divergence = type_path_divergence_step(alpha=alpha, indexer=indexer)

    return {
        "alpha": {
            **sep,
            "type_path_divergence_step": divergence,
        },
        "rollouts": per_type,
        "terminal_consistency_rate": consistency_rate,
        "all_finite": all_finite,
        "min_altitude_p1": min_alt_p1,
        "min_altitude_p2": min_alt_p2,
        "altitude_floor": altitude_floor,
        "physical_plausible": physical_plausible,
    }


def passes_correctness_gates(
    metrics: Dict[str, object],
    *,
    min_separation_fraction: float = 0.05,
    min_terminal_consistency_rate: float = 0.5,
) -> Tuple[bool, Dict[str, bool]]:
    """Boolean gates used for best-checkpoint selection."""
    alpha_stats = metrics["alpha"]  # type: ignore[index]
    sep_ok = float(alpha_stats["separation_fraction"]) >= min_separation_fraction
    term_ok = (
        float(metrics["terminal_consistency_rate"]) >= min_terminal_consistency_rate
    )
    phys_ok = bool(metrics["physical_plausible"])

    gates = {
        "separation_ok": sep_ok,
        "terminal_consistency_ok": term_ok,
        "physical_plausible_ok": phys_ok,
    }
    return bool(sep_ok and term_ok and phys_ok), gates
