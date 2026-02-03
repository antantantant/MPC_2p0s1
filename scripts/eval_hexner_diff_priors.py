# scripts/eval_hexner_mpc_priors.py
"""
Evaluate the compact representation with different prior distributions.

This script tests the signaling policy and Riccati solution:
1. Load a trained checkpoint (with uniform prior)
2. Extract compact representation
3. For each test prior:
   a. Recompute r values using recompute_for_prior()
   b. Run online policy rollouts
   c. Compare against ground truth (full tree solve with that prior)
4. Report error metrics across priors

Key insight: For non-uniform priors, linear interpolation of r_type is
approximate. The recompute_for_prior() method accounts for the actual
belief trajectory under the signaling policy α.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import numpy as np

from MPC_2p0s1.config.base_config import GameConfig, TrainingConfig, project_relative
from MPC_2p0s1.core.action_spaces import BoxActionSpace
from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.hexner_game import HexnerGame, HexnerParams
from MPC_2p0s1.outer_opt.alpha_param import AlphaParam, AlphaParamConfig
from MPC_2p0s1.outer_opt.checkpointing import load_checkpoint, CheckpointMeta
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.belief_tree import BeliefTree, build_belief_tree
from MPC_2p0s1.tree.riccati_tree import riccati_backward, RiccatiSolution
from MPC_2p0s1.tree.averaged_costs import compute_averaged_costs
from MPC_2p0s1.compact.compact_solution import (
    CompactLQSolution, extract_compact_solution, verify_r_linearity
)
from MPC_2p0s1.compact.online_policy import OnlinePolicy


# ============================================================================
# Analytical Ground Truth (Discrete-Time Riccati)
# ============================================================================

@dataclass
class GroundTruthTrajectory:
    """Ground truth trajectory for a specific initial state and type."""
    t_grid: Tensor           # (K+1,)
    x1_traj: Tensor          # (K+1, 4) P1 state
    x2_traj: Tensor          # (K+1, 4) P2 state
    u1_traj: Tensor          # (K, 2) P1 controls
    u2_traj: Tensor          # (K, 2) P2 controls
    theta: float             # Payoff type
    tr: float                # Revelation time
    prior: float             # Prior probability of type +1


def discrete_lqr(Ad: Tensor, Bd: Tensor, Q: Tensor, R: Tensor, Qf: Tensor, N: int) -> List[Tensor]:
    """
    Compute K_k feedback gain matrices for finite horizon discrete-time LQR.
    """
    K_matrices = []
    Pk = Qf.clone()
    
    for k in range(N, 0, -1):
        Fk = torch.linalg.inv(R + Bd.T @ Pk @ Bd) @ Bd.T @ Pk @ Ad
        Pk = Fk.T @ R @ Fk + (Ad - Bd @ Fk).T @ Pk @ (Ad - Bd @ Fk)
        K_matrices.insert(0, Fk)
    
    return K_matrices


def compute_ground_truth_trajectory(
    T: float,
    K: int,
    theta: float,
    x1_init: Tensor,
    x2_init: Tensor,
    tr: float = 0.5,
    prior: float = 0.5, 
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> GroundTruthTrajectory:
    """
    Compute ground truth trajectory using discrete-time Riccati.
    
    Parameters
    ----------
    T : float
        Time horizon
    K : int
        Number of time steps
    theta : float
        Payoff type (-1 or +1)
    x1_init, x2_init : Tensor
        Initial states for P1, P2 (4D each: pos_x, pos_y, vel_x, vel_y)
    tr : float
        Revelation time (default 0.5 for symmetric prior)
    """
    tau = T / float(K)
    
    # Discrete-time dynamics matrices
    A = torch.eye(4, dtype=dtype, device=device)
    A[0, 2] = tau
    A[1, 3] = tau
    
    B = torch.tensor([
        [0.5 * tau**2, 0],
        [0, 0.5 * tau**2],
        [tau, 0],
        [0, tau]
    ], dtype=dtype, device=device)
    
    # Cost matrices (matching HexnerParams)
    Qf = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=dtype, device=device))
    Q = torch.zeros((4, 4), dtype=dtype, device=device)
    R1 = torch.diag(torch.tensor([0.05, 0.025], dtype=dtype, device=device)) * tau
    R2 = torch.diag(torch.tensor([0.05, 0.10], dtype=dtype, device=device)) * tau
    
    # Compute feedback gains
    K1_gains = discrete_lqr(A, B, Q, R1, Qf, K)
    K2_gains = discrete_lqr(A, B, Q, R2, Qf, K)
    
    # Time grid
    t_grid = torch.linspace(0, T, K + 1, dtype=dtype, device=device)
    
    # Allocate trajectories
    x1 = torch.zeros((K + 1, 4), dtype=dtype, device=device)
    x2 = torch.zeros((K + 1, 4), dtype=dtype, device=device)
    u1 = torch.zeros((K, 2), dtype=dtype, device=device)
    u2 = torch.zeros((K, 2), dtype=dtype, device=device)
    
    x1[0] = x1_init.to(dtype=dtype, device=device)
    x2[0] = x2_init.to(dtype=dtype, device=device)
    
    # Forward simulation
    for k in range(K):
        t_k = t_grid[k].item()
        K1_k = K1_gains[k]
        K2_k = K2_gains[k]
        
        # Information control: reveal type after tr
        # prior here is P(θ=-1), so E[θ] = prior*(-1) + (1-prior)*(+1) = 1 - 2*prior
        th = theta if t_k > tr else (1 - 2*prior)
        
        # Target offset based on revealed type
        target = torch.tensor([0.0, th, 0.0, 0.0], dtype=dtype, device=device)
        
        # Controls: u = -K @ (x - target)
        u1_k = -K1_k @ (x1[k] - target)
        u2_k = -K2_k @ (x2[k] - target)
        
        u1[k] = u1_k
        u2[k] = u2_k
        
        # Dynamics
        x1[k + 1] = A @ x1[k] + B @ u1_k
        x2[k + 1] = A @ x2[k] + B @ u2_k
    
    return GroundTruthTrajectory(
        t_grid=t_grid,
        x1_traj=x1,
        x2_traj=x2,
        u1_traj=u1,
        u2_traj=u2,
        theta=theta,
        tr=tr,
        prior=prior,
    )


def compute_gt_cost(gt: GroundTruthTrajectory, theta: float, T: float, K: int) -> float:
    """Compute total cost for a ground truth trajectory."""
    tau = T / K
    dtype = gt.u1_traj.dtype
    device = gt.u1_traj.device
    
    # Control cost matrices (base values, without the 2x factor used in game storage)
    R1 = torch.diag(torch.tensor([0.05, 0.025], dtype=dtype, device=device))
    R2 = torch.diag(torch.tensor([0.05, 0.10], dtype=dtype, device=device))
    
    total_cost = 0.0
    
    # Running cost (control effort)
    # Cost is: tau * (u1^T R1 u1 - u2^T R2 u2) with NO 1/2 factor
    for k in range(K):
        u1 = gt.u1_traj[k]
        u2 = gt.u2_traj[k]
        # P1 minimizes, P2 maximizes
        total_cost += tau * (u1 @ R1 @ u1 - u2 @ R2 @ u2).item()
    
    # Terminal cost (position error, no 1/2 factor)
    target = torch.tensor([0.0, theta, 0.0, 0.0], dtype=dtype, device=device)
    Qf = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=dtype, device=device))
    
    x1_final = gt.x1_traj[-1]
    x2_final = gt.x2_traj[-1]
    
    # g(x) = ||x1 - target||^2_{Qf} - ||x2 - target||^2_{Qf}
    total_cost += ((x1_final - target) @ Qf @ (x1_final - target)).item()
    total_cost -= ((x2_final - target) @ Qf @ (x2_final - target)).item()
    
    return total_cost


# ============================================================================
# Evaluation Data Structures
# ============================================================================

@dataclass
class PriorEvalResult:
    """Container for evaluation results with a specific prior."""
    prior: Tensor
    prior_name: str
    # Errors per type
    position_errors: List[float]  # final position error for each type
    control_errors: List[float]   # control norm difference for each type
    trajectory_errors: List[float]  # state trajectory L2 error for each type
    value_errors: List[float]      # value function error at initial state
    # Aggregate
    mean_position_error: float
    mean_control_error: float
    mean_trajectory_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate signaling policy with different prior distributions. "
            "Compares recomputed feedforward against uniform prior baseline."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint file produced by train_hexner_primal.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/hexner_policy_priors",
        help="Directory in which to save figures and results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for device (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--num-rollouts-per-type",
        type=int,
        default=3,
        help="Number of rollouts to generate for each payoff type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def _rebuild_game_config(meta_dict) -> GameConfig:
    """Reconstruct a GameConfig from checkpoint metadata."""
    cfg = GameConfig()
    for k, v in meta_dict.items():
        setattr(cfg, k, v)
    if not hasattr(cfg, "dx"):
        cfg.dx = cfg.dx1 + cfg.dx2
    if not hasattr(cfg, "tau") and hasattr(cfg, "T") and hasattr(cfg, "K"):
        cfg.tau = cfg.T / cfg.K
    cfg.device_resolved = torch.device(getattr(cfg, "device", "cpu"))
    if not hasattr(cfg, "dtype"):
        cfg.dtype = torch.float32
    return cfg


def extract_alpha_for_online(
    alpha_param: AlphaParam,
    K: int,
    I: int,
) -> Tensor:
    """
    Extract a (K, I, I) signaling policy from AlphaParam.
    
    For online use, we only need the signaling policy at the root node
    for each time step (assuming pooling or uniform signaling).
    """
    device = next(alpha_param.parameters()).device
    dtype = next(alpha_param.parameters()).dtype
    
    # Get alpha from the module - shape is (K, max_nodes, I, I)
    alpha_full = alpha_param()  # Calls forward()
    
    # Get alpha at root node for each time step
    alpha_online = torch.zeros(K, I, I, device=device, dtype=dtype)
    
    for k in range(K):
        # alpha_full[k, 0] is the alpha at root node for depth k
        alpha_online[k] = alpha_full[k, 0]
    
    return alpha_online


def online_mpc_rollout_with_prior(
    game: HexnerGame,
    compact: CompactLQSolution,
    x0: Tensor,
    p0: Tensor,  # Initial prior
    type_idx: int,
    riccati_sol: RiccatiSolution,  # Use riccati solution for kappa
    indexer: FullIaryTreeIndexer,
    alpha_tree: Tensor,  # Full alpha for tree (K, max_nodes, I, I)
    action_space: Optional[BoxActionSpace] = None,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    Run an online policy rollout with a specific initial prior.
    
    Uses the riccati_sol kappa values (properly aggregated) rather than
    computing from compact.r_at() which uses immediate revelation r_type.
    
    Note: Uses alpha_tree for both signal selection and belief updates
    to ensure consistency with the tree-based solution.
    
    Returns
    -------
    x_traj : Tensor (K+1, dx)
    u_traj : Tensor (K, du)
    v_traj : Tensor (K, dv)
    total_cost : float
    """
    K = compact.K
    I = compact.I
    dx = compact.dx
    du = compact.game_params['B1'].shape[1]
    dv = compact.game_params['B2'].shape[1]
    
    device = x0.device
    dtype = x0.dtype
    
    A = compact.game_params['A']
    B1 = compact.game_params['B1']
    B2 = compact.game_params['B2']
    R1_inv = compact.game_params['R1_inv']
    R2_inv = compact.game_params['R2_inv']
    tau = compact.game_params['tau']
    
    R1 = torch.linalg.inv(R1_inv)
    R2 = torch.linalg.inv(R2_inv)
    
    # Storage
    x_traj = torch.zeros(K + 1, dx, device=device, dtype=dtype)
    u_traj = torch.zeros(K, du, device=device, dtype=dtype)
    v_traj = torch.zeros(K, dv, device=device, dtype=dtype)
    
    x_traj[0] = x0
    p = p0.clone()  # Current belief
    total_cost = 0.0
    node_idx = 0  # Track position in tree
    
    for k in range(K):
        x = x_traj[k]
        
        # Get alpha at current node in tree (for consistency)
        alpha_all_types = alpha_tree[k, node_idx]  # (I, I)
        
        # Choose signal for true type (argmax)
        signal_probs = alpha_all_types[type_idx]  # (I,)
        signal = signal_probs.argmax().item()
        
        # Compute edge probabilities: λₐ = Σᵢ pᵢ αᵢₐ
        lambda_edge = torch.einsum('i, ia -> a', p, alpha_all_types)  # (I,)
        
        # Get kappa_u for each possible action from the riccati solution
        kappa_u_all = riccati_sol.kappa_u[k][node_idx]  # (I, du)
        kappa_v_all = riccati_sol.kappa_v[k][node_idx]  # (I, dv)
        
        # Aggregate using lambda_edge (prior probability of each action)
        kappa_u = torch.einsum('a, ad -> d', lambda_edge, kappa_u_all)
        kappa_v = torch.einsum('a, ad -> d', lambda_edge, kappa_v_all)
        
        # Feedback gains
        K_u = compact.K_u[k]
        K_v = compact.K_v[k]
        # Full controls
        u = K_u @ x + kappa_u
        v = K_v @ x + kappa_v
        
        # Clip if needed
        if action_space is not None:
            u = action_space.clip_u(u)
            v = action_space.clip_v(v)
        
        u_traj[k] = u
        v_traj[k] = v
        
        # Running cost
        running_cost = 0.5 * tau * (u @ R1 @ u - v @ R2 @ v)
        total_cost = total_cost + running_cost.item()
        
        # Dynamics
        x_new = A @ x + B1 @ u + B2 @ v
        x_traj[k + 1] = x_new
        
        # Update belief via Bayes rule
        alpha_given_type = alpha_all_types[:, signal]  # (I,)
        lambda_signal = (alpha_given_type * p).sum()
        if lambda_signal > 1e-10:
            p = (alpha_given_type * p) / lambda_signal
        
        # Update node index for next step
        node_idx = node_idx * I + signal
    
    # Terminal cost
    x_final = x_traj[K]
    Q_i = game.Q[type_idx]
    q_i = game.q[type_idx]
    c_i = game.c[type_idx]
    terminal_cost = 0.5 * x_final @ Q_i @ x_final + q_i @ x_final + c_i
    total_cost = total_cost + terminal_cost.item()
    
    return x_traj, u_traj, v_traj, total_cost


def warmstart_mpc_rollout(
    game: HexnerGame,
    compact: CompactLQSolution,
    alpha: Tensor,  # (K, I, I)
    x0: Tensor,
    p0: Tensor,  # Initial prior
    type_idx: int,
    action_space: Optional[BoxActionSpace] = None,
    verbose_convergence: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    Run an online MPC rollout using the compact solution directly.
    
    Key insight: The compact r_type values are already correct for any belief!
    r(k, p) = p @ r_type[k] gives the proper edge-aggregated r value.
    
    No recomputation needed - just use compact.r_at() directly during rollout.
    
    This is the TRUE online approach:
    1. Offline: Store compact solution (P, K_u, K_v, Φ, r_type, c_type)
    2. Online rollout: At each step, compute r_bar = Σₐ λₐ r(k+1, post_a)
                       Then κ = -R⁻¹ Bᵀ r_bar
    
    Complexity: O(K × I² × dx) per rollout
    
    Returns
    -------
    x_traj : Tensor (K+1, dx)
    u_traj : Tensor (K, du)
    v_traj : Tensor (K, dv)
    total_cost : float
    """
    K = compact.K
    I = compact.I
    dx = compact.dx
    du = compact.game_params['B1'].shape[1]
    dv = compact.game_params['B2'].shape[1]
    
    device = x0.device
    dtype = x0.dtype
    
    # Use the original compact solution directly (r_type already encodes correct values)
    # No recompute_for_prior needed!
    
    A = compact.game_params['A']
    B1 = compact.game_params['B1']
    B2 = compact.game_params['B2']
    R1_inv = compact.game_params['R1_inv']
    R2_inv = compact.game_params['R2_inv']
    tau = compact.game_params['tau']
    
    R1 = torch.linalg.inv(R1_inv)
    R2 = torch.linalg.inv(R2_inv)
    
    # Storage
    x_traj = torch.zeros(K + 1, dx, device=device, dtype=dtype)
    u_traj = torch.zeros(K, du, device=device, dtype=dtype)
    v_traj = torch.zeros(K, dv, device=device, dtype=dtype)
    
    x_traj[0] = x0
    p = p0.clone()
    total_cost = 0.0
    
    for k in range(K):
        x = x_traj[k]
        
        # Choose signal for true type (argmax)
        alpha_k = alpha[k]  # (I, I)
        signal_probs = alpha_k[type_idx]
        signal = signal_probs.argmax().item()
        
        # Compute edge probabilities: λₐ = Σᵢ pᵢ αᵢₐ
        lambda_edge = torch.einsum('i, ia -> a', p, alpha_k)  # (I,)
        
        # Get r from the compact solution at next step
        # Aggregate r across children weighted by edge probabilities
        r_bar = torch.zeros(dx, device=device, dtype=dtype)
        for a in range(I):
            if lambda_edge[a] > 1e-10:
                # Posterior belief after signal a
                post_a = (alpha_k[:, a] * p) / lambda_edge[a]
                # r at child using compact solution (r is linear in beliefs)
                r_child_a = compact.r_at(k + 1, post_a)
                r_bar = r_bar + lambda_edge[a] * r_child_a
        
        # Get P at next step (same for all beliefs since P is belief-independent)
        P_plus = compact.P[k + 1]
        
        # Compute feedforward terms using correct formula: κ = -H^{-1} f
        # H = [tau*R + B1^T P B1,    B1^T P B2   ]
        #     [B2^T P B1,           -tau*S + B2^T P B2]
        # f = [B1^T r; B2^T r]
        
        H_uu = tau * R1 + B1.T @ P_plus @ B1
        H_uv = B1.T @ P_plus @ B2
        H_vu = H_uv.T
        H_vv = -tau * R2 + B2.T @ P_plus @ B2
        
        H = torch.zeros(du + dv, du + dv, device=device, dtype=dtype)
        H[:du, :du] = H_uu
        H[:du, du:] = H_uv
        H[du:, :du] = H_vu
        H[du:, du:] = H_vv
        
        f_u = B1.T @ r_bar
        f_v = B2.T @ r_bar
        f = torch.cat([f_u, f_v], dim=0)
        
        # Solve for kappa = -H^{-1} f
        kappa = -torch.linalg.solve(H, f)
        kappa_u = kappa[:du]
        kappa_v = kappa[du:]
        
        # Feedback gains (same as stored)
        K_u = compact.K_u[k]
        K_v = compact.K_v[k]
        
        # Full controls
        u = K_u @ x + kappa_u
        v = K_v @ x + kappa_v
        
        # Clip if needed
        if action_space is not None:
            u = action_space.clip_u(u)
            v = action_space.clip_v(v)
        
        u_traj[k] = u
        v_traj[k] = v
        
        # Running cost
        running_cost = 0.5 * tau * (u @ R1 @ u - v @ R2 @ v)
        total_cost = total_cost + running_cost.item()
        
        # Dynamics
        x_new = A @ x + B1 @ u + B2 @ v
        x_traj[k + 1] = x_new
        
        # Update belief via Bayes rule
        alpha_given_type = alpha_k[:, signal]
        lambda_signal = (alpha_given_type * p).sum()
        if lambda_signal > 1e-10:
            p = (alpha_given_type * p) / lambda_signal
    
    # Terminal cost
    x_final = x_traj[K]
    Q_i = game.Q[type_idx]
    q_i = game.q[type_idx]
    c_i = game.c[type_idx]
    terminal_cost = 0.5 * x_final @ Q_i @ x_final + q_i @ x_final + c_i
    total_cost = total_cost + terminal_cost.item()
    
    return x_traj, u_traj, v_traj, total_cost


def solve_ground_truth_with_prior(
    game: HexnerGame,
    K: int,
    I: int,
    alpha: Tensor,  # (K, I, I) signaling policy OR (K, max_nodes, I, I) full tree
    prior: Tensor,
    indexer: Optional[FullIaryTreeIndexer] = None,
) -> Tuple[BeliefTree, RiccatiSolution, FullIaryTreeIndexer, Tensor]:
    """
    Solve the game with a specific prior distribution.
    
    This creates a new belief tree starting from the given prior
    and solves the Riccati equations.
    
    Parameters
    ----------
    alpha : Tensor
        If shape is (K, I, I), it's the root-only alpha which gets copied to all nodes.
        If shape is (K, max_nodes, I, I), it's the full tree alpha used as-is.
    """
    device = prior.device
    dtype = prior.dtype
    
    # Create indexer if not provided
    if indexer is None:
        indexer = FullIaryTreeIndexer(I=I, K=K)
    
    max_nodes = indexer.max_nodes_per_depth
    
    # Check if alpha is full tree or root-only
    if alpha.ndim == 3:
        # alpha is (K, I, I) - expand to full tree by copying to all nodes
        alpha_tree = torch.zeros(K, max_nodes, I, I, device=device, dtype=dtype)
        for k in range(K):
            num_nodes = indexer.node_count(k)
            for n in range(num_nodes):
                alpha_tree[k, n] = alpha[k]
    else:
        # alpha is already (K, max_nodes, I, I) - use directly
        alpha_tree = alpha
    
    # Build belief tree with custom prior
    belief_tree = build_belief_tree(alpha_tree, prior, indexer)
    
    # Compute averaged costs
    avg_costs = compute_averaged_costs(game, belief_tree)
    
    # Solve Riccati
    riccati_sol = riccati_backward(game, belief_tree, avg_costs)
    
    return belief_tree, riccati_sol, indexer, alpha_tree


def evaluate_prior(
    game: HexnerGame,
    compact: CompactLQSolution,
    alpha: Tensor,  # (K, I, I) or (K, max_nodes, I, I) - signaling policy
    prior: Tensor,
    prior_name: str,
    x0: Tensor,
    action_space: Optional[BoxActionSpace] = None,
    verbose_warmstart: bool = False,
    indexer: Optional[FullIaryTreeIndexer] = None,
) -> Tuple[PriorEvalResult, PriorEvalResult]:
    """
    Evaluate both MPC approaches vs analytical ground truth.
    
    Returns two results:
    1. Riccati-based MPC (solves full tree for each prior)
    2. Warm-started MPC (uses recompute_for_prior from compact solution)
    
    Ground truth uses discrete-time Riccati with fixed revelation time t_r = 0.5.
    """
    K = compact.K
    I = compact.I
    T = game.cfg.T
    device = x0.device
    dtype = x0.dtype
    
    # Solve tree-based Riccati for Riccati-based MPC
    # Pass the full alpha if it's 4D, otherwise it will be expanded
    belief_tree, riccati_sol, indexer_out, alpha_tree = solve_ground_truth_with_prior(
        game, K, I, alpha, prior, indexer=indexer
    )
    
    # Extract root-only alpha (K, I, I) for signal selection in rollouts
    if alpha.ndim == 4:
        alpha_root = alpha[:, 0]  # (K, I, I) - root node at each depth
    else:
        alpha_root = alpha  # Already (K, I, I)
    
    # Extract P1 and P2 initial states from x0
    x1_init = x0[:4]
    x2_init = x0[4:]
    
    theta_vals = [-1.0, 1.0]  # Type 0 -> theta=-1, Type 1 -> theta=+1
    tr = 0.5  # Fixed revelation time for analytical GT
    
    # Results for Riccati-based approach
    pos_errs_riccati = []
    ctrl_errs_riccati = []
    traj_errs_riccati = []
    val_errs_riccati = []
    
    # Results for warm-started approach
    pos_errs_warmstart = []
    ctrl_errs_warmstart = []
    traj_errs_warmstart = []
    val_errs_warmstart = []
    
    for type_idx in range(I):
        theta = theta_vals[type_idx]
        
        # Analytical ground truth
        gt = compute_ground_truth_trajectory(
            T=T, K=K, theta=theta,
            x1_init=x1_init, x2_init=x2_init,
            tr=tr, prior=prior[0], dtype=dtype, device=device
        )
        gt_cost = compute_gt_cost(gt, theta, T, K)
        
        # === Riccati-based MPC (full tree solve) ===
        x_riccati, u_riccati, v_riccati, cost_riccati = online_mpc_rollout_with_prior(
            game, compact, x0, prior, type_idx, 
            riccati_sol, indexer_out, alpha_tree, action_space
        )
        
        # === Warm-started MPC (uses compact.recompute_for_prior) ===
        x_warmstart, u_warmstart, v_warmstart, cost_warmstart = warmstart_mpc_rollout(
            game, compact, alpha_root, x0, prior, type_idx, action_space,
            verbose_convergence=verbose_warmstart
        )
        
        # Compute errors for Riccati-based
        x1_riccati = x_riccati[:, :4]
        x2_riccati = x_riccati[:, 4:]
        
        pos_err_r = ((x1_riccati[-1, :2] - gt.x1_traj[-1, :2]).norm().item() + 
                     (x2_riccati[-1, :2] - gt.x2_traj[-1, :2]).norm().item()) / 2
        ctrl_err_r = (u_riccati - gt.u1_traj).norm().item() + (v_riccati - gt.u2_traj).norm().item()
        traj_err_r = (x1_riccati - gt.x1_traj).norm().item() + (x2_riccati - gt.x2_traj).norm().item()
        val_err_r = abs(cost_riccati - gt_cost)
        
        pos_errs_riccati.append(pos_err_r)
        ctrl_errs_riccati.append(ctrl_err_r)
        traj_errs_riccati.append(traj_err_r)
        val_errs_riccati.append(val_err_r)
        
        # Compute errors for warm-started
        x1_warmstart = x_warmstart[:, :4]
        x2_warmstart = x_warmstart[:, 4:]
        
        pos_err_w = ((x1_warmstart[-1, :2] - gt.x1_traj[-1, :2]).norm().item() + 
                     (x2_warmstart[-1, :2] - gt.x2_traj[-1, :2]).norm().item()) / 2
        ctrl_err_w = (u_warmstart - gt.u1_traj).norm().item() + (v_warmstart - gt.u2_traj).norm().item()
        traj_err_w = (x1_warmstart - gt.x1_traj).norm().item() + (x2_warmstart - gt.x2_traj).norm().item()
        val_err_w = abs(cost_warmstart - gt_cost)
        
        pos_errs_warmstart.append(pos_err_w)
        ctrl_errs_warmstart.append(ctrl_err_w)
        traj_errs_warmstart.append(traj_err_w)
        val_errs_warmstart.append(val_err_w)
    
    result_riccati = PriorEvalResult(
        prior=prior,
        prior_name=prior_name + " (Riccati)",
        position_errors=pos_errs_riccati,
        control_errors=ctrl_errs_riccati,
        trajectory_errors=traj_errs_riccati,
        value_errors=val_errs_riccati,
        mean_position_error=sum(pos_errs_riccati) / len(pos_errs_riccati),
        mean_control_error=sum(ctrl_errs_riccati) / len(ctrl_errs_riccati),
        mean_trajectory_error=sum(traj_errs_riccati) / len(traj_errs_riccati),
    )
    
    result_warmstart = PriorEvalResult(
        prior=prior,
        prior_name=prior_name + " (Warmstart)",
        position_errors=pos_errs_warmstart,
        control_errors=ctrl_errs_warmstart,
        trajectory_errors=traj_errs_warmstart,
        value_errors=val_errs_warmstart,
        mean_position_error=sum(pos_errs_warmstart) / len(pos_errs_warmstart),
        mean_control_error=sum(ctrl_errs_warmstart) / len(ctrl_errs_warmstart),
        mean_trajectory_error=sum(traj_errs_warmstart) / len(traj_errs_warmstart),
    )
    
    return result_riccati, result_warmstart


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint - first just get the metadata
    print(f"Loading checkpoint from {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu")
    meta_dict = payload.get("meta", {})
    
    # Rebuild config
    game_meta = meta_dict.get("game_config", {})
    cfg = _rebuild_game_config(game_meta)
    
    device = torch.device(args.device) if args.device else cfg.device_resolved
    print(f"Using device: {device}")
    
    # Create game
    K = cfg.K
    I = cfg.I
    
    # Use default HexnerParams
    hexner_params = HexnerParams()
    game = HexnerGame(cfg=cfg, params=hexner_params)
    
    # Create action space
    action_space = BoxActionSpace.from_config(cfg)
    
    # Create alpha parameterization and load weights
    indexer = FullIaryTreeIndexer(I=I, K=K)
    alpha_config = AlphaParamConfig()
    alpha_param = AlphaParam(indexer=indexer, alpha_cfg=alpha_config, device=device)
    
    # Load alpha state from checkpoint
    alpha_state = payload.get("alpha_state_dict", {})
    alpha_param.load_state_dict(alpha_state)
    
    # Extract compact solution
    print("\nExtracting compact solution...")
    compact = extract_compact_solution(game, K)
    
    # Get the FULL alpha from the trained model (K, max_nodes, I, I)
    alpha_full = alpha_param()  # Full tree alpha
    print(f"Alpha full shape: {alpha_full.shape}")
    
    # Also extract root-only alpha for compatibility with some functions
    alpha = extract_alpha_for_online(alpha_param, K, I)
    print(f"Alpha online shape: {alpha.shape}")
    
    # Define test priors
    test_priors = [
        (torch.tensor([0.5, 0.5], device=device), "Uniform [0.5, 0.5]"),
        (torch.tensor([0.3, 0.7], device=device), "Skewed [0.3, 0.7]"),
        (torch.tensor([0.7, 0.3], device=device), "Skewed [0.7, 0.3]"),
        (torch.tensor([0.1, 0.9], device=device), "Extreme [0.1, 0.9]"),
        (torch.tensor([0.9, 0.1], device=device), "Extreme [0.9, 0.1]"),
        (torch.tensor([0.2, 0.8], device=device), "Moderate [0.2, 0.8]"),
        (torch.tensor([0.8, 0.2], device=device), "Moderate [0.8, 0.2]"),
    ]
    
    # Default initial state
    x0 = game.default_initial_state().to(device)
    print(f"\nInitial state: {x0}")
    
    # Evaluate each prior
    print("\n" + "="*70)
    print("EVALUATING MPC vs GROUND TRUTH FOR DIFFERENT PRIORS")
    print("="*70)
    print("\nComparing two approaches:")
    print("  1. Riccati-based: Full tree solve for each prior (expensive)")
    print("  2. Warmstart: Use compact.recompute_for_prior (cheap, O(K × I² × dx))")
    
    # Debug: Compare r values for first prior
    debug_prior = test_priors[0][0]  # Uniform prior
    print("\n--- DEBUG: Verifying r linearity ---")
    
    # Solve tree for this prior using FULL alpha (for uniform prior, this is correct)
    belief_tree_debug, riccati_debug, indexer_debug, alpha_tree_debug = solve_ground_truth_with_prior(
        game, cfg.K, cfg.I, alpha_full, debug_prior, indexer=indexer
    )
    
    # Verify r linearity using the existing compact solution
    is_linear, max_err = verify_r_linearity(game, belief_tree_debug, riccati_debug, compact)
    print(f"  r linearity check: max_error = {max_err:.4f}, is_linear = {is_linear}")
    
    # Compare kappa and r_child values at k=3 where divergence occurs
    print("\n  Detailed comparison at k=3 (where kappa diverges):")
    p = debug_prior.clone()
    node_idx = 0
    for step in range(3):
        alpha_step = alpha_full[step, node_idx]  # Use full alpha at current node
        lam = p @ alpha_step
        signal = 0
        if lam[signal] > 1e-10:
            p = (alpha_step[:, signal] * p) / lam[signal]
        node_idx = node_idx * cfg.I + signal
    
    k = 3
    alpha_k = alpha_full[k, node_idx]  # Use full alpha at current node
    lam_k = p @ alpha_k
    print(f"    At k={k}, node_idx={node_idx}, belief={p.tolist()}")
    print(f"    Edge probabilities lambda={lam_k.tolist()}")
    
    # Tree's per-edge kappa
    kappa_u_tree_all = riccati_debug.kappa_u[k][node_idx]  # (I, du)
    print(f"    Tree kappa per edge: {kappa_u_tree_all.tolist()}")
    
    # Look at each edge's r_child
    for a in range(cfg.I):
        child_idx = node_idx * cfg.I + a
        r_tree_child = riccati_debug.r_nodes[k + 1][child_idx]
        belief_child = belief_tree_debug.beliefs[k + 1][child_idx]
        r_compact_child = compact.r_at(k + 1, belief_child)
        
        if lam_k[a] > 1e-10:
            post_a = (alpha_k[:, a] * p) / lam_k[a]
            r_compact_from_post = compact.r_at(k + 1, post_a)
        else:
            post_a = p
            r_compact_from_post = compact.r_at(k + 1, post_a)
        
        print(f"    Edge a={a}: child_idx={child_idx}")
        print(f"      belief_child (tree): {belief_child.tolist()}")
        print(f"      post_a (computed):   {post_a.tolist()}")
        print(f"      r_tree_child: {r_tree_child[:4].tolist()}")
        print(f"      r_compact(belief): {r_compact_child[:4].tolist()}")
        print(f"      r_compact(post_a): {r_compact_from_post[:4].tolist()}")
        
        # Compute kappa from r
        kappa_u_from_r = -compact.game_params['R1_inv'] @ compact.game_params['B1'].T @ r_tree_child
        print(f"      kappa from r_tree: {kappa_u_from_r.tolist()}")
        print(f"      kappa from tree:   {kappa_u_tree_all[a].tolist()}")
    
    print("-" * 70)
    
    results_riccati = []
    results_warmstart = []
    
    for i, (prior, prior_name) in enumerate(test_priors):
        print(f"\n--- {prior_name} ---")
        
        # Show convergence info for first prior only
        verbose_warmstart = (i == 0)
        
        # For uniform prior, use the FULL trained alpha tree
        # For other priors, the trained alpha is an approximation anyway
        is_uniform = (prior - torch.tensor([0.5, 0.5], device=prior.device)).abs().sum() < 0.01
        alpha_to_use = alpha_full if is_uniform else alpha
        
        result_r, result_w = evaluate_prior(
            game, compact, alpha_to_use, prior, prior_name, x0, action_space,
            verbose_warmstart=verbose_warmstart, indexer=indexer
        )
        results_riccati.append(result_r)
        results_warmstart.append(result_w)
        
        print(f"  Riccati-based:")
        print(f"    Position errors: {[f'{e:.4f}' for e in result_r.position_errors]}")
        print(f"    Mean position error: {result_r.mean_position_error:.4f}")
        print(f"  Warmstart:")
        print(f"    Position errors: {[f'{e:.4f}' for e in result_w.position_errors]}")
        print(f"    Mean position error: {result_w.mean_position_error:.4f}")
    
    # Summary table
    print("\n" + "="*90)
    print("SUMMARY: Riccati-based vs Warmstart MPC")
    print("="*90)
    print(f"{'Prior':<25} {'Riccati Pos':<12} {'Warmstart Pos':<14} {'Riccati Traj':<14} {'Warmstart Traj':<14}")
    print("-"*90)
    for r, w in zip(results_riccati, results_warmstart):
        prior_name = r.prior_name.replace(" (Riccati)", "")
        print(f"{prior_name:<25} {r.mean_position_error:<12.4f} {w.mean_position_error:<14.4f} "
              f"{r.mean_trajectory_error:<14.4f} {w.mean_trajectory_error:<14.4f}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "prior_eval_results.txt")
    with open(results_file, 'w') as f:
        f.write("Prior Evaluation Results: Riccati vs Warmstart MPC\n")
        f.write("="*70 + "\n\n")
        f.write("Riccati-based: Full tree solve for each prior\n")
        f.write("Warmstart: compact.recompute_for_prior (O(K × I² × dx))\n\n")
        
        for r, w in zip(results_riccati, results_warmstart):
            prior_name = r.prior_name.replace(" (Riccati)", "")
            f.write(f"{prior_name}\n")
            f.write(f"  Riccati-based:\n")
            f.write(f"    Position errors: {r.position_errors}\n")
            f.write(f"    Trajectory errors: {r.trajectory_errors}\n")
            f.write(f"    Value errors: {r.value_errors}\n")
            f.write(f"  Warmstart:\n")
            f.write(f"    Position errors: {w.position_errors}\n")
            f.write(f"    Trajectory errors: {w.trajectory_errors}\n")
            f.write(f"    Value errors: {w.value_errors}\n\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
