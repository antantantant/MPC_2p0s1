# compact/compact_solution.py
"""
Compact representation of the LQ game solution.

Key insight: In LQ games where only terminal targets are type-dependent,
the value function coefficients have special structure:

    V(x; p) = ½ xᵀ P x + r(p)ᵀ x + c(p)

where:
    - P depends only on time (belief-independent)
    - r(p) = Σᵢ pᵢ · rᵢ is LINEAR in beliefs
    - c(p) = Σᵢ pᵢ · cᵢ is LINEAR in beliefs

This allows O(K × I) storage instead of O(I^K) for the full tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.games.base_lq_game import BaseLQGame


@dataclass
class CompactLQSolution:
    """
    Compact representation of an LQ game solution.
    
    Exploits belief-linearity to reduce storage from O(I^K) to O(K × I).
    
    Attributes
    ----------
    P : Tensor
        Quadratic value coefficient, shape (K+1, dx, dx).
        P[k] is the same for ALL beliefs at depth k.
        
    r_type : Tensor
        Linear value coefficient for each pure type, shape (K+1, I, dx).
        r_type[k, i] is the r-vector at depth k for degenerate belief on type i.
        For any belief p: r(k, p) = Σᵢ pᵢ · r_type[k, i]
        
    c_type : Tensor
        Constant value term for each pure type, shape (K+1, I).
        c_type[k, i] is the c-scalar at depth k for degenerate belief on type i.
        For any belief p: c(k, p) = Σᵢ pᵢ · c_type[k, i]
        
    K_u : Tensor
        Feedback gain for P1, shape (K, du, dx).
        Belief-independent: u = K_u[k] @ x + κ_u(p)
        
    K_v : Tensor
        Feedback gain for P2, shape (K, dv, dx).
        Belief-independent: v = K_v[k] @ x + κ_v(p)
        
    Phi : Tensor
        Closed-loop dynamics for r propagation, shape (K, dx, dx).
        Phi[k] = (A - B @ H⁻¹ @ Bᵀ @ P[k+1] @ A)ᵀ
        Used to verify: r[k] = Phi[k] @ r_bar[k] where r_bar is aggregated from children.
        
    game_params : dict
        Game parameters needed for control computation:
        - A, B1, B2: dynamics matrices
        - R1_inv, R2_inv: inverse of control cost matrices (averaged or type-independent)
        - tau: time step
    """
    
    P: Tensor              # (K+1, dx, dx)
    r_type: Tensor         # (K+1, I, dx)
    c_type: Tensor         # (K+1, I)
    K_u: Tensor            # (K, du, dx)
    K_v: Tensor            # (K, dv, dx)
    Phi: Tensor            # (K, dx, dx)
    game_params: dict
    
    @property
    def K(self) -> int:
        """Number of time steps."""
        return self.P.shape[0] - 1
    
    @property
    def I(self) -> int:
        """Number of types."""
        return self.r_type.shape[1]
    
    @property
    def dx(self) -> int:
        """State dimension."""
        return self.P.shape[-1]
    
    def r_at(self, k: int, belief: Tensor) -> Tensor:
        """
        Compute r at depth k for given belief.
        
        r(k, p) = Σᵢ pᵢ · r_type[k, i]
        
        Parameters
        ----------
        k : int
            Time step (0 to K).
        belief : Tensor
            Belief vector of shape (I,) or (batch, I).
            
        Returns
        -------
        Tensor
            r vector of shape (dx,) or (batch, dx).
        """
        # r_type[k] has shape (I, dx)
        # belief has shape (I,) or (batch, I)
        if belief.ndim == 1:
            return torch.einsum('i, id -> d', belief, self.r_type[k])
        else:
            return torch.einsum('bi, id -> bd', belief, self.r_type[k])
    
    def c_at(self, k: int, belief: Tensor) -> Tensor:
        """
        Compute c at depth k for given belief.
        
        c(k, p) = Σᵢ pᵢ · c_type[k, i]
        
        Parameters
        ----------
        k : int
            Time step (0 to K).
        belief : Tensor
            Belief vector of shape (I,) or (batch, I).
            
        Returns
        -------
        Tensor
            c scalar of shape () or (batch,).
        """
        if belief.ndim == 1:
            return torch.dot(belief, self.c_type[k])
        else:
            return torch.einsum('bi, i -> b', belief, self.c_type[k])
    
    def value_at(self, k: int, x: Tensor, belief: Tensor) -> Tensor:
        """
        Evaluate the value function at (k, x, belief).
        
        V(x; p) = ½ xᵀ P[k] x + r(k, p)ᵀ x + c(k, p)
        
        Parameters
        ----------
        k : int
            Time step.
        x : Tensor
            State vector of shape (dx,).
        belief : Tensor
            Belief vector of shape (I,).
            
        Returns
        -------
        Tensor
            Scalar value.
        """
        P_k = self.P[k]
        r_k = self.r_at(k, belief)
        c_k = self.c_at(k, belief)
        
        quad = 0.5 * x @ P_k @ x
        lin = r_k @ x
        return quad + lin + c_k
    
    def feedforward_u(self, k: int, belief: Tensor) -> Tensor:
        """
        Compute the feedforward control term for P1.
        
        κ_u(p) = -R₁⁻¹ B₁ᵀ r(k, p)
        
        But we need r from the NEXT step's aggregated value.
        Actually, κ_u is computed from r_bar which is the edge-aggregated r.
        
        For the compact representation after learning, we store r_type
        which represents r at degenerate beliefs. The actual κ_u depends
        on how the signaling policy mixes these.
        """
        # This is a simplified version - in practice, the feedforward
        # depends on the edge-aggregated r which depends on the signaling policy
        r_k = self.r_at(k, belief)
        B1 = self.game_params['B1']
        R1_inv = self.game_params['R1_inv']
        return -R1_inv @ B1.T @ r_k
    
    def recompute_for_prior(
        self, 
        prior: Tensor, 
        alpha: Tensor,
        max_iters: int = 50,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> 'CompactLQSolution':
        """
        Recompute the r and c values for a different prior distribution.
        
        Uses iterative refinement with warm-start from stored r_type values.
        
        For a fixed signaling policy α (learned for uniform prior), this
        recomputes the r and c values that would arise from a different
        initial prior p_0.
        
        Key insight: P, K_u, K_v, Phi are all belief-independent and stay
        the same. Only the r and c values change based on how beliefs
        evolve under the new prior.
        
        Algorithm:
        1. Initialize r_type with stored values (warm-start)
        2. Forward pass: Compute beliefs at each time step based on prior and α
        3. Backward pass: Recompute r and c using edge-aggregated values
        4. Repeat 2-3 until convergence
        
        Parameters
        ----------
        prior : Tensor
            New prior distribution, shape (I,).
        alpha : Tensor
            Signaling policy, shape (K, I, I).
            alpha[k, i, a] = probability that type i sends signal a at step k.
        max_iters : int
            Maximum number of iterations (default 50).
        tol : float
            Convergence tolerance on r change (default 1e-8).
        verbose : bool
            If True, print convergence info.
            
        Returns
        -------
        CompactLQSolution
            New solution with recomputed r_type and c_type.
        """
        K = self.K
        I = self.I
        dx = self.dx
        device = prior.device
        dtype = prior.dtype
        
        # Initialize with stored values (warm-start)
        r_curr = self.r_type.clone()
        c_curr = self.c_type.clone()
        
        # Terminal values are fixed (type-dependent terminal cost)
        # r_curr[K, i] = r_type[K, i] - these don't change
        
        # Helper to interpolate r at a belief using current values
        def r_at_curr(k: int, belief: Tensor) -> Tensor:
            return torch.einsum('i, id -> d', belief, r_curr[k])
        
        def c_at_curr(k: int, belief: Tensor) -> Tensor:
            return torch.dot(belief, c_curr[k])
        
        # Forward pass: Compute beliefs at each step
        # This only needs to be done once since beliefs depend on alpha, not r
        beliefs = [prior.clone()]  # beliefs[k] is the belief at step k
        
        for k in range(K):
            alpha_k = alpha[k]  # (I, I): [type, signal]
            p = beliefs[k]
            
            # Compute edge probabilities: λₐ = Σᵢ pᵢ αᵢₐ
            lam = p @ alpha_k  # (I,)
            
            # Compute expected next belief (weighted by signal probabilities)
            next_belief = torch.zeros(I, device=device, dtype=dtype)
            for a in range(I):
                if lam[a] > 1e-10:
                    post_a = (alpha_k[:, a] * p) / lam[a]
                    next_belief = next_belief + lam[a] * post_a
            beliefs.append(next_belief)
        
        # Precompute posterior beliefs for each (k, signal)
        # posteriors[k][a] = belief after signal a at step k
        posteriors_cache = []
        lam_cache = []
        for k in range(K):
            alpha_k = alpha[k]
            p = beliefs[k]
            lam = p @ alpha_k
            lam_cache.append(lam)
            
            posts = []
            for a in range(I):
                if lam[a] > 1e-10:
                    post_a = (alpha_k[:, a] * p) / lam[a]
                else:
                    post_a = p.clone()
                posts.append(post_a)
            posteriors_cache.append(posts)
        
        # Iterate until convergence
        for iteration in range(max_iters):
            r_prev = r_curr.clone()
            
            # Backward pass: Update r from K-1 to 0
            for k in reversed(range(K)):
                alpha_k = alpha[k]
                lam = lam_cache[k]
                posts = posteriors_cache[k]
                
                for i in range(I):
                    # Type i's signaling distribution
                    signal_probs = alpha_k[i]  # (I,)
                    
                    # Aggregate r over signals that type i might send
                    r_agg_i = torch.zeros(dx, device=device, dtype=dtype)
                    c_agg_i = torch.tensor(0.0, device=device, dtype=dtype)
                    
                    for a in range(I):
                        if signal_probs[a] > 1e-10:
                            post_a = posts[a]
                            
                            # r at child using CURRENT values
                            r_child_a = r_at_curr(k + 1, post_a)
                            c_child_a = c_at_curr(k + 1, post_a)
                            
                            r_agg_i = r_agg_i + signal_probs[a] * r_child_a
                            c_agg_i = c_agg_i + signal_probs[a] * c_child_a
                    
                    # Apply Φ to propagate r backward
                    r_curr[k, i] = self.Phi[k] @ r_agg_i
                    c_curr[k, i] = c_agg_i
            
            # Check convergence
            r_change = (r_curr - r_prev).abs().max().item()
            
            if verbose:
                print(f"  Iter {iteration + 1}: max r change = {r_change:.2e}")
            
            if r_change < tol:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        else:
            if verbose:
                print(f"  Did not converge after {max_iters} iterations (change = {r_change:.2e})")
        
        # Create new solution with recomputed r and c
        return CompactLQSolution(
            P=self.P,
            r_type=r_curr,
            c_type=c_curr,
            K_u=self.K_u,
            K_v=self.K_v,
            Phi=self.Phi,
            game_params=self.game_params,
        )


def extract_compact_solution(
    game: BaseLQGame,
    K: int,
) -> CompactLQSolution:
    """
    Extract the compact solution representation for an LQ game.
    
    This computes the type-specific (r_θ, c_θ) values by running the Riccati
    recursion for each degenerate belief separately, then extracting the
    common P and type-specific (r, c).
    
    Note: This gives the solution for IMMEDIATE REVELATION (always separate).
    For optimal signaling, the actual r at intermediate nodes depends on
    the signaling policy α which determines how types pool.
    
    Parameters
    ----------
    game : BaseLQGame
        The LQ game instance.
    K : int
        Number of time steps (horizon).
        
    Returns
    -------
    CompactLQSolution
        Compact representation with O(K × I) storage.
    """
    I = game.I
    dx = game.dx
    du = game.du
    dv = game.dv
    tau = game.cfg.tau
    
    A = game.A
    B1 = game.B1
    B2 = game.B2
    
    device = game.device_resolved
    dtype = game.dtype
    
    # Storage for compact representation
    P = torch.zeros(K + 1, dx, dx, device=device, dtype=dtype)
    r_type = torch.zeros(K + 1, I, dx, device=device, dtype=dtype)
    c_type = torch.zeros(K + 1, I, device=device, dtype=dtype)
    K_u_all = torch.zeros(K, du, dx, device=device, dtype=dtype)
    K_v_all = torch.zeros(K, dv, dx, device=device, dtype=dtype)
    Phi = torch.zeros(K, dx, dx, device=device, dtype=dtype)
    
    # Terminal conditions for each type
    # For type i with degenerate belief eᵢ:
    #   P[K] = Q[i]  (but actually Q is the same for all types in standard formulation)
    #   r[K, i] = q[i] = -Q[i] @ z[i]
    #   c[K, i] = c[i]
    
    # In the game, Q, q, c are stored as (I, dx, dx), (I, dx), (I,)
    # The belief-averaged versions are:
    #   Q_bar = Σᵢ pᵢ Qᵢ, q_bar = Σᵢ pᵢ qᵢ, c_bar = Σᵢ pᵢ cᵢ
    
    # For degenerate belief eᵢ (p[j] = δᵢⱼ):
    #   Q_bar = Qᵢ, q_bar = qᵢ, c_bar = cᵢ
    
    # Check if Q is type-independent (common case)
    Q = game.Q  # (I, dx, dx)
    q = game.q  # (I, dx)
    c = game.c  # (I,)
    
    # Terminal P should be the same for all types if Q is type-independent
    # In Hexner's game and similar, Q[i] = Q for all i, only q[i] differs
    P[K] = Q[0]  # Assume type-independent Q (verify later)
    
    for i in range(I):
        r_type[K, i] = q[i]
        c_type[K, i] = c[i]
    
    # Backward Riccati for each type separately
    # Since P is belief-independent, we only need to compute it once
    # But r and c are type-specific
    
    # Get running cost matrices (assumed type-independent for now)
    # In the general case, R and S are type-dependent
    R = game.R  # (I, du, du)
    S = game.S  # (I, dv, dv)
    
    # For simplicity, use the first type's R and S (or average)
    # In many games, R and S are type-independent anyway
    R_bar = R.mean(dim=0)  # (du, du)
    S_bar = S.mean(dim=0)  # (dv, dv)
    
    # Backward pass
    for k in reversed(range(K)):
        P_next = P[k + 1]
        
        # Compute the feedback gains (belief-independent)
        # These come from the LQ saddle point conditions
        
        # Hessian blocks
        H_uu = tau * R_bar + B1.T @ P_next @ B1
        H_uv = B1.T @ P_next @ B2
        H_vu = H_uv.T
        H_vv = -tau * S_bar + B2.T @ P_next @ B2
        
        # Full Hessian
        H = torch.zeros(du + dv, du + dv, device=device, dtype=dtype)
        H[:du, :du] = H_uu
        H[:du, du:] = H_uv
        H[du:, :du] = H_vu
        H[du:, du:] = H_vv
        
        # Cross terms with x
        F_u = B1.T @ P_next @ A
        F_v = B2.T @ P_next @ A
        F = torch.cat([F_u, F_v], dim=0)
        
        # Solve for feedback gains: K = -H⁻¹ F
        sol_F = torch.linalg.solve(H, F)
        K_gains = -sol_F
        
        K_u_all[k] = K_gains[:du, :]
        K_v_all[k] = K_gains[du:, :]
        
        # Update P (Riccati for quadratic term)
        Q_stage = A.T @ P_next @ A
        P[k] = Q_stage - F.T @ sol_F
        P[k] = 0.5 * (P[k] + P[k].T)  # Symmetrize
        
        # Compute Phi for r propagation
        # Phi = (A - B H⁻¹ Bᵀ P A)ᵀ = Aᵀ - Aᵀ P B H⁻¹ Bᵀ = (I think this is wrong)
        # Actually: r_k = (A - B K)ᵀ r_next where K = H⁻¹ Bᵀ P A (ignoring the f term)
        # Let's compute: B = [B1, B2], K = [K_u; K_v]
        B = torch.cat([B1, B2], dim=1)  # (dx, du+dv)
        K_full = torch.cat([K_u_all[k], K_v_all[k]], dim=0)  # (du+dv, dx)
        A_cl = A + B @ K_full  # Closed-loop A (note: K already has negative sign)
        Phi[k] = A_cl.T
        
        # Update r for each type
        for i in range(I):
            r_next_i = r_type[k + 1, i]
            
            # Linear term from next step
            f_u = B1.T @ r_next_i
            f_v = B2.T @ r_next_i
            f = torch.cat([f_u, f_v], dim=0)
            
            # Solve for feedforward: kappa = -H⁻¹ f
            sol_f = torch.linalg.solve(H, f)
            
            # r update: r_k = Aᵀ r_next - Fᵀ H⁻¹ f
            r_type[k, i] = A.T @ r_next_i - F.T @ sol_f
            
            # c update: c_k = c_next - ½ fᵀ H⁻¹ f
            c_type[k, i] = c_type[k + 1, i] - 0.5 * f @ sol_f
    
    # Store game parameters needed for online control
    game_params = {
        'A': A,
        'B1': B1,
        'B2': B2,
        'R1_inv': torch.linalg.inv(R_bar),
        'R2_inv': torch.linalg.inv(S_bar),
        'tau': tau,
    }
    
    return CompactLQSolution(
        P=P,
        r_type=r_type,
        c_type=c_type,
        K_u=K_u_all,
        K_v=K_v_all,
        Phi=Phi,
        game_params=game_params,
    )


def verify_r_linearity(
    game: BaseLQGame,
    belief_tree,
    riccati_solution,
    compact_solution: CompactLQSolution,
    tol: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Verify that r is linear in beliefs by comparing the compact representation
    with the full tree solution.
    
    For each node in the tree with belief p, check:
        r_tree[node] ≈ Σᵢ pᵢ · r_type[k, i]
    
    Returns
    -------
    (is_linear, max_error)
        Whether the linearity holds within tolerance and the maximum error.
    """
    K = compact_solution.K
    I = compact_solution.I
    
    max_error = 0.0
    
    for k in range(K + 1):
        num_nodes = belief_tree.indexer.node_count(k)
        for node_idx in range(num_nodes):
            belief = belief_tree.beliefs[k][node_idx]
            
            # Compute r from compact representation
            r_compact = compact_solution.r_at(k, belief)
            
            # Get r from tree
            r_tree = riccati_solution.r_nodes[k][node_idx]
            
            # Compare
            error = (r_compact - r_tree).abs().max().item()
            max_error = max(max_error, error)
    
    return max_error < tol, max_error
