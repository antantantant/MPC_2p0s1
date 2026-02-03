# compact/online_policy.py
"""
Online policy using the compact LQ solution for information design control.

This module provides a stateless policy interface that:
1. Takes current state x and belief p
2. Returns optimal control without traversing a tree
3. Updates belief based on signaling decisions

The key insight is that once the signaling policy α is learned, we can:
- Precompute the type-specific (r_θ, c_θ) values
- At runtime, compute r(p) = Σᵢ pᵢ rᵢ in O(I × dx)
- Compute control in O(dx²) via u = K @ x + κ(p)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.compact.compact_solution import CompactLQSolution


@dataclass
class OnlinePolicy:
    """
    Online policy for LQ games with belief-linear structure.
    
    Given the compact solution and current (state, belief), computes
    optimal control without tree traversal.
    
    Attributes
    ----------
    compact : CompactLQSolution
        The precomputed compact solution.
    alpha : Tensor, optional
        Learned signaling policy, shape (K, I, I).
        alpha[k, i, a] = probability that type i sends signal a at step k.
        If None, assumes immediate revelation (separation from the start).
    """
    
    compact: CompactLQSolution
    alpha: Optional[Tensor] = None
    
    def get_control(
        self, 
        x: Tensor, 
        belief: Tensor, 
        k: int,
        type_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute optimal controls for both players.
        
        Parameters
        ----------
        x : Tensor
            Current state, shape (dx,).
        belief : Tensor
            Current public belief, shape (I,).
        k : int
            Current time step (0 to K-1).
        type_idx : int, optional
            If provided, the true type (for P1's signaling decision).
            
        Returns
        -------
        (u, v) : Tuple[Tensor, Tensor]
            Optimal controls for P1 and P2.
        """
        if k >= self.compact.K:
            raise ValueError(f"k={k} >= K={self.compact.K}")
        
        # Feedback terms (belief-independent)
        K_u = self.compact.K_u[k]  # (du, dx)
        K_v = self.compact.K_v[k]  # (dv, dx)
        
        # Feedforward terms (belief-dependent)
        # These depend on the aggregated r from the next step
        # For now, use the belief-averaged r
        r_next = self.compact.r_at(k + 1, belief)  # This is approximate!
        
        # Actually, the correct computation depends on the signaling policy.
        # If we're at belief p at step k, and signaling according to α,
        # then r_bar = Σ_a λ_a r_child(a) where:
        #   λ_a = Σᵢ pᵢ α[k,i,a] (probability of signal a)
        #   r_child(a) = r at child belief after signal a
        
        # For immediate revelation: each type goes to its own child
        # For pooling: all types send the same signal, belief unchanged
        
        # Simplified version: use current belief's r
        # This is correct for:
        #   - After revelation (belief is degenerate)
        #   - Under pooling (belief doesn't change)
        
        B1 = self.compact.game_params['B1']
        B2 = self.compact.game_params['B2']
        P_next = self.compact.P[k + 1]
        
        # Need to solve the local saddle point for feedforward
        # κ = -H⁻¹ Bᵀ r_bar where H is the Hessian
        # Simplified: use precomputed R_inv
        R1_inv = self.compact.game_params['R1_inv']
        R2_inv = self.compact.game_params['R2_inv']
        
        kappa_u = -R1_inv @ B1.T @ r_next
        kappa_v = R2_inv @ B2.T @ r_next
        
        # Full controls
        u = K_u @ x + kappa_u
        v = K_v @ x + kappa_v
        
        return u, v
    
    def get_p1_control(
        self,
        x: Tensor,
        belief: Tensor,
        k: int,
        r_aggregated: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute P1's optimal control.
        
        Parameters
        ----------
        x : Tensor
            Current state, shape (dx,).
        belief : Tensor
            Current public belief, shape (I,).
        k : int
            Current time step.
        r_aggregated : Tensor, optional
            Pre-aggregated r vector (if known from signaling policy).
            If None, uses belief-averaged r.
            
        Returns
        -------
        Tensor
            P1's control u, shape (du,).
        """
        K_u = self.compact.K_u[k]
        
        if r_aggregated is not None:
            r_bar = r_aggregated
        else:
            r_bar = self.compact.r_at(k + 1, belief)
        
        B1 = self.compact.game_params['B1']
        R1_inv = self.compact.game_params['R1_inv']
        
        kappa_u = -R1_inv @ B1.T @ r_bar
        u = K_u @ x + kappa_u
        
        return u
    
    def update_belief(
        self,
        belief: Tensor,
        k: int,
        signal: int,
    ) -> Tensor:
        """
        Update belief after observing a signal.
        
        Uses Bayes' rule:
            p'(i) = α[k, i, signal] × p(i) / λ_signal
            
        where λ_signal = Σᵢ α[k, i, signal] × p(i).
        
        Parameters
        ----------
        belief : Tensor
            Current belief, shape (I,).
        k : int
            Current time step.
        signal : int
            Observed signal (0 to I-1).
            
        Returns
        -------
        Tensor
            Updated belief, shape (I,).
        """
        if self.alpha is None:
            # Immediate revelation: signal = type, so belief becomes degenerate
            new_belief = torch.zeros_like(belief)
            new_belief[signal] = 1.0
            return new_belief
        
        # Bayes update
        alpha_k = self.alpha[k]  # (I, I): [type, signal]
        alpha_given_type = alpha_k[:, signal]  # (I,): α(signal | type)
        
        # Numerator: α(signal | type) × p(type)
        numerator = alpha_given_type * belief
        
        # Denominator: Σᵢ α(signal | i) × p(i)
        lambda_signal = numerator.sum()
        
        # Posterior
        new_belief = numerator / (lambda_signal + 1e-10)
        
        return new_belief
    
    def sample_signal(
        self,
        k: int,
        true_type: int,
    ) -> int:
        """
        Sample a signal according to the signaling policy.
        
        Parameters
        ----------
        k : int
            Current time step.
        true_type : int
            P1's true type (0 to I-1).
            
        Returns
        -------
        int
            Sampled signal (0 to I-1).
        """
        if self.alpha is None:
            # Immediate revelation: signal = type
            return true_type
        
        probs = self.alpha[k, true_type]  # (I,)
        signal = torch.multinomial(probs, 1).item()
        return signal
    
    def rollout(
        self,
        x0: Tensor,
        p0: Tensor,
        true_type: int,
        return_trajectory: bool = True,
    ) -> Tuple[Tensor, Optional[dict]]:
        """
        Execute a full rollout using the online policy.
        
        Parameters
        ----------
        x0 : Tensor
            Initial state, shape (dx,).
        p0 : Tensor
            Initial belief, shape (I,).
        true_type : int
            P1's true type.
        return_trajectory : bool
            If True, return full trajectory info.
            
        Returns
        -------
        (terminal_cost, trajectory) : Tuple
            Terminal cost for P1 and optional trajectory dict.
        """
        K = self.compact.K
        A = self.compact.game_params['A']
        B1 = self.compact.game_params['B1']
        B2 = self.compact.game_params['B2']
        tau = self.compact.game_params['tau']
        
        x = x0.clone()
        p = p0.clone()
        
        trajectory = {
            'x': [x.clone()],
            'u': [],
            'v': [],
            'belief': [p.clone()],
            'signal': [],
        } if return_trajectory else None
        
        total_cost = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for k in range(K):
            # Sample signal and update belief
            signal = self.sample_signal(k, true_type)
            p_new = self.update_belief(p, k, signal)
            
            # Compute controls using the NEW belief (after signal)
            # Important: use the aggregated r based on the prior belief and signaling
            u, v = self.get_control(x, p_new, k)
            
            # Running cost (for true type)
            # cost = ½ uᵀ R u - ½ vᵀ S v
            # (simplified: using averaged R, S)
            R1_inv = self.compact.game_params['R1_inv']
            R1 = torch.linalg.inv(R1_inv)
            R2_inv = self.compact.game_params['R2_inv']
            R2 = torch.linalg.inv(R2_inv)
            running_cost = 0.5 * tau * (u @ R1 @ u - v @ R2 @ v)
            total_cost = total_cost + running_cost
            
            # Dynamics
            x_new = A @ x + B1 @ u + B2 @ v
            
            if trajectory:
                trajectory['u'].append(u.clone())
                trajectory['v'].append(v.clone())
                trajectory['x'].append(x_new.clone())
                trajectory['belief'].append(p_new.clone())
                trajectory['signal'].append(signal)
            
            x = x_new
            p = p_new
        
        # Terminal cost (for true type)
        # This needs the game's terminal cost structure
        # Simplified: use compact representation
        # g_i(x) = ½ xᵀ Q x + qᵢᵀ x + cᵢ
        P_term = self.compact.P[K]
        r_term = self.compact.r_type[K, true_type]
        c_term = self.compact.c_type[K, true_type]
        
        terminal_cost = 0.5 * x @ P_term @ x + r_term @ x + c_term
        total_cost = total_cost + terminal_cost
        
        return total_cost, trajectory


def create_online_policy(
    compact: CompactLQSolution,
    alpha: Optional[Tensor] = None,
) -> OnlinePolicy:
    """
    Create an online policy from a compact solution.
    
    Parameters
    ----------
    compact : CompactLQSolution
        Precomputed compact solution.
    alpha : Tensor, optional
        Learned signaling policy, shape (K, I, I).
        
    Returns
    -------
    OnlinePolicy
        Ready-to-use online policy.
    """
    return OnlinePolicy(compact=compact, alpha=alpha)
