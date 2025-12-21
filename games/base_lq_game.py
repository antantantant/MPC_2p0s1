# games/base_lq_game.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn

from ..config.base_config import GameConfig
from ..core.types import Tensor, ValueQuad


class BaseLQGame(nn.Module, ABC):
    """
    Abstract base class for 2p0s1 linear–quadratic differential games.

    This class encapsulates the *fixed* game data:
    - Linear discrete-time dynamics:
        x_{k+1} = A x_k + B1 u_k + B2 v_k
      where x ∈ R^{dx}, u ∈ R^{du} (P1), v ∈ R^{dv} (P2).

    - Type-dependent quadratic running costs:
        ℓ_i(u, v) = 0.5 u^T R_i u - 0.5 v^T S_i v,

    - Type-dependent quadratic terminal costs:
        g_i(x) = 0.5 x^T Q_i x + q_i^T x + c_i.

    The solver only ever interacts with this interface; concrete games such as
    Hexner’s game provide particular A, B1, B2, {R_i, S_i, Q_i, q_i, c_i}.

    All tensors registered here are *buffers* (non-trainable); learnable
    parameters live in the outer α/logits modules.
    """

    def __init__(self, cfg: GameConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.I: int = cfg.I
        self.dx1: int = cfg.dx1
        self.dx2: int = cfg.dx2
        self.dx: int = cfg.dx
        self.du: int = cfg.du
        self.dv: int = cfg.dv
        self.dtype = cfg.dtype
        self.device_resolved = cfg.device_resolved

        # Concrete subclasses must set:
        #   self.A:  (dx, dx)
        #   self.B1: (dx, du)
        #   self.B2: (dx, dv)
        #   self.R:  (I, du, du)
        #   self.S:  (I, dv, dv)
        #   self.Q:  (I, dx, dx)
        #   self.q:  (I, dx)
        #   self.c:  (I,)
        #
        # via register_buffer(...) in their own __init__.

    # --------------------------------------------------------------------- #
    # Dynamics                                                              #
    # --------------------------------------------------------------------- #

    @property
    def A(self) -> Tensor:
        return self._A

    @property
    def B1(self) -> Tensor:
        return self._B1

    @property
    def B2(self) -> Tensor:
        return self._B2

    def step_dynamics(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        One-step discrete-time dynamics:

            x_{k+1} = A x_k + B1 u_k + B2 v_k.

        Parameters
        ----------
        x:
            State tensor of shape (..., dx).
        u:
            P1 control of shape (..., du).
        v:
            P2 control of shape (..., dv).

        Returns
        -------
        Tensor
            Next state with shape (..., dx).
        """
        if x.shape[-1] != self.dx:
            raise ValueError(f"step_dynamics: x last dim {x.shape[-1]} != dx={self.dx}")
        if u.shape[-1] != self.du:
            raise ValueError(f"step_dynamics: u last dim {u.shape[-1]} != du={self.du}")
        if v.shape[-1] != self.dv:
            raise ValueError(f"step_dynamics: v last dim {v.shape[-1]} != dv={self.dv}")

        # Broadcasting over leading dimensions is handled by torch.matmul:
        Ax = torch.matmul(x, self.A.T)          # (..., dx)
        B1u = torch.matmul(u, self.B1.T)        # (..., dx)
        B2v = torch.matmul(v, self.B2.T)        # (..., dx)
        return Ax + B1u + B2v

    # --------------------------------------------------------------------- #
    # Running cost                                                          #
    # --------------------------------------------------------------------- #

    @property
    def R(self) -> Tensor:
        """
        Type-dependent running-cost matrices for P1.

        Shape: (I, du, du), with R[i] ≻ 0.
        """
        return self._R

    @property
    def S(self) -> Tensor:
        """
        Type-dependent running-cost matrices for P2.

        Shape: (I, dv, dv), with S[i] ≻ 0.
        """
        return self._S

    def running_cost_mats(self, belief: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute belief-averaged running-cost matrices (R̄, S̄).

        For a belief p ∈ Δ(I),
            R̄ = Σ_i p[i] R_i,
            S̄ = Σ_i p[i] S_i.

        Parameters
        ----------
        belief:
            Tensor of shape (..., I) representing p.

        Returns
        -------
        (R_bar, S_bar):
            - R_bar: (..., du, du)
            - S_bar: (..., dv, dv)
        """
        if belief.shape[-1] != self.I:
            raise ValueError(
                f"running_cost_mats: belief last dim {belief.shape[-1]} != I={self.I}"
            )

        # (..., I) × (I, du, du) -> (..., du, du)
        R_bar = torch.einsum("...i, iab -> ...ab", belief, self.R)
        S_bar = torch.einsum("...i, iab -> ...ab", belief, self.S)
        return R_bar, S_bar

    # --------------------------------------------------------------------- #
    # Terminal cost                                                         #
    # --------------------------------------------------------------------- #

    @property
    def Q(self) -> Tensor:
        """Type-dependent terminal quadratic matrices: Q[i] ∈ R^{dx×dx}."""
        return self._Q

    @property
    def q(self) -> Tensor:
        """Type-dependent terminal linear coefficients: q[i] ∈ R^{dx}."""
        return self._q

    @property
    def c(self) -> Tensor:
        """Type-dependent terminal constants: c[i] ∈ R."""
        return self._c

    def terminal_cost_type(self, i: int, x: Tensor) -> Tensor:
        """
        Terminal cost g_i(x) for a specific type i.

        Parameters
        ----------
        i:
            Type index in {0, ..., I-1}.
        x:
            State tensor of shape (..., dx).

        Returns
        -------
        Tensor
            Scalar cost tensor of shape (...,).
        """
        if not (0 <= i < self.I):
            raise IndexError(f"type index i={i} out of range [0, {self.I})")
        if x.shape[-1] != self.dx:
            raise ValueError(f"terminal_cost_type: x last dim {x.shape[-1]} != dx={self.dx}")

        Qi = self.Q[i]  # (dx, dx)
        qi = self.q[i]  # (dx,)
        ci = self.c[i]  # ()

        Px = torch.matmul(x, Qi)           # (..., dx)
        quad = 0.5 * (Px * x).sum(dim=-1)  # (...)
        lin = torch.matmul(x, qi)          # (...)
        return quad + lin + ci

    def terminal_value_quad(self, belief: Tensor, node_mass: Tensor) -> ValueQuad:
        """
        Belief- and mass-weighted terminal value quad for a leaf node.

        For belief p and node mass λ, we consider

            E_{i∼p}[g_i(x)] = 0.5 x^T Q̄ x + q̄^T x + c̄
            where
                Q̄ = Σ_i p[i] Q_i,
                q̄ = Σ_i p[i] q_i,
                c̄ = Σ_i p[i] c_i.

        The tree recursion uses λ E[g_i(x)], so the returned ValueQuad encodes

            V(x) = λ E_{i∼p}[g_i(x)].

        Parameters
        ----------
        belief:
            Tensor of shape (I,) or (..., I) representing the belief at the leaf.
            For now we expect a single belief vector (I,).
        node_mass:
            Scalar tensor λ ∈ [0, 1] representing the total probability mass of
            reaching this leaf.

        Returns
        -------
        ValueQuad
            Quadratic representation V(x) = 0.5 x^T P x + r^T x + c.
        """
        if belief.ndim != 1 or belief.shape[0] != self.I:
            raise ValueError(
                f"terminal_value_quad: belief must be 1D of length I={self.I}, "
                f"got shape {tuple(belief.shape)}"
            )

        # Q̄, q̄, c̄
        Q_bar = torch.einsum("i, iab -> ab", belief, self.Q)  # (dx, dx)
        q_bar = torch.einsum("i, ia -> a", belief, self.q)    # (dx,)
        c_bar = torch.dot(belief, self.c)                    # ()

        lam = node_mass
        P = lam * Q_bar
        r = lam * q_bar
        c = lam * c_bar
        return ValueQuad(P=P, r=r, c=c)

    # --------------------------------------------------------------------- #
    # Abstract hooks for subclasses                                         #
    # --------------------------------------------------------------------- #

    @abstractmethod
    def default_initial_state(self) -> Tensor:
        """
        Optional convenience method: provide a canonical initial state x0.

        Concrete games (e.g., Hexner) implement this for quick tests and
        examples. Scripts are free to ignore it and supply their own x0.
        """
        raise NotImplementedError

    @abstractmethod
    def default_prior(self) -> Tensor:
        """
        Optional convenience method: provide a canonical prior p0 ∈ Δ(I).

        Concrete games (e.g., Hexner) implement this as a uniform prior, or
        something game-specific.
        """
        raise NotImplementedError