"""Vectorized belief propagation on a full I-ary public tree.

Identical to ``nl_sqp/belief_tree.py`` — belief dynamics are independent
of the underlying (linear vs nonlinear) dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from .indexing import FullIaryTreeIndexer


@dataclass
class BeliefTree:
    indexer: FullIaryTreeIndexer
    beliefs: List[Tensor]       # len K+1;  beliefs[k]: (I^k, I)
    lambda_node: List[Tensor]   # len K+1;  lambda_node[k]: (I^k,)
    lambda_edge: List[Tensor]   # len K;    lambda_edge[k]: (I^k, I)


def build_belief_tree_vectorized(
    *,
    alpha: Tensor,
    p0: Tensor,
    indexer: FullIaryTreeIndexer,
    eps: float = 1e-10,
) -> BeliefTree:
    """Build beliefs and path masses under Bayes updates.

    Parameters
    ----------
    alpha : (K, max_nodes, I, I)
        alpha[k, node, type, action].  Rows over actions sum to 1.
    p0 : (I,)
        Prior distribution.
    indexer : FullIaryTreeIndexer

    Returns
    -------
    BeliefTree
    """
    K = indexer.K
    I = indexer.I
    max_nodes = indexer.max_nodes_per_depth

    if alpha.shape != (K, max_nodes, I, I):
        raise ValueError(
            f"alpha must have shape ({K}, {max_nodes}, {I}, {I}), "
            f"got {tuple(alpha.shape)}"
        )

    device = alpha.device
    dtype = alpha.dtype

    beliefs: List[Tensor] = []
    lambda_node: List[Tensor] = []
    lambda_edge: List[Tensor] = []

    # Root
    beliefs.append(p0.view(1, I).to(device=device, dtype=dtype))
    lambda_node.append(torch.ones(1, device=device, dtype=dtype))

    for k in range(K):
        N = indexer.node_count(k)
        Nn = indexer.node_count(k + 1)   # == N * I

        b_k = beliefs[k]                 # (N, I)
        lam_node_k = lambda_node[k]      # (N,)
        alpha_k = alpha[k, :N]           # (N, I, I)

        # Edge probabilities  λ_edge[n, a] = Σ_type  p[type] α[type, a]
        lam_edge_k = torch.einsum("ni, nia -> na", b_k, alpha_k)
        lambda_edge.append(lam_edge_k)

        # Child node masses
        lam_node_next = (lam_node_k.unsqueeze(1) * lam_edge_k).reshape(Nn)
        lambda_node.append(lam_node_next)

        # Child beliefs via Bayes rule
        numer = b_k.unsqueeze(-1) * alpha_k              # (N, I, I)
        denom = lam_edge_k.unsqueeze(1).clamp_min(eps)    # (N, 1, I)
        b_child = numer / denom                           # (N, I, I)
        b_next = b_child.permute(0, 2, 1).reshape(Nn, I)
        b_next = b_next / b_next.sum(dim=-1, keepdim=True).clamp_min(eps)
        beliefs.append(b_next)

    return BeliefTree(
        indexer=indexer,
        beliefs=beliefs,
        lambda_node=lambda_node,
        lambda_edge=lambda_edge,
    )
