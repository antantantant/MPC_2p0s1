# tree/belief_tree.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from MPC_2p0s1.core.types import Tensor
from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer


@dataclass
class BeliefTree:
    """
    Belief and mass propagation on the public game tree for fixed α.

    Attributes
    ----------
    indexer:
        Tree indexer describing the full I-ary structure.
    beliefs:
        List of length K+1. beliefs[k] has shape (num_nodes_k, I) and
        stores the public belief p_{k,ω} at each node (k, ω).
    lambda_node:
        List of length K+1. lambda_node[k] has shape (num_nodes_k,)
        and stores the total path probability of reaching node (k, ω).
        This is not needed by the Riccati recursion itself but is
        useful for diagnostics and for reconstructing global costs.
    lambda_edge:
        List of length K. lambda_edge[k] has shape (num_nodes_k, I)
        and stores λ_{k,ω}^a, the conditional probability (given node)
        that P1 selects prototype a at node (k, ω).
    """

    indexer: FullIaryTreeIndexer
    beliefs: List[Tensor]
    lambda_node: List[Tensor]
    lambda_edge: List[Tensor]

    @property
    def I(self) -> int:
        return self.beliefs[0].shape[-1]

    @property
    def K(self) -> int:
        # number of steps is depth-1
        return self.indexer.K

    def belief_at(self, depth: int, node_idx: int) -> Tensor:
        return self.beliefs[depth][node_idx]

    def node_mass_at(self, depth: int, node_idx: int) -> Tensor:
        return self.lambda_node[depth][node_idx]

    def edge_mass_at(self, depth: int, node_idx: int, action_index: int) -> Tensor:
        return self.lambda_edge[depth][node_idx, action_index]


def build_belief_tree(
    alpha: Tensor,
    p0: Tensor,
    indexer: FullIaryTreeIndexer,
    eps: float = 1e-8,
) -> BeliefTree:
    """
    Construct the belief tree and path weights from α and the initial prior p0.

    Parameters
    ----------
    alpha:
        Tensor of shape (K, max_nodes_per_depth, I, I) where:
            - alpha[k, node_idx, i, a] is α_{k,ω,i}^a,
            - I is the number of payoff types and prototypes,
            - only the first num_nodes_per_depth[k] entries along node_idx
              are used at each depth; unused entries can be arbitrary.
        Each row alpha[k, node_idx, i, :] is interpreted as a probability
        vector over actions a (after softmax in the α-parameter module).
    p0:
        Initial public belief p0 ∈ Δ(I), tensor of shape (I,).
    indexer:
        Tree indexer defining the depth and node structure.
    eps:
        Small constant used to avoid division by zero when updating beliefs.

    Returns
    -------
    BeliefTree
        Object containing beliefs p_{k,ω}, node masses λ_{k,ω}, and
        edge masses λ_{k,ω}^a.
    """
    K = indexer.K
    I = indexer.I
    max_nodes = indexer.max_nodes_per_depth

    if alpha.ndim != 4 or alpha.shape[0] != K or alpha.shape[1] != max_nodes:
        raise ValueError(
            "build_belief_tree: expected alpha shape "
            f"(K={K}, max_nodes={max_nodes}, I={I}, I={I}), "
            f"got {tuple(alpha.shape)}"
        )
    if alpha.shape[2] != I or alpha.shape[3] != I:
        raise ValueError(
            f"build_belief_tree: alpha's last two dims must be (I={I}, I={I}), "
            f"got {tuple(alpha.shape[2:])}"
        )
    if p0.shape != (I,):
        raise ValueError(
            f"build_belief_tree: p0 must have shape ({I},), got {tuple(p0.shape)}"
        )

    device = alpha.device
    dtype = alpha.dtype

    beliefs: List[Tensor] = []
    lambda_node: List[Tensor] = []
    lambda_edge: List[Tensor] = []

    # Depth 0: root
    beliefs.append(p0.view(1, I).to(device=device, dtype=dtype))
    lambda_node.append(torch.ones(1, device=device, dtype=dtype))

    # Forward pass over depths
    for k in range(K):
        num_nodes = indexer.node_count(k)
        num_nodes_next = indexer.node_count(k + 1)
        if num_nodes_next != num_nodes * I:
            raise RuntimeError(
                "build_belief_tree expects a full I-ary layout with "
                f"num_nodes_next={num_nodes_next} == num_nodes*I={num_nodes * I}"
            )

        b_k = beliefs[k]            # (num_nodes, I)
        lam_node_k = lambda_node[k] # (num_nodes,)

        # Only active nodes at this depth are used.
        alpha_k = alpha[k, :num_nodes]  # (num_nodes, I(type), I(action))

        # Edge probabilities λ_{k,ω}^a = Σ_i α_{k,ω,i}^a p_{k,ω}[i]
        # Shapes: (num_nodes, I)
        lam_edge_k = torch.einsum("ni, nia -> na", b_k, alpha_k)

        # Path mass to children:
        # λ_{k+1,ωa} = λ_{k,ω} * λ_{k,ω}^a
        lam_child = lam_node_k.unsqueeze(-1) * lam_edge_k  # (num_nodes, I)
        # child_index(k, node, a) = node * I + a, which matches row-major flatten.
        lam_node_next = lam_child.reshape(num_nodes_next)

        # Posterior beliefs for each child edge:
        # p_{k+1,ωa}[i] ∝ α_{k,ω,i}^a * p_{k,ω}[i]
        numer = alpha_k * b_k.unsqueeze(-1)        # (num_nodes, I(type), I(action))
        numer = numer.permute(0, 2, 1)             # (num_nodes, I(action), I(type))

        safe_denom = lam_edge_k.clamp_min(eps).unsqueeze(-1)  # (num_nodes, I, 1)
        b_child = numer / safe_denom
        b_child = b_child / b_child.sum(dim=-1, keepdim=True).clamp_min(eps)

        # Degenerate edge: retain parent belief (edge mass is ~0 anyway).
        keep_parent = (lam_edge_k <= eps).unsqueeze(-1)       # (num_nodes, I, 1)
        parent_expanded = b_k.unsqueeze(1).expand(-1, I, -1)  # (num_nodes, I, I(type))
        b_child = torch.where(keep_parent, parent_expanded, b_child)

        # child_index layout matches row-major flatten.
        b_next = b_child.reshape(num_nodes_next, I)

        beliefs.append(b_next)
        lambda_node.append(lam_node_next)
        lambda_edge.append(lam_edge_k)

    return BeliefTree(
        indexer=indexer,
        beliefs=beliefs,
        lambda_node=lambda_node,
        lambda_edge=lambda_edge,
    )
