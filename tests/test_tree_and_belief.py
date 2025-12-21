# test/test_tree_and_belief.py
from __future__ import annotations

import math

import pytest
import torch

from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.belief_tree import build_belief_tree


def test_tree_indexer_parent_child_inverses():
    I = 3
    K = 3
    indexer = FullIaryTreeIndexer(I=I, K=K)

    # Node counts must be powers of I
    for k in range(K + 1):
        assert indexer.node_count(k) == I**k

    # Parent/child relations must be consistent
    for k in range(K):
        num_nodes = indexer.node_count(k)
        for node_idx in range(num_nodes):
            for a in range(I):
                child_idx = indexer.child_index(k, node_idx, a)
                assert 0 <= child_idx < indexer.node_count(k + 1)
                parent_idx = indexer.parent_index(k + 1, child_idx)
                assert parent_idx == node_idx


def test_belief_tree_uniform_alpha_no_information_revealed():
    """
    With uniform α over prototypes and a uniform prior, beliefs should remain
    constant across the tree and node masses should sum to 1 at each depth.
    """
    I = 2
    K = 2
    indexer = FullIaryTreeIndexer(I=I, K=K)
    max_nodes = indexer.max_nodes_per_depth

    device = torch.device("cpu")
    dtype = torch.float32

    # α[k, node, i, a] = 1/I for all entries.
    alpha = torch.full(
        (K, max_nodes, I, I),
        1.0 / I,
        device=device,
        dtype=dtype,
    )

    p0 = torch.full((I,), 1.0 / I, device=device, dtype=dtype)

    belief_tree = build_belief_tree(alpha=alpha, p0=p0, indexer=indexer)

    # Root checks
    assert belief_tree.beliefs[0].shape == (1, I)
    assert torch.allclose(belief_tree.beliefs[0][0], p0)
    assert torch.allclose(belief_tree.lambda_node[0][0], torch.tensor(1.0, dtype=dtype))

    # Beliefs should stay uniform and node masses should sum to 1 at each depth.
    for k in range(K + 1):
        num_nodes = indexer.node_count(k)
        # beliefs
        b_k = belief_tree.beliefs[k][:num_nodes]
        assert b_k.shape == (num_nodes, I)
        assert torch.allclose(b_k, p0.expand(num_nodes, I))

        # node masses
        lam_k = belief_tree.lambda_node[k][:num_nodes]
        assert torch.allclose(lam_k.sum(), torch.tensor(1.0, dtype=dtype), atol=1e-6)

    # Edge masses at each depth should be uniform over actions.
    for k in range(K):
        num_nodes = indexer.node_count(k)
        lam_edge_k = belief_tree.lambda_edge[k][:num_nodes]  # (num_nodes, I)
        assert torch.allclose(
            lam_edge_k,
            torch.full_like(lam_edge_k, 1.0 / I),
            atol=1e-6,
        )