"""Tests for the belief tree construction and signaling parameterisation.

Verifies:
  1. Tree indexer basics (node counts, child/parent)
  2. Belief tree with uniform α → uniform beliefs throughout
  3. AlphaParam output sums to 1 over actions
  4. Belief tree Bayes updates are correct
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.tree.indexing import FullIaryTreeIndexer
from src.tree.belief_tree import BeliefTree, build_belief_tree_vectorized
from src.tree.signaling import AlphaParam, AlphaParamConfig


DTYPE = torch.float64
DEVICE = "cpu"


class TestIndexer:

    def test_node_counts(self):
        idx = FullIaryTreeIndexer(I=2, K=3)
        assert idx.node_count(0) == 1
        assert idx.node_count(1) == 2
        assert idx.node_count(2) == 4
        assert idx.node_count(3) == 8

    def test_child_index(self):
        idx = FullIaryTreeIndexer(I=2, K=3)
        # Root (k=0, node=0), action 0 → child = 0*2+0=0
        assert idx.child_index(0, 0, 0) == 0
        # Root (k=0, node=0), action 1 → child = 0*2+1=1
        assert idx.child_index(0, 0, 1) == 1

    def test_parent_index(self):
        idx = FullIaryTreeIndexer(I=2, K=3)
        assert idx.parent_index(1, 0) == 0
        assert idx.parent_index(1, 1) == 0
        assert idx.parent_index(2, 2) == 1
        assert idx.parent_index(2, 3) == 1

    def test_max_nodes(self):
        idx = FullIaryTreeIndexer(I=3, K=2)
        assert idx.max_nodes_per_depth == 9  # 3^2

    def test_invalid_depth_raises(self):
        idx = FullIaryTreeIndexer(I=2, K=2)
        with pytest.raises(ValueError):
            idx.node_count(3)
        with pytest.raises(ValueError):
            idx.child_index(2, 0, 0)
        with pytest.raises(ValueError):
            idx.parent_index(0, 0)


class TestBeliefTree:

    @pytest.fixture
    def uniform_setup(self):
        I, K = 2, 3
        indexer = FullIaryTreeIndexer(I=I, K=K)
        # Uniform signaling: α[type, action] = 1/I  (no info revelation)
        alpha = torch.ones(K, indexer.max_nodes_per_depth, I, I,
                           dtype=DTYPE) / I
        p0 = torch.tensor([0.5, 0.5], dtype=DTYPE)
        return indexer, alpha, p0

    def test_uniform_beliefs_stay_uniform(self, uniform_setup):
        indexer, alpha, p0 = uniform_setup
        bt = build_belief_tree_vectorized(alpha=alpha, p0=p0, indexer=indexer)

        for k in range(indexer.K + 1):
            beliefs_k = bt.beliefs[k]
            expected = torch.full_like(beliefs_k, 0.5)
            torch.testing.assert_close(beliefs_k, expected, atol=1e-10, rtol=0)

    def test_lambda_node_sum_to_one(self, uniform_setup):
        """At each depth, lambda_node should sum to 1."""
        indexer, alpha, p0 = uniform_setup
        bt = build_belief_tree_vectorized(alpha=alpha, p0=p0, indexer=indexer)

        for k in range(indexer.K + 1):
            total = bt.lambda_node[k].sum()
            assert float(total.item()) == pytest.approx(1.0, abs=1e-10)

    def test_lambda_edge_sums(self, uniform_setup):
        """λ_edge[k] rows should sum to 1 (they are the action probs)."""
        indexer, alpha, p0 = uniform_setup
        bt = build_belief_tree_vectorized(alpha=alpha, p0=p0, indexer=indexer)

        for k in range(indexer.K):
            row_sums = bt.lambda_edge[k].sum(dim=-1)
            torch.testing.assert_close(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-10, rtol=0,
            )

    def test_belief_tree_shapes(self, uniform_setup):
        indexer, alpha, p0 = uniform_setup
        bt = build_belief_tree_vectorized(alpha=alpha, p0=p0, indexer=indexer)

        assert len(bt.beliefs) == indexer.K + 1
        assert len(bt.lambda_node) == indexer.K + 1
        assert len(bt.lambda_edge) == indexer.K

        for k in range(indexer.K + 1):
            N = indexer.node_count(k)
            assert bt.beliefs[k].shape == (N, indexer.I)
            assert bt.lambda_node[k].shape == (N,)

        for k in range(indexer.K):
            N = indexer.node_count(k)
            assert bt.lambda_edge[k].shape == (N, indexer.I)

    def test_informative_signaling_updates_beliefs(self):
        """With informative signaling, beliefs should polarise."""
        I, K = 2, 1
        indexer = FullIaryTreeIndexer(I=I, K=K)
        # Perfectly informative: type 0 → action 0, type 1 → action 1
        alpha = torch.zeros(K, indexer.max_nodes_per_depth, I, I, dtype=DTYPE)
        alpha[0, 0, 0, 0] = 1.0  # type 0 → action 0
        alpha[0, 0, 1, 1] = 1.0  # type 1 → action 1
        p0 = torch.tensor([0.5, 0.5], dtype=DTYPE)

        bt = build_belief_tree_vectorized(alpha=alpha, p0=p0, indexer=indexer)

        # After one step: child 0 should have belief (1, 0)
        #                 child 1 should have belief (0, 1)
        b1 = bt.beliefs[1]  # (2, 2)
        torch.testing.assert_close(
            b1[0], torch.tensor([1.0, 0.0], dtype=DTYPE), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(
            b1[1], torch.tensor([0.0, 1.0], dtype=DTYPE), atol=1e-10, rtol=0
        )


class TestAlphaParam:

    def test_output_sums_to_one(self):
        I, K = 2, 3
        indexer = FullIaryTreeIndexer(I=I, K=K)
        alpha_mod = AlphaParam(indexer, dtype=DTYPE)
        alpha = alpha_mod()

        # Sum over actions (last dim) should be 1
        action_sums = alpha.sum(dim=-1)
        torch.testing.assert_close(
            action_sums,
            torch.ones_like(action_sums),
            atol=1e-10, rtol=0,
        )

    def test_output_shape(self):
        I, K = 3, 2
        indexer = FullIaryTreeIndexer(I=I, K=K)
        alpha_mod = AlphaParam(indexer, dtype=DTYPE)
        alpha = alpha_mod()
        assert alpha.shape == (K, indexer.max_nodes_per_depth, I, I)

    def test_gradients_flow(self):
        I, K = 2, 2
        indexer = FullIaryTreeIndexer(I=I, K=K)
        alpha_mod = AlphaParam(indexer, dtype=DTYPE)
        alpha = alpha_mod()
        loss = alpha.sum()
        loss.backward()
        assert alpha_mod.logits.grad is not None
        assert alpha_mod.logits.grad.shape == alpha_mod.logits.shape
