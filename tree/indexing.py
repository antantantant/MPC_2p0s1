# tree/indexing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List

from MPC_2p0s1.core.types import NodeId


@dataclass
class FullIaryTreeIndexer:
    """
    Indexer for a full I-ary public game tree of depth K.

    Nodes are grouped by depth:
        depth k = 0, 1, ..., K
        number of nodes at depth k: I^k

    We assign a 0-based *local* index within each depth:
        node_idx ∈ {0, ..., I^k - 1} at depth k.

    The children of node (k, node_idx) at depth k+1 are:
        child_idx = node_idx * I + a,   a ∈ {0, ..., I-1}.

    This indexing matches the natural identification of a node with
    its base-I representation (the action sequence ω ∈ [I]^k).
    """

    I: int  # branching factor (number of P1 prototypes per node)
    K: int  # horizon in steps

    def __post_init__(self) -> None:
        if self.I <= 0:
            raise ValueError(f"FullIaryTreeIndexer: I must be positive, got {self.I}")
        if self.K < 0:
            raise ValueError(f"FullIaryTreeIndexer: K must be nonnegative, got {self.K}")

        self._num_nodes_per_depth: List[int] = [self.I**k for k in range(self.K + 1)]
        self._max_nodes: int = max(self._num_nodes_per_depth)

    # ------------------------------------------------------------------ #
    # Basic properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def num_depths(self) -> int:
        """Number of depths including root; equals K + 1."""
        return self.K + 1

    @property
    def num_nodes_per_depth(self) -> List[int]:
        """List of node counts at each depth 0..K."""
        return list(self._num_nodes_per_depth)

    @property
    def max_nodes_per_depth(self) -> int:
        """
        Maximum number of nodes on any depth.

        Useful for allocating depth-major tensors with a fixed second
        dimension, masking out unused entries at shallow depths.
        """
        return self._max_nodes

    def node_count(self, depth: int) -> int:
        """Number of nodes at a given depth."""
        if not (0 <= depth <= self.K):
            raise ValueError(f"depth must be in [0, K={self.K}], got {depth}")
        return self._num_nodes_per_depth[depth]

    # ------------------------------------------------------------------ #
    # Parent / children relationships                                    #
    # ------------------------------------------------------------------ #

    def child_index(self, depth: int, node_idx: NodeId, action_index: int) -> NodeId:
        """
        Local index of the child at depth+1 for a given node and action.

        Parameters
        ----------
        depth:
            Current depth (0 ≤ depth < K).
        node_idx:
            Local node index at this depth.
        action_index:
            Prototype index a ∈ {0, ..., I-1}.

        Returns
        -------
        int
            Local index of the child node at depth+1.

        Notes
        -----
        This assumes a *full* I-ary tree. If in later applications
        you prune parts of the tree, you should provide a subclass
        that overrides this logic accordingly.
        """
        if not (0 <= depth < self.K):
            raise ValueError(f"child_index: depth must be in [0, {self.K-1}], got {depth}")
        if not (0 <= node_idx < self.node_count(depth)):
            raise ValueError(
                f"child_index: node_idx must be in [0, {self.node_count(depth)-1}], "
                f"got {node_idx}"
            )
        if not (0 <= action_index < self.I):
            raise ValueError(
                f"child_index: action_index must be in [0, {self.I-1}], got {action_index}"
            )
        return node_idx * self.I + action_index

    def parent_index(self, depth: int, node_idx: NodeId) -> NodeId:
        """
        Local index of the parent node at depth-1.

        Parameters
        ----------
        depth:
            Current depth (1 ≤ depth ≤ K).
        node_idx:
            Local node index at this depth.

        Returns
        -------
        int
            Local index of the parent node at depth-1.
        """
        if not (1 <= depth <= self.K):
            raise ValueError(f"parent_index: depth must be in [1, {self.K}], got {depth}")
        if not (0 <= node_idx < self.node_count(depth)):
            raise ValueError(
                f"parent_index: node_idx must be in [0, {self.node_count(depth)-1}], "
                f"got {node_idx}"
            )
        return node_idx // self.I

    # ------------------------------------------------------------------ #
    # Convenience iterators                                              #
    # ------------------------------------------------------------------ #

    def iter_depth(self, depth: int) -> Iterator[NodeId]:
        """Iterate over all local node indices at a given depth."""
        return iter(range(self.node_count(depth)))

    def all_nodes(self) -> Iterable[tuple[int, NodeId]]:
        """Iterate over (depth, node_idx) pairs for all nodes in the tree."""
        for k in range(self.num_depths):
            for node_idx in self.iter_depth(k):
                yield (k, node_idx)