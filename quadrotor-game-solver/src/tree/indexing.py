"""Full I-ary public tree indexing.

Identical to ``nl_sqp/indexing.py`` — the tree structure does not change
when the inner dynamics become nonlinear.

- Depth k has I^k nodes.
- Children of (k, node) are at (k+1, node*I + a).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List


@dataclass(frozen=True)
class FullIaryTreeIndexer:
    I: int
    K: int

    def __post_init__(self) -> None:
        if self.I <= 0:
            raise ValueError(f"I must be positive, got {self.I}")
        if self.K < 0:
            raise ValueError(f"K must be nonnegative, got {self.K}")

        object.__setattr__(
            self, "_num_nodes_per_depth",
            [self.I ** k for k in range(self.K + 1)]
        )
        object.__setattr__(
            self, "_max_nodes", max(self._num_nodes_per_depth)
        )

    @property
    def num_nodes_per_depth(self) -> List[int]:
        return list(self._num_nodes_per_depth)

    @property
    def max_nodes_per_depth(self) -> int:
        return self._max_nodes

    def node_count(self, depth: int) -> int:
        if not (0 <= depth <= self.K):
            raise ValueError(f"depth must be in [0, K={self.K}], got {depth}")
        return self._num_nodes_per_depth[depth]

    def child_index(self, depth: int, node_idx: int, action_index: int) -> int:
        if not (0 <= depth < self.K):
            raise ValueError(
                f"child_index: depth must be in [0, {self.K-1}], got {depth}"
            )
        return node_idx * self.I + action_index

    def parent_index(self, depth: int, node_idx: int) -> int:
        if not (1 <= depth <= self.K):
            raise ValueError(
                f"parent_index: depth must be in [1, {self.K}], got {depth}"
            )
        return node_idx // self.I

    def iter_depth(self, depth: int) -> Iterator[int]:
        return iter(range(self.node_count(depth)))
