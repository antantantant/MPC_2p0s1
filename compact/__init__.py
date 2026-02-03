# compact/__init__.py
"""
Compact representation for LQ games with belief-linear structure.

This module provides a memory-efficient representation that exploits the fact
that in LQ games with type-dependent terminal costs:

1. The quadratic coefficient P is belief-independent (depends only on time)
2. The linear coefficient r is LINEAR in beliefs: r(p) = Σᵢ pᵢ · r_θᵢ
3. The constant c is also linear in beliefs: c(p) = Σᵢ pᵢ · c_θᵢ

This allows us to store O(K × I) vectors instead of O(I^K) tree nodes.
"""

from .compact_solution import CompactLQSolution, extract_compact_solution
from .online_policy import OnlinePolicy

# Backward compatibility alias
OnlineMPCPolicy = OnlinePolicy

__all__ = [
    "CompactLQSolution",
    "extract_compact_solution",
    "OnlinePolicy",
    "OnlineMPCPolicy",  # deprecated alias
]
