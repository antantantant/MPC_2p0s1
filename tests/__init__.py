# test/__init__.py
"""
Test suite for the 2p0s1 LQ differential game solver.

These tests are intentionally lightweight and CPU-only. They provide smoke
and sanity checks for:

- Tree indexing and belief propagation.
- Belief-averaged costs and the tree-structured Riccati solver.
- The primal outer objective and its gradients w.r.t. α/logits.
- Rollout + basic visualization utilities (using a non-interactive backend).

All tests use very small instances (e.g., Hexner’s game with I=2, K≤3) so
they run quickly on a laptop yet still exercise the core computation paths.
"""