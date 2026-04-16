# tree/__init__.py
"""
Tree-structured public-game representation and solvers.

This subpackage provides:
- A full I-ary tree indexer for the primal game (P1 atomic structure).
- Belief and path-mass propagation given belief-splitting parameters α.
- Construction of belief-averaged running and terminal cost data.
- A tree-structured Riccati solver for the inner LQ saddle problem.
- Optional QP backend (debug / validation only).
- Rollout utilities for simulating trajectories under a learned α
  and the corresponding feedback prototypes.

These components implement the tree-level structure described in the
2p0s1 primal reformulation and its atomic equilibrium characterization.  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""

from MPC_2p0s1.tree.indexing import FullIaryTreeIndexer
from MPC_2p0s1.tree.belief_tree import BeliefTree, build_belief_tree
from MPC_2p0s1.tree.averaged_costs import AveragedCostData, compute_averaged_costs
from MPC_2p0s1.tree.riccati_tree import RiccatiSolution, riccati_backward
from MPC_2p0s1.tree.qp_tree_backend import solve_tree_qp_open_loop
from MPC_2p0s1.tree.rollout import rollout_trajectory

__all__ = [
    "FullIaryTreeIndexer",
    "BeliefTree",
    "build_belief_tree",
    "AveragedCostData",
    "compute_averaged_costs",
    "RiccatiSolution",
    "riccati_backward",
    "solve_tree_qp_open_loop",
    "rollout_trajectory",
]