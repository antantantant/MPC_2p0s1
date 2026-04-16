# outer_opt/__init__.py
"""
Outer-loop optimization utilities for belief-splitting parameters (α/logits).

This subpackage provides:
- A logit-based parameterization of α over the public game tree.
- A scalar primal objective (P1’s value with P2 best-responding via Riccati).
- A simple depth-wise scheduling mechanism for staged optimization in time.
- An optimizer loop that ties everything together and supports checkpointing.

These pieces implement the outer optimization layer of the CAMS-style solver
for 2p0s1 games described in the accompanying paper.  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""

from .alpha_param import AlphaParam
from .objective_primal import primal_objective
from .depth_schedule import DepthSchedule, make_default_depth_schedule
from .optimizer_loop import run_outer_optimization
from .checkpointing import save_checkpoint, load_checkpoint

__all__ = [
    "AlphaParam",
    "primal_objective",
    "DepthSchedule",
    "make_default_depth_schedule",
    "run_outer_optimization",
    "save_checkpoint",
    "load_checkpoint",
]