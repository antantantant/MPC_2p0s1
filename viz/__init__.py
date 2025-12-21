# viz/__init__.py
"""
Visualization utilities for the 2p0s1 LQ differential game solver.

This subpackage provides:
- Training progress plots (loss vs. iteration, etc.).
- State, action, and belief trajectory plots in 2D/3D.
- A small report generator tailored to the Hexner test case.

These tools are intended to document and inspect optimization progress and
the structure of the converged solution (e.g., conceal–reveal patterns
in Hexner’s game and football-style examples).  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""

from .plot_training_curves import (
    plot_training_curve,
    plot_multiple_training_curves,
)
from .plot_trajectories import (
    plot_hexner_trajectory_2d,
    plot_hexner_trajectory_3d,
)
from .plot_beliefs import (
    plot_belief_trajectory,
    plot_belief_ensemble,
)
from .make_report_hexner import make_hexner_report

__all__ = [
    "plot_training_curve",
    "plot_multiple_training_curves",
    "plot_hexner_trajectory_2d",
    "plot_hexner_trajectory_3d",
    "plot_belief_trajectory",
    "plot_belief_ensemble",
    "make_hexner_report",
]