# logging/__init__.py
"""
Lightweight logging utilities for the 2p0s1 LQ differential game solver.

This subpackage provides:
- A simple in-memory metric logger that can also export to JSONL/CSV.
- A streaming JSONL writer for long-running optimization runs.

These tools are designed to work both on a laptop and at cluster scale, and they
are used to track optimization progress in experiments such as Hexner’s game
and the football case study described in the accompanying paper.  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""

from .metrics import MetricLogger, RunningStat
from .jsonl_writer import JsonlWriter, load_jsonl

__all__ = [
    "MetricLogger",
    "RunningStat",
    "JsonlWriter",
    "load_jsonl",
]