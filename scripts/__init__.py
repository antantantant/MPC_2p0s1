# scripts/__init__.py
"""
Entry-point scripts for running experiments with the 2p0s1 LQ differential game solver.

Current scripts include:
- `train_hexner_primal`: Train the primal solver on Hexner’s game with fixed (x0, p0).
- `eval_hexner_rollouts`: Load a trained α / checkpoint and generate rollouts + figures.

These scripts are intended as minimal, end-to-end examples that:
- run on a laptop (e.g., M1/M2 MacBook),
- scale up to multi-GPU or H100 clusters,
- and reproduce the Hexner experiments described in the accompanying paper.  [oai_citation:0‡16010_Solving_Football_by_Expl-2.pdf](sediment://file_000000001b0871f58cd1c91b483083b4)
"""