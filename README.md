# MPC_2p0s1
Real-time strategic planning in two-player zero-sum differential games with one-sided payoff information

This repository implements a solver for two-player zero-sum differential games
with one-sided payoff information (“2p0s1”) and linear–quadratic structure.

It exploits the **atomic NE structure** of 2p0s1 games to collapse game-tree
complexity from `U^(2K)` to `I^K` (primal) and `(I+1)^K` (dual), and provides
a PyTorch-based implementation of the tree-structured Riccati solver plus
outer optimization over belief-splitting parameters `α`. Hexner’s game is used
as a minimal LQ test case.

## Layout

- `config/` – basic configs for the game, training loop, and paths.
- `core/` – linear dynamics, quadratic costs, and action spaces.
- `games/` – game definitions (`HexnerGame` etc.).
- `tree/` – public game tree indexing, belief propagation, averaged
  costs, and tree-structured Riccati recursion.
- `outer_opt/` – α/logit parameterization, primal objective, outer
  optimizer loop, checkpointing.
- `viz/` – plotting utilities and report generation for Hexner.
- `logging/` – metric logging and JSONL writer.
- `scripts/` – command-line entry points for training and evaluation.
- `test/` – small CPU-only smoke tests.

## Local setup (Python virtual environment)

Below is a simple workflow that works on macOS (Intel or Apple Silicon).

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>.git
cd <your-repo-name>