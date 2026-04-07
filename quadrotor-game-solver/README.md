# Quadrotor Game Solver

This project implements a solver for a **3-D two-player zero-sum differential game** with **asymmetric information** and **nonlinear 6-DoF quadrotor dynamics**. The solver extends the classical 2-D LQ Hexner game (Hexner, 1979) to three spatial dimensions with full quadrotor physics, using an **SQP (Sequential Quadratic Programming)** inner loop with **Riccati saddle-point recursion** and a **backtracking line search**.

## Overview

### Problem Setup

- **Two players**: Player 1 (informed minimizer) and Player 2 (uninformed maximizer)
- **Asymmetric information**: P1 knows the payoff type θ ∈ {θ₁, ..., θᵢ}, P2 knows only the prior distribution
- **Signaling mechanism**: P1's actions can signal information about θ to P2 via a learnable signaling strategy α
- **State space**: Each player controls a 6-DoF quadrotor (12-D per player, 24-D joint state)
- **Control inputs**: 4-D per player (thrust + 3 body-frame torques)
- **Dynamics**: Nonlinear quadrotor dynamics with RK4 integration

### Solution Approach

1. **Outer loop** (gradient-based): Optimize the signaling strategy parameters α using Adam
2. **Inner loop** (SQP with line search): For fixed α, solve the finite-horizon game over a belief tree
   - Linearize dynamics around current trajectory
   - Solve saddle-point LQ game via time-varying Riccati recursion
   - Forward rollout with backtracking line search (try step sizes 0.5, 0.25, 0.125, ...)
   - Repeat until convergence or maximum iterations

See `docs/mathematical_formulation.md` for the complete mathematical details.

## Project Structure

```
quadrotor-game-solver/
├── src/
│   ├── main.py                      # Entry point for the quadrotor game solver
│   ├── game/
│   │   ├── __init__.py
│   │   └── quadrotor_game.py        # QuadrotorGame class: dynamics, linearization, cost functions
│   ├── dynamics/
│   │   ├── __init__.py
│   │   └── quadrotor_model.py       # Nonlinear 6-DoF quadrotor dynamics (RK4 integration)
│   ├── objectives/
│   │   ├── __init__.py
│   │   └── cost_tree.py             # Cost function tree structure (running + terminal costs)
│   ├── tree/
│   │   ├── __init__.py
│   │   ├── belief_tree.py           # Belief propagation tree for asymmetric information
│   │   ├── signaling.py             # Signaling strategy α parameterization (softmax logits)
│   │   └── indexing.py              # Tree indexing utilities for efficient batch operations
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── sqp_tree.py              # SQP solver with backtracking line search
│   │   ├── riccati.py               # Time-varying Riccati recursion for saddle-point LQ games
│   │   └── action_spaces.py         # Control constraints and damping utilities
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── objective_primal_sqp.py  # Outer-loop objective: expected cost over belief tree
│   ├── rollout/
│   │   ├── __init__.py
│   │   └── trajectory.py            # Forward trajectory rollout utilities
│   └── utils/
│       ├── __init__.py
│       ├── linalg.py                # Linear algebra utilities (SVD, matrix ops)
│       ├── visualization.py         # Trajectory and belief visualization tools
│       ├── config.py                # Configuration data structures
│       └── types.py                 # Type definitions and data classes
├── tests/
│   ├── __init__.py
│   ├── test_dynamics.py             # Unit tests for quadrotor dynamics
│   ├── test_game.py                 # Unit tests for game logic
│   ├── test_tree.py                 # Unit tests for belief tree operations
│   ├── test_solvers.py              # Unit tests for Riccati and SQP solvers
│   ├── test_nonlinear_sqp.py        # Integration tests for nonlinear SQP
│   ├── test_sqp_on_lq_game.py       # Tests SQP on linearized (LQ) games
│   ├── test_linearized_mode.py      # Tests for linearized dynamics mode
│   ├── test_with_lq_warmstart.py    # Tests for LQ warmstart strategies
│   ├── test_small_angle_sqp.py      # Tests for small-angle approximations
│   ├── test_conservative_sqp.py     # Tests for conservative SQP settings
│   ├── test_cost_recomputation.py   # Tests for cost function correctness
│   ├── test_rescaled_costs.py       # Tests for rescaled cost functions
│   ├── check_rollout_paths.py       # Diagnostic script for trajectory validation
│   ├── check_alpha_separation.py    # Diagnostic script for signaling strategy
│   └── check_saved_trajectories.py  # Diagnostic script for saved trajectory files
├── scripts/
│   ├── train.py                     # Training script for optimizing signaling strategy α
│   └── evaluate.py                  # Evaluation script for testing trained policies
├── docs/
│   └── mathematical_formulation.md  # Complete mathematical formulation (937 lines)
├── requirements.txt                 # Python dependencies (PyTorch, numpy, etc.)
├── pyproject.toml                   # Modern Python package configuration
├── setup.py                         # Legacy setup script for pip install
└── README.md                        # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+ with CUDA support (optional, for GPU acceleration)

### Install from source

```bash
git clone <repository-url>
cd quadrotor-game-solver
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a signaling strategy α from scratch:

```bash
python scripts/train.py \
  --epochs 100 \
  --sqp-iters 10 \
  --lr 0.01 \
  --run-dir runs/my_experiment
```

**Key training arguments:**

- `--epochs`: Number of outer-loop (gradient) iterations (default: 100)
- `--sqp-iters`: Number of inner SQP iterations per epoch (default: 20, **recommended: 10-15**)
- `--sqp-step-size`: Initial step size for line search (default: 0.5)
- `--riccati-reg`: Regularization for Riccati recursion (default: 0.1)
- `--lr`: Adam learning rate for outer loop (default: 0.01)
- `--run-dir`: Directory to save checkpoints and logs
- `--sqp-verbose`: Print detailed SQP convergence info
- `--sqp-early-stop`: Stop SQP early if convergence criteria met
- `--no-line-search`: Disable backtracking line search (use fixed step size)

See `python scripts/train.py --help` for the full list of options.

### Evaluation

Evaluate a trained policy:

```bash
python scripts/evaluate.py --checkpoint runs/my_experiment/checkpoint_epoch_100.pt
```

### Running Tests

Run all tests with pytest:

```bash
python -m pytest tests/ -v
```

Run specific test suites:

```bash
python -m pytest tests/test_solvers.py -v          # SQP solver tests
python -m pytest tests/test_nonlinear_sqp.py -v    # Nonlinear integration tests
python -m pytest tests/test_dynamics.py -v         # Dynamics tests
```

## Key Features

### 1. Nonlinear 6-DoF Quadrotor Dynamics

- Full 12-dimensional state per player: position, velocity, Euler angles (roll-pitch-yaw), angular rates
- 4-dimensional control: thrust magnitude + body-frame torques
- RK4 (Runge-Kutta 4th order) integration for accurate time-stepping
- Automatic differentiation via `torch.func.jacfwd` for linearization

### 2. Sequential Quadratic Programming (SQP) with Line Search

- Iteratively linearize dynamics around current trajectory
- Solve linearized saddle-point LQ game via time-varying Riccati recursion
- **Backtracking line search**: Try step sizes {0.5, 0.25, 0.125, 0.0625, ...}, pick the best
- Designed for saddle-point problems (not monotone descent)
- **Recommended iteration count: 10-15** (see diagnostic analysis in `docs/mathematical_formulation.md`)

### 3. Belief Tree and Signaling

- P1's signaling strategy α encoded as learnable softmax logits
- Belief tree propagates P2's posterior beliefs via Bayes' rule
- I-ary signaling alphabet (default I=2)
- Tree depth K (default 10 time steps)

### 4. Gradient-Based Outer Optimization

- Differentiate through the entire SQP solve (PyTorch autograd)
- Adam optimizer for signaling parameters
- Checkpointing and logging support

## Performance Notes

- **SQP iterations**: Based on empirical analysis, 10-15 iterations provide a good balance between convergence and wall time. Beyond ~15 iterations, the solution often oscillates or degrades due to the saddle-point structure.
- **Wall time**: ~2 seconds per SQP iteration (CPU, default settings). GPU acceleration available for large batch sizes.
- **Convergence**: The SQP solver may not achieve classical convergence (du_max, dv_max below tolerance) due to saddle-point oscillations. This is expected behavior.

## Mathematical Background

The solver implements the following pipeline:

1. **Outer loop** (epochs 0, 1, 2, ...):
   - Current signaling parameters α
   - Build belief tree with α
   - Solve game via SQP (inner loop) → get expected cost J(α)
   - Compute ∇J(α) via PyTorch autograd
   - Update α ← α - lr · ∇J(α) (Adam)

2. **Inner loop** (SQP iterations 0, 1, 2, ..., N):
   - Current trajectory (x, u, v)
   - Linearize dynamics around (x, u, v) → (A, B₁, B₂, d)
   - Riccati backward pass → feedback gains (K₁, K₂, k)
   - Forward rollout with line search → new (x', u', v')
   - Repeat until convergence or max iterations

For complete mathematical details, see [`docs/mathematical_formulation.md`](docs/mathematical_formulation.md).

## References

- Hexner, G. (1979). "Differential games with incomplete information." PhD thesis.
- PyTorch automatic differentiation: `torch.func.jacfwd`, `torch.autograd`


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.