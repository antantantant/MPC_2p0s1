# Hexner 3D Rollout Bundle

This directory is a shareable rollout bundle for the amortized 3D Hexner-mod policy.

The main entry point is:

- `hexner_mod_3d_rollout.py`

It supports:

- one full simulated rollout at a time
- single-step action queries for an externally maintained state
- optional online re-solve of the remaining subtree from the current time step
- timing stats for both control evaluation and online re-solve
- saving rollout JSON, a static 3D trajectory figure, and an interactive 3D animation

The tree topology is fixed by the initial solve. During step queries and online re-solves, the path is identified by the realized prototype history, passed as `--prototypes-so-far`.

## Quick Start

From this directory:

```bash
uv run python hexner_mod_3d_rollout.py \
  --checkpoint checkpoints/latest.pt
```

That samples one context, solves the tree once, and rolls out one game.

## Full Rollout

Example with deterministic prototype selection and saved artifacts:

```bash
uv run python hexner_mod_3d_rollout.py \
  --checkpoint checkpoints/latest.pt \
  --no-sample-actions \
  --output-json /tmp/hexner_rollout.json \
  --save-trajectory-png /tmp/hexner_rollout.png \
  --save-animation-html /tmp/hexner_rollout.html
```

Useful rollout flags:

- `--x0 1.2,-1.9,1.7,0,0,0,-1.6,2.0,0.3,0,0,0`
- `--prior 0.5,0.5`
- `--type-index 0`
- `--no-sample-actions`
- `--action-clip`
- `--online-solve`

## Single-Step Query

Use `--mode query` when you already have the current state and want the next `(u, v)` pair.

Example:

```bash
uv run python hexner_mod_3d_rollout.py \
  --mode query \
  --checkpoint checkpoints/latest.pt \
  --type-index 0 \
  --x-current 1.0,-1.5,1.2,0,0,0,-1.0,1.8,0.6,0,0,0 \
  --belief-current 0.5,0.5 \
  --prototypes-so-far 0,1 \
  --no-sample-actions
```

Important:

- `--prototypes-so-far` means the realized public prototype history, not the actual continuous controls.
- `--time-step` is optional and only acts as a sanity check against the prototype-history length.
- if `--belief-current` is omitted, the script infers belief from the offline root tree and the prototype history

## Online Re-Solve

To re-solve the remaining subtree from the current `(x_t, p_t)`:

```bash
uv run python hexner_mod_3d_rollout.py \
  --mode query \
  --checkpoint checkpoints/latest.pt \
  --type-index 0 \
  --x-current 1.0,-1.5,1.2,0,0,0,-1.0,1.8,0.6,0,0,0 \
  --belief-current 0.5,0.5 \
  --prototypes-so-far 0 \
  --online-solve \
  --online-alpha-iters 0 \
  --no-sample-actions
```

Online-solve controls:

- `--online-alpha-iters`
- `--online-alpha-lr`
- `--online-alpha-early-stop`
- `--online-alpha-min-iters`
- `--online-alpha-patience`
- `--online-alpha-min-delta`
- `--online-skip-solve-at-t0`

Notes:

- `--online-alpha-iters 0` means “rebuild the remaining problem and solve Riccati once from the rebased alpha warm start,” without gradient refinement.
- the reported `online_solve_ms` is separate from `control_compute_ms`, so you can benchmark both pieces independently.
- no noise is added anywhere in this runner

## Python API

You can also import the module and call it from another controller stack:

```python
from pathlib import Path
import torch

from hexner_mod_3d_rollout import (
    OnlineSolveConfig,
    load_policy_runtime,
    query_policy_action,
)

runtime = load_policy_runtime(
    checkpoint_path=Path("checkpoints/latest.pt"),
    device=torch.device("cpu"),
)

step = query_policy_action(
    runtime=runtime,
    type_index=0,
    x_current=runtime.x0,
    belief_current=runtime.p0,
    prototypes_so_far=[],
    sample_actions=False,
    online_solve=OnlineSolveConfig(enabled=False),
)

u = step.u
v = step.v
proto = step.prototype_index
```

Most useful functions:

- `load_setup(...)`
- `build_policy_runtime(...)`
- `load_policy_runtime(...)`
- `query_policy_action(...)`
- `rollout_one_game(...)`
- `save_trajectory_png(...)`
- `save_animation_html(...)`

## Real-Time Loop Pattern

For a real robot/controller stack, the usual pattern is:

1. load the checkpoint and solve the offline root tree once
2. keep track of:
   - current measured state `x_current`
   - current public belief `belief`
   - realized prototype history `prototypes_so_far`
3. call `query_policy_action(...)` every control cycle
4. apply `u` and/or `v`
5. update:
   - `belief = step.next_belief`
   - `prototypes_so_far.append(step.prototype_index)`

Important:

- this tree policy needs the realized prototype history, not just the current belief
- `type_index` is the informed player type row used to read the correct alpha branch
- the one-time expensive setup is `load_policy_runtime(...)`; the per-step call is `query_policy_action(...)`

Example:

```python
from pathlib import Path
import torch

from hexner_mod_3d_rollout import load_policy_runtime, query_policy_action

device = torch.device("cpu")

x0 = torch.tensor(
    [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
    device=device,
)
p0 = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)

runtime = load_policy_runtime(
    checkpoint_path=Path("checkpoints/latest.pt"),
    device=device,
    game_type="original",
    x0=x0,
    p0=p0,
)

belief = p0.clone()
prototypes_so_far = []

for k in range(runtime.setup.game_cfg.K):
    x_current = get_current_state_from_robot()   # torch tensor, shape (12,)

    step = query_policy_action(
        runtime=runtime,
        type_index=0,                  # example: choose the controller row for type 0
        x_current=x_current,
        belief_current=belief,
        prototypes_so_far=prototypes_so_far,
        sample_actions=True,           # or False for deterministic argmax
        use_action_clip=False,
    )

    u = step.u
    v = step.v

    send_control_to_robot(u, v)

    belief = step.next_belief
    prototypes_so_far.append(step.prototype_index)

    print(
        f"step={k}, proto={step.prototype_index}, "
        f"control_ms={step.control_compute_ms:.4f}, "
        f"u={u.tolist()}, v={v.tolist()}"
    )
```

The same call pattern applies to `hexner_mod_3d_rollout_diffmpc.py`; the only difference is that its optional online re-solve path can use the DiffMPC-style backend.

## Timing Fields

The JSON output includes:

- `alpha_eval_ms`
- `belief_tree_ms`
- `avg_costs_ms`
- `riccati_ms`
- `offline_total_ms`
- `control_compute_per_step_ms`
- `online_solve_per_step_ms`

Interpretation:

- `offline_total_ms` is the one-time root solve for the initial context
- `control_compute_ms` is the per-step time to pick the prototype and compute `(u, v)`
- `online_solve_ms` is the extra per-step cost of re-solving the remaining subtree, if enabled

## Checkpoint Support

`--game-type auto` is the default.

The bundle can rebuild both:

- the original target-based 3D game from `hexner_mod_3d_game_original.py`
- the newer coupled-terminal-matrix game from `hexner_mod_3d_game.py`

If auto-detection is ambiguous, pass:

- `--game-type original`
- `--game-type coupled`

## Dependencies

This bundle is configured through `pyproject.toml` and `uv.lock`.

Main runtime dependencies:

- Python `3.12`
- `torch`
- `numpy`
- `matplotlib`
- `plotly`

The easiest way to run it is:

```bash
uv run python hexner_mod_3d_rollout.py --checkpoint checkpoints/latest.pt
```

## Files To Share

If you hand this to another team, the simplest safe option is to zip the whole `rollout_codes/` directory.

That bundle already contains:

- the runner script
- `pyproject.toml`
- `uv.lock`
- the copied `src/MPC_2p0s1/` package subtree
- the example checkpoint under `checkpoints/latest.pt`
