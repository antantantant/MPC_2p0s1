# Drone3D Variant-2 Rollout Bundle

This README is for the direct-solve Drone3D Variant-2 runner:

- `drone3d_variant2_rollout.py`

It is the direct-solve counterpart to the amortized rollout bundle. Instead of loading a neural checkpoint, it builds the Drone3D Variant-2 game directly and solves it with the vendored JAX solver.

It supports:

- one full simulated rollout at a time
- single-step control queries for an externally maintained state
- optional online re-solve of the remaining horizon
- fixed-shape padded remaining-horizon online solves for the unconstrained exact-LQ path
- separate solver budgets for the root solve and the online re-solves
- saved JSON, static trajectory PNG, and interactive animation HTML
- timing stats for the initial solve and the per-step control queries

## Current Behavior

The runner currently uses a hybrid control flow:

1. `load_policy_runtime(...)` solves the full root game once for the initial `(x0, p0)`
2. `query_policy_action(...)` returns the current control from that solved policy
3. if online solve is enabled, it re-solves from the current state and belief

For unconstrained exact-LQ games, online solve defaults to a fixed-shape padded remaining-horizon solve. The high-quality root solve still happens once first. The online solve keeps the elapsed public history and true remaining horizon, while padding the graph shape so repeated JAX calls reuse the same compiled shape.

So the current modes are:

- offline-root only:
  - do the root solve once
  - use that solved tree for every step
- offline first, online afterward:
  - do the root solve once
  - precompile the online solver once when supported
  - use the root solve directly at step 0
  - re-solve from step 1 onward
- online every step including step 0:
  - do the root solve once
  - precompile the online solver once when supported
  - then also re-solve again at step 0 and later steps

The pure “no offline root solve at all, solve from scratch every control cycle” version is not what this file does today.

## Real-Time Deployment Status

For a controller running at `dt = 0.3 s`, the currently recommended deployment mode is:

- solve the root game once when the initial states are available
- execute the solved affine feedback law online with `--no-online-solve`
- update the measured state each control cycle and evaluate `u = K_u x + k_u`, `v = K_v x + k_v`

This is still closed-loop state feedback, not nominal open-loop playback. The selected public edge gives the affine law, and the measured current state is used directly in the controller.

The correct padded online re-solve mode is policy-equivalent to the root-tree rollout, but it is not yet real-time for `dt = 0.3 s` in the current Python/JAX implementation. In the T10 unconstrained example, padded online solves have been observed around `1.3-1.6 s` per control step.

The experimental full-horizon receding mode can be much faster, around `60 ms` per online solve in the same example, but it solves a different online problem. It restarts a fresh full-horizon public game at every step, so it can choose the wrong public branch and should not be used as the policy-equivalent online controller for this T10 signaling example.

The next optimization target for real-time online re-solving is a dedicated fixed-shape JAX function that caches all tree/problem constants, avoids rebuilding the problem each step, and returns only the current root action/edge feedback instead of reconstructing the full policy and belief tree.

## Quick Start

From `rollout_codes/`:

```bash
python drone3d_variant2_rollout.py \
  --mode rollout \
  --total-steps 10 \
  --reveal-step 10 \
  --no-forced-reveal \
  --unconstrained \
  --p1-init-pos 0,0,1 \
  --p2-init-pos=-1,0,1 \
  --target0-pos 5,1,0 \
  --target1-pos 5,-1,0 \
  --running-r 0.05,0.025,0.05 \
  --running-s 0.05,0.10,0.05 \
  --terminal-type-diags '20,20,20,0,0,0;20,20,20,0,0,0' \
  --inner-iters 24 \
  --outer-steps 200 \
  --outer-min-steps 4 \
  --outer-lr 0.03 \
  --outer-alpha-clip 8 \
  --seed 123 \
  --type-index 0 \
  --no-sample-actions
```

That builds the problem, solves the root game once, and rolls out one trajectory.

## Rollout Modes

### 1. Offline Root Solve Only

```bash
python drone3d_variant2_rollout.py \
  --mode rollout \
  --output-json out/drone3d_variant2_rollout_one_solve.json \
  --save-animation-html out/drone3d_variant2_rollout_one_solve.html \
  --save-trajectory-png out/drone3d_variant2_rollout_one_solve.png \
  --figure-title "Drone3D Variant-2 one solve" \
  --total-steps 10 \
  --reveal-step 10 \
  --no-forced-reveal \
  --unconstrained \
  --p1-init-pos 0,0,1 \
  --p2-init-pos=-1,0,1 \
  --target0-pos 5,1,0 \
  --target1-pos 5,-1,0 \
  --running-r 0.05,0.025,0.05 \
  --running-s 0.05,0.10,0.05 \
  --terminal-type-diags '20,20,20,0,0,0;20,20,20,0,0,0' \
  --inner-iters 24 \
  --verbose-history \
  --outer-steps 200 \
  --outer-min-steps 4 \
  --outer-lr 0.03 \
  --outer-alpha-clip 8 \
  --seed 123 \
  --type-index 0 \
  --no-sample-actions
```

### 2. Offline First, Online After Step 0

This is often the most practical mode if you want the controller to “think” once before the first move, then re-solve afterward.

```bash
python drone3d_variant2_rollout.py \
  --mode rollout \
  --output-json out/drone3d_variant2_rollout_offline_then_online.json \
  --save-animation-html out/drone3d_variant2_rollout_offline_then_online.html \
  --save-trajectory-png out/drone3d_variant2_rollout_offline_then_online.png \
  --figure-title "Drone3D Variant-2 offline first, online afterward" \
  --total-steps 10 \
  --reveal-step 10 \
  --no-forced-reveal \
  --unconstrained \
  --p1-init-pos 0,0,1 \
  --p2-init-pos=-1,0,1 \
  --target0-pos 5,1,0 \
  --target1-pos 5,-1,0 \
  --running-r 0.05,0.025,0.05 \
  --running-s 0.05,0.10,0.05 \
  --terminal-type-diags '20,20,20,0,0,0;20,20,20,0,0,0' \
  --inner-iters 24 \
  --verbose-history \
  --outer-steps 200 \
  --outer-min-steps 4 \
  --outer-lr 0.03 \
  --outer-alpha-clip 8 \
  --seed 123 \
  --type-index 0 \
  --no-sample-actions \
  --online-solve
```

Because `--online-skip-solve-at-t0` defaults to `True`, step 0 uses the root solve and re-solving starts at step 1.

If the online re-solves are too slow, you can give them a smaller budget than the root solve:

```bash
python drone3d_variant2_rollout.py \
  --mode rollout \
  --output-json out/drone3d_variant2_rollout_offline_then_online_fast.json \
  --save-animation-html out/drone3d_variant2_rollout_offline_then_online_fast.html \
  --save-trajectory-png out/drone3d_variant2_rollout_offline_then_online_fast.png \
  --figure-title "Drone3D Variant-2 offline first, lighter online solves" \
  --total-steps 10 \
  --reveal-step 10 \
  --no-forced-reveal \
  --unconstrained \
  --p1-init-pos 0,0,1 \
  --p2-init-pos=-1,0,1 \
  --target0-pos 5,1,0 \
  --target1-pos 5,-1,0 \
  --running-r 0.05,0.025,0.05 \
  --running-s 0.05,0.10,0.05 \
  --terminal-type-diags '20,20,20,0,0,0;20,20,20,0,0,0' \
  --inner-iters 24 \
  --verbose-history \
  --outer-steps 200 \
  --outer-min-steps 4 \
  --outer-lr 0.03 \
  --outer-alpha-clip 8 \
  --seed 123 \
  --type-index 0 \
  --no-sample-actions \
  --online-solve \
  --online-precompile-fixed-shape \
  --no-online-full-horizon-receding \
  --online-outer-steps 2 \
  --online-outer-min-steps 2 \
  --online-inner-iters 8 \
  --online-grad-tolerance 1e-3 \
  --online-loss-change-tolerance 1e-4
```

These extra online-only controls are:

- `--online-outer-steps`
- `--online-outer-min-steps`
- `--online-inner-iters`
- `--online-grad-tolerance`
- `--online-loss-change-tolerance`
- `--online-full-horizon-receding` / `--no-online-full-horizon-receding`
- `--online-fixed-shape-padding` / `--no-online-fixed-shape-padding`
- `--online-precompile-fixed-shape` / `--no-online-precompile-fixed-shape`

`--online-fixed-shape-padding` is enabled by default for unconstrained exact-LQ problems. It solves from the measured current state and belief, but keeps the true remaining mixed/tail depths active so the online policy stays consistent with the already-observed public branch. Box-constrained problems fall back to the older shrinking remaining-horizon solve.

`--online-full-horizon-receding` is available as an experimental speed path, but it is not the recommended policy-equivalent mode for the T10 signaling example because it restarts a fresh full-horizon public game at each step.

`--online-precompile-fixed-shape` is also enabled by default. Despite the older flag name, it now precompiles the supported online graph, records the setup cost as `online_precompile_ms`, and keeps that compile cost out of the first online control query.

### 3. Re-Solve at Every Step Including Step 0

```bash
python drone3d_variant2_rollout.py \
  --mode rollout \
  --output-json out/drone3d_variant2_rollout_resolve_each_step.json \
  --save-animation-html out/drone3d_variant2_rollout_resolve_each_step.html \
  --save-trajectory-png out/drone3d_variant2_rollout_resolve_each_step.png \
  --figure-title "Drone3D Variant-2 re-solve each step" \
  --total-steps 10 \
  --reveal-step 10 \
  --no-forced-reveal \
  --unconstrained \
  --p1-init-pos 0,0,1 \
  --p2-init-pos=-1,0,1 \
  --target0-pos 5,1,0 \
  --target1-pos 5,-1,0 \
  --running-r 0.05,0.025,0.05 \
  --running-s 0.05,0.10,0.05 \
  --terminal-type-diags '20,20,20,0,0,0;20,20,20,0,0,0' \
  --inner-iters 24 \
  --verbose-history \
  --outer-steps 200 \
  --outer-min-steps 4 \
  --outer-lr 0.03 \
  --outer-alpha-clip 8 \
  --seed 123 \
  --type-index 0 \
  --no-sample-actions \
  --online-solve \
  --no-online-skip-solve-at-t0
```

## Single-Step Query

Use `--mode query` when the state is maintained outside this script and you just want the next control.

Example:

```bash
python drone3d_variant2_rollout.py \
  --mode query \
  --total-steps 10 \
  --reveal-step 10 \
  --no-forced-reveal \
  --unconstrained \
  --p1-init-pos 0,0,1 \
  --p2-init-pos=-1,0,1 \
  --target0-pos 5,1,0 \
  --target1-pos 5,-1,0 \
  --running-r 0.05,0.025,0.05 \
  --running-s 0.05,0.10,0.05 \
  --terminal-type-diags '20,20,20,0,0,0;20,20,20,0,0,0' \
  --inner-iters 24 \
  --outer-steps 200 \
  --outer-min-steps 4 \
  --outer-lr 0.03 \
  --outer-alpha-clip 8 \
  --seed 123 \
  --type-index 0 \
  --x-current 0,0,1,0,0,0,-1,0,1,0,0,0 \
  --belief-current 0.5,0.5 \
  --prototypes-so-far 0,1 \
  --no-sample-actions
```

Important:

- `--type-index` is the realized hidden P1 type, not the public belief
- `--belief-current` is the current public belief
- `--prototypes-so-far` is the realized public prototype history
- `--time-step` is optional and only checks consistency with the history length

## Python API

The file can also be imported directly.

Most useful functions:

- `load_policy_runtime(...)`
- `query_policy_action(...)`
- `rollout_one_game(...)`
- `save_trajectory_png(...)`
- `save_animation_html(...)`

### One Control Query

```python
import torch

from drone3d_variant2_rollout import (
    OnlineSolveConfig,
    load_policy_runtime,
    query_policy_action,
)

runtime = load_policy_runtime(
    {
        "total_steps": 10,
        "reveal_step": 10,
        "no_forced_reveal": True,
        "unconstrained": True,
        "p1_init_pos": "0,0,1",
        "p2_init_pos": "-1,0,1",
        "target0_pos": "5,1,0",
        "target1_pos": "5,-1,0",
        "running_r": "0.05,0.025,0.05",
        "running_s": "0.05,0.10,0.05",
        "terminal_type_diags": "20,20,20,0,0,0;20,20,20,0,0,0",
        "inner_iters": 24,
        "outer_steps": 200,
        "outer_min_steps": 4,
        "outer_lr": 0.03,
        "outer_alpha_clip": 8.0,
        "seed": 123,
    },
    device="cpu",
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

### Real-Time Control Loop

This is the recommended real-time pattern for `dt = 0.3 s`: solve once before the first move, then evaluate the solved affine feedback at each measured state. This does not re-solve the game inside the control cycle.

```python
import torch

from drone3d_variant2_rollout import (
    OnlineSolveConfig,
    load_policy_runtime,
    query_policy_action,
)

device = torch.device("cpu")

runtime = load_policy_runtime(
    {
        "total_steps": 10,
        "reveal_step": 10,
        "no_forced_reveal": True,
        "unconstrained": True,
        "p1_init_pos": "0,0,1",
        "p2_init_pos": "-1,0,1",
        "target0_pos": "5,1,0",
        "target1_pos": "5,-1,0",
        "running_r": "0.05,0.025,0.05",
        "running_s": "0.05,0.10,0.05",
        "terminal_type_diags": "20,20,20,0,0,0;20,20,20,0,0,0",
        "inner_iters": 24,
        "outer_steps": 200,
        "outer_min_steps": 4,
        "outer_lr": 0.03,
        "outer_alpha_clip": 8.0,
        "seed": 123,
    },
    device=device,
)

belief = runtime.p0.clone()
prototypes_so_far = []

online_cfg = OnlineSolveConfig(
    enabled=False,  # real-time-safe mode: use solved affine feedback only
)

for k in range(runtime.problem.tree.total_horizon_steps):
    x_current = get_current_state_from_robot()   # torch tensor, shape (12,)

    step = query_policy_action(
        runtime=runtime,
        type_index=0,
        x_current=x_current,
        belief_current=belief,
        prototypes_so_far=prototypes_so_far,
        sample_actions=False,
        use_action_clip=False,
        online_solve=online_cfg,
    )

    u = step.u
    v = step.v

    send_control_to_robot(u, v)

    belief = step.next_belief
    prototypes_so_far.append(step.prototype_index)

    print(
        f"step={k}, proto={step.prototype_index}, "
        f"source={step.policy_source}, "
        f"control_total_ms={step.control_total_ms:.3f}, "
        f"online_solve_ms={step.online_solve_ms:.3f}"
    )
```

The returned controls are still closed-loop state feedback:

```python
u = K_u_edge @ x_current[:6] + kappa_u_edge
v = K_v_edge @ x_current[6:12] + kappa_v_edge
```

`query_policy_action(...)` handles selecting the correct public node/edge from `prototypes_so_far`, extracting the corresponding affine law, applying it to `x_current`, and returning the next belief.

### Optional Online Re-Solve Loop

This is policy-equivalent for the padded unconstrained exact-LQ path, but currently too slow for `dt = 0.3 s` in the T10 example.

```python
from drone3d_variant2_rollout import OnlineSolveConfig, precompile_online_solver

online_cfg = OnlineSolveConfig(
    enabled=True,
    skip_solve_at_t0=True,
    fixed_shape_padding=True,
    full_horizon_receding=False,
    precompile_fixed_shape=True,
    outer_steps=2,
    outer_min_steps=2,
    inner_iters=8,
    grad_tolerance=1e-3,
    loss_change_tolerance=1e-4,
)

precompile_ms, precompile_diag = precompile_online_solver(runtime, online_cfg)

# Use the same loop above, but pass online_solve=online_cfg.
```

## Timing Fields

The rollout JSON includes:

- `offline_total_ms`
- `solver_ms`
- `control_compute_per_step_ms`
- `control_total_per_step_ms`
- `online_solve_per_step_ms`
- `online_precompile_ms`
- `step_total_per_step_ms`

The root and online solve summaries also include:

- `configured_outer_steps`
- `configured_outer_min_steps`
- `configured_inner_iters`
- `configured_grad_tolerance`
- `configured_loss_change_tolerance`

Interpretation:

- `offline_total_ms` is the one-time root solve for the initial context
- `online_precompile_ms` is the one-time fixed-shape online warmup, when enabled
- `control_compute_per_step_ms` is only the final prototype-selection and affine-control computation
- `control_total_per_step_ms` is the whole `query_policy_action(...)` call
- `online_solve_per_step_ms` is only the remaining-horizon solve portion, when enabled
- `step_total_per_step_ms` is the broader rollout-loop timing around each step

For `dt = 0.3 s`, compare `control_total_per_step_ms` against `300 ms`. If online re-solving is disabled, this should mostly measure prototype selection plus affine feedback evaluation. If padded online re-solving is enabled, the current implementation is expected to exceed `300 ms` for the T10 example.

For `--mode query`, the JSON reports:

- `offline_total_ms`
- `control_compute_ms`
- `control_total_ms`
- `online_solve_ms`
- `online_precompile_ms`, when fixed-shape precompile is enabled

## HTML Output

The saved animation HTML shows:

- the 3D trajectory
- belief evolution over time
- per-step control-query latency
- a summary box with:
  - initial root solve time
  - root and online solver budgets
  - root and online convergence thresholds
  - fixed-shape online/precompile status
  - step-0 policy source
  - number of online re-solves used

This makes it easy to compare the one-time initial solve against the later online solves.

## Notes

- `type_index` chooses the realized hidden P1 type row used by the controller
- the public belief and the realized type are different objects
- the current direct-solve runner still uses the root solve to define the initial tree topology and warm-start later online solves
- if you want separate trajectories for type 0 and type 1, run the script twice with different `--type-index` values
