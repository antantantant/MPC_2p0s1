# Hexner 3D Hardware Rollout

This directory contains a small rollout runner for the amortized 3D Hexner-mod checkpoints.

The main entry point is:

- `MPC_2p0s1.hardware_rollout.hexner_mod_3d_rollout`

It does the following for one sampled game at a time:

- loads a checkpoint
- rebuilds the 3D game and amortized alpha model from checkpoint metadata
- samples one initial state `x0` and one prior `p0` unless you provide them
- samples a realized type unless you provide it
- solves the tree once for that context
- rolls out one trajectory
- reports offline solve timing and per-step control-computation timing
- optionally saves everything to JSON

## Quick Use

From the repo root, run:

```bash
python hardware_rollout/hexner_mod_3d_rollout.py \
  --checkpoint runs/hexner_mod_3d_amortized_vel_low_running_cost/latest.pt
```

A more explicit example with JSON output:

```bash
python hardware_rollout/hexner_mod_3d_rollout.py \
  --checkpoint runs/hexner_mod_3d_amortized_vel_low_running_cost/latest.pt \
  --seed 7 \
  --sample-actions \
  --output-json /tmp/hexner_3d_rollout.json
```

You can also force a fixed context:

```bash
python hardware_rollout/hexner_mod_3d_rollout.py \
  --checkpoint runs/hexner_mod_3d_amortized_vel_low_running_cost/latest.pt \
  --prior 0.5,0.5 \
  --x0 1.2,-1.9,1.7,0,0,0,-1.6,2.0,0.3,0,0,0 \
  --type-index 0
```

If the package is installed in a normal Python environment, module invocation also works:

```bash
python -m MPC_2p0s1.hardware_rollout.hexner_mod_3d_rollout \
  --checkpoint MPC_2p0s1/runs/hexner_mod_3d_amortized_vel_low_running_cost/latest.pt
```

## What Timing Means

The JSON / printed summary includes:

- `alpha_eval_ms`: time to evaluate the amortized network once
- `belief_tree_ms`: time to build the public belief tree
- `avg_costs_ms`: time to form the averaged game data on the tree
- `riccati_ms`: time for the backward Riccati solve
- `control_compute_per_step_ms`: time, at each step, to:
  - read the current `alpha` row
  - choose or sample the prototype/action index
  - compute `u_k` and `v_k`
  - optionally clip actions

So `control_compute_per_step_ms` is the closest thing to the online control latency.

## Checkpoint Support

`--game-type auto` is the default.

The runner can rebuild both:

- the original target-based game from `hexner_mod_3d_game_original.py`
- the newer coupled-terminal-matrix game from `hexner_mod_3d_game.py`

If auto-detection is ever ambiguous, pass one of:

- `--game-type original`
- `--game-type coupled`

## Files Needed For Sharing

This runner is light, but it is not a single-file export. It imports parts of the repo package.

The safest bundle to share is:

- `MPC_2p0s1/__init__.py`
- `MPC_2p0s1/hardware_rollout/`
- `MPC_2p0s1/config/`
- `MPC_2p0s1/core/`
- `MPC_2p0s1/games/`
- `MPC_2p0s1/outer_opt/`
- `MPC_2p0s1/tree/`
- the desired checkpoint `.pt` file

If the hardware team wants the absolute minimum set later, we can prune that bundle more aggressively, but the list above is the low-risk version that should run without chasing transitive imports.

## Dependencies

Required:

- Python 3.10+
- `torch`

Optional:

- none for this runner

## Notes

- The runner samples one game per invocation.
- By default it samples a random prior and random initial state around the checkpoint's default context.
- For original-game checkpoints it uses the original per-player jitter fields when available.
- For coupled-game checkpoints it uses the shared jitter fields when available.
- If you want deterministic public signaling during rollout, pass `--no-sample-actions`.
