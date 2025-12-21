# test/test_rollout_and_viz.py
from __future__ import annotations

# Use a non-interactive backend for matplotlib before importing pyplot.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from MPC_2p0s1.tests.utils_small_hexner import make_small_hexner_setup
from MPC_2p0s1.outer_opt.objective_primal import primal_objective
from MPC_2p0s1.tree.rollout import rollout_trajectory
from MPC_2p0s1.viz.plot_trajectories import plot_hexner_trajectory_2d
from MPC_2p0s1.viz.plot_beliefs import plot_belief_trajectory


def test_rollout_shapes():
    """
    Roll out trajectories under the current α and feedback prototypes and
    verify basic shapes and finiteness of state, control, and belief paths.
    """
    setup = make_small_hexner_setup(I=2, K=2, T=1.0, device="cpu")
    game = setup.game
    indexer = setup.indexer
    alpha_module = setup.alpha_module
    action_space = setup.action_space

    with torch.no_grad():
        loss, details = primal_objective(
            game=game,
            alpha_module=alpha_module,
            indexer=indexer,
            x0=None,
            p0=None,
            action_space=action_space,
            return_details=True,
        )

    alpha = details["alpha"]
    belief_tree = details["belief_tree"]
    riccati_solution = details["riccati_solution"]

    dx = game.dx
    du = game.du
    dv = game.dv
    I = game.I
    K = indexer.K

    for type_index in range(I):
        ro = rollout_trajectory(
            game=game,
            indexer=indexer,
            belief_tree=belief_tree,
            riccati_sol=riccati_solution,
            alpha=alpha,
            x0=game.default_initial_state(),
            type_index=type_index,
            action_space=action_space,
            sample_actions=False,
            generator=None,
        )

        assert ro.x_traj.shape == (K + 1, dx)
        assert ro.u_traj.shape == (K, du)
        assert ro.v_traj.shape == (K, dv)
        assert ro.belief_traj.shape == (K + 1, I)
        assert ro.proto_indices.shape == (K,)

        assert torch.isfinite(ro.x_traj).all()
        assert torch.isfinite(ro.u_traj).all()
        assert torch.isfinite(ro.v_traj).all()
        assert torch.isfinite(ro.belief_traj).all()


def test_viz_smoke_hexner_trajectory_and_belief():
    """
    Smoke-test the core visualization functions by drawing a single trajectory
    and belief curve onto matplotlib axes and forcing a render.
    """
    setup = make_small_hexner_setup(I=2, K=2, T=1.0, device="cpu")
    game = setup.game
    indexer = setup.indexer
    alpha_module = setup.alpha_module
    action_space = setup.action_space

    with torch.no_grad():
        _, details = primal_objective(
            game=game,
            alpha_module=alpha_module,
            indexer=indexer,
            x0=None,
            p0=None,
            action_space=action_space,
            return_details=True,
        )

    alpha = details["alpha"]
    belief_tree = details["belief_tree"]
    riccati_solution = details["riccati_solution"]

    # Single rollout for type 0
    ro = rollout_trajectory(
        game=game,
        indexer=indexer,
        belief_tree=belief_tree,
        riccati_sol=riccati_solution,
        alpha=alpha,
        x0=game.default_initial_state(),
        type_index=0,
        action_space=action_space,
        sample_actions=False,
        generator=None,
    )

    # 2D trajectory plot
    fig1, ax1 = plt.subplots()
    plot_hexner_trajectory_2d(
        game=game,
        rollout=ro,
        ax=ax1,
        show_targets=True,
        show_belief_colormap=True,
        title=None,
    )
    fig1.canvas.draw()  # Force rendering
    plt.close(fig1)

    # Belief trajectory plot
    fig2, ax2 = plt.subplots()
    plot_belief_trajectory(
        belief_traj=ro.belief_traj,
        T=game.cfg.T,
        ax=ax2,
        labels=None,
        title=None,
    )
    fig2.canvas.draw()
    plt.close(fig2)