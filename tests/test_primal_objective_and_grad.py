# test/test_primal_objective_and_grad.py
from __future__ import annotations

import torch

from MPC_2p0s1.tests.utils_small_hexner import make_small_hexner_setup
from MPC_2p0s1.outer_opt.objective_primal import primal_objective


def test_primal_objective_scalar_and_gradients():
    """
    The primal objective should produce a scalar loss whose gradient w.r.t.
    α/logits is finite and non-zero at a generic (non-optimal) initialization.
    """
    setup = make_small_hexner_setup(I=2, K=2, T=1.0, device="cpu")
    game = setup.game
    indexer = setup.indexer
    alpha_module = setup.alpha_module

    # Forward pass
    loss, details = primal_objective(
        game=game,
        alpha_module=alpha_module,
        indexer=indexer,
        x0=None,
        p0=None,
        action_space=None,
        return_details=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Backward pass
    loss.backward()
    grad = alpha_module.logits.grad

    assert grad is not None
    assert torch.isfinite(grad).all()

    # For a generic configuration, we expect some nonzero gradient entries
    assert (grad.abs() > 0).any()