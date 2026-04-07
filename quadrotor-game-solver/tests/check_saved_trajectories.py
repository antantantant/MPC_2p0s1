"""Load and compare the saved trajectories from evaluation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Load the checkpoint to get config
checkpoint_path = Path("/Users/mghimire/Research/MPC_2p0s1/runs/quadrotor_no_linesearch/checkpoint_0019.pt")
checkpoint = torch.load(checkpoint_path, weights_only=False)

config = checkpoint["config"]
print(f"Config: {config}")

# The eval script saves animations, but let me try to find the actual trajectory data
# Let's re-run a rollout here to see what we get

from src.game.quadrotor_game import Hexner3DQuadrotorGame
from src.tree.indexing import FullIaryTreeIndexer
from src.tree.belief_tree import BeliefTree
from src.solvers.sqp_tree import sqp_tree_layer
from src.rollout.trajectory import rollout_trajectory

# Load alpha
alpha_path = Path("/Users/mghimire/Research/MPC_2p0s1/runs/quadrotor_no_linesearch/final_alpha.pt")
alpha_data = torch.load(alpha_path, weights_only=False)
alpha_logits = alpha_data["logits"]
alpha = torch.softmax(alpha_logits, dim=0)  # Convert to probabilities

# Create game
game = Hexner3DQuadrotorGame(
    T=1.0,
    K=10,
    dtype=torch.float64,
)

# Create indexer and belief tree
indexer = FullIaryTreeIndexer(K=10, I=2)
p0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
belief_tree = BeliefTree.from_prior(indexer=indexer, p0=p0, alpha=alpha)

# Solve SQP to get Riccati solution
result = sqp_tree_layer(
    game=game,
    belief_tree=belief_tree,
    num_iters=50,
    step_size=0.05,
    verbose=False,
)

# Get initial state from config
x0_p1_pos = torch.tensor(config["x0_p1_pos"], dtype=torch.float64)
x0_p2_pos = torch.tensor(config["x0_p2_pos"], dtype=torch.float64)
x0 = torch.cat([
    x0_p1_pos,
    torch.zeros(3, dtype=torch.float64),  # P1 velocity
    x0_p2_pos,
    torch.zeros(3, dtype=torch.float64),  # P2 velocity
    torch.zeros(12, dtype=torch.float64),  # Omega, Lambda (not used)
])

# Rollout for both types
print("\n" + "="*60)
print("Rollout for Type 0 (θ=-1.0):")
print("="*60)
rollout_0 = rollout_trajectory(
    game=game,
    indexer=indexer,
    belief_tree=belief_tree,
    riccati_sol=result.riccati_sol,
    alpha=alpha,
    x0=x0,
    type_index=0,
)
print(f"Type 0 proto_indices: {rollout_0.proto_indices.cpu().numpy()}")
print(f"Type 0 P1 positions:")
for k in range(11):
    pos = rollout_0.x_traj[k, :3].cpu().numpy()
    print(f"  k={k:2d}: {pos}")

print("\n" + "="*60)
print("Rollout for Type 1 (θ=+1.0):")
print("="*60)
rollout_1 = rollout_trajectory(
    game=game,
    indexer=indexer,
    belief_tree=belief_tree,
    riccati_sol=result.riccati_sol,
    alpha=alpha,
    x0=x0,
    type_index=1,
)
print(f"Type 1 proto_indices: {rollout_1.proto_indices.cpu().numpy()}")
print(f"Type 1 P1 positions:")
for k in range(11):
    pos = rollout_1.x_traj[k, :3].cpu().numpy()
    print(f"  k={k:2d}: {pos}")

# Check if positions are identical
print("\n" + "="*60)
print("Comparing trajectories:")
print("="*60)
pos_diff = torch.norm(rollout_0.x_traj[:, :3] - rollout_1.x_traj[:, :3], dim=1)
print(f"Position differences (P1): {pos_diff.cpu().numpy()}")
if torch.all(pos_diff < 1e-6):
    print("⚠️  POSITIONS ARE IDENTICAL!")
else:
    print(f"✓ Positions differ, max diff: {pos_diff.max().item():.6f}")
