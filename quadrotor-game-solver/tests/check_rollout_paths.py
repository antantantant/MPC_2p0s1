"""Check if the specific rollout paths show separation."""

from pathlib import Path
import torch

# Load alpha
alpha_path = Path("/Users/mghimire/Research/MPC_2p0s1/runs/quadrotor_no_linesearch/final_alpha.pt")
checkpoint = torch.load(alpha_path, weights_only=False)
alpha_logits = checkpoint["logits"]  # (K, max_nodes, I, I)

K, max_nodes, I, _ = alpha_logits.shape
alpha = torch.softmax(alpha_logits, dim=-1)  # (K, max_nodes, I, I)

print("Following the greedy rollout paths for each type:")
print("="*60)

# Type 0 rollout
print("\nType 0 rollout path:")
node_idx = 0
for k in range(K):
    alpha_row = alpha[k, node_idx, 0]  # Type 0's action distribution
    action = torch.argmax(alpha_row).item()
    print(f"  k={k:2d}, node={node_idx:4d}: action={action}, probs={alpha_row.numpy()}")
    # Move to child node
    child_idx = node_idx * I + action
    node_idx = child_idx

# Type 1 rollout
print("\nType 1 rollout path:")
node_idx = 0
for k in range(K):
    alpha_row = alpha[k, node_idx, 1]  # Type 1's action distribution
    action = torch.argmax(alpha_row).item()
    print(f"  k={k:2d}, node={node_idx:4d}: action={action}, probs={alpha_row.numpy()}")
    # Move to child node
    child_idx = node_idx * I + action
    node_idx = child_idx

# Check if paths diverge
print("\n" + "="*60)
print("Checking if rollout paths are identical:")
print("="*60)

node_idx_0 = 0
node_idx_1 = 0
path_identical = True
first_divergence = None

for k in range(K):
    action_0 = torch.argmax(alpha[k, node_idx_0, 0]).item()
    action_1 = torch.argmax(alpha[k, node_idx_1, 1]).item()
    
    if action_0 != action_1:
        path_identical = False
        if first_divergence is None:
            first_divergence = k
    
    node_idx_0 = node_idx_0 * I + action_0
    node_idx_1 = node_idx_1 * I + action_1

if path_identical:
    print("⚠️  PATHS ARE IDENTICAL - both types follow the same trajectory")
    print("   This explains why rollout plots show identical movement!")
else:
    print(f"✓ Paths diverge at step {first_divergence}")
