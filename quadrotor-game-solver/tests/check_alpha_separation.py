"""Diagnostic: Check if learned alpha separates the two types."""

from pathlib import Path
import torch

# Load checkpoint
checkpoint_path = Path("/Users/mghimire/Research/MPC_2p0s1/runs/quadrotor_no_linesearch/final_alpha.pt")
if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, weights_only=False)
alpha_logits = checkpoint["logits"]  # (K, max_nodes, I, I)

print(f"Alpha logits shape: {alpha_logits.shape}")
K, max_nodes, I, _ = alpha_logits.shape

# Convert to probabilities
alpha = torch.softmax(alpha_logits, dim=-1)  # (K, max_nodes, I, I)

# For each depth k and each node n, check if type 0 and type 1 choose different actions
total_nodes = 0
separated_nodes = 0

for k in range(K):
    # At depth k, there are I^k nodes
    num_nodes = I ** k
    for n in range(num_nodes):
        # Get action distributions for both types
        alpha_type0 = alpha[k, n, 0]  # (I,) - type 0's action distribution
        alpha_type1 = alpha[k, n, 1]  # (I,) - type 1's action distribution

        # Get argmax actions
        action_type0 = torch.argmax(alpha_type0).item()
        action_type1 = torch.argmax(alpha_type1).item()

        total_nodes += 1
        if action_type0 != action_type1:
            separated_nodes += 1

separation_fraction = separated_nodes / total_nodes if total_nodes > 0 else 0

print(f"\n{'='*60}")
print(f"Alpha Separation Diagnostic")
print(f"{'='*60}")
print(f"Total nodes checked: {total_nodes}")
print(f"Nodes with separation: {separated_nodes}")
print(f"Separation fraction: {separation_fraction:.2%}")
print(f"{'='*60}\n")

if separation_fraction < 0.01:
    print("⚠️  WARNING: Alpha appears to be in POOLING EQUILIBRIUM")
    print("   Both types choose the same actions at almost all nodes.")
else:
    print(f"✓ Alpha shows {separation_fraction:.1%} separation")

# Show some examples
print("\nExample alpha values at root (k=0, node=0):")
print(f"  Type 0 chooses: {alpha[0, 0, 0].detach().cpu().numpy()}")
print(f"  Type 1 chooses: {alpha[0, 0, 1].detach().cpu().numpy()}")
print(f"  Argmax type 0: {torch.argmax(alpha[0, 0, 0]).item()}")
print(f"  Argmax type 1: {torch.argmax(alpha[0, 0, 1]).item()}")
