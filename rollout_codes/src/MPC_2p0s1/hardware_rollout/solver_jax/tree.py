from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

F32 = jnp.float32
I32 = jnp.int32


def _enumerate_paths(branching_factor: int, depth: int) -> jnp.ndarray:
  if depth == 0:
    return jnp.zeros((1, 0), dtype=I32)

  count = branching_factor**depth
  indices = jnp.arange(count, dtype=I32)
  digits = []
  for power in range(depth - 1, -1, -1):
    digits.append((indices // (branching_factor**power)) % branching_factor)
  return jnp.stack(digits, axis=1).astype(I32)


def _prefix_node_indices(paths: jnp.ndarray, branching_factor: int) -> jnp.ndarray:
  if paths.shape[1] == 0:
    return jnp.zeros((paths.shape[0], 0), dtype=I32)

  prefix = jnp.zeros((paths.shape[0],), dtype=I32)
  indices = []
  for depth in range(paths.shape[1]):
    indices.append(prefix)
    prefix = prefix * branching_factor + paths[:, depth]
  return jnp.stack(indices, axis=1).astype(I32)


@dataclass(frozen=True)
class MixedPrefixTreeSpec:
  type_count: int
  mixed_horizon_steps: int
  tail_horizon_steps: int = 0
  force_identity_reveal: bool = False

  def __post_init__(self) -> None:
    if self.type_count <= 0:
      raise ValueError("type_count must be positive")
    if self.mixed_horizon_steps < 0:
      raise ValueError("mixed_horizon_steps must be non-negative")
    if self.tail_horizon_steps < 0:
      raise ValueError("tail_horizon_steps must be non-negative")

    paths = _enumerate_paths(self.type_count, self.mixed_horizon_steps)
    node_indices = _prefix_node_indices(paths, self.type_count)
    edge_indices = node_indices * self.type_count + paths if self.mixed_horizon_steps else node_indices

    object.__setattr__(self, "paths", paths)
    object.__setattr__(self, "node_indices", edge_indices * 0 + node_indices)
    object.__setattr__(self, "edge_indices", edge_indices.astype(I32))

  @property
  def total_horizon_steps(self) -> int:
    return self.mixed_horizon_steps + self.tail_horizon_steps

  @property
  def frontier_count(self) -> int:
    return self.type_count**self.mixed_horizon_steps

  @property
  def max_node_count(self) -> int:
    if self.mixed_horizon_steps == 0:
      return 1
    return self.type_count ** (self.mixed_horizon_steps - 1)

  @property
  def max_edge_count(self) -> int:
    if self.mixed_horizon_steps == 0:
      return 1
    return self.type_count**self.mixed_horizon_steps

  def node_count(self, depth: int) -> int:
    if not (0 <= depth < self.mixed_horizon_steps):
      raise ValueError(f"depth must be in [0, {self.mixed_horizon_steps})")
    return self.type_count**depth

  def edge_count(self, depth: int) -> int:
    if not (0 <= depth < self.mixed_horizon_steps):
      raise ValueError(f"depth must be in [0, {self.mixed_horizon_steps})")
    return self.type_count ** (depth + 1)


def build_alpha_from_logits(
  logits: jnp.ndarray,
  tree: MixedPrefixTreeSpec,
) -> jnp.ndarray:
  if tree.mixed_horizon_steps == 0:
    return jnp.zeros((0, 1, tree.type_count, tree.type_count), dtype=F32)

  alpha = jax_softmax(logits, axis=-1).astype(F32)

  if tree.force_identity_reveal:
    depth = tree.mixed_horizon_steps - 1
    used_nodes = tree.node_count(depth)
    identity = jnp.eye(tree.type_count, dtype=F32)
    reveal_block = jnp.broadcast_to(identity, (used_nodes, tree.type_count, tree.type_count))
    alpha = alpha.at[depth, :used_nodes].set(reveal_block)

  return alpha


def jax_softmax(logits: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
  shifted = logits - jnp.max(logits, axis=axis, keepdims=True)
  exp_shifted = jnp.exp(shifted)
  return exp_shifted / jnp.sum(exp_shifted, axis=axis, keepdims=True)


def path_type_probabilities(alpha: jnp.ndarray, tree: MixedPrefixTreeSpec) -> jnp.ndarray:
  if tree.mixed_horizon_steps == 0:
    return jnp.ones((1, tree.type_count), dtype=F32)

  probs = jnp.ones((tree.frontier_count, tree.type_count), dtype=F32)
  for depth in range(tree.mixed_horizon_steps):
    node_idx = tree.node_indices[:, depth]
    action_idx = tree.paths[:, depth]
    alpha_node = alpha[depth, node_idx]  # (S, I, I)
    edge_prob = jnp.take_along_axis(
      alpha_node,
      jnp.broadcast_to(action_idx[:, None, None], (tree.frontier_count, tree.type_count, 1)),
      axis=2,
    ).squeeze(-1)
    probs = probs * edge_prob
  return probs


def unconditional_path_probabilities(
  alpha: jnp.ndarray,
  tree: MixedPrefixTreeSpec,
  prior: jnp.ndarray,
) -> jnp.ndarray:
  type_probs = path_type_probabilities(alpha, tree)
  return jnp.sum(type_probs * prior[None, :], axis=1)


def frontier_beliefs(
  alpha: jnp.ndarray,
  tree: MixedPrefixTreeSpec,
  prior: jnp.ndarray,
) -> jnp.ndarray:
  type_probs = path_type_probabilities(alpha, tree) * prior[None, :]
  normalizer = jnp.clip(jnp.sum(type_probs, axis=1, keepdims=True), 1e-8, None)
  return type_probs / normalizer

