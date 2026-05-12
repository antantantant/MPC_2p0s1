from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .tree import I32, F32, MixedPrefixTreeSpec


@dataclass(frozen=True)
class PublicTreeTopology:
  branch_factor: int
  mixed_depth: int
  tail_depth: int
  total_depth: int
  node_offsets_py: tuple[int, ...]
  edge_offsets_py: tuple[int, ...]
  node_offsets: jnp.ndarray
  edge_offsets: jnp.ndarray
  node_depths: jnp.ndarray
  node_local_indices: jnp.ndarray
  node_parents: jnp.ndarray
  node_parent_edges: jnp.ndarray
  edge_parents: jnp.ndarray
  edge_children: jnp.ndarray
  edge_depths: jnp.ndarray
  edge_local_indices: jnp.ndarray
  edge_actions: jnp.ndarray
  leaf_nodes: jnp.ndarray
  leaf_node_paths: jnp.ndarray
  leaf_edge_paths: jnp.ndarray

  @property
  def node_count(self) -> int:
    return int(self.node_depths.shape[0])

  @property
  def edge_count(self) -> int:
    return int(self.edge_parents.shape[0])

  @property
  def leaf_count(self) -> int:
    return int(self.leaf_nodes.shape[0])

  @property
  def frontier_count(self) -> int:
    return self.branch_factor**self.mixed_depth


def build_public_tree_topology(tree: MixedPrefixTreeSpec) -> PublicTreeTopology:
  branch = tree.type_count
  mixed_depth = tree.mixed_horizon_steps
  tail_depth = tree.tail_horizon_steps
  total_depth = tree.total_horizon_steps
  frontier_count = branch**mixed_depth

  node_offsets = []
  total_nodes = 0
  for depth in range(total_depth + 1):
    node_offsets.append(total_nodes)
    if depth <= mixed_depth:
      total_nodes += branch**depth
    else:
      total_nodes += frontier_count

  edge_offsets = []
  total_edges = 0
  for depth in range(total_depth):
    edge_offsets.append(total_edges)
    if depth < mixed_depth:
      total_edges += branch ** (depth + 1)
    else:
      total_edges += frontier_count

  node_depths: list[int] = []
  node_local_indices: list[int] = []
  node_parents: list[int] = []
  node_parent_edges: list[int] = []
  for depth in range(total_depth + 1):
    count = branch**depth if depth <= mixed_depth else frontier_count
    for local in range(count):
      node_depths.append(depth)
      node_local_indices.append(local)
      if depth == 0:
        node_parents.append(-1)
        node_parent_edges.append(-1)
      elif depth <= mixed_depth:
        parent_local = local // branch
        action = local % branch
        node_parents.append(node_offsets[depth - 1] + parent_local)
        node_parent_edges.append(edge_offsets[depth - 1] + parent_local * branch + action)
      else:
        node_parents.append(node_offsets[depth - 1] + local)
        node_parent_edges.append(edge_offsets[depth - 1] + local)

  edge_parents: list[int] = []
  edge_children: list[int] = []
  edge_depths: list[int] = []
  edge_local_indices: list[int] = []
  edge_actions: list[int] = []
  for depth in range(total_depth):
    if depth < mixed_depth:
      parent_count = branch**depth
      for parent_local in range(parent_count):
        parent_global = node_offsets[depth] + parent_local
        for action in range(branch):
          edge_local = parent_local * branch + action
          child_global = node_offsets[depth + 1] + edge_local
          edge_parents.append(parent_global)
          edge_children.append(child_global)
          edge_depths.append(depth)
          edge_local_indices.append(edge_local)
          edge_actions.append(action)
    else:
      for local in range(frontier_count):
        parent_global = node_offsets[depth] + local
        child_global = node_offsets[depth + 1] + local
        edge_parents.append(parent_global)
        edge_children.append(child_global)
        edge_depths.append(depth)
        edge_local_indices.append(local)
        edge_actions.append(0)

  leaf_count = frontier_count
  leaf_node_paths = jnp.zeros((leaf_count, total_depth + 1), dtype=I32)
  leaf_edge_paths = jnp.zeros((leaf_count, total_depth), dtype=I32)
  leaf_node_paths = leaf_node_paths.at[:, 0].set(0)
  if mixed_depth > 0:
    for depth in range(mixed_depth):
      leaf_node_paths = leaf_node_paths.at[:, depth].set(tree.node_indices[:, depth] + node_offsets[depth])
      leaf_edge_paths = leaf_edge_paths.at[:, depth].set(tree.edge_indices[:, depth] + edge_offsets[depth])

    final_local = jnp.zeros((leaf_count,), dtype=I32)
    for depth in range(mixed_depth):
      final_local = final_local * branch + tree.paths[:, depth]
    mixed_frontier_nodes = final_local + node_offsets[mixed_depth]
  else:
    mixed_frontier_nodes = jnp.array([0], dtype=I32)

  leaf_node_paths = leaf_node_paths.at[:, mixed_depth].set(mixed_frontier_nodes)
  for tail_step in range(tail_depth):
    depth = mixed_depth + tail_step
    leaf_edge_paths = leaf_edge_paths.at[:, depth].set(edge_offsets[depth] + jnp.arange(frontier_count, dtype=I32))
    leaf_node_paths = leaf_node_paths.at[:, depth + 1].set(node_offsets[depth + 1] + jnp.arange(frontier_count, dtype=I32))

  leaf_nodes = leaf_node_paths[:, -1]

  return PublicTreeTopology(
    branch_factor=branch,
    mixed_depth=mixed_depth,
    tail_depth=tail_depth,
    total_depth=total_depth,
    node_offsets_py=tuple(int(value) for value in node_offsets),
    edge_offsets_py=tuple(int(value) for value in edge_offsets),
    node_offsets=jnp.array(node_offsets, dtype=I32),
    edge_offsets=jnp.array(edge_offsets, dtype=I32) if edge_offsets else jnp.zeros((0,), dtype=I32),
    node_depths=jnp.array(node_depths, dtype=I32),
    node_local_indices=jnp.array(node_local_indices, dtype=I32),
    node_parents=jnp.array(node_parents, dtype=I32),
    node_parent_edges=jnp.array(node_parent_edges, dtype=I32),
    edge_parents=jnp.array(edge_parents, dtype=I32),
    edge_children=jnp.array(edge_children, dtype=I32),
    edge_depths=jnp.array(edge_depths, dtype=I32),
    edge_local_indices=jnp.array(edge_local_indices, dtype=I32),
    edge_actions=jnp.array(edge_actions, dtype=I32),
    leaf_nodes=leaf_nodes.astype(I32),
    leaf_node_paths=leaf_node_paths.astype(I32),
    leaf_edge_paths=leaf_edge_paths.astype(I32),
  )


def propagate_type_probabilities(
  alpha: jnp.ndarray,
  topology: PublicTreeTopology,
  prior: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  hidden_type_count = int(prior.shape[0])
  node_type_probs = jnp.zeros((topology.node_count, hidden_type_count), dtype=F32)
  node_type_probs = node_type_probs.at[0].set(prior.astype(F32))

  edge_type_probs_parts = []
  edge_public_probs_parts = []

  for depth in range(topology.mixed_depth):
    node_offset = topology.node_offsets_py[depth]
    next_offset = topology.node_offsets_py[depth + 1]
    node_count = topology.branch_factor**depth
    child_count = topology.branch_factor ** (depth + 1)

    depth_node_type_probs = node_type_probs[node_offset : node_offset + node_count]
    alpha_depth = alpha[depth, :node_count]
    child_type_probs = depth_node_type_probs[:, :, None] * alpha_depth
    child_type_probs = jnp.transpose(child_type_probs, (0, 2, 1)).reshape((child_count, hidden_type_count))
    node_type_probs = node_type_probs.at[next_offset : next_offset + child_count].set(child_type_probs)
    edge_type_probs_parts.append(child_type_probs)
    edge_public_probs_parts.append(jnp.sum(child_type_probs, axis=1))

  if topology.tail_depth > 0:
    frontier_type_probs = node_type_probs[topology.leaf_node_paths[:, topology.mixed_depth]]
    for tail_step in range(topology.tail_depth):
      node_offset = topology.node_offsets_py[topology.mixed_depth + tail_step + 1]
      node_type_probs = node_type_probs.at[node_offset : node_offset + topology.frontier_count].set(frontier_type_probs)
      edge_type_probs_parts.append(frontier_type_probs)
      edge_public_probs_parts.append(jnp.sum(frontier_type_probs, axis=1))

  if edge_type_probs_parts:
    edge_type_probs = jnp.concatenate(edge_type_probs_parts, axis=0)
    edge_public_probs = jnp.concatenate(edge_public_probs_parts, axis=0)
  else:
    edge_type_probs = jnp.zeros((0, hidden_type_count), dtype=F32)
    edge_public_probs = jnp.zeros((0,), dtype=F32)

  node_public_probs = jnp.sum(node_type_probs, axis=1)
  return node_type_probs, node_public_probs, edge_type_probs, edge_public_probs
