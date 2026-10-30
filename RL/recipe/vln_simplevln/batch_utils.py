"""Batch layout shared by rollout flattening and DAPO resampling."""

from math import lcm

import numpy as np


def padding_multiple(config) -> int:
    arr = config.actor_rollout_ref
    world_size = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
    actor_micro = int(arr.actor.ppo_micro_batch_size_per_gpu)
    rollout_micro = int(arr.rollout.log_prob_micro_batch_size_per_gpu)
    mini_batch = int(arr.actor.ppo_mini_batch_size) * int(arr.rollout.n)
    if min(world_size, actor_micro, rollout_micro, mini_batch) <= 0:
        raise ValueError("Static batch sizes and GPU counts must be positive")
    if mini_batch % (world_size * actor_micro):
        raise ValueError("PPO mini-batch size times rollout n must divide evenly into per-GPU actor micro-batches")
    return lcm(world_size * actor_micro, world_size * rollout_micro, mini_batch)


def object_vector(values) -> np.ndarray:
    """Preserve each list-valued row as one object, even when lengths are equal."""
    values = list(values)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result
