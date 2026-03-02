"""
Nicholas M. Boffi
10/5/25

Simple utilities for single-node multi-GPU data parallelism.
"""

from typing import Any, Tuple

import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from ml_collections import config_dict


def safe_index(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Extract first element if using multiple devices, otherwise return as-is."""
    if cfg.training.ndevices > 1:
        return x[0]
    else:
        return x


def safe_replicate(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Replicate data across devices if using multiple GPUs."""
    if cfg.training.ndevices > 1:
        return replicate(x)
    else:
        return x


def safe_unreplicate(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Unreplicate data from devices if using multiple GPUs."""
    if cfg.training.ndevices > 1:
        return unreplicate(x)
    else:
        return x


def replicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Shard batch across local devices for data parallelism."""
    if (
        cfg.training.ndevices > 1
        and x is not None
        and not getattr(cfg.training, "batch_is_sharded", False)
    ):
        x = x.reshape((cfg.training.ndevices, -1, *x.shape[1:]))
    return x


def unreplicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Merge batch from local devices."""
    if cfg.training.ndevices > 1 and x is not None:
        x = x.reshape((-1, *x.shape[2:]))
    return x


def replicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
    """Replicate all loss function arguments for data parallelism."""
    return tuple(replicate_batch(cfg, arg) for arg in loss_fn_args)


def unreplicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
    """Unreplicate all loss function arguments."""
    return tuple(unreplicate_batch(cfg, arg) for arg in loss_fn_args)
