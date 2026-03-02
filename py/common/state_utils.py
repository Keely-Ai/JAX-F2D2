"""
Nicholas M. Boffi
10/5/25

Utilities for storing training state.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax import struct
from flax.serialization import from_bytes
from flax.training import train_state
from ml_collections import config_dict

from . import flow_map, interpolant
from . import dist_utils


def _print_model_summary(
    net: nn.Module,
    params: Dict[str, Any],
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Print model structure and parameter count once."""
    if jax.process_index() != 0:
        return prng_key

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Number of parameters: {param_count}")

    ex_s = ex_t = 0.0
    ex_label = 0
    prng_key, skey = jax.random.split(prng_key)
    try:
        print("Model structure:")
        print(
            net.tabulate(
                {"params": prng_key, "constants": skey},
                ex_s,
                ex_t,
                ex_input,
                ex_label,
                train=False,
                init_weights=True,
            )
        )
    except Exception as exc:
        print(f"Model tabulate failed: {exc}")

    return prng_key


class EMATrainState(train_state.TrainState):
    """Train state including EMA parameters."""

    ema_params: Dict[float, Any] = struct.field(default_factory=dict)


class StaticArgs(NamedTuple):
    net: nn.Module
    schedule: optax.Schedule
    loss: Callable
    get_loss_fn_args: Callable
    train_step: Callable
    update_ema_params: Callable
    ds: tf.data.Dataset
    interp: interpolant.Interpolant
    sample_rho0: Callable
    inception_fn: Callable = None  # For FID computation
    teacher_params: Optional[Dict[str, Any]] = None  # Optional external teacher


def load_checkpoint(
    cfg: config_dict.ConfigDict,
    train_state: EMATrainState,
) -> EMATrainState:
    """Load a training checkpoint."""
    with open(cfg.network.load_path, "rb") as f:
        raw_bytes = f.read()
        train_state = from_bytes(train_state, raw_bytes)

    return train_state


def load_teacher_params(
    cfg: config_dict.ConfigDict,
    train_state: EMATrainState,
) -> Optional[Dict[str, Any]]:
    """Load teacher params from checkpoint if configured."""
    teacher_cfg = getattr(cfg, "teacher", None)
    teacher_path = ""
    teacher_ema_fac = None
    if teacher_cfg is not None:
        teacher_path = getattr(teacher_cfg, "load_path", "")
        teacher_ema_fac = getattr(teacher_cfg, "ema_fac", None)
    if not teacher_path:
        return None

    print(f"Loading teacher checkpoint from {teacher_path}.")
    with open(teacher_path, "rb") as f:
        raw_bytes = f.read()
        teacher_state = from_bytes(train_state, raw_bytes)
    print("Loaded teacher checkpoint.")
    if teacher_ema_fac is not None:
        ema_params = teacher_state.ema_params.get(teacher_ema_fac, None)
        if ema_params is None:
            print(f"Warning: teacher EMA {teacher_ema_fac} not found, using params.")
            return teacher_state.params
        return ema_params
    return teacher_state.params


def setup_schedule(
    cfg: config_dict.ConfigDict,
) -> optax.Schedule:
    """Set up the learning rate schedule."""
    if cfg.optimization.schedule_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=cfg.optimization.learning_rate,
            decay_steps=cfg.optimization.decay_steps,
            alpha=0.0,
        )
    elif cfg.optimization.schedule_type == "sqrt":
        return lambda step: cfg.optimization.learning_rate / jnp.sqrt(
            jnp.maximum(step / cfg.optimization.decay_steps, 1.0)
        )
    elif cfg.optimization.schedule_type == "constant":
        return lambda step: cfg.optimization.learning_rate
    else:
        raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")


def setup_optimizer(cfg: config_dict.ConfigDict):
    """Set up the optimizer."""
    schedule = setup_schedule(cfg)

    # optimizer mask for positional embeddings (which do not have a constants key)
    def mask_fn(variables):
        masks = {
            "params": jax.tree_util.tree_map(lambda _: True, variables["params"]),
        }
        if "constants" in variables:  # network has Fourier tables
            masks["constants"] = jax.tree_util.tree_map(
                lambda _: False, variables["constants"]
            )
        return masks

    # define optimizer
    tx = optax.masked(
        optax.chain(
            optax.clip_by_global_norm(cfg.optimization.clip),
            optax.radam(
                learning_rate=schedule,
                b1=getattr(cfg.optimization, "b1", 0.9),
                b2=getattr(cfg.optimization, "b2", 0.999),
                eps=getattr(cfg.optimization, "eps", 1e-8),
            ),
        ),
        mask_fn,
    )

    return tx, schedule


def setup_training_state(
    cfg: config_dict.ConfigDict,
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[EMATrainState, flow_map.FlowMap, optax.Schedule, jnp.ndarray]:
    """Load flax training state."""

    # Initialize flow map network
    net, params, prng_key = flow_map.initialize_flow_map(
        cfg.network, ex_input, prng_key
    )
    ema_params = {ema_fac: deepcopy(params) for ema_fac in cfg.training.ema_facs}
    prng_key = _print_model_summary(net, params, ex_input, prng_key)

    # define training state
    tx, schedule = setup_optimizer(cfg)
    train_state = EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params=ema_params,
        tx=tx,
    )

    # load training state from checkpoint, if desired
    if cfg.network.load_path != "":
        print("Loading full training state checkpoint.")
        train_state = load_checkpoint(cfg, train_state)
        print("Loaded training state checkpoint.")

        init_from_ema_fac = getattr(cfg.network, "init_from_ema_factor", None)
        if init_from_ema_fac is not None:
            ema_params = train_state.ema_params.get(init_from_ema_fac, None)
            if ema_params is None:
                print(
                    f"Warning: EMA factor {init_from_ema_fac} not found for init. "
                    "Keeping checkpoint params."
                )
            else:
                print(f"Initializing params from EMA {init_from_ema_fac}.")
                train_state = train_state.replace(params=ema_params)

        if cfg.network.reset_optimizer:
            print("Resetting optimizer state.")
            train_state = train_state.replace(
                opt_state=tx.init(train_state.params),
                step=0,
            )

    return train_state, net, schedule, prng_key