"""
Nicholas M. Boffi
10/5/25

Update functions for learning.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import ml_collections.config_dict as config_dict
from jax import value_and_grad

from . import state_utils, edm2_net

Parameters = Dict[str, Dict]


def setup_train_step(cfg: config_dict.ConfigDict) -> Callable:
    """Setup the training step function for single or multi-device training."""

    if cfg.training.ndevices > 1:
        decorator = lambda f: jax.pmap(
            f,
            in_axes=(0, None, 0),
            static_broadcasted_argnums=(1,),
            axis_name="data",
        )
    else:
        decorator = lambda f: functools.partial(jax.jit, static_argnums=(1,))(f)

    @decorator
    def train_step(
        state: state_utils.EMATrainState,
        loss_func: Callable[[Parameters], Tuple[float, Dict[str, jnp.ndarray]]],
        loss_func_args=tuple(),
    ) -> Tuple[
        state_utils.EMATrainState,
        float,
        Parameters,
        Dict[str, jnp.ndarray],
    ]:
        """Single training step for the neural network.

        Args:
            state: Training state.
            loss_func: Loss function for the parameters.
            loss_func_args: Argument other than the parameters for the loss function.
        """
        (loss_value, metrics), grads = value_and_grad(loss_func, has_aux=True)(
            state.params, *loss_func_args
        )

        if cfg.training.ndevices > 1:
            loss_value = jax.lax.pmean(loss_value, axis_name="data")
            grads = jax.lax.pmean(grads, axis_name="data")
            metrics = jax.tree_util.tree_map(
                lambda x: jax.lax.pmean(x, axis_name="data"), metrics
            )

        def _all_finite(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            if not leaves:
                return jnp.array(True)
            finite_flags = [jnp.all(jnp.isfinite(x)) for x in leaves]
            return jnp.all(jnp.stack(finite_flags))

        grads_finite = _all_finite(grads)
        loss_finite = jnp.all(jnp.isfinite(loss_value))
        all_finite = grads_finite & loss_finite
        if cfg.training.ndevices > 1:
            all_finite = jax.lax.pmin(all_finite.astype(jnp.int32), axis_name="data")
            all_finite = all_finite.astype(jnp.bool_)

        def _apply_updates(curr_state):
            next_state = curr_state.apply_gradients(grads=grads)
            return next_state.replace(
                params=edm2_net.safe_project_to_sphere(cfg, next_state.params)
            )

        state = jax.lax.cond(all_finite, _apply_updates, lambda s: s, state)
        metrics = dict(metrics)
        metrics["train/loss_finite"] = loss_finite
        metrics["train/grads_finite"] = grads_finite
        metrics["train/nan_detected"] = jnp.logical_not(all_finite)

        return state, loss_value, grads, metrics

    return train_step


def setup_ema_update(
    cfg: config_dict.ConfigDict,
) -> Callable:
    """Setup the function for updating the EMA parameters on single or multiple devices."""

    decorator = jax.jit if cfg.training.ndevices == 1 else jax.pmap

    @decorator
    def update_ema_params(
        state: state_utils.EMATrainState,
    ) -> state_utils.EMATrainState:
        """Update EMA parameters."""
        new_ema_params = {}
        for ema_fac, ema_params in state.ema_params.items():
            new_ema_params[ema_fac] = jax.tree_util.tree_map(
                lambda param, ema_param: ema_fac * ema_param + (1 - ema_fac) * param,
                state.params,
                ema_params,
            )

        return state.replace(ema_params=new_ema_params)

    return update_ema_params
