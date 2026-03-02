"""
Nicholas M. Boffi
10/5/25

Basic routines for flow map class.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from ml_collections import config_dict

from . import edm2_net, network_utils

Parameters = Dict[str, Dict]


class FlowMap(nn.Module):
    """Basic class for a flow map."""

    config: config_dict.ConfigDict

    def setup(self):
        """Set up the flow map."""
        self.flow_map = network_utils.setup_network(self.config)

    def __call__(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        """Apply the flow map."""
        return self.flow_map(
            s,
            t,
            x,
            label,
            train,
            calc_weight,
            return_X_and_phi,
            init_weights,
            return_div,
        )

    def partial_t(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        """Compute the partial derivative with respect to time."""
        Xst, dt_Xst = jax.jvp(
            lambda t: self.flow_map(
                s,
                t,
                x,
                label,
                train,
                calc_weight,
                return_X_and_phi=False,
                return_div=return_div,
            ),
            primals=(t,),
            tangents=(jnp.ones_like(t),),
        )

        return Xst, dt_Xst

    def partial_s(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        """Compute the partial derivative with respect to space."""
        Xst, ds_Xst = jax.jvp(
            lambda s: self.flow_map(s, t, x, label, train, calc_weight),
            primals=(s,),
            tangents=(jnp.ones_like(s),),
        )

        return Xst, ds_Xst

    def calc_weight(self, s: float, t: float) -> jnp.ndarray:
        """Compute the weights for the flow map."""
        return self.flow_map.calc_weight(s, t)

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        """Compute the flow map."""
        return self.flow_map.calc_phi(
            s,
            t,
            x,
            label=label,
            train=train,
            calc_weight=calc_weight,
            return_div=return_div,
        )

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        """Apply the flow map."""
        return self.flow_map.calc_b(
            t,
            x,
            label=label,
            train=train,
            calc_weight=calc_weight,
            return_div=return_div,
        )


def sample(
    apply_flow_map: Callable, variables: Dict, x0: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Unconditional sampling."""
    ts = jnp.linspace(0.0, 1.0, N + 1)

    def step(x, idx):
        return (
            apply_flow_map(
                variables,
                ts[idx],
                ts[idx + 1],
                x,
                label=label,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            ),
            None,
        )

    final_state, _ = jax.lax.scan(step, x0, jnp.arange(N))
    return final_state


@functools.partial(jax.jit, static_argnums=(0, 3))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, 0))
def batch_sample(
    apply_flow_map: Callable, variables: Dict, x0s: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Batch unconditional sampling."""
    return sample(apply_flow_map, variables, x0s, N, label)


@functools.partial(
    jax.pmap,
    in_axes=(None, 0, 0, None, 0),
    static_broadcasted_argnums=(0, 3),
    axis_name="data",
)
def pmap_batch_sample(
    apply_flow_map: Callable, variables: Dict, x0s: jnp.ndarray, N: int, labels
) -> jnp.ndarray:
    """Parallel batch sampling across devices."""
    return batch_sample(apply_flow_map, variables, x0s, N, labels)


def initialize_flow_map(
    network_config: config_dict.ConfigDict, ex_input: jnp.ndarray, prng_key: jnp.ndarray
) -> Tuple[nn.Module, Parameters, jnp.ndarray]:
    # define the network
    net = FlowMap(network_config)

    # initialize the parameters
    ex_s = ex_t = 0.0
    ex_label = 0
    prng_key, skey = jax.random.split(prng_key)

    params = net.init(
        {"params": prng_key, "constants": skey},
        ex_s,
        ex_t,
        ex_input,
        ex_label,
        train=False,
        init_weights=True,  # This triggers initialization of all weight params
    )

    prng_key = jax.random.split(prng_key)[0]

    if network_config.network_type == "edm2":
        params = edm2_net.project_to_sphere(params)

    return net, params, prng_key
