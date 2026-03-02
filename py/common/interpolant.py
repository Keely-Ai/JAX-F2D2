"""
Nicholas M. Boffi
10/5/25

Basic class for a stochastic interpolant.
"""

import dataclasses
import functools
from typing import Callable

import jax
import jax.numpy as jnp
from ml_collections import config_dict


@dataclasses.dataclass
class Interpolant:
    """Basic class for a stochastic interpolant"""

    alpha: Callable[[float], float]
    beta: Callable[[float], float]
    alpha_dot: Callable[[float], float]
    beta_dot: Callable[[float], float]

    def calc_It(self, t: float, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(self, t: float, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It(
        self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return self.calc_It(t, x0, x1)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It_dot(
        self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return self.calc_It_dot(t, x0, x1)

    def __hash__(self):
        return hash((self.alpha, self.beta))

    def __eq__(self, other):
        return self.alpha == other.alpha and self.beta == other.beta


def setup_interpolant(cfg: config_dict.ConfigDict) -> Interpolant:
    if cfg.problem.interp_type == "linear":
        interp = Interpolant(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )
    elif cfg.problem.interp_type == "trig":
        interp = Interpolant(
            alpha=lambda t: jnp.cos(jnp.pi * t / 2),
            beta=lambda t: jnp.sin(jnp.pi * t / 2),
            alpha_dot=lambda t: -0.5 * jnp.pi * jnp.sin(jnp.pi * t / 2),
            beta_dot=lambda t: 0.5 * jnp.pi * jnp.cos(jnp.pi * t / 2),
        )
    else:
        raise NotImplementedError("Interpolant type not implemented.")

    return interp
