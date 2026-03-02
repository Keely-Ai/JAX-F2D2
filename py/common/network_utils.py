"""
Nicholas M. Boffi
10/5/25

Helper routines for neural network definitions.
"""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import edm2_net as edm2_net


class MLP(nn.Module):
    """Simple MLP network with square weight pattern."""

    n_hidden: int
    n_neurons: int
    output_dim: int
    act: Callable
    use_residual: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_neurons)(x)
        x = self.act(x)

        for _ in range(self.n_hidden):
            if self.use_residual:
                x = x + nn.Dense(self.n_neurons)(x)
            else:
                x = nn.Dense(self.n_neurons)(x)
            x = self.act(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class FlowMapMLP(nn.Module):
    """Simple MLP network with square weight pattern, for flow map representation."""

    config: config_dict.ConfigDict

    def setup(self):
        self.act = get_act(self.config)
        self.has_div = self.config.output_dim == 3
        self.velocity_dim = self._get_velocity_dim()

        self.trunk_mlp = MLP(
            self.config.n_hidden,
            self.config.n_neurons,
            self.config.n_neurons,
            self.act,
            self.config.use_residual,
        )

        self.v_head_dense0 = nn.Dense(self.config.n_neurons)
        self.v_head_dense1 = nn.Dense(self.velocity_dim)

        if self.has_div:
            self.div_head_dense0 = nn.Dense(self.config.n_neurons)
            self.div_head_dense1 = nn.Dense(1)

        self.weight_mlp = MLP(
            n_hidden=1,
            n_neurons=self.config.n_neurons,
            output_dim=1,
            act=jax.nn.gelu,
            use_residual=False,
        )

    def _get_velocity_dim(self) -> int:
        if hasattr(self.config, "input_dims") and self.config.input_dims is not None:
            if isinstance(self.config.input_dims, tuple):
                return int(self.config.input_dims[0])
            return int(self.config.input_dims)
        if isinstance(self.config.rescale, (list, tuple)):
            return len(self.config.rescale)
        if self.config.output_dim == 3:
            return 2
        return int(self.config.output_dim)

    def calc_weight(self, s: float, t: float) -> float:
        st = jnp.stack([s, t], axis=-1)
        # return self.weight_mlp(st)
        return 1.0

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        del label
        del train
        del init_weights  # MLP doesn't have dual weights to initialize
        rescale = jnp.asarray(self.config.rescale)
        if x.ndim == 1:
            st = jnp.stack([s, t], axis=-1)
            inp = jnp.concatenate((st, x / rescale), axis=-1)
        else:
            s_vec = jnp.broadcast_to(s, (x.shape[0],))
            t_vec = jnp.broadcast_to(t, (x.shape[0],))
            st = jnp.stack([s_vec, t_vec], axis=-1)
            inp = jnp.concatenate((st, x / rescale), axis=-1)
        features = self.trunk_mlp(inp)
        phi_st = self.v_head_dense1(self.act(self.v_head_dense0(features)))
        phi_st = rescale * phi_st

        div_st = None
        if self.has_div:
            div_st = self.div_head_dense1(self.act(self.div_head_dense0(features)))
            div_st = jnp.squeeze(div_st, axis=-1)

        if calc_weight:
            weight = self.calc_weight(s, t)
            if return_div and div_st is not None:
                return phi_st, div_st, weight
            return phi_st, weight

        if return_div and div_st is not None:
            return phi_st, div_st
        return phi_st

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        return self.calc_phi(t, t, x, label, train, calc_weight, False, return_div)

    def __call__(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight=False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        del label
        phi_rslt = self.calc_phi(
            s,
            t,
            x,
            label=None,
            train=train,
            calc_weight=calc_weight,
            init_weights=init_weights,
            return_div=return_div,
        )
        div_st = None
        weight = None

        if calc_weight:
            if return_div and self.has_div:
                phi_st, div_st, weight = phi_rslt
            else:
                phi_st, weight = phi_rslt
        else:
            if return_div and self.has_div:
                phi_st, div_st = phi_rslt
            else:
                phi_st = phi_rslt

        X_st = x + (t - s) * phi_st

        if calc_weight:
            if return_div and div_st is not None:
                return X_st, phi_st, div_st, weight
            return X_st, weight
        elif return_X_and_phi:
            if return_div and div_st is not None:
                return X_st, phi_st, div_st
            return X_st, phi_st
        else:
            if return_div and div_st is not None:
                return X_st, div_st
            return X_st


class EDM2FlowMap(nn.Module):
    """UNet architecture based on EDM2.
    Note: assumes that there is no batch dimension, to interface with the rest of the code.
    Adds a padded batch dimension to handle this.
    """

    config: config_dict.ConfigDict

    def setup(self):
        self.one_hot_dim = (
            self.config.label_dim + 1 if self.config.use_cfg else self.config.label_dim
        )
        self.net = edm2_net.PrecondFlowMap(
            img_resolution=self.config.img_resolution,
            img_channels=self.config.img_channels,
            label_dim=self.one_hot_dim,
            sigma_data=self.config.rescale,
            logvar_channels=self.config.logvar_channels,
            use_bfloat16=self.config.use_bfloat16,
            use_weight=self.config.use_weight,
            unet_kwargs=self.config.unet_kwargs,
        )

    def process_inputs(self, s: float, t: float, x: jnp.ndarray, label: float = None):
        # add batch dimension when needed
        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        if x.ndim == 3:
            x = x[None, ...]
            is_single = True
        elif x.ndim == 4:
            is_single = False
        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")

        batch_size = x.shape[0]
        s = jnp.broadcast_to(s, (batch_size,))
        t = jnp.broadcast_to(t, (batch_size,))

        # one-hot encode
        if label is not None:
            label = jnp.asarray(label)
            if label.ndim == 0:
                label = jnp.broadcast_to(label, (batch_size,))
            elif label.ndim == 1 and label.shape[0] != batch_size:
                label = jnp.broadcast_to(label, (batch_size,))
            label = jax.nn.one_hot(label, num_classes=self.one_hot_dim).reshape(
                (batch_size, -1)
            )

        return s, t, x, label, is_single

    def calc_weight(self, s: float, t: float) -> jnp.ndarray:
        # add batch dimension
        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        return self.net.calc_weight(s, t)

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        s, t, x, label, is_single = self.process_inputs(s, t, x, label)
        rslt = self.net.calc_phi(
            s, t, x, label, train, calc_weight, init_weights, return_div
        )
        div_st = None
        if calc_weight:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                phi_st, div_st, logvar = rslt
            else:
                phi_st, logvar = rslt
            if return_div and div_st is not None:
                if is_single:
                    return phi_st[0], div_st[0], logvar[0]
                return phi_st, div_st, logvar
            if is_single:
                return phi_st[0], logvar[0]
            return phi_st, logvar
        else:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 2:
                phi_st, div_st = rslt
            else:
                phi_st = rslt
            if return_div and div_st is not None:
                if is_single:
                    return phi_st[0], div_st[0]
                return phi_st, div_st
            if is_single:
                return phi_st[0]
            return phi_st

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        _, t, x, label, is_single = self.process_inputs(t, t, x, label)
        rslt = self.net.calc_b(t, x, label, train, calc_weight, return_div)
        div_st = None
        if calc_weight:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                bt, div_st, logvar = rslt
            else:
                bt, logvar = rslt
            if return_div and div_st is not None:
                if is_single:
                    return bt[0], div_st[0], logvar[0]
                return bt, div_st, logvar
            if is_single:
                return bt[0], logvar[0]
            return bt, logvar
        else:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 2:
                bt, div_st = rslt
            else:
                bt = rslt
            if return_div and div_st is not None:
                if is_single:
                    return bt[0], div_st[0]
                return bt, div_st
            if is_single:
                return bt[0]
            return bt

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
    ):
        s, t, x, label, is_single = self.process_inputs(s, t, x, label)
        rslt = self.net(
            s, t, x, label, train, calc_weight, return_X_and_phi, init_weights, return_div
        )
        div_st = None

        if calc_weight:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                Xst, div_st, logvar = rslt
            else:
                Xst, logvar = rslt
            if return_div and div_st is not None:
                if is_single:
                    return Xst[0], div_st[0], logvar[0]
                return Xst, div_st, logvar
            if is_single:
                return Xst[0], logvar[0]
            return Xst, logvar
        elif return_X_and_phi:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                Xst, phi_st, div_st = rslt
            else:
                Xst, phi_st = rslt
            if return_div and div_st is not None:
                if is_single:
                    return Xst[0], phi_st[0], div_st[0]
                return Xst, phi_st, div_st
            if is_single:
                return Xst[0], phi_st[0]
            return Xst, phi_st
        else:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 2:
                Xst, div_st = rslt
            else:
                Xst = rslt
            if return_div and div_st is not None:
                if is_single:
                    return Xst[0], div_st[0]
                return Xst, div_st
            if is_single:
                return Xst[0]
            return Xst


def get_act(
    config: config_dict.ConfigDict,
) -> Callable:
    """Get the activation function for the network.

    Args:
        config: Configuration dictionary.
    """
    if config.act == "gelu":
        return jax.nn.gelu
    elif config.act == "swish" or config.act == "silu":
        return jax.nn.silu
    else:
        raise ValueError(f"Activation function {config.activation} not recognized.")


def setup_network(
    network_config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """
    if "mlp" in network_config.network_type:
        return FlowMapMLP(config=network_config)
    elif network_config.network_type == "edm2":
        return EDM2FlowMap(config=network_config)
    else:
        raise ValueError(f"Network type {network_config.network_type} not recognized.")
