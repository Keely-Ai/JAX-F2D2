"""
Nicholas M. Boffi
10/5/25

jax port of the EDM2 UNet architecture with positional embeddings.
"""

import functools
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

Parameters = Dict[str, Dict]


def safe_project_to_sphere(cfg: config_dict.ConfigDict, params: Parameters):
    if cfg.network.network_type == "edm2":
        return project_to_sphere(params)
    else:
        return params


@functools.partial(jax.pmap, axis_name="data")
def pmap_project_to_sphere(params: dict):
    """Project parameter dictionary to sphere."""
    return project_to_sphere(params)


@jax.jit
def project_to_sphere(params: dict):
    """Project parameter dictionary to sphere."""
    flat = flax.traverse_util.flatten_dict(params)
    projected = {k: project_weight_to_sphere(k, v) for k, v in flat.items()}
    return flax.traverse_util.unflatten_dict(projected)


def project_weight_to_sphere(key: str, val: np.ndarray):
    """Project weight to sphere only if it is an MPConv weight."""
    return jax.lax.cond(
        "mpconv_weight" in key, lambda _: normalize(val), lambda _: val, None
    )


def multi_axis_norm(x: jnp.ndarray, axis: Tuple[int] = (1, 2, 3)):
    """Compute the norm of a tensor over multiple axes."""
    return jnp.sqrt(jnp.sum(x.astype(jnp.float32) ** 2, axis=axis, keepdims=True))


def normalize(x: jnp.ndarray, dim: Tuple = None, eps: float = 1e-4):
    """Normalize tensor to unit magnitude with respect to given dimensions."""
    if dim is None:
        dim = tuple(np.arange(1, x.ndim))

    # calculate norm along specified dimensions
    norm = multi_axis_norm(x, axis=dim).astype(jnp.float32)

    # calculate the scaling factor for the norm
    norm_size = np.prod(norm.shape)
    x_size = np.prod(x.shape)

    # add epsilon and scale
    norm = eps + norm * jnp.sqrt(norm_size / x_size)

    return x / norm.astype(x.dtype)


def resample(x: jnp.ndarray, f: List = [1, 1], mode: str = "keep"):
    """Upsample or downsample tensor with given filter."""
    if mode == "keep":
        return x

    f = jnp.array(f, dtype=jnp.float32)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / jnp.sum(f)
    f = jnp.outer(f, f)[jnp.newaxis, jnp.newaxis, :, :]
    f = f.astype(x.dtype)
    c = x.shape[1]  # number of channels: x == [B, C, H, W]

    # Expand filter to match input channels (for depthwise convolution)
    f_expanded = jnp.tile(f, (c, 1, 1, 1))

    if mode == "down":
        # Depthwise convolution for downsampling
        return jax.lax.conv_general_dilated(
            x,
            f_expanded,
            window_strides=(2, 2),
            padding=((pad, pad), (pad, pad)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=c,
        )

    assert mode == "up"
    # Transpose convolution for upsampling
    return jax.lax.conv_general_dilated(
        x,
        f_expanded * 4,
        lhs_dilation=(2, 2),
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=c,
    )


def mp_silu(x: jnp.ndarray):
    """Magnitude-preserving SiLU activation."""
    return jax.nn.swish(x) / 0.596


def mp_sum(a: jnp.ndarray, b: jnp.ndarray, t: float = 0.5):
    """Magnitude-preserving sum."""
    # Linear interpolation: a * (1-t) + b * t
    lerp_result = a * (1 - t) + b * t
    return lerp_result / jnp.sqrt((1 - t) ** 2 + t**2)


def mp_cat(a: jnp.ndarray, b: jnp.ndarray, dim: int = 1, t: float = 0.5):
    """Magnitude-preserving concatenation."""
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = jnp.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / jnp.sqrt(Na) * (1 - t)
    wb = C / jnp.sqrt(Nb) * t
    return jnp.concatenate([wa * a, wb * b], axis=dim)


class MPPositionalEmbedding(nn.Module):
    """
    Deterministic positional embedding with magnitude-preserving scaling.
    """

    dim: int
    max_period: float = 10000.0

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        half = self.dim // 2
        assert half % 2 == 0

        # create logarithmically spaced frequencies
        freqs = jnp.exp(
            -jnp.log(self.max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )

        # compute embeddings
        args = jnp.outer(timesteps.astype(jnp.float32), freqs)

        # apply √2 scaling factor for magnitude preservation
        cos_embeddings = jnp.cos(args) * jnp.sqrt(2.0)
        sin_embeddings = jnp.sin(args) * jnp.sqrt(2.0)

        # concatenate cosine and sine components
        embedding = jnp.concatenate([cos_embeddings, sin_embeddings], axis=-1)

        return embedding.astype(timesteps.dtype)


class MPConv(nn.Module):
    in_channels: int
    out_channels: int
    kernel: tuple = ()

    def setup(self):
        if not self.kernel:  # Empty kernel means linear layer
            self.kernel_shape = (self.out_channels, self.in_channels)
        else:
            self.kernel_shape = (self.out_channels, self.in_channels, *self.kernel)

        self.weight = self.param("mpconv_weight", jax.random.normal, self.kernel_shape)

    def __call__(self, x, gain=1):
        w = self.weight.astype(jnp.float32)
        w = normalize(w)
        w_size = np.prod(w.shape[1:])
        w = w * (gain / jnp.sqrt(w_size))
        w = w.astype(x.dtype)

        if len(w.shape) == 2:  # linear layer
            return x @ w.T
        else:  # conv layer
            assert len(w.shape) == 4
            padding = [
                (w.shape[-1] // 2, w.shape[-1] // 2),
                (w.shape[-1] // 2, w.shape[-1] // 2),
            ]

            return jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding=padding,
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )


class Block(nn.Module):
    # Keep the same parameter order and defaults as PyTorch
    in_channels: int  # Number of input channels
    out_channels: int  # Number of output channels
    emb_channels: int  # Number of embedding channels
    flavor: str = "enc"  # Flavor: 'enc' or 'dec'
    resample_mode: str = "keep"  # Resampling: 'keep', 'up', or 'down'
    resample_filter: List[int] = field(
        default_factory=lambda: [1, 1]
    )  # Resampling filter
    attention: bool = False  # Include self-attention?
    channels_per_head: int = 64  # Number of channels per attention head
    dropout: float = 0  # Dropout probability
    res_balance: float = 0.3  # Balance between main branch (0) and residual branch (1)
    attn_balance: float = 0.3  # Balance between main branch (0) and self-attention (1)
    clip_act: Optional[int] = 256  # Clip output activations

    def setup(self):
        self.num_heads = (
            self.out_channels // self.channels_per_head if self.attention else 0
        )

        self.emb_gain = self.param("emb_gain", nn.initializers.zeros, ())
        self.conv_res0 = MPConv(
            self.out_channels if self.flavor == "enc" else self.in_channels,
            self.out_channels,
            kernel=(3, 3),
        )
        self.emb_linear = MPConv(self.emb_channels, self.out_channels)
        self.conv_res1 = MPConv(self.out_channels, self.out_channels, kernel=(3, 3))
        self.dropout_layer = nn.Dropout(rate=self.dropout) if self.dropout > 0 else None

        if self.in_channels != self.out_channels:
            self.conv_skip = MPConv(self.in_channels, self.out_channels, kernel=(1, 1))
        else:
            self.conv_skip = None

        if self.num_heads != 0:
            self.attn_qkv = MPConv(
                self.out_channels, self.out_channels * 3, kernel=(1, 1)
            )
            self.attn_proj = MPConv(self.out_channels, self.out_channels, kernel=(1, 1))
        else:
            self.attn_qkv = None
            self.attn_proj = None

    def __call__(self, x, emb, train=False):
        # Main branch
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        c = jnp.expand_dims(jnp.expand_dims(c, 2), 3)
        y = mp_silu(y * c.astype(y.dtype))

        if train and self.dropout_layer is not None:
            y = self.dropout_layer(y, deterministic=not train)

        y = self.conv_res1(y)

        # Connect branches
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention
        if self.num_heads > 0:
            y = self.attn_qkv(x)

            batch_size = y.shape[0]
            height, width = y.shape[2], y.shape[3]
            y = y.reshape(batch_size, self.num_heads, -1, 3, height * width)
            y = normalize(y, dim=2)
            q = y[:, :, :, 0, :]
            k = y[:, :, :, 1, :]
            v = y[:, :, :, 2, :]

            scale = jnp.sqrt(q.shape[2])
            attention_weights = jnp.einsum("nhcq,nhck->nhqk", q, k / scale)
            attention_weights = jax.nn.softmax(attention_weights, axis=3)
            y = jnp.einsum("nhqk,nhck->nhcq", attention_weights, v)
            y = y.reshape(*x.shape)
            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations
        if self.clip_act is not None:
            x = jnp.clip(x, -self.clip_act, self.clip_act)

        return x


class EDM2FlowMapUNet(nn.Module):
    img_resolution: int  # Image resolution
    img_channels: int  # Image channels
    label_dim: int  # Class label dimensionality. 0 = unconditional
    model_channels: int = 192  # Base multiplier for the number of channels
    channel_mult: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4]
    )  # Channel multipliers
    channel_mult_noise: Optional[int] = None  # Noise embedding multiplier
    channel_mult_emb: Optional[int] = None  # Final embedding multiplier
    num_blocks: int = 3  # Number of residual blocks per resolution
    attn_resolutions: List[int] = field(
        default_factory=lambda: [16, 8]
    )  # Resolutions with attention
    label_balance: float = 0.5  # Balance between noise and class embedding
    concat_balance: float = 0.5  # Balance between skip connections and main path
    block_kwargs: dict = field(default_factory=dict)  # Arguments for Block
    predict_divergence: bool = False  # Whether to output a scalar divergence head

    def setup(self):
        cblock = [self.model_channels * x for x in self.channel_mult]
        cst = (
            self.model_channels * self.channel_mult_noise
            if self.channel_mult_noise is not None
            else cblock[0]
        )
        cemb = (
            self.model_channels * self.channel_mult_emb
            if self.channel_mult_emb is not None
            else max(cblock)
        )

        # store parameters
        self.cblock = cblock
        self.cst = cst
        self.cemb = cemb
        self.out_gain = self.param("out_gain", nn.initializers.zeros, ())

        # Embedding layers - using positional embeddings
        self.emb_s_fourier = MPPositionalEmbedding(cst)
        self.emb_t_fourier = MPPositionalEmbedding(cst)

        self.emb_s_linear = MPConv(cst, cemb)
        self.emb_t_linear = MPConv(cst, cemb)

        # Class embedding if needed
        if self.label_dim != 0:
            self.emb_label = MPConv(self.label_dim, cemb)
        else:
            self.emb_label = None

        # Encoder modules dictionary
        enc = {}
        cout = self.img_channels + 1  # Start with image channels + 1 for constant

        # Create encoder modules
        for level, channels in enumerate(cblock):
            res = self.img_resolution >> level

            if level == 0:
                # Initial convolution
                cin = cout
                cout = channels
                enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=(3, 3))
            else:
                # Downsample block
                enc[f"{res}x{res}_down"] = Block(
                    cout,
                    cout,
                    cemb,
                    flavor="enc",
                    resample_mode="down",
                    **self.block_kwargs,
                )

            # Regular blocks at this resolution
            for idx in range(self.num_blocks):
                cin = cout
                cout = channels
                enc[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="enc",
                    attention=(res in self.attn_resolutions),
                    **self.block_kwargs,
                )

        # assign to construct encoder dictionary
        self.enc = enc

        # Decoder
        dec = {}
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = self.img_resolution >> level

            if level == len(cblock) - 1:
                dec[f"{res}x{res}_in0"] = Block(
                    cout, cout, cemb, flavor="dec", attention=True, **self.block_kwargs
                )

                dec[f"{res}x{res}_in1"] = Block(
                    cout, cout, cemb, flavor="dec", **self.block_kwargs
                )
            else:
                dec[f"{res}x{res}_up"] = Block(
                    cout,
                    cout,
                    cemb,
                    flavor="dec",
                    resample_mode="up",
                    **self.block_kwargs,
                )
            for idx in range(self.num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels

                dec[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="dec",
                    attention=(res in self.attn_resolutions),
                    **self.block_kwargs,
                )

        # assign to construct decoder dictionary
        self.dec = dec

        self.out_conv = MPConv(cout, self.img_channels, kernel=(3, 3))

        if self.predict_divergence:
            # Scalar head for divergence prediction.
            self.scalar_dense0 = nn.Dense(32)
            self.scalar_dense1 = nn.Dense(8)
            self.scalar_dense2 = nn.Dense(1)

    def __call__(self, x, ss, ts, class_labels, train=False):
        # process embeddings
        emb_s = self.emb_s_linear(self.emb_s_fourier(ss))
        emb_t = self.emb_t_linear(self.emb_t_fourier(ts))
        emb = mp_sum(emb_s, emb_t, t=0.5)
        if self.emb_label is not None:
            class_emb = self.emb_label(class_labels * jnp.sqrt(class_labels.shape[1]))
            emb = mp_sum(emb, class_emb, t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder forward pass
        x = jnp.concatenate([x, jnp.ones_like(x[:, :1])], axis=1)
        skips = []
        for name, block in self.enc.items():
            if "conv" in name:
                x = block(x)  # convolution doesn't use embedding
            else:
                x = block(x, emb, train=train)

            # Store skip connection
            skips.append(x)

        # Decoder forward pass
        for name, block in self.dec.items():
            if "block" in name:
                skip = skips.pop()  # get skip connection
                x = mp_cat(x, skip, t=self.concat_balance)
            x = block(x, emb, train=train)

        div = None
        if self.predict_divergence:
            flat = x.reshape((x.shape[0], -1))
            div = self.scalar_dense0(flat)
            div = nn.silu(div)
            div = self.scalar_dense1(div)
            div = nn.silu(div)
            div = self.scalar_dense2(div)
            div = jnp.squeeze(div, axis=-1)

        # final convolution
        x = self.out_conv(x, gain=self.out_gain)

        if self.predict_divergence:
            return x, div
        return x


class PrecondFlowMap(nn.Module):
    img_resolution: int  # Image resolution
    img_channels: int  # Image channels
    label_dim: int  # Class label dimensionality
    sigma_data: float = 0.5  # Expected std of training data
    logvar_channels: int = 128  # Dimensionality for uncertainty estimation
    use_bfloat16: bool = False  # Use bfloat16 precision
    use_weight: bool = True  # Use uncertainty weights
    unet_kwargs: dict = field(default_factory=dict)  # Additional UNet kwargs

    def setup(self):
        self.unet = EDM2FlowMapUNet(
            img_resolution=self.img_resolution,
            img_channels=self.img_channels,
            label_dim=self.label_dim,
            **self.unet_kwargs,
        )

        # Single weight function - using positional embeddings
        self.logvar_fourier_s = MPPositionalEmbedding(self.logvar_channels)
        self.logvar_fourier_t = MPPositionalEmbedding(self.logvar_channels)

        self.logvar_linear = MPConv(self.logvar_channels, 1)

    def calc_weight(self, ss: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
        """Calculate weight function."""
        if not self.use_weight:
            return jnp.ones_like(ss, dtype=jnp.float32)

        embed_s = self.logvar_fourier_s(ss).reshape(-1, self.logvar_channels)
        embed_t = self.logvar_fourier_t(ts).reshape(-1, self.logvar_channels)
        embed = mp_sum(embed_s, embed_t, t=0.5)
        logvar = self.logvar_linear(embed).reshape(-1, 1, 1, 1)
        return logvar

    def process_input(
        self, ss: jnp.ndarray, ts: jnp.ndarray, xs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Process inputs for the model using s_dt parameterization."""
        ss = ss.astype(jnp.float32).reshape(-1, 1, 1, 1)
        ts = ts.astype(jnp.float32).reshape(-1, 1, 1, 1)
        xs = xs.astype(jnp.float32)

        # Transform from (s, t) to (s, dt) where dt = t - s
        ts = ts - ss  # ts now represents the interval dt = t - s

        return ss, ts, xs

    def process_label(self, class_labels: jnp.ndarray) -> Optional[jnp.ndarray]:
        """Process class labels for the model."""
        if self.label_dim == 0 or class_labels is None:
            return None
        else:
            return class_labels.astype(jnp.float32).reshape(-1, self.label_dim)

    def calc_phi(
        self,
        ss: jnp.ndarray,
        ts: jnp.ndarray,
        xs: jnp.ndarray,
        class_labels: jnp.ndarray = None,
        train: bool = False,
        calc_weight: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        ss, ts, xs = self.process_input(ss, ts, xs)
        class_labels = self.process_label(class_labels)

        c_out = self.sigma_data
        c_in = 1.0 / self.sigma_data

        # Run the model
        xs_in = (c_in * xs).astype(dtype)
        unet_out = self.unet(xs_in, ss, ts, class_labels, train=train)
        div_st = None
        if isinstance(unet_out, tuple):
            phi_raw, div_raw = unet_out
        else:
            phi_raw, div_raw = unet_out, None
        phi_st = c_out * phi_raw.astype(jnp.float32)
        if div_raw is not None:
            div_st = c_out * div_raw.astype(jnp.float32)

        if init_weights:
            # During initialization, ensure weight params are created
            _ = self.calc_weight(ss, ts)

        if calc_weight:
            logvar = self.calc_weight(ss, ts)[0]
            if return_div and div_st is not None:
                return phi_st, div_st, logvar
            return phi_st, logvar

        if return_div and div_st is not None:
            return phi_st, div_st
        return phi_st

    def calc_b(
        self,
        ts: float,
        xs: jnp.ndarray,
        class_labels: jnp.ndarray = None,
        train: bool = False,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        return self.calc_phi(
            ts,
            ts,
            xs,
            class_labels=class_labels,
            train=train,
            calc_weight=calc_weight,
            return_div=return_div,
        )

    def __call__(
        self,
        ss: jnp.ndarray,
        ts: jnp.ndarray,
        xs: jnp.ndarray,
        class_labels: jnp.ndarray = None,
        train: bool = False,
        calc_weight: bool = False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        phi_rslt = self.calc_phi(
            ss,
            ts,
            xs,
            class_labels,
            train=train,
            calc_weight=calc_weight,
            init_weights=init_weights,
            return_div=return_div,
        )

        div_st = None
        if calc_weight:
            if return_div and isinstance(phi_rslt, tuple) and len(phi_rslt) == 3:
                phi_st, div_st, logvar = phi_rslt
            else:
                phi_st, logvar = phi_rslt
        else:
            if return_div and isinstance(phi_rslt, tuple) and len(phi_rslt) == 2:
                phi_st, div_st = phi_rslt
            else:
                phi_st = phi_rslt

        Xst = xs + (ts - ss) * phi_st

        # estimate uncertainty if requested
        if calc_weight:
            if return_div and div_st is not None:
                return Xst, div_st, logvar
            return Xst, logvar

        if return_X_and_phi:
            if return_div and div_st is not None:
                return Xst, phi_st, div_st
            return Xst, phi_st

        if return_div and div_st is not None:
            return Xst, div_st
        return Xst
