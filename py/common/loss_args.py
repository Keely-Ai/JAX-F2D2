"""
Nicholas M. Boffi
10/5/25

Code for setting up arguments for loss functions.
"""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import state_utils
from . import dist_utils


def safe_resize(curr_bs: int, bs: int, x: jnp.ndarray) -> jnp.ndarray:
    """Resize the input array to the current batch size."""
    if curr_bs < bs:
        x = x[:curr_bs]
    return x


def _sample_diagonal(
    key: jnp.ndarray, bs: int, tmin: float, tmax: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample points on the diagonal (s=t)."""
    s = jax.random.uniform(key, shape=(bs,), minval=tmin, maxval=tmax)
    return s, s


def _sample_triangle(
    key1: jnp.ndarray,
    key2: jnp.ndarray,
    bs: int,
    tmin: float,
    tmax: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample uniformly for whole region"""
    s = jax.random.uniform(key1, shape=(bs,), minval=tmin, maxval=tmax)
    t = jax.random.uniform(key2, shape=(bs,), minval=tmin, maxval=tmax)
    return s, t


def _get_diag_offdiag_bs(cfg: config_dict.ConfigDict, bs: int) -> Tuple[int, int]:
    """Get diagonal and off-diagonal batch sizes."""
    if hasattr(cfg.optimization, "diag_fraction"):
        diag_bs = max(1, int(bs * cfg.optimization.diag_fraction))
    elif hasattr(cfg.optimization, "diag_bs"):
        diag_bs = cfg.optimization.diag_bs
    else:
        raise ValueError("Either diag_fraction or diag_bs must be specified")

    offdiag_bs = bs - diag_bs

    return diag_bs, offdiag_bs


def _concat_diag_offdiag(
    s_diag: jnp.ndarray,
    t_diag: jnp.ndarray,
    s_offdiag: jnp.ndarray,
    t_offdiag: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Concatenate diagonal and off-diagonal samples."""
    sbatch = jnp.concatenate([s_diag, s_offdiag])
    tbatch = jnp.concatenate([t_diag, t_offdiag])
    return sbatch, tbatch


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_loss_fn_args_randomness(
    prng_key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
    diag_bs: int,
    offdiag_bs: int,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    (
        tkey,
        skey,
        ukey,
        x0key,
        tkey2,
    ) = jax.random.split(prng_key, num=5)
    x0batch = sample_rho0(cfg.optimization.bs, x0key)

    bs = cfg.optimization.bs
    tmin = cfg.training.tmin
    tmax = cfg.training.tmax

    # If offdiag_bs is 0, use full batch on diagonal
    if offdiag_bs == 0:
        sbatch, tbatch = _sample_diagonal(skey, bs, tmin, tmax)
    else:
        # sample diagonal and off-diagonal points
        s_diag, t_diag = (
            _sample_diagonal(skey, diag_bs, tmin, tmax)
            if diag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )
        s_offdiag, t_offdiag = (
            _sample_triangle(tkey, tkey2, offdiag_bs, tmin, tmax)
            if offdiag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )

        sbatch, tbatch = _concat_diag_offdiag(s_diag, t_diag, s_offdiag, t_offdiag)

    if cfg.training.psd_type == "midpoint":
        ubatch = 0.5 * (sbatch + tbatch)
        hbatch = None  # Not used for midpoint interpolation
    elif cfg.training.psd_type == "uniform":
        minval = 0.0
        maxval = 1.0

        hbatch = jax.random.uniform(
            ukey, shape=(cfg.optimization.bs,), minval=minval, maxval=maxval
        )

        ubatch = hbatch * sbatch + (1 - hbatch) * tbatch
    elif cfg.training.psd_type == None:
        ubatch = None
        hbatch = None
    else:
        raise ValueError(f"Unknown psd_type: {cfg.training.psd_type}")

    dropout_keys = jax.random.split(tkey, num=cfg.optimization.bs).reshape(
        (cfg.optimization.bs, -1)
    )
    prng_key = jax.random.split(dropout_keys[0])[0]

    if getattr(cfg.training, "batch_is_sharded", False):
        def _shard_batch(x):
            if x is None:
                return None
            return x.reshape((cfg.training.ndevices, -1, *x.shape[1:]))

        x0batch = _shard_batch(x0batch)
        sbatch = _shard_batch(sbatch)
        tbatch = _shard_batch(tbatch)
        ubatch = _shard_batch(ubatch)
        hbatch = _shard_batch(hbatch)
        dropout_keys = _shard_batch(dropout_keys)
    return (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
    )


def get_batch(
    cfg: config_dict.ConfigDict, statics: state_utils.StaticArgs, prng_key: jnp.ndarray
) -> int:
    """Extract a batch based on the structure expected for image
    or non-image datasets."""
    is_image_dataset = (cfg.problem.target in ["cifar10", "celeb_a"]) or (
        "afhq" in cfg.problem.target
    )

    batch = next(statics.ds)
    if is_image_dataset:
        x1batch = batch["image"]
        label_batch = batch["label"]
    else:
        x1batch = batch
        label_batch = None

    # add droput to randomly replace fraction cfg.class_dropout of labels by num_classes
    # if not conditional, we don't need the labels
    if not cfg.training.conditional:
        label_batch = None

    elif cfg.training.class_dropout > 0:
        assert cfg.network.use_cfg  # class dropout doesn't make sense without cfg
        if getattr(cfg.training, "batch_is_sharded", False):
            keys = jax.random.split(prng_key, cfg.training.ndevices)

            def _apply_dropout(key, labels):
                mask = jax.random.bernoulli(
                    key, cfg.training.class_dropout, shape=labels.shape
                )
                return labels.at[mask].set(cfg.problem.num_classes)

            label_batch = jax.vmap(_apply_dropout)(keys, label_batch)
            prng_key = jax.random.split(keys[0])[0]
        else:
            mask = jax.random.bernoulli(
                prng_key, cfg.training.class_dropout, shape=label_batch.shape
            )
            label_batch = label_batch.at[mask].set(cfg.problem.num_classes)
            prng_key = jax.random.split(prng_key)[0]

    return x1batch, label_batch, prng_key


def get_loss_fn_args(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> Tuple:

    # Determine batch sizes based on splitting configuration
    bs = cfg.optimization.bs

    # Normal batch splitting
    diag_bs, offdiag_bs = _get_diag_offdiag_bs(cfg, bs)

    # drew randomness needed for the objective
    (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
    ) = get_loss_fn_args_randomness(
        prng_key,
        cfg,
        statics.sample_rho0,
        diag_bs,
        offdiag_bs,
    )

    # grab next batch of samples and labels
    x1batch, label_batch, prng_key = get_batch(cfg, statics, prng_key)

    # set up teacher params
    external_teacher = statics.teacher_params
    if external_teacher is None:
        external_teacher = train_state.params
    else:
        external_teacher = jax.tree_util.tree_map(
            jax.lax.stop_gradient, external_teacher
        )
        external_teacher = dist_utils.safe_replicate(cfg, external_teacher)

    self_teacher = train_state.params

    def _select_teacher(source: str):
        if source == "external":
            return external_teacher
        if source == "self":
            return self_teacher
        raise ValueError(f"Unknown teacher source: {source}")

    diag_source = getattr(cfg.training, "diag_teacher_source", None)
    offdiag_source = getattr(cfg.training, "offdiag_teacher_source", None)
    if diag_source is None:
        diag_source = "external" if statics.teacher_params is not None else "self"
    if offdiag_source is None:
        offdiag_source = diag_source

    teacher_params_diag = _select_teacher(diag_source)
    teacher_params_offdiag = _select_teacher(offdiag_source)

    # for training flow map
    loss_fn_args = (
        x0batch,
        x1batch,
        label_batch,
        sbatch,
        tbatch,
        ubatch,
        hbatch,
        dropout_keys,
    )
    loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)
    loss_fn_args = (teacher_params_diag, teacher_params_offdiag, *loss_fn_args)

    return loss_fn_args, prng_key
