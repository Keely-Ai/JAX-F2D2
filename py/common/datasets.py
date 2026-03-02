"""
Nicholas M. Boffi
10/5/25

Code for initializing common datasets.
"""

import functools
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils
from ml_collections import config_dict


def unnormalize_image(image: jnp.ndarray):
    """Unnormalize an image from [-1, 1] to [0, 1] by scaling and clipping."""
    image = (image + 1) / 2
    image = jnp.clip(image, 0.0, 1.0)
    return image


def normalize_image_tf(image: tf.Tensor):
    """Normalize an image to have pixel values in the range [-1, 1]."""
    return (2 * (image / 255)) - 1


def preprocess_celeb_a(image: tf.Tensor) -> tf.Tensor:
    """Crop an image to 140x140, then resize to 64x64 pixels."""
    image = normalize_image_tf(image)
    crop = tf.image.resize_with_crop_or_pad(image, 140, 140)
    crop64 = tf.image.resize(crop, [64, 64], method="area", antialias=True)
    return crop64


def preprocess_image(cfg, x: Dict) -> Dict:
    """Preprocess the image for TensorFlow datasets."""
    image = x["image"]

    if cfg.problem.target == "celeb_a":
        # celeb_a doesn't have labels; artificially pad them all to 1
        label = 1.0
    else:
        label = x["label"]

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    if cfg.problem.target == "cifar10" or "afhq" in cfg.problem.target:
        image = normalize_image_tf(image)
    elif cfg.problem.target == "celeb_a":
        image = preprocess_celeb_a(image)
    else:
        raise ValueError("Unknown dataset type.")

    # ensure (N, C, H, W)
    image = tf.transpose(image, [2, 0, 1])

    return {"image": image, "label": label}


def get_image_dataset(cfg: config_dict.ConfigDict):
    """Assemble a TensorFlow dataset for the specified problem target."""
    small_image_datasets = ["cifar10", "celeb_a"]
    is_small_image_dataset = cfg.problem.target in small_image_datasets
    is_afhq = "afhq" in cfg.problem.target

    if is_small_image_dataset:
        if cfg.problem.target == "cifar10":
            ds = tfds.load(
                "cifar10",
                split="train",
                shuffle_files=True,
                data_dir=cfg.problem.dataset_location,
            )
        elif cfg.problem.target == "celeb_a":
            ds = tfds.load(
                "celeb_a",
                split="train",
                shuffle_files=True,
                data_dir=cfg.problem.dataset_location,
            )
    elif is_afhq:
        load_str = f"{cfg.problem.dataset_location}/{cfg.problem.target}"
        ds = tf.data.experimental.load(load_str)

    ds = (
        ds.shuffle(10_000, reshuffle_each_iteration=True)
        .map(
            lambda x: preprocess_image(cfg, x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    return ds


def _shard_batch(cfg: config_dict.ConfigDict, batch):
    """Shard a batch along the leading axis for multi-device training."""
    if cfg.training.ndevices <= 1:
        return batch

    def _reshape(x):
        if x is None:
            return None
        return x.reshape((cfg.training.ndevices, -1, *x.shape[1:]))

    if isinstance(batch, dict):
        return {key: _reshape(value) for key, value in batch.items()}
    return _reshape(batch)


def prefetch_to_device(
    cfg: config_dict.ConfigDict, ds, buffer_size: int = 2
):
    """Prefetch dataset batches onto device(s)."""
    if buffer_size <= 0:
        return ds
    # For single-device training, avoid device_put_sharded which expects
    # a leading device axis (len(shards) == ndevices).
    if cfg.training.ndevices <= 1:
        return ds

    def _iterator():
        for batch in ds:
            yield _shard_batch(cfg, batch)

    return jax_utils.prefetch_to_device(_iterator(), buffer_size)


def sample_checkerboard(
    n_samples: int, key: jnp.ndarray, *, n_squares: int
) -> np.ndarray:
    """
    Samples the checkerboard dataset using the RealNVP-style construction
    used in try_ll.py.
    """
    del key
    del n_squares
    x1 = np.random.rand(n_samples) * 4.0 - 2.0
    x2 = np.random.rand(n_samples) - (np.random.randint(0, 2, size=n_samples) * 2.0)
    x2 = x2 + (np.floor(x1) % 2)
    x = np.stack([x1, x2], axis=1)
    return x * 2.0


def setup_base(cfg: config_dict.ConfigDict, ex_input: jnp.ndarray) -> Callable:
    """Set up the base density for the system."""
    if cfg.problem.base == "gaussian":

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: jnp.ndarray):
            rescale = jnp.asarray(cfg.network.rescale)
            return rescale * jax.random.normal(key, shape=(bs, *ex_input.shape))

    else:
        raise ValueError("Specified base density is not implemented.")

    return sample_rho0


def np_to_tfds(cfg: config_dict.ConfigDict, x1s: np.ndarray) -> tf.data.Dataset:
    """Given a NumPy array, convert to a TensorFlow dataset with batching and shuffling."""
    return (
        tf.data.Dataset.from_tensor_slices(x1s)
        .shuffle(50_000, reshuffle_each_iteration=True)
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )


def setup_target(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray):
    """Set up the target density for the system."""
    if cfg.problem.target == "checker":
        assert cfg.problem.d == 2, "Checkerboard only implemented for d=2."

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho1(num_samples: int, key: jnp.ndarray) -> jnp.ndarray:
            return sample_checkerboard(num_samples, key, n_squares=4)

        n_samples = cfg.problem.n
        key, prng_key = jax.random.split(prng_key)
        x1s = sample_rho1(n_samples, key)
        rescale_value = np.std(x1s, axis=0).tolist()
        ds = np_to_tfds(cfg, x1s)

    elif (
        cfg.problem.target == "cifar10"
        or cfg.problem.target == "celeb_a"
        or "afhq" in cfg.problem.target
    ):
        ds = get_image_dataset(cfg)
        print("Loaded image dataset.")

    else:
        raise ValueError("Specified target density is not implemented.")

    # compute standard deviation of the dataset
    if cfg.problem.gaussian_scale == "adaptive":
        # hard code
        if (
            cfg.problem.target == "cifar10"
            or cfg.problem.target == "celeb_a"
            or "afhq" in cfg.problem.target
        ):
            rescale_value = 0.5

        # for generated datasets, it's computed above
        cfg.network.rescale = rescale_value
    else:
        cfg.network.rescale = 1.0

    return cfg, ds, prng_key
