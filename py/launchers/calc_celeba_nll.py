import argparse
import functools
import os
import sys
from typing import Iterable, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import common.flow_map as flow_map
import common.state_utils as state_utils
from configs import celeba64


def _preprocess_celeb_a_with_dequant(
    example: dict, *, dequantize: bool
) -> tf.Tensor:
    image = tf.cast(example["image"], tf.float32)
    if dequantize:
        noise = tf.random.uniform(tf.shape(image), 0.0, 1.0, dtype=tf.float32)
        image = (image + noise) / 256.0
    else:
        image = image / 255.0

    image = (2.0 * image) - 1.0
    crop = tf.image.resize_with_crop_or_pad(image, 140, 140)
    crop64 = tf.image.resize(crop, [64, 64], method="area", antialias=True)
    return tf.transpose(crop64, [2, 0, 1])


def _get_test_dataset(
    data_dir: str, batch_size: int, *, dequantize: bool
) -> Iterable[np.ndarray]:
    ds = tfds.load(
        "celeb_a",
        split="test",
        shuffle_files=False,
        data_dir=data_dir,
    )
    ds = (
        ds.map(
            lambda x: _preprocess_celeb_a_with_dequant(x, dequantize=dequantize),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds.as_numpy_iterator()


def _base_log_prob(x: jnp.ndarray, rescale: float) -> jnp.ndarray:
    sigma = jnp.asarray(rescale, dtype=x.dtype)
    dims = tuple(range(1, x.ndim))
    quad = jnp.sum((x / sigma) ** 2, axis=dims)
    d = x[0].size
    log_det = d * jnp.log(sigma**2)
    log_norm = 0.5 * (d * jnp.log(2 * jnp.pi) + log_det)
    return -0.5 * quad - log_norm


@functools.partial(jax.jit, static_argnums=(0, 3))
def _inverse_logp_euler(
    apply_fn,
    params,
    x_t: jnp.ndarray,
    n_steps: int,
    rescale: float,
) -> jnp.ndarray:
    dt = -1.0 / n_steps
    t_curr = jnp.ones((x_t.shape[0],), dtype=jnp.float32)
    x = x_t
    delta_logp = jnp.zeros((x_t.shape[0],), dtype=jnp.float32)

    def body(_, state):
        t_curr, x, delta_logp = state
        t_next = t_curr + dt
        b, div = apply_fn(
            params,
            t_curr,
            t_next,
            x,
            None,
            train=False,
            method="calc_phi",
            return_div=True,
        )
        div = div * 10000.0
        jax.debug.print("t={}, div={}", t_curr[0], div.mean())
        x = x + dt * b
        delta_logp = delta_logp - dt * div
        return t_next, x, delta_logp

    t_curr, x0, delta_logp = jax.lax.fori_loop(
        0, n_steps, body, (t_curr, x, delta_logp)
    )
    logp0 = _base_log_prob(x0, rescale)
    # jax.debug.print("rescale={}",rescale)
    return logp0 - delta_logp


def _compute_bpd(
    logp_z: jnp.ndarray, d: int, *, dequantize: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    log2_const = jnp.log(2.0)
    log_det_transform = d * jnp.log(2.0)
    total_logp = logp_z + log_det_transform
    if dequantize:
        total_logp = total_logp - d * jnp.log(256.0)
    bpd = -(total_logp / (d * log2_const))
    return bpd, total_logp


def _load_teacher_params(cfg, prng_key):
    net, params, prng_key = flow_map.initialize_flow_map(
        cfg.network, jnp.zeros(cfg.problem.image_dims), prng_key
    )
    tx, _ = state_utils.setup_optimizer(cfg)
    train_state = state_utils.EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params={ema_fac: params for ema_fac in cfg.training.ema_facs},
        tx=tx,
    )
    with open(cfg.teacher.load_path, "rb") as f:
        train_state = flax.serialization.from_bytes(train_state, f.read())
    ema_fac = getattr(cfg.teacher, "ema_fac", None)
    if ema_fac is not None:
        return net.apply, train_state.ema_params[ema_fac]
    return net.apply, train_state.params


def parse_args():
    parser = argparse.ArgumentParser("CelebA NLL (Euler, flow-maps)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/xinyueai/Experiments/data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/user_data/xinyueai/flow-maps/celeba-lsd-100-0.5/celeba_paper_lsd_12.pkl",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--no_dequantize",
        action="store_true",
        help="Disable uniform dequantization noise.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prng_key = jax.random.PRNGKey(args.seed)

    dequantize = not args.no_dequantize

    cfg = celeba64.get_config(
        slurm_id=0,
        dataset_location=args.data_dir,
        output_folder="/tmp",
    )
    cfg.teacher.load_path = args.checkpoint
    cfg.network.load_path = args.checkpoint
    cfg.training.ndevices = 1

    apply_fn, params = _load_teacher_params(cfg, prng_key)
    data_iter = _get_test_dataset(args.data_dir, args.batch_size, dequantize=dequantize)

    n_seen = 0
    bpd_sum = 0.0
    logp_sum = 0.0
    d = int(np.prod(cfg.problem.image_dims))

    for batch in data_iter:
        x = jnp.asarray(batch)
        logp_z = _inverse_logp_euler(
            apply_fn,
            params,
            x,
            args.n_steps,
            cfg.network.rescale,
        )
        bpd, total_logp = _compute_bpd(logp_z, d, dequantize=dequantize)
        bpd_sum += float(jnp.sum(bpd))
        logp_sum += float(jnp.sum(total_logp))
        n_seen += x.shape[0]
        print(f"Seen {n_seen} samples. Current bpd={bpd_sum / n_seen:.4f}")

    avg_bpd = bpd_sum / max(n_seen, 1)
    avg_logp = logp_sum / max(n_seen, 1)
    print(f"Average BPD over test set: {avg_bpd:.4f}")
    print(f"Average logp over test set: {avg_logp:.4f}")


if __name__ == "__main__":
    main()
