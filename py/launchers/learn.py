"""
Nicholas M. Boffi
10/5/25

Main training loop for self-distillation of flow maps.
"""

# isort: off
import os
import pathlib
import sys

# Set up path for imports FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

# Suppress TensorFlow logging before any TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# Optional: enable PTX-only compilation when supported
def _append_xla_flag(flag: str) -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if flag not in flags:
        flags = f"{flags} {flag}".strip()
        os.environ["XLA_FLAGS"] = flags

if os.environ.get("ENABLE_XLA_PTX", "0") == "1":
    _append_xla_flag("--xla_gpu_force_ptx_compilation")

# Force TensorFlow to use CPU only for data loading - no GPU ops
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # Hide all GPUs from TensorFlow
# isort: on
#
import argparse
import importlib
import time
from typing import Dict, Tuple

import common.datasets as datasets
import common.dist_utils as dist_utils
import common.fid_utils as fid_utils
import common.interpolant as interpolant
import common.logging as logging
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import wandb
from ml_collections import config_dict  # type: ignore
from tqdm.auto import tqdm as tqdm

Parameters = Dict[str, Dict]
mpl.rc_file(f"{pathlib.Path(__file__).resolve().parent}/matplotlibrc")


def train_loop(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: np.ndarray,
) -> None:
    """Carry out the training loop."""

    logging.register_signal_handlers(cfg, train_state)
    log_freq = getattr(cfg.logging, "log_freq", 1)

    pbar = tqdm(range(cfg.optimization.total_steps))
    for step in pbar:
        # construct loss function arguments
        start_time = time.time()
        loss_fn_args, prng_key = statics.get_loss_fn_args(
            cfg, statics, train_state, prng_key
        )

        # take a step on the loss
        train_state, loss_value, grads, metrics = statics.train_step(
            train_state, statics.loss, loss_fn_args
        )
        end_time = time.time()

        # compute update to EMA params
        train_state = statics.update_ema_params(train_state)

        # log to wandb
        if log_freq <= 1 or (step % log_freq) == 0:
            prng_key = logging.log_metrics(
                cfg,
                statics,
                train_state,
                grads,
                loss_value,
                metrics,
                loss_fn_args,
                prng_key,
                end_time - start_time,
            )

            pbar.set_postfix(loss=loss_value)

    # dump one final time
    logging.save_state(train_state, cfg)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Direct flow map learning.")
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--slurm_id", type=int)
    parser.add_argument("--dataset_location", type=str)
    parser.add_argument("--output_folder", type=str)
    return parser.parse_args()


def setup_config_dict():
    args = parse_command_line_arguments()
    cfg_module = importlib.import_module(args.cfg_path)
    return cfg_module.get_config(
        args.slurm_id, args.dataset_location, args.output_folder
    )


def setup_state(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray) -> Tuple[
    config_dict.ConfigDict,
    state_utils.StaticArgs,
    state_utils.EMATrainState,
    jnp.ndarray,
]:
    """Construct static arguments and training state objects."""
    # define dataset
    cfg, ds, prng_key = datasets.setup_target(cfg, prng_key)
    ex_input = next(ds)
    if isinstance(ex_input, dict):  # handle image datasets
        ex_input = ex_input["image"][0]
    else:
        ex_input = ex_input[0]
    interp = interpolant.setup_interpolant(cfg)

    # prefetch batches to device(s) after grabbing an example
    cfg.training.batch_is_sharded = cfg.training.ndevices > 1
    ds = datasets.prefetch_to_device(cfg, ds, buffer_size=2)
    cfg = config_dict.FrozenConfigDict(cfg)

    # define training state
    train_state, net, schedule, prng_key = state_utils.setup_training_state(
        cfg,
        ex_input,
        prng_key,
    )

    # load optional teacher params
    teacher_params = state_utils.load_teacher_params(cfg, train_state)

    # define the loss
    loss = losses.setup_loss(cfg, net, interp)

    # initialize FID network if FID computation is enabled
    inception_fn = None
    if hasattr(cfg.logging, "fid_freq") and cfg.logging.fid_freq > 0:
        print("Initializing Inception network for FID computation...")
        inception_fn = fid_utils.get_fid_network()
        print("Inception network initialized.")

    # define static object
    statics = state_utils.StaticArgs(
        net=net,
        schedule=schedule,
        loss=loss,
        get_loss_fn_args=loss_args.get_loss_fn_args,
        train_step=updates.setup_train_step(cfg),
        update_ema_params=updates.setup_ema_update(cfg),
        ds=ds,
        interp=interp,
        sample_rho0=datasets.setup_base(cfg, ex_input),
        inception_fn=inception_fn,
        teacher_params=teacher_params,
    )

    train_state = dist_utils.safe_replicate(cfg, train_state)

    return cfg, statics, train_state, prng_key


if __name__ == "__main__":
    print("Entering main. Setting up config dict and PRNG key.")
    cfg = setup_config_dict()

    # Populate JAX device information for single-node multi-GPU training
    cfg.training.ndevices = jax.device_count()
    print(f"Initialized with {cfg.training.ndevices} local GPUs")

    prng_key = jax.random.PRNGKey(cfg.training.seed)

    # Set up weights and biases tracking
    print("Setting up wandb.")
    wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.wandb_name,
        config=cfg.to_dict(),
    )

    print("Config dict set up. Setting up static arguments and training state.")
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)

    train_loop(cfg, statics, train_state, prng_key)
