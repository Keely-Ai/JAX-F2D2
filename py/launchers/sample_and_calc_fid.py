import functools
import importlib
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)


import click
import common.fid_utils as fid_utils
import common.flow_map as flow_map
import common.state_utils as state_utils
import flax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


@click.command()
@click.option(
    "--checkpoint",
    required=False,
    type=click.Path(exists=True, file_okay=True),
    help="Checkpoint path (if not provided, will use config.checkpoint_path)",
)
@click.option("--cfg_path", required=True)
@click.option("--n_steps", required=True, type=int)
@click.option(
    "--ema_fac",
    required=False,
    type=float,
    help="EMA factor (if not provided, will use config.sweep_params.ema_fac)",
)
@click.option("--slurm_id", type=int, required=True)
@click.option("--n_images", default=50_000, show_default=True)
@click.option("--bs", default=256, show_default=True)
@click.option("--stats", required=True, type=click.Path(exists=True))
@click.option("--out_dir", required=True)
def main(
    checkpoint: str,
    cfg_path: str,
    n_steps: int,
    ema_fac: float,
    slurm_id: int,
    n_images: int,
    bs: int,
    stats: str,
    out_dir: str,
):
    # fixed random seed
    prng_key = jax.random.PRNGKey(42)

    # load in the config
    cfg_mod = importlib.import_module(cfg_path)
    cfg = cfg_mod.get_config(slurm_id, "", "")

    # Get checkpoint and ema_fac from config if available
    if not checkpoint and hasattr(cfg, "checkpoint_path"):
        checkpoint = cfg.checkpoint_path
    if ema_fac is None and hasattr(cfg, "sweep_params") and hasattr(cfg.sweep_params, "ema_fac"):
        ema_fac = cfg.sweep_params.ema_fac

    if cfg.problem.gaussian_scale == "adaptive":
        if (
            cfg.problem.target == "cifar10"
            or cfg.problem.target == "celeb_a"
            or "afhq" in cfg.problem.target
        ):
            rescale_value = 0.5
        cfg.network.rescale = rescale_value

    if cfg.training.conditional:
        raise NotImplementedError(
            "Conditional sampling not implemented. Please set cfg.training.conditional = False."
        )

    # set up the network
    net, params, prng_key = flow_map.initialize_flow_map(
        cfg.network, jnp.zeros(cfg.problem.image_dims), prng_key
    )

    # define dummy train state for loading
    tx, _ = state_utils.setup_optimizer(cfg)
    train_state = state_utils.EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params={ema_fac: params for ema_fac in cfg.training.ema_facs},
        tx=tx,
    )
    with open(checkpoint, "rb") as f:
        train_state = flax.serialization.from_bytes(train_state, f.read())

    # unpack ema parameters
    params = train_state.ema_params[ema_fac] if ema_fac else train_state.params

    # set up FID computation
    stats = np.load(stats)
    mu_real, sigma_real = stats["mu"], stats["sigma"]
    inception = fid_utils.get_fid_network()

    # running mean / covariance online
    n_seen, mu, M2 = 0, None, None  # Welford vars

    # define base sampler
    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_rho0(bs, key):
        return cfg.network.rescale * jax.random.normal(key, (bs, *cfg.problem.image_dims))

    # sample and calculate the FID
    num_full, rem = divmod(n_images, bs)
    for _ in tqdm(range(num_full + (rem > 0))):
        curr_bs = bs if n_seen + bs <= n_images else rem
        prng_key, step_key = jax.random.split(prng_key)
        x0 = sample_rho0(curr_bs, step_key)
        imgs = flow_map.batch_sample(net.apply, params, x0, n_steps, None)
        imgs = jnp.clip(imgs, -1, 1)
        imgs = imgs.transpose(0, 2, 3, 1)  # NCHW→NHWC
        acts = fid_utils.resize_and_incept(imgs, inception)  # (B, 2048)

        # online mean and covariance
        acts = np.asarray(np.squeeze(acts))
        n_seen += curr_bs
        if mu is None:
            mu = acts.mean(0)
            M2 = np.cov(acts, rowvar=False) * (curr_bs - 1)
        else:
            delta = acts.mean(0) - mu
            mu += delta * curr_bs / n_seen
            M2 += (
                np.cov(acts, rowvar=False) * (curr_bs - 1)
                + np.outer(delta, delta) * (n_seen - curr_bs) * curr_bs / n_seen
            )

    sigma_gen = M2 / (n_seen - 1)
    fid = fid_utils.fid_from_stats(mu, sigma_gen, mu_real, sigma_real)
    print(f"FID = {fid:.4f}")

    # save fid calculation
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_name = os.path.basename(checkpoint)
    checkpoint_name = os.path.splitext(checkpoint_name)[0]
    tag = f"{checkpoint_name}_ema={ema_fac}_N={n_steps}"
    np.savez(
        os.path.join(out_dir, f"fid_{tag}.npz"),
        fid=float(fid),
        mu_gen=mu,
        sigma_gen=sigma_gen,
        mu_real=mu_real,
        sigma_real=sigma_real,
    )

    # Write to human-readable summary files if sweep_params exists in config
    if hasattr(cfg, "sweep_params"):
        import fcntl
        from datetime import datetime

        # Summary text file
        summary_file = os.path.join(os.path.dirname(out_dir), "fid_results_summary.txt")
        csv_file = os.path.join(os.path.dirname(out_dir), "fid_results.csv")

        # Get timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write to text summary with file locking
        try:
            with open(summary_file, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                if not f.read():  # File is empty, write header
                    f.write("===== CIFAR-10 Velocity Model FID Results =====\n")
                    f.write(f"Generated on: {timestamp}\n\n")
                    f.write(
                        f"{'Model':<10} {'calc_weight':<12} {'ema_fac':<10} {'FID Score':<12} {'N_steps':<10} {'Timestamp':<25}\n"
                    )
                    f.write("-" * 90 + "\n")
                # Write result
                f.write(
                    f"{cfg.sweep_params.model:<10} {str(cfg.sweep_params.calc_weight):<12} "
                    f"{cfg.sweep_params.ema_fac:<10.4f} {fid:<12.4f} {n_steps:<10} {timestamp:<25}\n"
                )
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Could not write to summary file: {e}")

        # Write to CSV with file locking
        try:
            with open(csv_file, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                if not f.read():  # File is empty, write header
                    f.write(
                        "model,calc_weight,ema_fac,fid_score,n_steps,n_images,timestamp,checkpoint\n"
                    )
                # Write result
                f.write(
                    f"{cfg.sweep_params.model},{cfg.sweep_params.calc_weight},{cfg.sweep_params.ema_fac},"
                    f"{fid},{n_steps},{n_images},{timestamp},{checkpoint_name}\n"
                )
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Could not write to CSV file: {e}")


if __name__ == "__main__":
    main()
