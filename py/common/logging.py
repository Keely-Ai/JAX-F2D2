"""
Nicholas M. Boffi
10/5/25

Code for basic wandb visualization and logging.
"""

import functools
import signal
import sys
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import wandb
from flax.serialization import to_bytes
from jax.flatten_util import ravel_pytree
from matplotlib import pyplot as plt
from ml_collections import config_dict

from . import datasets, dist_utils, fid_utils, flow_map, state_utils

Parameters = Dict[str, Dict]


def get_params_for_sampling(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    param_type: str = "visual",
) -> jnp.ndarray:
    """
    Get the appropriate parameters for sampling (visualization or FID).

    Args:
        cfg: Configuration
        train_state: Current training state
        param_type: Either "visual" or "fid" to select the right config parameter

    Returns:
        Parameters to use for sampling (unreplicated for single-device use)
    """
    # Determine which config parameter to check based on param_type
    if param_type == "visual":
        config_param = "visual_ema_factor"
    elif param_type == "fid":
        config_param = "fid_ema_factor"
    elif param_type == "bpd":
        config_param = "bpd_ema_factor"
    else:
        raise ValueError(f"Unknown param_type: {param_type}")

    # Select which parameters to use
    if (
        hasattr(cfg.logging, config_param)
        and getattr(cfg.logging, config_param) is not None
    ):
        ema_factor = getattr(cfg.logging, config_param)
        # Use EMA parameters with specified factor
        if ema_factor in train_state.ema_params:
            params = train_state.ema_params[ema_factor]
        else:
            print(
                f"Warning: EMA factor {ema_factor} not found in ema_params. Using instantaneous params."
            )
            params = train_state.params
    else:
        # Use instantaneous parameters (default)
        params = train_state.params

    # Visual/BPD uses unreplicated params, FID uses replicated params for pmap
    if param_type in ("visual", "bpd"):
        return dist_utils.safe_unreplicate(cfg, params)
    else:
        return params


def compute_fid_on_the_fly(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    n_samples: int = 10000,
    batch_size: int = 256,
    n_steps_flow: int = 8,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute FID on the fly during training using distributed sampling.

    Args:
        cfg: Configuration
        statics: Static arguments containing dataset stats
        train_state: Current training state
        prng_key: Random key
        n_samples: Number of samples to generate (default 10,000)
        batch_size: Batch size for sampling (will be split across devices)
        n_steps_flow: Number of steps for flow map models

    Returns:
        Tuple of FID score and updated PRNG key
    """
    # Check if FID reference statistics are available
    if not hasattr(cfg.logging, "fid_stats_path"):
        print(
            "Warning: No FID reference statistics path configured. Set cfg.logging.fid_stats_path"
        )
        return jnp.nan, prng_key

    # Load reference statistics
    try:
        fid_stats = np.load(cfg.logging.fid_stats_path)
        mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
    except FileNotFoundError:
        print(f"Warning: FID stats file not found at {cfg.logging.fid_stats_path}")
        return jnp.nan, prng_key
    except Exception as e:
        print(f"Warning: Error loading FID stats: {e}")
        return jnp.nan, prng_key

    # Get number of devices and adjust batch size
    per_device_batch_size = batch_size // cfg.training.ndevices
    if batch_size % cfg.training.ndevices != 0:
        per_device_batch_size += 1
        batch_size = per_device_batch_size * cfg.training.ndevices

    # Use flow map steps
    n_steps = n_steps_flow

    # Get pre-initialized FID network from statics
    if statics.inception_fn is None:
        print("Warning: Inception network not initialized. FID computation disabled.")
        return jnp.nan, prng_key
    inception_fn = statics.inception_fn

    # Get pmap sampler function based on number of devices
    if cfg.training.ndevices == 1:
        sampler = flow_map.batch_sample
    else:
        sampler = flow_map.pmap_batch_sample

    # Initialize statistics for Welford's online algorithm
    n_seen = 0
    mu_gen = None
    M2_gen = None

    # Generate samples in batches
    n_full_batches = n_samples // batch_size
    remainder = n_samples % batch_size

    for batch_idx in range(n_full_batches + (1 if remainder > 0 else 0)):
        # Determine current batch size
        if batch_idx == n_full_batches and remainder > 0:
            current_batch_size = remainder
            current_per_device_batch = (
                remainder + cfg.training.ndevices - 1
            ) // cfg.training.ndevices
            padded_batch_size = current_per_device_batch * cfg.training.ndevices
        else:
            current_batch_size = batch_size
            current_per_device_batch = per_device_batch_size
            padded_batch_size = batch_size

        # Generate noise and reshape for pmap
        prng_key, sample_key = jax.random.split(prng_key)
        x0_full = statics.sample_rho0(padded_batch_size, sample_key)

        if cfg.training.ndevices > 1:
            x0_batched = x0_full.reshape(
                cfg.training.ndevices, current_per_device_batch, *cfg.problem.image_dims
            )
        else:
            x0_batched = x0_full

        # Handle labels for conditional generation
        if cfg.training.conditional:
            if cfg.training.class_dropout > 0:
                labels = jnp.array(
                    np.random.choice(cfg.problem.num_classes + 1, padded_batch_size)
                ).reshape(cfg.training.ndevices, current_per_device_batch)
            else:
                labels = jnp.array(
                    np.random.choice(cfg.problem.num_classes, padded_batch_size)
                ).reshape(cfg.training.ndevices, current_per_device_batch)
        else:
            labels = None

        # Get parameters for FID sampling
        params_for_fid = get_params_for_sampling(cfg, train_state, param_type="fid")

        # Sample images across devices
        imgs_batched = sampler(
            train_state.apply_fn,
            params_for_fid,
            x0_batched,
            n_steps,
            labels,
        )

        # Flatten from devices and clip
        imgs = imgs_batched.reshape(padded_batch_size, *cfg.problem.image_dims)
        imgs = jnp.clip(imgs, -1, 1)

        # Only keep the actual samples we need
        imgs = imgs[:current_batch_size]

        # Convert from NCHW to NHWC for Inception
        imgs = imgs.transpose(0, 2, 3, 1)

        # Extract Inception features (no need for pmap here since we're back to single batch)
        features = fid_utils.resize_and_incept(imgs, inception_fn)
        features = np.asarray(np.squeeze(features))

        # Update running statistics using Welford's method
        batch_mean = features.mean(0)
        batch_cov = (
            np.cov(features, rowvar=False)
            if features.shape[0] > 1
            else np.zeros((features.shape[1], features.shape[1]))
        )

        n_seen += current_batch_size

        if mu_gen is None:
            mu_gen = batch_mean
            M2_gen = (
                batch_cov * (current_batch_size - 1)
                if current_batch_size > 1
                else np.zeros_like(batch_cov)
            )
        else:
            delta = batch_mean - mu_gen
            mu_gen += delta * current_batch_size / n_seen
            M2_gen += (
                batch_cov * (current_batch_size - 1)
                + np.outer(delta, delta)
                * (n_seen - current_batch_size)
                * current_batch_size
                / n_seen
            )

    # Compute final covariance and FID
    sigma_gen = M2_gen / (n_seen - 1)
    fid_score = fid_utils.fid_from_stats(mu_gen, sigma_gen, mu_real, sigma_real)

    return float(fid_score), prng_key


def _save_ckpt_on_signal(
    cfg: config_dict.ConfigDict, train_state: state_utils.EMATrainState
) -> None:
    save_state(train_state, cfg)
    sys.exit(0)


def register_signal_handlers(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
) -> None:
    """Drop a checkpoint on SIGTERM or SIGINT."""
    handler = functools.partial(_save_ckpt_on_signal, cfg, train_state)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def save_state(
    train_state: state_utils.EMATrainState,
    cfg: config_dict.ConfigDict,
) -> None:
    """Save flax training state."""

    with open(
        f"{cfg.logging.output_folder}/{cfg.logging.output_name}_{dist_utils.safe_index(cfg, train_state.step)//cfg.logging.save_freq}.pkl",
        "wb",
    ) as f:
        state = jax.device_get(dist_utils.safe_unreplicate(cfg, train_state))
        f.write(to_bytes(state))


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return jnp.linalg.norm(flat_params)


def log_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    grads: jnp.ndarray,
    loss_value: float,
    loss_metrics: Dict[str, jnp.ndarray],
    loss_fn_args: Tuple,
    prng_key: jnp.ndarray,
    step_time: float,
) -> jnp.ndarray:
    """Log some metrics to wandb, make a figure, and checkpoint the parameters."""

    grads = dist_utils.safe_unreplicate(cfg, grads)
    loss_value = dist_utils.safe_index(cfg, jnp.array(loss_value))
    loss_metrics = dist_utils.safe_unreplicate(cfg, loss_metrics)
    step = dist_utils.safe_index(cfg, train_state.step)
    learning_rate = statics.schedule(step)

    # Standard metrics
    metrics = {
        f"loss": loss_value,
        f"grad": compute_grad_norm(grads),
        f"learning_rate": learning_rate,
        f"step_time": step_time,
    }
    metrics.update(loss_metrics)

    # Compute FID on-the-fly if enabled and at the right frequency
    if (
        hasattr(cfg.logging, "fid_freq")
        and cfg.logging.fid_freq > 0
        and (step % cfg.logging.fid_freq) == 0
        and step > 0
    ):
        try:
            # Get step counts configuration - can be a list or single value
            steps_config = getattr(cfg.logging, "fid_n_steps_flow", 8)

            # Convert to list if single value
            if isinstance(steps_config, (list, tuple)):
                n_steps_list = list(steps_config)
            else:
                n_steps_list = [steps_config]

            # Compute FID for each step count
            for n_steps in n_steps_list:
                fid_score, prng_key = compute_fid_on_the_fly(
                    cfg,
                    statics,
                    train_state,
                    prng_key,
                    n_samples=getattr(cfg.logging, "fid_n_samples", 10000),
                    batch_size=getattr(cfg.logging, "fid_batch_size", 256),
                    n_steps_flow=n_steps,
                )
                # Log with step-specific key
                metrics[f"fid_{n_steps}_steps"] = fid_score
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")

    # Compute CelebA BPD on a single batch if enabled
    if (
        cfg.problem.target == "celeb_a"
        and hasattr(cfg.logging, "bpd_freq")
        and cfg.logging.bpd_freq > 0
        and (step % cfg.logging.bpd_freq) == 0
        and step > 0
    ):
        try:
            steps_config = getattr(cfg.logging, "bpd_n_steps", [1, 2, 4, 8])
            if isinstance(steps_config, (list, tuple)):
                n_steps_list = tuple(steps_config)
            else:
                n_steps_list = (int(steps_config),)
            bpd_metrics = compute_celeba_bpd_on_batch(
                cfg, train_state, loss_fn_args, n_steps_list
            )
            metrics.update(bpd_metrics)
        except Exception as e:
            print(f"Warning: BPD computation failed: {e}")

    wandb.log(metrics)

    if (dist_utils.safe_index(cfg, train_state.step) % cfg.logging.visual_freq) == 0:
        if cfg.problem.target == "checker":
            prng_key = make_lowd_plot(cfg, statics, train_state, prng_key)
            prng_key = make_likelihood_heatmap_plot(
                cfg, statics, train_state, prng_key
            )
        else:
            prng_key = make_image_plot(cfg, statics, train_state, prng_key)

        make_loss_fn_args_plot(cfg, statics, train_state, loss_fn_args)

    if (dist_utils.safe_index(cfg, train_state.step) % cfg.logging.save_freq) == 0:
        save_state(train_state, cfg)

    return prng_key


def make_lowd_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> None:
    # Use flow map batch sampler for single-device visualization
    batch_sample = flow_map.batch_sample

    # Get parameters for visualization
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    steps = [1, 2, 5, 10, 25]
    titles = ["base and target"] + [rf"${step}$-step" for step in steps]

    ## extract target samples
    plot_x1s = next(statics.ds)[: cfg.logging.plot_bs]

    ## draw multi-step samples from the model
    x0s = statics.sample_rho0(cfg.logging.plot_bs, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), cfg.logging.plot_bs, cfg.problem.d))
    for kk, step in enumerate(steps):
        xhats[kk] = batch_sample(
            train_state.apply_fn,
            params_for_visual,
            x0s,
            step,
            -jnp.ones(cfg.logging.plot_bs),
        )

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if cfg.problem.target == "checker":
        xmin, xmax = -4.25, 4.25
        ymin, ymax = -4.25, 4.25

    for ax in axs.ravel():
        if cfg.problem.target == "checker":
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[jj]
        ax.set_title(title, fontsize=fontsize)

        if jj == 0:
            ax.scatter(x0s[:, 0], x0s[:, 1], s=0.1, alpha=0.5, marker="o", c="black")
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )
        else:
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )

            ax.scatter(
                xhats[jj - 1, :, 0],
                xhats[jj - 1, :, 1],
                s=0.1,
                alpha=0.5,
                marker="o",
                c="black",
            )

    wandb.log({"samples": wandb.Image(fig)})
    return prng_key


def make_image_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> None:
    """Make a plot of the generated images."""
    # Use flow map batch sampler for single-device visualization
    batch_sample = flow_map.batch_sample

    # Get parameters for visualization (already unreplicated)
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 1, 1
    fontsize = 12.5

    ## set up plot array
    steps = [1, 2, 4, 8, 16]

    titles = [rf"{step}-step" for step in steps]

    ## draw multi-step samples from the model
    n_images = 16
    x0s = statics.sample_rho0(n_images, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), n_images, *cfg.problem.image_dims))

    ## set up conditioning information
    if cfg.training.conditional:
        if cfg.training.class_dropout > 0:
            assert cfg.network.use_cfg  # class dropout doesn't make sense without cfg
            labels = jnp.array(np.random.choice(cfg.problem.num_classes + 1, n_images))
        else:
            labels = jnp.array(np.random.choice(cfg.problem.num_classes, n_images))
        prng_key = jax.random.split(prng_key)[0]
    else:
        labels = None

    for kk, step in enumerate(steps):
        xhats[kk] = batch_sample(
            train_state.apply_fn,
            params_for_visual,
            x0s,
            step,
            labels,
        )

    # transpose (S, N, C, H, W) -> (S, N, H, W, C)
    xhats = xhats.transpose(0, 1, 3, 4, 2)

    ## make the image grids
    nrows = 2 if n_images > 8 else 1
    ncols = n_images // nrows

    for kk, title in enumerate(titles):
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fw * ncols, fh * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs = axs.reshape((nrows, ncols))

        fig.suptitle(title, fontsize=fontsize)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect("equal")

        ## visualize the generated images
        for ii in range(nrows):
            for jj in range(ncols):
                index = ii * ncols + jj
                image = datasets.unnormalize_image(xhats[kk, index])
                axs[ii, jj].imshow(image)

        wandb.log({titles[kk]: wandb.Image(fig)})

    return prng_key


def make_loss_fn_args_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
) -> None:
    """Make a plot of the loss function arguments."""
    # unpack the full loss arguments
    data_args = loss_fn_args[2:]
    (x0batch, x1batch, _, sbatch, tbatch, _, _, _) = (
        dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    )

    # remove pmap reshaping
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    sbatch = jnp.squeeze(sbatch)
    tbatch = jnp.squeeze(tbatch)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    # compute xts
    xtbatch = statics.interp.batch_calc_It(tbatch, x0batch, x1batch)

    ## set up plot array
    if cfg.problem.target == "checker":
        titles = [r"$x_0$", r"$x_1$", r"$x_t$", r"$(s, t)$"]
    else:
        titles = [r"$(s, t)$"]

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )

    if cfg.problem.target == "checker":
        all_x = np.concatenate(
            [np.asarray(x0batch), np.asarray(x1batch), np.asarray(xtbatch)],
            axis=0,
        )
        margin = 0.5
        xmin = float(all_x[:, 0].min()) - margin
        xmax = float(all_x[:, 0].max()) + margin
        ymin = float(all_x[:, 1].min()) - margin
        ymax = float(all_x[:, 1].max()) + margin

    for kk, ax in enumerate(axs.ravel()):
        if kk == (len(titles) - 1):
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
        else:
            if cfg.problem.target == "checker":
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[0, jj]
        ax.set_title(title, fontsize=fontsize)

        if cfg.problem.target == "checker":
            if jj == 0:
                ax.scatter(x0batch[:, 0], x0batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 1:
                ax.scatter(x1batch[:, 0], x1batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 2:
                ax.scatter(xtbatch[:, 0], xtbatch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 3:
                ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")
        else:
            ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})


def _base_log_prob(cfg: config_dict.ConfigDict, x: jnp.ndarray) -> jnp.ndarray:
    """Log prob of N(0, diag(rescale^2)) for base distribution."""
    sigma = jnp.asarray(cfg.network.rescale)
    if sigma.ndim == 0:
        sigma = jnp.ones((x.shape[-1],)) * sigma
    quad = jnp.sum((x / sigma) ** 2, axis=-1)
    log_det = jnp.sum(jnp.log(sigma**2))
    log_norm = 0.5 * (x.shape[-1] * jnp.log(2 * jnp.pi) + log_det)
    return -0.5 * quad - log_norm


def _base_log_prob_image(x: jnp.ndarray, rescale: float) -> jnp.ndarray:
    """Log prob of N(0, diag(rescale^2)) for image-shaped tensors."""
    sigma = jnp.asarray(rescale, dtype=x.dtype)
    dims = tuple(range(1, x.ndim))
    quad = jnp.sum((x / sigma) ** 2, axis=dims)
    d = x[0].size
    log_det = d * jnp.log(sigma**2)
    log_norm = 0.5 * (d * jnp.log(2 * jnp.pi) + log_det)
    return -0.5 * quad - log_norm


def _sample_model_nsteps_with_logp(
    apply_fn: Callable,
    params: Dict,
    x0: jnp.ndarray,
    n_steps: int,
    label: jnp.ndarray,
    *,
    cfg: config_dict.ConfigDict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample with Euler steps and track logp using model divergence."""
    dt = 1.0 / n_steps
    t_curr = jnp.zeros((x0.shape[0],), dtype=jnp.float32)
    x = x0
    logp = _base_log_prob(cfg, x0)

    divergence_rescale = getattr(cfg.network, "divergence_rescale", 10.0)
    for _ in range(n_steps):
        t_next = t_curr + dt
        phi, div = apply_fn(
            params,
            t_curr,
            t_next,
            x,
            label,
            train=False,
            method="calc_phi",
            return_div=True,
        )
        div = div * divergence_rescale
        x = x + dt * phi
        logp = logp - dt * div
        t_curr = t_next

    return np.asarray(x), np.asarray(logp)


def _make_mean_logp_heatmap(xs, logp, xlim, ylim, bins=100):
    """Mean logp per bin, matching try_ll.py."""
    x = xs[:, 0]
    y = xs[:, 1]
    hist_sum, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim], weights=logp
    )
    hist_count, _, _ = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    mean_logp = hist_sum / (hist_count + 1e-6)
    mean_logp[hist_count == 0] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return mean_logp, extent


def _make_log_density_heatmap(xs, xlim, ylim, bins=100):
    """Numerical log-density estimate from samples."""
    x = xs[:, 0]
    y = xs[:, 1]
    hist_count, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim]
    )
    bin_area = (
        (xedges[-1] - xedges[0]) / bins * (yedges[-1] - yedges[0]) / bins
    )
    density = hist_count / (hist_count.sum() * bin_area + 1e-12)
    logp = np.full_like(density, np.nan, dtype=np.float64)
    mask = hist_count > 0
    logp[mask] = np.log(density[mask])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return logp, extent


def _make_uniform_logp_heatmap(xs, xlim, ylim, bins=100, logp_value=-np.log(32.0)):
    """Uniform logp over occupied bins for checkerboard."""
    x = xs[:, 0]
    y = xs[:, 1]
    hist_count, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim]
    )
    logp = np.full_like(hist_count, np.nan, dtype=np.float64)
    logp[hist_count > 0] = logp_value
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return logp, extent


def _make_checker_logp_heatmap(
    xlim, ylim, bins=100, logp_value=-np.log(32.0)
):
    """Analytical logp heatmap for the checkerboard target."""
    xedges = np.linspace(xlim[0], xlim[1], bins + 1)
    yedges = np.linspace(ylim[0], ylim[1], bins + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")

    u = Xc / 2.0
    v = Yc / 2.0
    in_range = (u >= -2.0) & (u < 2.0) & (v >= -2.0) & (v < 2.0)
    parity = np.mod(np.floor(u), 2.0)
    v_shift = v - parity
    in_band = ((v_shift >= 0.0) & (v_shift < 1.0)) | (
        (v_shift >= -2.0) & (v_shift < -1.0)
    )
    support = in_range & in_band

    logp = np.full_like(Xc, np.nan, dtype=np.float64)
    logp[support] = logp_value
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return logp, extent


def _make_grid_points(xlim, ylim, grid_size=200):
    xs = jnp.linspace(xlim[0], xlim[1], grid_size)
    ys = jnp.linspace(ylim[0], ylim[1], grid_size)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    pts = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return pts, X, Y


def _inverse_logp_points_with_divhead(
    apply_fn: Callable,
    params: Dict,
    x_t: jnp.ndarray,
    n_steps: int,
    label: jnp.ndarray,
    *,
    cfg: config_dict.ConfigDict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse logp on points using divergence head."""
    dt = -1.0 / n_steps
    t_curr = jnp.ones((x_t.shape[0],), dtype=jnp.float32)
    x = x_t
    delta_logp = jnp.zeros((x_t.shape[0],), dtype=jnp.float32)

    divergence_rescale = getattr(cfg.network, "divergence_rescale", 10.0)
    for _ in range(n_steps):
        t_next = t_curr + dt
        phi, div = apply_fn(
            params,
            t_curr,
            t_next,
            x,
            label,
            train=False,
            method="calc_phi",
            return_div=True,
        )
        print(
            "t_curr:", np.asarray(t_curr),
            "t_next:", np.asarray(t_next),
            "div:", np.asarray(div.mean()),
        )
        div = div * divergence_rescale
        delta_logp = delta_logp - dt * div
        x = x + dt * phi
        t_curr = t_next

    logp0 = _base_log_prob(cfg, x)
    logp1 = logp0 - delta_logp
    return np.asarray(x), np.asarray(logp1)


@functools.partial(jax.jit, static_argnums=(0, 3))
def _inverse_logp_euler_divhead(
    apply_fn: Callable,
    params: Dict,
    x_t: jnp.ndarray,
    n_steps: int,
    rescale: float,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    dt = -1.0 / n_steps
    t_curr = jnp.ones((x_t.shape[0],), dtype=jnp.float32)
    x = x_t
    delta_logp = jnp.zeros((x_t.shape[0],), dtype=jnp.float32)

    divergence_rescale = getattr(cfg.network, "divergence_rescale", 10000.0)

    def body(_, state):
        t_curr, x, delta_logp = state
        t_next = t_curr + dt
        phi, div = apply_fn(
            params,
            t_curr,
            t_next,
            x,
            None,
            train=False,
            method="calc_phi",
            return_div=True,
        )
        x = x + dt * phi
        delta_logp = delta_logp - dt * div * divergence_rescale
        jax.debug.print("t={}, div_mean={}", t_curr[0], div.mean())
        return t_next, x, delta_logp

    t_curr, x0, delta_logp = jax.lax.fori_loop(
        0, n_steps, body, (t_curr, x, delta_logp)
    )
    logp0 = _base_log_prob_image(x0, rescale)
    return logp0 - delta_logp


def _compute_bpd_from_logp(
    logp_z: jnp.ndarray, d: int, *, dequantize: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    log2_const = jnp.log(2.0)
    log_det_transform = d * jnp.log(2.0)
    total_logp = logp_z + log_det_transform
    if dequantize:
        total_logp = total_logp - d * jnp.log(256.0)
    bpd = -(total_logp / (d * log2_const))
    return bpd, total_logp


def compute_celeba_bpd_on_batch(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
    n_steps_list: Tuple[int, ...],
) -> Dict[str, float]:
    """Compute per-step BPD on a single CelebA batch using div head."""
    # loss_fn_args = (teacher_diag, teacher_offdiag, x0, x1, label, s, t, u, h, dropout)
    x1batch = loss_fn_args[3]
    x1batch = dist_utils.unreplicate_batch(cfg, x1batch)
    batch_size = getattr(cfg.logging, "bpd_batch_size", None)
    if batch_size is not None:
        x1batch = x1batch[:batch_size]

    params_for_bpd = get_params_for_sampling(cfg, train_state, param_type="bpd")
    d = int(np.prod(cfg.problem.image_dims))
    metrics = {}
    for n_steps in n_steps_list:
        logp_z = _inverse_logp_euler_divhead(
            train_state.apply_fn,
            params_for_bpd,
            x1batch,
            n_steps,
            cfg.network.rescale,
            cfg,
        )
        bpd, _ = _compute_bpd_from_logp(logp_z, d, dequantize=True)
        metrics[f"bpd_{n_steps}_steps"] = float(jnp.mean(bpd))
    return metrics


def _clip_heatmaps(heatmaps):
    """Replace NaNs and compute shared vmin/vmax."""
    vals = []
    for h in heatmaps:
        finite = np.isfinite(h)
        if finite.any():
            vals.append(h[finite])
    if not vals:
        return heatmaps, 0.0, 1.0
    all_vals = np.concatenate(vals)
    vmin = all_vals.min()
    vmax = all_vals.max()
    clipped = []
    for h in heatmaps:
        h2 = h.copy()
        h2[~np.isfinite(h2)] = vmin
        clipped.append(np.clip(h2, vmin, vmax))
    return clipped, vmin, vmax


def make_likelihood_heatmap_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Plot likelihood heatmaps for 1/2/4/8 steps and target checkerboard."""
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")
    steps = [1, 2, 4, 8]
    n_samples = cfg.logging.plot_bs
    if cfg.problem.target == "checker":
        n_samples = getattr(cfg.logging, "likelihood_n_samples", max(n_samples, 500_000))
    # Analytical target heatmap for checkerboard (uniform over support)
    margin = 0.0
    base_range = 4.0
    xlim = (-base_range - margin, base_range + margin)
    ylim = (-base_range - margin, base_range + margin)
    h_target, extent = _make_checker_logp_heatmap(
        xlim, ylim, bins=100, logp_value=-np.log(32.0)
    )

    # Model samples + logp for each step
    prng_key, sample_key = jax.random.split(prng_key)
    x0s = statics.sample_rho0(n_samples, sample_key)
    labels = -jnp.ones((n_samples,))

    heatmaps = [h_target]
    titles = ["Target log p(x) = -ln(32)"]
    for step in steps:
        xs, logp = _sample_model_nsteps_with_logp(
            train_state.apply_fn,
            params_for_visual,
            x0s,
            step,
            labels,
            cfg=cfg,
        )
        h_step, _ = _make_mean_logp_heatmap(xs, logp, xlim, ylim, bins=100)
        heatmaps.append(h_step)
        titles.append(f"Mean log p(x), {step} steps")

        grid_pts, X, Y = _make_grid_points(xlim, ylim, grid_size=200)
        grid_labels = -jnp.ones((grid_pts.shape[0],))
        _, logp_inv = _inverse_logp_points_with_divhead(
            train_state.apply_fn,
            params_for_visual,
            grid_pts,
            step,
            grid_labels,
            cfg=cfg,
        )
        h_inv = logp_inv.reshape(X.shape)
        heatmaps.append(np.asarray(h_inv))
        titles.append(f"Inverse log p(x), {step} steps")

    heatmaps, _, _ = _clip_heatmaps(heatmaps)
    vmin, vmax = -4, -3
    heatmaps = [np.clip(h, vmin, vmax) for h in heatmaps]

    plt.close("all")
    ncols = len(heatmaps)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(4 * ncols, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if ncols == 1:
        axs = [axs]
    for ax, h, title in zip(axs, heatmaps, titles):
        im = ax.imshow(
            h.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    wandb.log({"likelihood_heatmap": wandb.Image(fig)})
    return prng_key
