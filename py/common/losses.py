"""
Nicholas M. Boffi
10/5/25

Loss functions for learning.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import flow_map as flow_map
from . import interpolant as interpolant
from . import loss_args

Parameters = Dict[str, Dict]


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jax.tree_util.tree_map(lambda x: jnp.mean(x), batched_outputs)

    return wrapper


def _hutchinson_divergence_with_phi(
    params: Parameters,
    t: float,
    x: jnp.ndarray,
    label: jnp.ndarray,
    rng: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    method: str = "calc_b",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate div(phi) with Hutchinson's estimator and return phi."""
    eps_key = rng["dropout"] if isinstance(rng, dict) else rng
    eps = jax.random.normal(eps_key, shape=x.shape)

    def phi_fn(x_in):
        return X.apply(
            params, t, x_in, label, train=False, method=method, rngs=rng
        )

    phi, vjp_fn = jax.vjp(phi_fn, x)
    def _div_axes(x_in, label_in):
        if x_in.ndim == 4:
            return (1, 2, 3)
        if x_in.ndim == 3:
            return (0, 1, 2)
        if x_in.ndim == 2:
            if label_in is not None and jnp.ndim(label_in) > 0:
                if label_in.shape[0] == x_in.shape[0]:
                    return (1,)
            return (0, 1)
        if x_in.ndim == 1:
            return (0,)
        return tuple(range(x_in.ndim))

    div_axes = _div_axes(x, label)
    vjp_eps = vjp_fn(eps)[0]
    div_hat = jnp.sum(vjp_eps * eps, axis=div_axes)
    return phi, div_hat


def diagonal_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the diagonal (interpolant) term of the loss."""

    # compute interpolant and the target
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)

    # compute the weighted loss
    bt_rslt = X.apply(
        params,
        t,
        It,
        label,
        train=True,
        method="calc_b",
        rngs=rng,
        return_div=True,
    )
    
    bt_teacher, div_hat = _hutchinson_divergence_with_phi(
        teacher_params, t, It, label, rng, X=X, method="calc_b"
    )
    divergence_rescale = getattr(X.config, "divergence_rescale", 10000.0)
    bt_teacher = jax.lax.stop_gradient(bt_teacher)
    div_hat = jax.lax.stop_gradient(div_hat) / divergence_rescale
    if isinstance(bt_rslt, tuple):
        bt, div_tt = bt_rslt
    else:
        bt = bt_rslt
        div_tt = None
    
    # match the teacher velocity
    base_velocity_loss = jnp.sum((bt - bt_teacher) ** 2)
    div_loss = jnp.array(0.0, dtype=base_velocity_loss.dtype)
    if div_tt is not None:
        # match the teacher divergence
        div_loss = jnp.sum((div_tt - div_hat) ** 2)
        velocity_loss = base_velocity_loss + div_loss
    else:
        jax.debug.print(
            "warning: diag div head missing (div_tt is None)",
        )
        velocity_loss = base_velocity_loss

    # Diagonal uses s=t
    weight_tt = X.apply(params, t, t, method="calc_weight")
    loss_value = jnp.exp(-weight_tt) * velocity_loss + weight_tt
    metrics = {
        "diag/velocity_loss": jax.lax.stop_gradient(base_velocity_loss),
        "diag/div_loss": jax.lax.stop_gradient(div_loss),
    }
    return loss_value, metrics



def lsd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the LSD term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    def _split_output(value):
        if isinstance(value, tuple):
            return value[0], value[1]
        return value, None

    # Compute the distillation loss
    Xst_Is, dt_Xst = X.apply(
        params,
        s,
        t,
        Is,
        label,
        train=False,
        method="partial_t",
        rngs=rng,
        return_div=True,
    )
    Xst_Is, D_st = _split_output(Xst_Is)
    dt_Xst, dt_Dst = _split_output(dt_Xst)
    dt_Dst = jax.lax.stop_gradient(dt_Dst)

    if stopgrad_type == "convex":
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval_v, b_eval_div = _hutchinson_divergence_with_phi(
            teacher_params, t, Xst_Is, label, rng, X=X, method="calc_b"
        )
        divergence_rescale = getattr(X.config, "divergence_rescale", 10000.0)
        b_eval_v = jax.lax.stop_gradient(b_eval_v)
        b_eval_div = jax.lax.stop_gradient(b_eval_div) / divergence_rescale
    elif stopgrad_type == "none":
        b_eval_v, b_eval_div = _hutchinson_divergence_with_phi(
            params, t, Xst_Is, label, rng, X=X, method="calc_b"
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    weight_st = X.apply(params, s, t, method="calc_weight")
    
    # LSD term for velocity alignment
    error = b_eval_v - dt_Xst
    
    if D_st is not None:
        A_dot = D_st + (t - s) * jax.lax.stop_gradient(dt_Dst)
        
        # LSD term for divergence alignment
        error_div = A_dot - b_eval_div
        
        lsd_div_loss = jnp.sum(error_div**2)
    else:
        jax.debug.print(
            "warning: lsd div head missing (D_st is None)",
        )
        lsd_div_loss = jnp.array(0.0, dtype=error.dtype)
    lsd_v_loss = jnp.sum(error**2)
    lsd_loss = lsd_v_loss + lsd_div_loss
    loss_value = jnp.exp(-weight_st) * lsd_loss + weight_st
    metrics = {
        "lsd/lsd_v_loss": jax.lax.stop_gradient(lsd_v_loss),
        "lsd/lsd_div_loss": jax.lax.stop_gradient(lsd_div_loss),
    }
    return loss_value, metrics


def psd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    u: float,
    h: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    psd_type: str,
    stopgrad_type: str,
) -> float:
    """Compute the PSD (Progressive Self-Distillation) term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # compute the full jump
    X_st, phi_st, div_st = X.apply(
        params,
        s,
        t,
        Is,
        label,
        train=False,
        rngs=rng,
        return_X_and_phi=True,
        return_div=True,
    )

    # break it down into two jumps
    if stopgrad_type == "convex":
        X_su, phi_su, div_su = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                u,
                Is,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
                return_div=True,
            )
        )

        X_ut, phi_ut, div_ut = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                u,
                t,
                X_su,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
                return_div=True,
            )
        )
    elif stopgrad_type == "none":
        X_su, phi_su, div_su = X.apply(
            params,
            s,
            u,
            Is,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
            return_div=True,
        )

        X_ut, phi_ut, div_ut = X.apply(
            params,
            u,
            t,
            X_su,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
            return_div=True,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut
        div_teacher = (1 - h) * div_su + h * div_ut
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)
        div_teacher = 0.5 * (div_su + div_ut)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")

    psd_loss = jnp.sum((student - teacher) ** 2)
    if div_st is not None:
        # psd_loss = psd_loss + jnp.sum((div_st - div_teacher) ** 2)
        pass

    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * psd_loss + weight_st


def esd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the ESD term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # compute the derivative with respect to the first time
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_s", rngs=rng
    )

    # stopgrad everything to avoid backpropagating through the UNet spatial Jacobian
    if stopgrad_type == "full":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )

        # compute the advective term
        _, grad_Xst_b = jax.lax.stop_gradient(
            jax.jvp(
                lambda x: X.apply(
                    teacher_params, s, t, x, label, train=False, rngs=rng
                ),
                primals=(Is,),
                tangents=(b_eval,),
            )
        )

    # stopgrad the b, so it's like EMD
    elif stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )

        # compute the advective term
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )

    # pure residual minimization -- no stopgrad
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            s,
            Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )

        # compute the advective term
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )

    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    esd_loss = jnp.sum((ds_Xst + grad_Xst_b) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * esd_loss + weight_st


def setup_loss(
    cfg: config_dict.ConfigDict, net: flow_map.FlowMap, interp: interpolant.Interpolant
) -> Callable:
    """Setup the loss function."""

    print(f"Setting up loss: {cfg.training.loss_type}")
    print(f"Stopgrad type: {cfg.training.stopgrad_type}")

    # Pure diagonal loss
    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
    def diagonal_only_loss(params, teacher_params, x0, x1, label, t, rng):
        return diagonal_term(
            params,
            teacher_params,
            x0,
            x1,
            label,
            t,
            rng,
            interp=interp,
            X=net,
        )

    # Pure off-diagonal loss
    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
    def offdiagonal_only_loss(
        params, teacher_params, x0, x1, label, s, t, u, h, dropout_keys
    ):
        rng = {"dropout": dropout_keys}

        if cfg.training.loss_type == "psd":
            return psd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                u,
                h,
                rng,
                interp=interp,
                X=net,
                psd_type=cfg.training.psd_type,
                stopgrad_type=cfg.training.stopgrad_type,
            ), {}
        elif cfg.training.loss_type == "lsd":
            return lsd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
            )
        elif cfg.training.loss_type == "esd":
            return esd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
            ), {}
        else:
            raise ValueError(f"Unknown loss_type: {cfg.training.loss_type}")

    def loss(
        params,
        teacher_params_diag,
        teacher_params_offdiag,
        x0,
        x1,
        label,
        s,
        t,
        u,
        h,
        dropout_keys,
    ):
        """Split batch into diagonal and off-diagonal portions."""
        total_bs = x0.shape[0]
        diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, total_bs)

        total_loss = 0.0
        total_metrics = {}

        def _add_metrics(metrics, new_metrics):
            for key, value in new_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
            return metrics

        # Compute diagonal loss on first portion
        if diag_bs > 0:
            label_diag = None if label is None else label[:diag_bs]
            diag_loss, diag_metrics = diagonal_only_loss(
                params,
                teacher_params_diag,
                x0[:diag_bs],
                x1[:diag_bs],
                label_diag,
                t[:diag_bs],
                dropout_keys[:diag_bs],
            )
            total_loss += diag_loss * diag_bs
            total_metrics = _add_metrics(
                total_metrics,
                {k: v * diag_bs for k, v in diag_metrics.items()},
            )

        # Compute off-diagonal loss on second portion
        if offdiag_bs > 0:
            label_offdiag = None if label is None else label[diag_bs:]
            u_offdiag = None if u is None else u[diag_bs:]
            h_offdiag = None if h is None else h[diag_bs:]

            offdiag_loss, offdiag_metrics = offdiagonal_only_loss(
                params,
                teacher_params_offdiag,
                x0[diag_bs:],
                x1[diag_bs:],
                label_offdiag,
                s[diag_bs:],
                t[diag_bs:],
                u_offdiag,
                h_offdiag,
                dropout_keys[diag_bs:],
            )
            total_loss += offdiag_loss * offdiag_bs
            total_metrics = _add_metrics(
                total_metrics,
                {k: v * offdiag_bs for k, v in offdiag_metrics.items()},
            )

        # Normalize by total batch size
        total_loss = total_loss / total_bs
        if cfg.training.loss_type != "lsd":
            total_metrics = {}
        elif total_metrics:
            total_metrics = jax.tree_util.tree_map(
                lambda x: jax.lax.stop_gradient(x / total_bs),
                total_metrics,
            )
        return total_loss, total_metrics

    return loss