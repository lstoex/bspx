"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ._typing import ControlPoints, CurvePoints, KnotVector, Order, Time
from .deboor import differentiate, propagate
from .utils import build_alpha_lut, clamped_uniform_knot_vector, get_indices_nonuniform, get_indices_uniform


@jax.jit(static_argnames=["k", "n_points"])
def bspline(
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    T: KnotVector | None = None,
    t: Time | None = None,
) -> CurvePoints:
    """Evaluate a B-spline curve using de Boor's algorithm.
    Args:
        control_points: An array of shape (np1, d) representing the control points of the B-spline curve.
        n_points: The number of output points to generate on the B-spline curve.
        k: The order of the B-spline curve (default is 4 for cubic).
        T: Optional knot vector of shape (n_in + k + 1,). If not provided, a uniform knot vector will be used.
        t: Optional array of shape (n_points,) representing the parameter values at which to evaluate the B-spline curve. If not provided, uniform parameter values will be used.
    Returns:
        An array of shape (n_points, d) representing the points on the B-spline curve.
    """

    n = control_points.shape[0] - 1
    if T is None and t is None:
        with jax.ensure_compile_time_eval():
            t_ = np.linspace(0.0, 1.0, n_points)
            T_ = clamped_uniform_knot_vector(n, k, use_jax=False)
            j = get_indices_uniform(n, k, t_)
            assert isinstance(j, np.ndarray) and isinstance(T_, np.ndarray) and isinstance(t_, np.ndarray)
            alphas_flat = build_alpha_lut(k, j, T_, t_)
        return propagate(control_points, k, j, alphas_flat)
    elif T is not None and t is not None:
        j = get_indices_nonuniform(n, T, t)
        return propagate(control_points, k, j, None, T, t)
    elif T is None and t is not None:
        T_ = clamped_uniform_knot_vector(n, k, use_jax=True)
        j = get_indices_nonuniform(n, T_, t)
        return propagate(control_points, k, j, None, T_, t)
    else:
        raise ValueError("Both T and t or just t must be provided.")


@jax.jit(static_argnames=["k", "n_points", "n_fine"])
def bspline_arclength_adjusted(
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    n_fine: int | None = None,
) -> tuple[CurvePoints, Time]:
    """
    Compute a B-spline curve at approximately uniform arclength distribution. Evaluates B-spline function twice.

    Args:
        control_points: Control points of shape (np1, d).
        n_points: Number of arc-length-uniform output points.
        k: B-spline order (4 = cubic).
        n_fine: Number of dense samples for arc-length estimation (defaults to n_points if not provided).

    Returns:
        Points on the B-spline with approximately uniform arc-length spacing, and the corresponding t values used for evaluation.
    """
    n_fine = n_fine or n_points
    c_fine = bspline(control_points, n_points=n_fine, k=k)  # get dense samples for arc-length estimation
    segments = jnp.linalg.norm(jnp.diff(c_fine, axis=0), axis=-1)
    cum_len = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(segments)])
    t_old = jnp.linspace(0.0, 1.0, n_fine)  # arc-lengths correspond to these parameter values
    normalized_length_distribution = cum_len / cum_len[-1]  # normalize to [0, 1]
    t_new = jnp.interp(jnp.linspace(0.0, 1.0, n_points), normalized_length_distribution, t_old)  # map (t,ac) -> t_new
    t_new = jax.lax.stop_gradient(t_new)  # Prevent gradients from flowing through the reparameterization
    return bspline(control_points, n_points=n_points, k=k, t=t_new), t_new


def chord_length_params(
    data: Float[Array, "n d"],
) -> Time:
    """Chord-length parameterization of data points → t in [0, 1]."""
    diffs = jnp.diff(data, axis=0)
    seg_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=-1))
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lengths)])
    total = cum[-1]
    return jnp.where(total > 0, cum / total, jnp.linspace(0.0, 1.0, data.shape[0]))


@jax.jit(static_argnames=["n_ctrl", "order"])
def bspline_lsq_fit(
    *,
    data: Float[Array, "n d"],
    n_ctrl: int,
    order: Order = 4,
) -> ControlPoints:
    """Fit a B-spline with ``n_ctrl`` control points to ``data`` via weighted LSQ.

    Uses chord-length parameterization and high endpoint weights so the
    fitted curve interpolates the first and last data points exactly.

    Args:
        data: Data points of shape ``(n, d)``.
        n_ctrl: Number of control points for the fitted B-spline.
        order: B-spline order (4 = cubic).

    Returns:
        Fitted control points of shape ``(n_ctrl, d)``.
    """
    n = data.shape[0]
    t_data = chord_length_params(data)

    eye_ctrl = jnp.eye(n_ctrl)
    B = bspline(eye_ctrl, n_points=n, k=order, t=t_data)

    w = jnp.ones(n)
    w = w.at[0].set(1e6)
    w = w.at[-1].set(1e6)
    W = jnp.diag(w)

    BtW = B.T @ W
    A = BtW @ B
    rhs = BtW @ data

    return jnp.linalg.solve(A, rhs)


@jax.jit(static_argnames=["n_points", "k", "n_fine"])
def bspline_arclength_subrange(
    *,
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    t_lo: Float[Array, ""],
    t_hi: Float[Array, ""],
    n_fine: int | None = None,
) -> CurvePoints:
    """Evaluate a B-spline with arc-length spacing over a sub-range [t_lo, t_hi].

    Args:
        control_points: Control points of shape ``(np1, d)``.
        n_points: Number of output points.
        k: B-spline order (4 = cubic).
        t_lo: Start of the parameter sub-range.
        t_hi: End of the parameter sub-range.
        n_fine: Dense samples for arc-length estimation (default: ``n_points * 4``).

    Returns:
        Points on the B-spline with arc-length spacing over ``[t_lo, t_hi]``.
    """
    if n_fine is None:
        n_fine = n_points * 4

    t_fine = t_lo + (t_hi - t_lo) * jnp.linspace(0.0, 1.0, n_fine)
    c_fine = bspline(control_points, n_points=n_fine, k=k, t=t_fine)

    seg = jnp.sqrt(jnp.sum(jnp.diff(c_fine, axis=0) ** 2, axis=-1))
    cum_arc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    cum_arc_norm = cum_arc / jnp.maximum(cum_arc[-1], 1e-12)

    t_uniform = jnp.interp(jnp.linspace(0.0, 1.0, n_points), cum_arc_norm, t_fine)
    t_uniform = jax.lax.stop_gradient(t_uniform)
    return bspline(control_points, n_points=n_points, k=k, t=t_uniform)


@jax.jit(static_argnames=["k", "n_points", "derivative_order", "emit_intermediates"])
def bspline_derivative(
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    T: KnotVector | None = None,
    t: Time | None = None,
    derivative_order: int = 1,
    emit_intermediates=False,
) -> list[CurvePoints] | CurvePoints:
    """Evaluate the derivative of a B-spline curve using de Boor's algorithm.
    Args:
        control_points: An array of shape (n+1, d) representing the control points of the B-spline curve.
        k: The order of the B-spline curve (default is 4 for cubic).
        T: Optional knot vector of shape (n + k + 1,). If not provided, a uniform knot vector will be used.
        t: Optional array of shape (n_points,) representing the parameter values at which to evaluate the B-spline curve. If not provided, uniform parameter values will be used.
        n_points: The number of output points to generate on the B-spline curve.
        derivative_order: The order of the derivative to compute (default is 1 for first derivative).
    Returns:
        An array of shape (n_points, d) representing the points on the derivative of the B-spline curve.
    """

    n = control_points.shape[0] - 1
    T_ = clamped_uniform_knot_vector(n, k) if T is None else T
    t_ = jnp.linspace(0.0, 1.0, n_points) if t is None else t
    k_ = k
    Q = control_points

    def assemble(Q, k_, T):
        return bspline(Q, n_points, k_, T=T_, t=t_)

    out = []
    for _ in range(derivative_order):
        Q, k_, T_ = differentiate(Q, k_, T_)
        if emit_intermediates:
            out.append(assemble(Q, k_, T))
    out = assemble(Q, k_, T) if not emit_intermediates else out
    return out
