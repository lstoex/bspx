"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .deboor import differentiate, propagate
from .utils import build_alpha_lut, clamped_uniform_knot_vector, get_indices_nonuniform, get_indices_uniform


@jax.jit(static_argnames=["k", "n_points"])
def bspline(
    control_points: Float[Array, "np1 d"],
    n_points: int,
    k: int,
    T: Float[Array, " mp1"] | None = None,
    t: Float[Array, " n_points"] | None = None,
) -> Float[Array, "n_points d"]:
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
            T_ = clamped_uniform_knot_vector(n, k, False)
            j = get_indices_uniform(n, k, t_)
            assert isinstance(j, np.ndarray) and isinstance(T_, np.ndarray) and isinstance(t_, np.ndarray)
            alphas_flat = build_alpha_lut(k, j, T_, t_)
        return propagate(control_points, k, j, alphas_flat)
    elif T is not None and t is not None:
        j = get_indices_nonuniform(n, T, t)
        return propagate(control_points, k, j, None, T, t)
    else:
        raise ValueError("Either both T and t must be provided, or neither.")


@jax.jit(static_argnames=["k", "n_points", "n_fine"])
def resample_arc_length(
    control_points: Float[Array, "np1 d"],
    n_points: int,
    k: int,
    n_fine: int = 200,
) -> Float[Array, "n_points d"]:
    """Re-evaluate a B-spline at arc-length-uniform parameter values.

    Densely samples the curve (``n_fine`` points), computes cumulative chord
    lengths, maps uniform arc-length targets back to parameter values via
    linear interpolation, then re-evaluates the B-spline at those parameters.

    The output lies exactly on the original B-spline curve, preserving its
    continuity class (e.g. C2 for cubic).

    Args:
        control_points: Control points of shape (np1, d).
        n_points: Number of arc-length-uniform output points.
        k: B-spline order (4 = cubic).
        n_fine: Number of dense samples for arc-length estimation (default 200).

    Returns:
        Points on the B-spline with approximately uniform arc-length spacing.
    """
    n = control_points.shape[0] - 1
    q_fine = bspline(control_points, n_fine, k)
    steps = q_fine[1:] - q_fine[:-1]
    lengths = jnp.linalg.norm(steps, axis=-1)
    cum_len = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(lengths)])
    total_len = cum_len[-1]
    target_arc = jnp.linspace(0.0, total_len, n_points)
    t_fine = jnp.linspace(0.0, 1.0, n_fine)
    t_uniform = jnp.interp(target_arc, cum_len, t_fine)
    T = jnp.array(clamped_uniform_knot_vector(n, k))
    return bspline(control_points, n_points, k, T=T, t=t_uniform)


@jax.jit(static_argnames=["k", "n_points", "derivative_order", "emit_intermediates"])
def bspline_derivative(
    control_points: Float[Array, "np1 d"],
    n_points: int,
    k: int,
    T: Float[Array, " mp1"] | None = None,
    t: Float[Array, " n_points"] | None = None,
    derivative_order: int = 1,
    emit_intermediates=False,
) -> list[Float[Array, "n_points d"]] | Float[Array, "n_points d"]:
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
    t_ = np.linspace(0.0, 1.0, n_points) if t is None else t
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
