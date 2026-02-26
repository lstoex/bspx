"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import jax
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
