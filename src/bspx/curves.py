"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .deboor import de_boor, de_boor_static


def _evaluate(control_points, order, n_output, ts):
    """Dispatch to static or dynamic de Boor based on how evaluation points are specified.

    - n_output given: generates uniform numpy linspace → compile-time optimized static path.
    - ts is a concrete numpy array (e.g. in eager mode): static path.
    - ts is a JAX array or tracer (inside jit): dynamic runtime path.
    """
    if n_output is not None:
        ts = np.linspace(0.0, 1.0, n_output)
    if isinstance(ts, np.ndarray):
        return de_boor_static(control_points, order, ts)
    return de_boor(control_points, order, ts)


@jax.jit(static_argnames=["order", "n_output"])
def bspline(
    control_points: Float[Array, "n_in d"],
    n_output: int | None = None,
    order: int = 4,
    *,
    ts: Float[Array, " n_points"] | None = None,
) -> Float[Array, "n_output d"]:
    """Evaluate a B-spline curve using de Boor's algorithm.

    Args:
        control_points: An array of shape (n_in, d) representing the control points of the B-spline curve.
        n_output: Number of uniformly spaced output points in [0, 1]. Uses compile-time optimized path.
        order: The order of the B-spline curve (default is 4 for cubic).
        ts: Explicit parameter values to evaluate at, shape (n_points,). Uses runtime path.
            Specify either n_output or ts, not both.

    Returns:
        An array of shape (n_output, d) or (n_points, d) representing the points on the B-spline curve.
    """
    if ts is not None and n_output is not None:
        raise ValueError("Specify either n_output or ts, not both.")
    if ts is None and n_output is None:
        raise ValueError("Specify either n_output or ts.")
    return _evaluate(control_points, order, n_output, ts)


@jax.jit(static_argnames=["order", "n_output", "derivative_order", "emit_intermediates"])
def bspline_derivative(
    control_points: Float[Array, "n_in d"],
    n_output: int | None = None,
    order: int = 4,
    *,
    ts: Float[Array, " n_points"] | None = None,
    derivative_order: int = 1,
    emit_intermediates: bool = False,
) -> list[Float[Array, "n_output d"]] | Float[Array, "n_output d"]:
    """Evaluate the derivative of a B-spline curve using de Boor's algorithm.

    Args:
        control_points: An array of shape (n_in, d) representing the control points of the B-spline curve.
        n_output: Number of uniformly spaced output points in [0, 1]. Uses compile-time optimized path.
        order: The order of the B-spline curve (default is 4 for cubic).
        ts: Explicit parameter values to evaluate at, shape (n_points,). Uses runtime path.
            Specify either n_output or ts, not both.
        derivative_order: The order of the derivative to compute (default is 1 for first derivative).
        emit_intermediates: Whether to return all intermediate derivatives (default is False).

    Returns:
        An array of shape (n_output, d) representing the points on the derivative of the B-spline curve.
        If emit_intermediates is True, returns a list of arrays for each derivative order.
    """
    if ts is not None and n_output is not None:
        raise ValueError("Specify either n_output or ts, not both.")
    if ts is None and n_output is None:
        raise ValueError("Specify either n_output or ts.")

    from .deboor import diff_spline
    from .utils import make_uniform_knot_vector

    n = control_points.shape[0] - 1
    T = make_uniform_knot_vector(n, order)
    k_ = order
    Q = control_points

    def assemble(Q, k_):
        return _evaluate(Q, k_, n_output, ts)

    out = []
    for _ in range(derivative_order):
        Q, k_, T = diff_spline(Q, k_, T)
        if emit_intermediates:
            out.append(assemble(Q, k_))
    out = assemble(Q, k_) if not emit_intermediates else out
    return out
