"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .deboor import de_boor, de_boor_static


@jax.jit(static_argnames=["order", "use_static", "n_output"])
def bspline(
    control_points: Float[Array, "n_in d"], n_output: int, order: int = 4, use_static: bool = True
) -> Float[Array, "n_output d"]:
    """Evaluate a B-spline curve using de Boor's algorithm.
    Args:
        control_points: An array of shape (n_in, d) representing the control points of the B-spline curve.
        n_output: The number of output points to generate on the B-spline curve.
        order: The order of the B-spline curve (default is 4 for cubic).
        use_static: Whether to use the static version of de Boor's algorithm (default is True).
    Returns:
        An array of shape (n_output, d) representing the points on the B-spline curve.
    """
    if use_static:
        ts = np.linspace(0.0, 1.0, n_output)
        return de_boor_static(control_points, order, ts)
    else:
        ts = jnp.linspace(0.0, 1.0, n_output)
        return de_boor(control_points, order, ts)


def bspline_derivative(
    control_points: Float[Array, "n_in d"],
    order: int,
    n_output: int,
    use_static=False,
    derivative_order: int = 1,
    emit_intermediates=False,
) -> list[Float[Array, "n_output d"]] | Float[Array, "n_output d"]:
    """Evaluate the derivative of a B-spline curve using de Boor's algorithm.
    Args:
        control_points: An array of shape (n_in, d) representing the control points of the B-spline curve.
        order: The order of the B-spline curve (default is 4 for cubic).
        n_output: The number of output points to generate on the B-spline curve.
        use_static: Whether to use the static version of de Boor's algorithm (default is False).
        derivative_order: The order of the derivative to compute (default is 1 for first derivative).
    Returns:
        An array of shape (n_output, d) representing the points on the derivative of the B-spline curve.
    """
    from .deboor import diff_spline
    from .utils import make_uniform_knot_vector

    n = control_points.shape[0] - 1
    T = make_uniform_knot_vector(n, order)
    k_ = order
    Q = control_points

    def assemble(Q, k_, T):
        return (
            de_boor_static(Q, k_, np.linspace(0.0, 1.0, n_output))
            if use_static
            else de_boor(Q, k_, jnp.linspace(0.0, 1.0, n_output))
        )

    out = []
    for _ in range(derivative_order):
        Q, k_, T = diff_spline(Q, k_, T)
        if emit_intermediates:
            out.append(assemble(Q, k_, T))
    out = assemble(Q, k_, T) if not emit_intermediates else out
    return out
