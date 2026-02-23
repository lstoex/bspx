"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import jax
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
    alg = de_boor_static if use_static else de_boor
    return alg(control_points, order, n_output)
