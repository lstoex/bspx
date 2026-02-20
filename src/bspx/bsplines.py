"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .deboor import de_boor, de_boor_static


class BSpline(eqx.Module):
    """B-Spline curve evaluator"""
    n_output: int = eqx.field(static=True)  # number of points to evaluate on the curve
    order: int = eqx.field(static=True, default=4)  # default to cubic B-splines
    use_static: bool = eqx.field(
        static=True, default=True
    )  # whether to use the static version of de Boor's algorithm for maximum efficiency.

    @jax.jit
    def __call__(self, control_points: Float[Array, "n_in d"]) -> Float[Array, "n_output d"]:
        if self.use_static:
            return de_boor_static(control_points, self.order, self.n_output)
        else:
            ts = jnp.linspace(0.0, 1.0, self.n_output)
            return de_boor(control_points, self.order, ts)

    def __str__(self):
        return f"BSpline(n_output={self.n_output}, order={self.order}, use_static={self.use_static})"
