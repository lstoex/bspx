import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from bspx.bsplines import bspline


@pytest.mark.parametrize(
    "n_in, order, n_output",
    [
        (4, 3, 10),  # cubic Bezier
        (5, 4, 20),  # cubic B-spline
        (6, 5, 15),  # quartic B-spline
        (7, 6, 25),  # quintic B-spline
        (3, 2, 5),  # linear B-spline
        (4, 5, 10),  # invalid case: order > n_in + 1, should raise an error
    ],
)
def test_bspline_against_scipy(n_in, order, n_output):
    if order > n_in + 1:
        with pytest.raises(AssertionError):
            control_points = jnp.random.rand(n_in + 1, 2)  # 2D control points
            bspline(control_points, n_output=n_output, order=order, use_static=True)
    else:
        # Create random control points
        control_points = np.random.rand(n_in + 1, 2)  # 2D control points

        # Create our B-spline and evaluate
        result = bspline(control_points, n_output=n_output, order=order, use_static=True)

        # Create SciPy B-spline and evaluate
        k = order
        from bspx.utils_static import get_knots_static

        T = get_knots_static(n_in, order)
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(np.linspace(0, 1, n_output))

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "n_in, order, n_output",
    [
        (4, 3, 10),  # cubic Bezier
        (5, 4, 20),  # cubic B-spline
        (6, 5, 15),  # quartic B-spline
        (7, 6, 25),  # quintic B-spline
        (3, 2, 5),  # linear B-spline
        (4, 5, 10),  # invalid case: order > n_in + 1, should raise an error
    ],
)
def test_bspline_against_scipy_use_static_false(n_in, order, n_output):
    if order > n_in + 1:
        with pytest.raises(AssertionError):
            control_points = jnp.random.rand(n_in + 1, 2)  # 2D control points
            bspline(control_points, n_output=n_output, order=order, use_static=False)
    else:
        # Create random control points
        control_points = np.random.rand(n_in + 1, 2)  # 2D control points

        # Create our B-spline with use_static=False and evaluate
        result = bspline(control_points, n_output=n_output, order=order, use_static=False)

        # Create SciPy B-spline and evaluate
        k = order
        from bspx.utils_static import get_knots_static

        T = get_knots_static(n_in, order)
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(np.linspace(0, 1, n_output))

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
