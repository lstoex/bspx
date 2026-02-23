import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from bspx.bsplines import bspline


@pytest.mark.parametrize(
    "n_in, order, n_output, static",
    [
        (4, 3, 10, True),  # cubic Bezier
        (5, 4, 20, True),  # cubic B-spline
        (6, 5, 15, True),  # quartic B-spline
        (7, 6, 25, True),  # quintic B-spline
        (3, 2, 5, True),  # linear B-spline
        (4, 5, 10, True),  # invalid case: order > n_in + 1, should raise an error
        (4, 3, 10, False),  # cubic Bezier
        (5, 4, 20, False),  # cubic B-spline
        (6, 5, 15, False),  # quartic B-spline
        (7, 6, 25, False),  # quintic B-spline
        (3, 2, 5, False),  # linear B-spline
        (4, 5, 10, False),  # invalid case: order > n_in + 1, should raise an error
    ],
)
def test_bspline_against_scipy(n_in, order, n_output, static):
    if order > n_in + 1:
        with pytest.raises(AssertionError):
            control_points = jnp.random.rand(n_in + 1, 2)  # 2D control points
            bspline(control_points, n_output=n_output, order=order, use_static=static)
    else:
        # Create random control points
        control_points = np.random.rand(n_in + 1, 2)  # 2D control points

        # Create our B-spline and evaluate
        result = bspline(control_points, n_output=n_output, order=order, use_static=static)

        # Create SciPy B-spline and evaluate
        k = order
        from bspx.utils_static import get_knots_static

        T = get_knots_static(n_in, order)
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(np.linspace(0, 1, n_output))

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize(
    "n_in, order, n_output, static, derivative_order",
    [
        (5, 4, 20, True, 1),  # cubic B-spline first derivative
        (5, 4, 20, False, 1),  # cubic B-spline first derivative with use_static=False
        (6, 5, 15, True, 2),  # quartic B-spline second derivative
        (6, 5, 15, False, 2),  # quartic B-spline second derivative with use_static=False
        (5, 4, 20, False, 3),  # cubic B-spline third derivative (should be piecewise constant)
        (5, 4, 20, True, 4), # cubic B-spline fourth derivative (should throw an error since it's not defined)
    ],
)
def test_bspline_derivative_against_scipy(n_in, order, n_output, static, derivative_order):
    if derivative_order >= order:
        with pytest.raises(AssertionError):
            control_points = np.random.rand(n_in + 1, 2)  # 2D control points
            from bspx.bsplines import bspline_derivative

            bspline_derivative(jnp.array(control_points), order=order, n_output=n_output, use_static=static, derivative_order=derivative_order)
    else:
        # Create random control points
        control_points = np.random.rand(n_in + 1, 2)  # 2D control points

        # Create our B-spline derivative and evaluate
        from bspx.bsplines import bspline_derivative

        result = bspline_derivative(jnp.array(control_points), order=order, n_output=n_output, use_static=static, derivative_order=derivative_order)

        # Create SciPy B-spline and evaluate its derivative
        k = order
        from bspx.utils_static import get_knots_static

        T = get_knots_static(n_in, order)
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(np.linspace(0, 1, n_output), nu=derivative_order)

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
